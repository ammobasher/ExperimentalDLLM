"""
Training script for Episodic-Centric Small Language Model.

This script demonstrates the novel training approach:
1. Pre-train base model (optional, can load pretrained)
2. Freeze model weights
3. Personalize via episodic memory only
4. Trigger sleep consolidation when memory fills
5. Zero catastrophic forgetting

Usage:
    python train_episodic.py --config small --mode personalize
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
from pathlib import Path

from src.model import PCModel
from src.config import Config, ConfigSmall, ConfigMicro
from src.memory_optimized import OptimizedEpisodicMemory
from src.memory_generate import MemoryAugmentedModel, compute_surprise
from src.sleep import SleepConsolidation, SleepScheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train episodic-centric LLM')
    parser.add_argument('--config', type=str, default='small',
                       choices=['base', 'small', 'micro'],
                       help='Model configuration')
    parser.add_argument('--mode', type=str, default='personalize',
                       choices=['pretrain', 'personalize', 'full'],
                       help='Training mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Load model from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of training steps')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    return parser.parse_args()


def get_config(config_name: str):
    """Get configuration by name."""
    configs = {
        'base': Config,
        'small': ConfigSmall,
        'micro': ConfigMicro,
    }
    return configs[config_name]


def pretrain_model(model, data_loader, device, config, steps, log_interval):
    """
    Pre-training phase: Standard training with PC loss.

    This creates the base model with general knowledge.
    """
    print("\n" + "="*60)
    print("PRE-TRAINING PHASE")
    print("="*60)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_llm)

    for step in range(steps):
        # Get batch (placeholder - would use real data loader)
        batch = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)

        # Forward pass
        logits, pc_loss = model(batch, inference=False)

        # Compute cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        ce_loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1)
        )

        # Combined loss (CE + PC)
        loss = ce_loss + 0.1 * pc_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if (step + 1) % log_interval == 0:
            print(f"Step {step+1}/{steps}: Loss={loss.item():.4f}, CE={ce_loss.item():.4f}, PC={pc_loss.item():.4f}")

    print("\n✓ Pre-training complete\n")
    return model


def personalize_model(model, memory, sleep, device, config, steps, log_interval):
    """
    Personalization phase: Frozen model + episodic memory.

    This demonstrates the novel episodic-centric approach:
    - Model weights are frozen
    - New experiences stored in episodic memory
    - Sleep consolidation triggered automatically
    - Zero catastrophic forgetting
    """
    print("\n" + "="*60)
    print("PERSONALIZATION PHASE (Episodic-Centric)")
    print("="*60)

    # Ensure model is frozen
    if not model.is_frozen():
        model.freeze()

    # Create memory-augmented wrapper
    mem_model = MemoryAugmentedModel(model, memory, config)

    # Setup sleep scheduler
    scheduler = SleepScheduler(sleep, strategy='threshold')

    # Statistics
    stats = {
        'memories_added': 0,
        'memories_rejected': 0,
        'sleep_cycles': 0,
        'total_surprise': 0.0,
    }

    for step in range(steps):
        # Simulate user interaction (in real use, this would be actual user queries)
        batch = torch.randint(0, config.vocab_size, (1, config.seq_len), device=device)

        # Compute surprise for this interaction
        surprise = compute_surprise(model, batch)
        stats['total_surprise'] += surprise

        # Try to add to episodic memory
        added = mem_model.add_memory(batch)

        if added:
            stats['memories_added'] += 1
        else:
            stats['memories_rejected'] += 1

        # Check if sleep should be triggered
        sleep_result = scheduler.check_and_consolidate()

        if sleep_result:
            stats['sleep_cycles'] += 1

        # Logging
        if (step + 1) % log_interval == 0:
            mem_stats = memory.get_stats()
            avg_surprise = stats['total_surprise'] / (step + 1)

            print(f"\nStep {step+1}/{steps}:")
            print(f"  Memories: {mem_stats['count']}/{mem_stats['capacity']} ({mem_stats['usage_percent']:.1f}%)")
            print(f"  Added: {stats['memories_added']}, Rejected: {stats['memories_rejected']}")
            print(f"  Avg Surprise: {avg_surprise:.4f}, Threshold: {mem_stats['threshold']:.4f}")
            print(f"  Sleep Cycles: {stats['sleep_cycles']}")
            print(f"  Retrieval Time: {mem_stats['avg_retrieval_time_ms']:.2f}ms")

    print("\n✓ Personalization complete\n")
    print(f"Final Statistics:")
    print(f"  Total memories: {memory.count}")
    print(f"  Sleep cycles: {stats['sleep_cycles']}")
    print(f"  Memories added: {stats['memories_added']}")

    return model, memory


def main():
    args = parse_args()

    print("="*60)
    print("EPISODIC-CENTRIC SMALL LLM TRAINING")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Steps: {args.steps}")
    print("="*60)

    # Get configuration
    config_class = get_config(args.config)
    config = config_class()

    # Initialize model
    device = torch.device(args.device)

    if args.checkpoint:
        print(f"\nLoading model from {args.checkpoint}...")
        model = PCModel(config=config)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("✓ Model loaded")
    else:
        print(f"\nInitializing new model...")
        model = PCModel(config=config)
        print("✓ Model initialized")

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    # Initialize episodic memory
    print(f"\nInitializing episodic memory (capacity: {config.memory_capacity})...")
    memory = OptimizedEpisodicMemory(
        dim=config.memory_dim,
        capacity=config.memory_capacity,
        use_faiss=True
    )
    print("✓ Memory initialized")

    # Initialize sleep consolidation
    print("\nInitializing sleep consolidation...")
    sleep = SleepConsolidation(model, memory, config)
    print("✓ Sleep consolidation initialized")

    # Training modes
    if args.mode == 'pretrain':
        # Pre-training only
        model = pretrain_model(
            model, None, device, config,
            steps=args.steps,
            log_interval=args.log_interval
        )

    elif args.mode == 'personalize':
        # Personalization only (assumes pretrained model)
        model, memory = personalize_model(
            model, memory, sleep, device, config,
            steps=args.steps,
            log_interval=args.log_interval
        )

    elif args.mode == 'full':
        # Full pipeline: pretrain then personalize
        pretrain_steps = args.steps // 2
        personalize_steps = args.steps - pretrain_steps

        # Pre-train
        model = pretrain_model(
            model, None, device, config,
            steps=pretrain_steps,
            log_interval=args.log_interval
        )

        # Personalize
        model, memory = personalize_model(
            model, memory, sleep, device, config,
            steps=personalize_steps,
            log_interval=args.log_interval
        )

    # Save checkpoint
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    model_path = save_dir / f"model_{args.config}_{args.mode}.pt"
    memory_path = save_dir / f"memory_{args.config}_{args.mode}.npz"

    print(f"\nSaving checkpoint...")
    torch.save(model.state_dict(), model_path)
    memory.save(str(memory_path))
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Memory saved to {memory_path}")

    # Final statistics
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model: {n_params/1e6:.1f}M params")
    print(f"Memory: {memory.count} experiences")
    print(f"Sleep cycles: {sleep.stats['total_sleep_cycles']}")
    print("="*60)


if __name__ == "__main__":
    main()
