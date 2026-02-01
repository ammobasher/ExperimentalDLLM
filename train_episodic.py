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
import os

from src.model import PCModel
from src.config import Config, ConfigSmall, ConfigMicro
from src.memory import EpisodicMemory
from src.sleep import SleepConsolidation, SleepScheduler
from src.cached_loader import CachedDataLoader

# --- Helper Classes (Polyfills for missing files) ---

class MemoryAugmentedModel:
    """Wrapper to handle interaction between Model and EpisodicMemory."""
    def __init__(self, model, memory, config):
        self.model = model
        self.memory = memory
        self.config = config
        
    def add_memory(self, batch, surprise_score):
        """Add batch to memory if surprising."""
        with torch.no_grad():
            # Get embedding for storage (mean of sequence)
            embed = self.model(batch, return_embeds=True).mean(dim=1)
            
        # Add to memory
        return self.memory.add(embed[0], batch[0].cpu().tolist(), surprise_score)

def compute_surprise(model, batch):
    """Compute predictive coding loss as surprise metric."""
    with torch.no_grad():
        _, pc_loss = model(batch, inference=False)
    return pc_loss.mean().item()

# ----------------------------------------------------

def get_best_device():
    """Detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'
    return 'cpu'


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
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint directory')
    parser.add_argument('--device', type=str, default=get_best_device(),
                       help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of training steps')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--checkpoint_interval', type=int, default=5000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--cache_dir', type=str, default='cached_data',
                       help='Directory with cached data')
    return parser.parse_args()


def get_config(config_name: str):
    """Get configuration by name."""
    configs = {
        'base': Config,
        'small': ConfigSmall,
        'micro': ConfigMicro,
    }
    return configs[config_name]


def save_checkpoint(model, optimizer, step, save_dir, config_name, mode, data_loader_pos=None):
    """Save training checkpoint for resumption."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config_name': config_name,
        'mode': mode,
        'data_loader_pos': data_loader_pos,
    }
    
    checkpoint_path = save_dir / f"checkpoint_{config_name}_{mode}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save a "latest" symlink/copy for easy resumption
    latest_path = save_dir / f"checkpoint_{config_name}_{mode}_latest.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"💾 Checkpoint saved: step {step} → {checkpoint_path.name}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """Load training checkpoint for resumption."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('step', 0), checkpoint.get('data_loader_pos', 0)


def pretrain_model(model, data_loader, device, config, steps, log_interval, 
                   checkpoint_interval=5000, save_dir='checkpoints', config_name='small',
                   start_step=0, optimizer=None):
    """
    Pre-training phase: Standard training with PC loss.

    This creates the base model with general knowledge.
    Now with intermediate checkpointing for resumption.
    """
    print("\n" + "="*60)
    print("PRE-TRAINING PHASE")
    print("="*60)
    if start_step > 0:
        print(f"Resuming from step {start_step}")

    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_llm)
    
    start_time = time.time()
    total_steps = start_step + steps

    for step in range(start_step, total_steps):
        # Get batch from real data loader
        batch, _ = data_loader.get_train_batch()
        # Advance loader
        data_loader.advance()
        
        # Ensure batch is on device
        batch = batch.to(device)
        if step == start_step:
            print(f"DEBUG: Batch shape: {batch.shape}")

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
            elapsed = time.time() - start_time
            print(f"Step {step+1}/{total_steps}: Loss={loss.item():.4f}, CE={ce_loss.item():.4f}, PC={pc_loss.item():.4f} ({elapsed:.2f}s)")
            start_time = time.time()
        
        # Checkpointing
        if (step + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, step + 1, save_dir, config_name, 'pretrain',
                data_loader_pos=data_loader.text_steps if hasattr(data_loader, 'text_steps') else None
            )

    print("\n✓ Pre-training complete\n")
    return model, optimizer


def personalize_model(model, memory, data_loader, device, config, steps, log_interval):
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
    model.freeze()

    # Create memory-augmented wrapper
    mem_model = MemoryAugmentedModel(model, memory, config)

    # Initialize sleep consolidation
    print("\nInitializing sleep consolidation...")
    sleep = SleepConsolidation(model, memory, config)
    print("✓ Sleep consolidation initialized")

    # Setup sleep scheduler
    scheduler = SleepScheduler(sleep, strategy='threshold')

    # Statistics
    stats = {
        'memories_added': 0,
        'memories_rejected': 0,
        'sleep_cycles': 0,
        'total_surprise': 0.0,
    }

    start_time = time.time()

    for step in range(steps):
        # Simulate user interaction using real data
        batch, _ = data_loader.get_train_batch()
        data_loader.advance()
        
        batch = batch.to(device)

        # Compute surprise for this interaction
        surprise = compute_surprise(model, batch)
        stats['total_surprise'] += surprise

        # Try to add to episodic memory
        added = mem_model.add_memory(batch, surprise)

        if added:
            stats['memories_added'] += 1
        else:
            stats['memories_rejected'] += 1

        # Check if sleep should be triggered
        if config.enable_sleep:
             sleep_result = scheduler.check_and_consolidate()
             if sleep_result:
                 stats['sleep_cycles'] += 1
                 # Note: model is re-frozen automatically by sleep.consolidate

        # Logging
        if (step + 1) % log_interval == 0:
            # Re-calculate usage
            mem_stats = memory.get_stats()
            avg_surprise = stats['total_surprise'] / (step + 1)
            elapsed = time.time() - start_time

            print(f"\nStep {step+1}/{steps}:")
            print(f"  Memories: {mem_stats['count']}/{mem_stats['capacity']} ({mem_stats['usage_percent']:.1f}%)")
            print(f"  Added: {stats['memories_added']}, Rejected: {stats['memories_rejected']}")
            print(f"  Avg Surprise: {avg_surprise:.4f}, Threshold: {mem_stats['threshold']:.4f}")
            print(f"  Sleep Cycles: {stats['sleep_cycles']}")
            
            start_time = time.time()

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
    if args.device == 'mps':
        print("🚀 Using Apple Silicon GPU (MPS) for acceleration!")
    elif args.device == 'cuda':
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Using CPU - training will be slow. Consider using --device mps")
    print(f"Steps: {args.steps}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print("="*60)

    # Get configuration
    config_class = get_config(args.config)
    config = config_class()

    # Initialize model
    device = torch.device(args.device)

    if args.checkpoint:
        print(f"\nLoading model from {args.checkpoint}...")
        model = PCModel(config=config)
        # Handle state dict loading
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
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
    # Using standard EpisodicMemory from src.memory
    memory = EpisodicMemory(
        dim=config.memory_dim,
        capacity=config.memory_capacity,
        threshold=config.memory_threshold
    )
    print("✓ Memory initialized")
    
    # Initialize CachedDataLoader
    print(f"\nInitializing CachedDataLoader from {args.cache_dir}...")
    try:
        data_loader = CachedDataLoader(args.cache_dir, device, target_batch_size=config.batch_size)
        print(f"✓ Data loader initialized with {data_loader.n_text_steps} batches (Batch Size: {config.batch_size})")
    except Exception as e:
        print(f"!! Error initializing data loader: {e}")
        print("Please run cache_data.py first!")
        return

    # Check for resume
    start_step = 0
    optimizer = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / f"checkpoint_{args.config}_{args.mode}_latest.pt"
        if resume_path.exists():
            print(f"\n📂 Resuming from {resume_path}...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_llm)
            start_step, loader_pos = load_checkpoint(resume_path, model, optimizer, device)
            if loader_pos:
                data_loader.text_steps = loader_pos
            print(f"✓ Resumed from step {start_step}")
        else:
            print(f"⚠️  Resume checkpoint not found: {resume_path}")

    # Training modes
    if args.mode == 'pretrain':
        # Pre-training only
        model, optimizer = pretrain_model(
            model, data_loader, device, config,
            steps=args.steps,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            save_dir=args.save_dir,
            config_name=args.config,
            start_step=start_step,
            optimizer=optimizer
        )

    elif args.mode == 'personalize':
        # Personalization only (assumes pretrained model)
        model, memory = personalize_model(
            model, memory, data_loader, device, config,
            steps=args.steps,
            log_interval=args.log_interval
        )

    elif args.mode == 'full':
        # Full pipeline: pretrain then personalize
        pretrain_steps = args.steps // 2
        personalize_steps = args.steps - pretrain_steps

        # Pre-train
        model = pretrain_model(
            model, data_loader, device, config,
            steps=pretrain_steps,
            log_interval=args.log_interval
        )

        # Personalize
        model, memory = personalize_model(
            model, memory, data_loader, device, config,
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
    print("="*60)


if __name__ == "__main__":
    main()
