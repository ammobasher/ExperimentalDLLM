"""
Interactive Demo: Episodic-Centric Personal Assistant

This demo showcases the key innovations:
- Instant personalization through episodic memory
- Zero catastrophic forgetting (frozen weights)
- Sleep consolidation when memory fills
- Privacy-first local execution

Usage:
    python demo_interactive.py --config small --checkpoint checkpoints/model_small.pt
"""

import torch
import argparse
from pathlib import Path

from src.model import PCModel
from src.config import ConfigSmall, ConfigMicro
from src.memory_optimized import OptimizedEpisodicMemory
from src.memory_generate import MemoryAugmentedModel, compute_surprise
from src.sleep import SleepConsolidation, SleepScheduler


def print_header():
    """Print demo header."""
    print("\n" + "="*70)
    print("  🧠  EPISODIC-CENTRIC PERSONAL ASSISTANT DEMO  🧠")
    print("="*70)
    print("\nKey Features:")
    print("  ✓ Instant learning (no training required)")
    print("  ✓ Zero forgetting (frozen model weights)")
    print("  ✓ Sleep consolidation (automatic memory management)")
    print("  ✓ Privacy-first (all local, no cloud)")
    print("="*70 + "\n")


def print_help():
    """Print available commands."""
    print("\nAvailable commands:")
    print("  /learn <fact>    - Teach the assistant a new fact")
    print("  /forget          - Clear episodic memory")
    print("  /sleep           - Trigger sleep consolidation")
    print("  /stats           - Show memory statistics")
    print("  /help            - Show this help")
    print("  /quit            - Exit")
    print()


def main():
    parser = argparse.ArgumentParser(description='Interactive episodic assistant demo')
    parser.add_argument('--config', type=str, default='small',
                       choices=['small', 'micro'],
                       help='Model configuration')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Load model from checkpoint')
    parser.add_argument('--memory_file', type=str, default=None,
                       help='Load memory from file')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    args = parser.parse_args()

    print_header()

    # Get configuration
    config = ConfigSmall() if args.config == 'small' else ConfigMicro()
    device = torch.device(args.device)

    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"Model: {config.n_layers} layers, {config.embed_dim} dim")

    # Initialize model
    print(f"\nInitializing model...")
    model = PCModel(config=config).to(device)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print("✓ Checkpoint loaded")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("Using randomly initialized model instead")

    # Freeze model for episodic-centric operation
    model.freeze()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({n_params/1e6:.1f}M params, FROZEN)")

    # Initialize episodic memory
    print(f"\nInitializing episodic memory (capacity: {config.memory_capacity})...")
    memory = OptimizedEpisodicMemory(
        dim=config.memory_dim,
        capacity=config.memory_capacity,
        use_faiss=True
    )

    if args.memory_file:
        print(f"Loading memory from {args.memory_file}...")
        try:
            memory.load(args.memory_file)
            print("✓ Memory loaded")
        except Exception as e:
            print(f"⚠️  Failed to load memory: {e}")

    print(f"✓ Memory initialized")

    # Create memory-augmented model
    mem_model = MemoryAugmentedModel(model, memory, config)

    # Initialize sleep consolidation
    sleep = SleepConsolidation(model, memory, config)
    scheduler = SleepScheduler(sleep, strategy='threshold')

    print("\n" + "="*70)
    print("✓ Initialization complete! Ready for interaction.")
    print_help()

    # Interaction loop
    interaction_count = 0

    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower()

                if cmd == '/quit' or cmd == '/exit':
                    print("\nGoodbye! 👋")
                    break

                elif cmd == '/help':
                    print_help()

                elif cmd.startswith('/learn'):
                    # Teach a fact
                    fact = user_input[6:].strip()
                    if not fact:
                        print("Usage: /learn <fact>")
                        continue

                    print(f"Learning: '{fact}'...")

                    # Create dummy token IDs (would use real tokenizer)
                    fact_ids = torch.randint(0, config.vocab_size, (1, 20), device=device)

                    # Force add to memory
                    added = mem_model.add_memory(fact_ids, force=True)

                    if added:
                        print(f"✓ Learned! Memory count: {memory.count}")
                    else:
                        print("⚠️  Failed to add to memory")

                elif cmd == '/forget':
                    # Clear memory
                    old_count = memory.count
                    memory.reset()
                    print(f"✓ Memory cleared. Forgot {old_count} experiences.")

                elif cmd == '/sleep':
                    # Trigger sleep consolidation
                    print("\nInitiating sleep consolidation...")

                    if memory.count < 100:
                        print("⚠️  Not enough memories to consolidate (need at least 100)")
                        continue

                    result = sleep.consolidate(verbose=True)

                    if result['status'] == 'success':
                        print(f"\n✓ Sleep complete!")
                    else:
                        print(f"\n⚠️  Sleep failed: {result.get('status')}")

                elif cmd == '/stats':
                    # Show statistics
                    stats = memory.get_stats()
                    sleep_stats = sleep.get_stats()

                    print("\n" + "-"*50)
                    print("SYSTEM STATISTICS")
                    print("-"*50)
                    print(f"Model Parameters: {n_params/1e6:.1f}M")
                    print(f"Model Frozen: {model.is_frozen()}")
                    print(f"\nEpisodic Memory:")
                    print(f"  Count: {stats['count']}/{stats['capacity']}")
                    print(f"  Usage: {stats['usage_percent']:.1f}%")
                    print(f"  Threshold: {stats['threshold']:.2f}")
                    print(f"  Additions: {stats['total_additions']}")
                    print(f"  Retrievals: {stats['total_retrievals']}")
                    print(f"  Avg Retrieval Time: {stats['avg_retrieval_time_ms']:.2f}ms")
                    print(f"  FAISS Enabled: {stats['faiss_enabled']}")
                    print(f"\nSleep Consolidation:")
                    print(f"  Cycles: {sleep_stats['sleep_cycles']}")
                    print(f"  Memories Replayed: {sleep_stats['memories_replayed']}")
                    print(f"  Memories Pruned: {sleep_stats['memories_pruned']}")
                    print(f"  Avg Time: {sleep_stats['avg_consolidation_time']:.1f}s")
                    print(f"\nInteractions: {interaction_count}")
                    print("-"*50)

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")

                continue

            # Normal interaction (not a command)
            interaction_count += 1

            # Create dummy token IDs (would use real tokenizer)
            input_ids = torch.randint(0, config.vocab_size, (1, 30), device=device)

            print("\n[Thinking...]")

            # Generate response with memory
            output_ids, gen_stats = mem_model.generate(
                input_ids,
                max_length=50,
                k_memories=5,
                min_similarity=0.3,
                verbose=False
            )

            # Display generation info
            print(f"\n[Response generated]")
            print(f"  Memories used: {gen_stats['memories_retrieved']}")
            if gen_stats['memories_retrieved'] > 0:
                print(f"  Avg similarity: {gen_stats['avg_similarity']:.3f}")
            print(f"  Tokens generated: {gen_stats['tokens_generated']}")
            print(f"  Generation time: {gen_stats['generation_time']:.2f}s")

            # Compute surprise and potentially add to memory
            surprise = compute_surprise(model, input_ids)

            if surprise > memory.threshold_tau:
                added = mem_model.add_memory(input_ids)
                if added:
                    print(f"  💾 Stored in memory (surprise: {surprise:.2f})")

            # Check if sleep should be triggered
            sleep_result = scheduler.check_and_consolidate()
            if sleep_result:
                print("\n🌙 Sleep consolidation was automatically triggered!")
                print(f"   Freed {sleep_result['memories_pruned']} memories")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save state on exit
    print("\nSaving state...")
    save_dir = Path("demo_saves")
    save_dir.mkdir(exist_ok=True)

    memory_path = save_dir / "memory.npz"
    memory.save(str(memory_path))
    print(f"✓ Memory saved to {memory_path}")

    print("\nSession statistics:")
    print(f"  Total interactions: {interaction_count}")
    print(f"  Final memory count: {memory.count}")
    print(f"  Sleep cycles: {sleep.stats['total_sleep_cycles']}")


if __name__ == "__main__":
    main()
