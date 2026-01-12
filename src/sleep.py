"""
Sleep-Based Consolidation for Episodic-Centric Architecture.

This module implements offline memory consolidation that updates model weights
without catastrophic forgetting. It mimics biological sleep where hippocampal
memories are replayed and consolidated into neocortical long-term storage.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import time
import numpy as np


class SleepConsolidation:
    """
    Implements sleep-based memory consolidation.

    Process:
    1. Sample high-priority memories (frequent, recent, surprising)
    2. Unfreeze model temporarily
    3. Replay memories with PC-guided learning
    4. Prune redundant memories
    5. Re-freeze model

    This allows the model to consolidate episodic knowledge into weights
    without online gradient updates that cause catastrophic forgetting.
    """

    def __init__(self, model, memory, config):
        """
        Initialize sleep consolidation.

        Args:
            model: PCModel instance
            memory: OptimizedEpisodicMemory instance
            config: Configuration object (Config, ConfigSmall, or ConfigMicro)
        """
        self.model = model
        self.memory = memory
        self.config = config

        # Consolidation statistics
        self.stats = {
            'total_sleep_cycles': 0,
            'total_memories_replayed': 0,
            'total_memories_pruned': 0,
            'avg_consolidation_time': 0.0,
            'last_pc_loss': 0.0,
            'last_reconstruction_loss': 0.0,
        }

    def should_sleep(self) -> bool:
        """
        Determine if sleep consolidation should be triggered.

        Triggers when:
        1. Memory usage exceeds threshold (default 80%)
        2. Sleep is enabled in config
        """
        if not self.config.enable_sleep:
            return False

        usage = self.memory.count / self.memory.capacity
        return usage > self.config.sleep_trigger_threshold

    def consolidate(
        self,
        n_replay: Optional[int] = None,
        n_epochs: Optional[int] = None,
        strategy: str = 'priority',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Main consolidation algorithm.

        Args:
            n_replay: Number of memories to replay (default from config)
            n_epochs: Number of training epochs (default from config)
            strategy: Sampling strategy ('priority', 'random', 'recent', 'frequent')
            verbose: Print progress information

        Returns:
            Dictionary with consolidation statistics
        """
        if verbose:
            print("\n" + "="*60)
            print("🌙 SLEEP CONSOLIDATION INITIATED")
            print("="*60)

        start_time = time.time()

        # Use config defaults if not specified
        n_replay = n_replay or self.config.sleep_replay_samples
        n_epochs = n_epochs or self.config.sleep_epochs

        # Get initial statistics
        initial_memory_count = self.memory.count
        initial_memory_usage = (initial_memory_count / self.memory.capacity) * 100

        if verbose:
            print(f"Memory usage: {initial_memory_count}/{self.memory.capacity} ({initial_memory_usage:.1f}%)")
            print(f"Replay samples: {n_replay}, Epochs: {n_epochs}, Strategy: {strategy}")

        # 1. Sample memories for replay
        memories = self.memory.sample_for_consolidation(n=n_replay, strategy=strategy)

        if len(memories) == 0:
            if verbose:
                print("⚠️  No memories to consolidate. Sleep cancelled.")
            return {'status': 'no_memories'}

        if verbose:
            print(f"✓ Sampled {len(memories)} memories for replay")

        # 2. Unfreeze model
        was_frozen = self.model.is_frozen()
        if was_frozen:
            self.model.unfreeze()
            if verbose:
                print("✓ Model unfrozen for consolidation")

        # 3. Setup optimizer (low learning rate for careful updates)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr_sleep,
            weight_decay=0.01
        )

        # 4. Replay memories (multiple epochs)
        total_pc_loss = 0.0
        total_recon_loss = 0.0

        self.model.train()

        for epoch in range(n_epochs):
            epoch_pc_loss = 0.0
            epoch_recon_loss = 0.0
            n_batches = 0

            # Shuffle memories each epoch
            np.random.shuffle(memories)

            # Process in batches
            for i in range(0, len(memories), self.config.batch_size):
                batch = memories[i:i+self.config.batch_size]

                # Extract embeddings from memories
                embeddings = torch.tensor(
                    [m['embedding'] for m in batch],
                    dtype=torch.float32,
                    device=next(self.model.parameters()).device
                )

                # Forward pass with PC loss
                logits, pc_loss = self.model(
                    inputs_embeds=embeddings.unsqueeze(1),  # Add seq dimension
                    inference=False
                )

                # Reconstruction loss: Try to reconstruct the same embedding
                # This ensures the model learns the patterns in the memories
                recon_embedding = logits.mean(dim=1)  # Mean pool over sequence
                recon_loss = torch.nn.functional.mse_loss(recon_embedding, embeddings)

                # Combined loss: Task + PC consistency
                # PC loss ensures hierarchical consistency
                # Reconstruction loss ensures pattern learning
                loss = recon_loss + 0.1 * pc_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                # Track losses
                epoch_pc_loss += pc_loss.item()
                epoch_recon_loss += recon_loss.item()
                n_batches += 1

            # Epoch statistics
            avg_pc_loss = epoch_pc_loss / n_batches if n_batches > 0 else 0
            avg_recon_loss = epoch_recon_loss / n_batches if n_batches > 0 else 0

            total_pc_loss += avg_pc_loss
            total_recon_loss += avg_recon_loss

            if verbose:
                print(f"  Epoch {epoch+1}/{n_epochs}: "
                      f"PC Loss={avg_pc_loss:.4f}, Recon Loss={avg_recon_loss:.4f}")

        # Average over epochs
        avg_pc_loss = total_pc_loss / n_epochs if n_epochs > 0 else 0
        avg_recon_loss = total_recon_loss / n_epochs if n_epochs > 0 else 0

        if verbose:
            print(f"✓ Consolidation training complete")

        # 5. Prune redundant memories
        if verbose:
            print(f"🧹 Pruning redundant memories...")

        final_memory_count = self.memory.prune_redundant(similarity_threshold=0.95)
        memories_pruned = initial_memory_count - final_memory_count

        if verbose:
            print(f"✓ Pruned {memories_pruned} redundant memories")
            print(f"  Memory count: {initial_memory_count} → {final_memory_count}")

        # 6. Re-freeze model
        if was_frozen:
            self.model.freeze()
            if verbose:
                print("✓ Model re-frozen")

        # Calculate final statistics
        consolidation_time = time.time() - start_time
        final_memory_usage = (final_memory_count / self.memory.capacity) * 100

        # Update running statistics
        self.stats['total_sleep_cycles'] += 1
        self.stats['total_memories_replayed'] += len(memories)
        self.stats['total_memories_pruned'] += memories_pruned
        self.stats['last_pc_loss'] = avg_pc_loss
        self.stats['last_reconstruction_loss'] = avg_recon_loss

        # Update average consolidation time
        alpha = 0.2
        self.stats['avg_consolidation_time'] = (
            alpha * consolidation_time +
            (1 - alpha) * self.stats['avg_consolidation_time']
        )

        result = {
            'status': 'success',
            'memories_replayed': len(memories),
            'memories_pruned': memories_pruned,
            'initial_memory_count': initial_memory_count,
            'final_memory_count': final_memory_count,
            'memory_usage_before': initial_memory_usage,
            'memory_usage_after': final_memory_usage,
            'pc_loss': avg_pc_loss,
            'reconstruction_loss': avg_recon_loss,
            'consolidation_time': consolidation_time,
            'epochs': n_epochs,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ SLEEP CONSOLIDATION COMPLETE")
            print(f"{'='*60}")
            print(f"Time: {consolidation_time:.1f}s")
            print(f"Memory usage: {initial_memory_usage:.1f}% → {final_memory_usage:.1f}%")
            print(f"Memories freed: {memories_pruned}")
            print(f"PC Loss: {avg_pc_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}")
            print(f"{'='*60}\n")

        return result

    def force_consolidate(self, **kwargs) -> Dict[str, Any]:
        """
        Force consolidation even if threshold not reached.

        Useful for testing or manual triggering.
        """
        return self.consolidate(**kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            'sleep_cycles': self.stats['total_sleep_cycles'],
            'memories_replayed': self.stats['total_memories_replayed'],
            'memories_pruned': self.stats['total_memories_pruned'],
            'avg_consolidation_time': self.stats['avg_consolidation_time'],
            'last_pc_loss': self.stats['last_pc_loss'],
            'last_reconstruction_loss': self.stats['last_reconstruction_loss'],
        }

    def estimate_time(self, n_replay: Optional[int] = None, n_epochs: Optional[int] = None) -> float:
        """
        Estimate consolidation time based on previous cycles.

        Returns:
            Estimated time in seconds
        """
        if self.stats['avg_consolidation_time'] == 0:
            # No previous data, use rough estimate
            n_replay = n_replay or self.config.sleep_replay_samples
            n_epochs = n_epochs or self.config.sleep_epochs
            # Rough estimate: 0.1s per sample per epoch
            return (n_replay * n_epochs * 0.1)
        else:
            return self.stats['avg_consolidation_time']


def validate_consolidation(
    model,
    memory,
    test_data,
    before_stats: Dict[str, Any],
    after_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate that sleep consolidation didn't cause catastrophic forgetting.

    Args:
        model: PCModel
        memory: OptimizedEpisodicMemory
        test_data: Test dataset for validation
        before_stats: Model performance before consolidation
        after_stats: Model performance after consolidation

    Returns:
        Validation results
    """
    # This would compare performance metrics before/after sleep
    # For now, return placeholder
    return {
        'forgetting_score': 0.0,  # 0 = no forgetting
        'consolidation_quality': 1.0,  # 1 = perfect
        'memory_efficiency': after_stats['final_memory_count'] / before_stats['initial_memory_count']
    }


class SleepScheduler:
    """
    Automatic sleep scheduling based on various criteria.

    Strategies:
    - 'threshold': Trigger when memory exceeds threshold
    - 'periodic': Trigger every N interactions
    - 'adaptive': Adjust based on performance metrics
    """

    def __init__(self, sleep_consolidation: SleepConsolidation, strategy: str = 'threshold'):
        self.sleep = sleep_consolidation
        self.strategy = strategy
        self.interaction_count = 0
        self.last_sleep_interaction = 0

    def check_and_consolidate(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check if consolidation should be triggered and execute if needed.

        Args:
            force: Force consolidation regardless of criteria

        Returns:
            Consolidation results if triggered, None otherwise
        """
        self.interaction_count += 1

        if force:
            return self.sleep.consolidate()

        if self.strategy == 'threshold':
            if self.sleep.should_sleep():
                return self.sleep.consolidate()

        elif self.strategy == 'periodic':
            # Trigger every 1000 interactions
            if self.interaction_count - self.last_sleep_interaction >= 1000:
                self.last_sleep_interaction = self.interaction_count
                return self.sleep.consolidate()

        elif self.strategy == 'adaptive':
            # Adaptive: Consider both memory usage and interaction count
            memory_usage = self.sleep.memory.count / self.sleep.memory.capacity
            time_since_sleep = self.interaction_count - self.last_sleep_interaction

            if memory_usage > 0.7 or time_since_sleep > 2000:
                self.last_sleep_interaction = self.interaction_count
                return self.sleep.consolidate()

        return None
