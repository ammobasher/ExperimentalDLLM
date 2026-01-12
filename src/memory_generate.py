"""
Memory-Augmented Generation for Episodic-Centric Personalization.

This module implements text generation using frozen model + episodic memory retrieval.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import time


def generate_with_memory(
    model,
    memory,
    input_ids: torch.Tensor,
    max_length: int = 100,
    k_memories: int = 5,
    min_similarity: float = 0.5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    verbose: bool = False
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate text using frozen model + episodic memory augmentation.

    Process:
    1. Encode input prompt to get query embedding
    2. Retrieve relevant memories from episodic memory
    3. Augment input context with memory embeddings
    4. Generate using frozen model (no gradient updates!)

    Args:
        model: Frozen PCModel
        memory: OptimizedEpisodicMemory instance
        input_ids: Input token IDs [batch_size, seq_len]
        max_length: Maximum sequence length to generate
        k_memories: Number of memories to retrieve
        min_similarity: Minimum similarity threshold for retrieval
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        verbose: Print debug information

    Returns:
        (generated_ids, generation_stats)
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    if batch_size > 1:
        raise NotImplementedError("Batch generation not yet supported")

    start_time = time.time()
    stats = {
        'memories_retrieved': 0,
        'avg_similarity': 0.0,
        'generation_time': 0.0,
        'tokens_generated': 0,
    }

    # Ensure model is in eval mode and frozen
    model.eval()
    if not model.is_frozen():
        print("[Warning] Model is not frozen. Freezing now for inference.")
        model.freeze()

    with torch.no_grad():
        # 1. Encode input to get query embedding
        query_embedding = model.encode(input_ids)  # [embed_dim]

        # 2. Retrieve relevant memories
        memories = memory.retrieve(
            query_embedding.cpu().numpy(),
            k=k_memories,
            min_similarity=min_similarity
        )

        stats['memories_retrieved'] = len(memories)
        if memories:
            stats['avg_similarity'] = sum(score for _, score in memories) / len(memories)

        if verbose:
            print(f"[Generate] Retrieved {len(memories)} memories")
            for i, (mem_data, score) in enumerate(memories):
                print(f"  Memory {i+1}: similarity={score:.3f}")

        # 3. Augment context with memory embeddings
        if memories:
            # Extract memory embeddings
            memory_embeds = []
            for mem_data, score in memories:
                # If metadata is a dict with embedding, use it
                if isinstance(mem_data, dict) and 'embedding' in mem_data:
                    mem_emb = torch.tensor(mem_data['embedding'], dtype=torch.float32, device=device)
                # Otherwise encode the text/tokens (placeholder - needs implementation)
                else:
                    # For now, skip if we can't get embedding
                    continue
                memory_embeds.append(mem_emb)

            if memory_embeds:
                # Stack memory embeddings [k, embed_dim]
                memory_embeds = torch.stack(memory_embeds)

                # Get input embeddings
                input_embeds = model.embedding(input_ids)  # [batch, seq_len, embed_dim]

                # Concatenate memories before input: [mem1, mem2, ..., input]
                # Add batch dimension to memories [1, k, embed_dim]
                memory_embeds = memory_embeds.unsqueeze(0)

                # Concatenate [batch, k + seq_len, embed_dim]
                augmented_embeds = torch.cat([memory_embeds, input_embeds], dim=1)
            else:
                augmented_embeds = model.embedding(input_ids)
        else:
            augmented_embeds = model.embedding(input_ids)

        # 4. Generate tokens autoregressively
        current_embeds = augmented_embeds
        generated_ids = input_ids.clone()

        for step in range(max_length):
            # Forward pass
            logits, _ = model(inputs_embeds=current_embeds, inference=True)

            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token (assuming token_id=2 for EOS)
            if next_token.item() == 2:
                break

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            stats['tokens_generated'] += 1

            # Get embedding for next token
            next_embed = model.embedding(next_token)

            # Append to current embeddings
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

            # Check max length
            if generated_ids.shape[1] >= max_length:
                break

    stats['generation_time'] = time.time() - start_time

    if verbose:
        print(f"[Generate] Generated {stats['tokens_generated']} tokens in {stats['generation_time']:.2f}s")
        print(f"[Generate] Tokens/sec: {stats['tokens_generated'] / stats['generation_time']:.1f}")

    return generated_ids, stats


def compute_surprise(model, input_ids: torch.Tensor) -> float:
    """
    Compute surprise score (PC loss) for an input.

    This is used to determine if an interaction should be added to episodic memory.

    Args:
        model: PCModel
        input_ids: Input token IDs [batch_size, seq_len]

    Returns:
        Surprise score (PC loss value)
    """
    model.eval()
    with torch.no_grad():
        _, pc_loss = model(input_ids, inference=False)
        return pc_loss.item()


def memory_aware_forward(
    model,
    memory,
    input_ids: torch.Tensor,
    k_memories: int = 5,
    min_similarity: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with memory augmentation (for training/evaluation).

    Similar to generate_with_memory but returns logits instead of generating.

    Args:
        model: PCModel
        memory: OptimizedEpisodicMemory
        input_ids: Input token IDs
        k_memories: Number of memories to retrieve
        min_similarity: Minimum similarity threshold

    Returns:
        (logits, pc_loss)
    """
    device = input_ids.device

    with torch.no_grad():
        # Encode input to get query
        query_embedding = model.encode(input_ids)

        # Retrieve memories
        memories = memory.retrieve(
            query_embedding.cpu().numpy(),
            k=k_memories,
            min_similarity=min_similarity
        )

        # Augment with memory embeddings
        if memories:
            memory_embeds = []
            for mem_data, score in memories:
                if isinstance(mem_data, dict) and 'embedding' in mem_data:
                    mem_emb = torch.tensor(mem_data['embedding'], dtype=torch.float32, device=device)
                    memory_embeds.append(mem_emb)

            if memory_embeds:
                memory_embeds = torch.stack(memory_embeds).unsqueeze(0)
                input_embeds = model.embedding(input_ids)
                augmented_embeds = torch.cat([memory_embeds, input_embeds], dim=1)
            else:
                augmented_embeds = model.embedding(input_ids)
        else:
            augmented_embeds = model.embedding(input_ids)

    # Forward pass
    logits, pc_loss = model(inputs_embeds=augmented_embeds, inference=False)

    return logits, pc_loss


class MemoryAugmentedModel:
    """
    Wrapper class that combines frozen model + episodic memory.

    This provides a clean interface for personalized inference.
    """

    def __init__(self, model, memory, config):
        self.model = model
        self.memory = memory
        self.config = config

        # Ensure model is frozen
        if not model.is_frozen():
            model.freeze()

    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Generate with memory augmentation."""
        return generate_with_memory(
            self.model,
            self.memory,
            input_ids,
            k_memories=kwargs.get('k_memories', 5),
            min_similarity=kwargs.get('min_similarity', 0.5),
            max_length=kwargs.get('max_length', 100),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            top_k=kwargs.get('top_k', 50),
            verbose=kwargs.get('verbose', False),
        )

    def add_memory(self, input_ids: torch.Tensor, force: bool = False) -> bool:
        """
        Add interaction to episodic memory if sufficiently surprising.

        Args:
            input_ids: Input token IDs
            force: Force addition regardless of surprise

        Returns:
            True if added, False otherwise
        """
        # Compute surprise
        surprise_score = compute_surprise(self.model, input_ids)

        if force:
            surprise_score = max(surprise_score, self.memory.threshold_tau + 1.0)

        # Get embedding
        embedding = self.model.encode(input_ids)

        # Store
        metadata = {
            'input_ids': input_ids.cpu().numpy(),
            'embedding': embedding.cpu().numpy(),
            'surprise': surprise_score,
        }

        return self.memory.add(embedding.cpu().numpy(), metadata, surprise_score)

    def should_sleep(self) -> bool:
        """Check if sleep consolidation should be triggered."""
        if not self.config.enable_sleep:
            return False

        usage = self.memory.count / self.memory.capacity
        return usage > self.config.sleep_trigger_threshold

    def get_stats(self) -> Dict:
        """Get combined model + memory statistics."""
        return {
            'model_frozen': self.model.is_frozen(),
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'memory_stats': self.memory.get_stats(),
        }
