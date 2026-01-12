"""
Optimized Episodic Memory System with FAISS for fast similarity search.

This module implements the core episodic memory component for the
episodic-centric personalization architecture.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional

# Try to import FAISS, fall back to numpy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[Memory] Warning: FAISS not available. Using numpy fallback (slower).")


class OptimizedEpisodicMemory:
    """
    High-performance episodic memory with FAISS indexing.

    Features:
    - Fast retrieval (<10ms for 50K vectors)
    - Surprise-based storage (adaptive threshold)
    - Access tracking for sleep consolidation priority
    - Automatic index rebuilding
    - Memory pruning for redundancy reduction
    """

    def __init__(self, dim: int, capacity: int = 50000, use_faiss: bool = True):
        self.dim = dim
        self.capacity = capacity
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Storage arrays
        self.keys = np.zeros((capacity, dim), dtype=np.float32)
        self.values = [None] * capacity  # Metadata (text, tokens, etc.)
        self.timestamps = np.zeros(capacity, dtype=np.int64)
        self.access_counts = np.zeros(capacity, dtype=np.int32)
        self.surprise_scores = np.zeros(capacity, dtype=np.float32)

        # FAISS index for fast retrieval
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        else:
            self.index = None

        # Counters and thresholds
        self.count = 0
        self.threshold_tau = 10.0  # Dynamic surprise threshold
        self.total_queries = 0

        # Statistics
        self.stats = {
            'total_additions': 0,
            'total_retrievals': 0,
            'avg_retrieval_time': 0.0,
            'last_prune_count': 0,
        }

    def add(self, vector, metadata, surprise_score: float) -> bool:
        """
        Add a memory if surprise score exceeds threshold.

        Args:
            vector: Embedding vector [dim] (numpy or torch)
            metadata: Associated data (text, tokens, etc.)
            surprise_score: PC loss or other surprise metric

        Returns:
            True if added, False if below threshold
        """
        # Convert torch tensor to numpy if needed
        if hasattr(vector, 'cpu'):
            vector = vector.detach().cpu().numpy()

        # Ensure 1D array
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim > 1:
            vector = vector.flatten()

        # Check surprise threshold
        if surprise_score > self.threshold_tau:
            # Normalize for cosine similarity
            vector = vector / (np.linalg.norm(vector) + 1e-8)

            # Store
            idx = self.count % self.capacity
            self.keys[idx] = vector
            self.values[idx] = metadata
            self.timestamps[idx] = self.count
            self.access_counts[idx] = 0
            self.surprise_scores[idx] = surprise_score

            self.count += 1
            self.stats['total_additions'] += 1

            # Update FAISS index periodically
            if self.use_faiss and self.count % 100 == 0:
                self._rebuild_index()

            # Adaptive threshold (slowly increase when adding)
            self.threshold_tau = 0.95 * self.threshold_tau + 0.05 * surprise_score

            return True
        else:
            # Decay threshold to allow new additions eventually
            self.threshold_tau = 0.99 * self.threshold_tau
            return False

    def retrieve(self, query_vector, k: int = 5, min_similarity: float = 0.0) -> List[Tuple[Any, float]]:
        """
        Fast retrieval using FAISS (<10ms for 50K vectors).

        Args:
            query_vector: Query embedding [dim]
            k: Number of nearest neighbors
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of (metadata, similarity_score) tuples
        """
        if self.count == 0:
            return []

        start_time = time.time()

        # Convert torch tensor to numpy if needed
        if hasattr(query_vector, 'cpu'):
            query_vector = query_vector.detach().cpu().numpy()

        # Ensure correct shape and normalization
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_vector = query_vector.reshape(1, -1)

        # Determine valid memory size
        n = min(self.count, self.capacity)
        k = min(k, n)

        # Retrieve using FAISS or numpy fallback
        if self.use_faiss and self.index.ntotal > 0:
            scores, indices = self.index.search(query_vector, k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            valid_keys = self.keys[:n]
            k_norms = np.linalg.norm(valid_keys, axis=1, keepdims=True) + 1e-8
            keys_norm = valid_keys / k_norms
            scores = keys_norm @ query_vector.T
            scores = scores.flatten()
            indices = np.argsort(scores)[-k:][::-1]
            scores = scores[indices]

        # Update access counts for consolidation priority
        for idx in indices:
            if 0 <= idx < n:
                self.access_counts[idx] += 1

        # Filter by minimum similarity and build results
        results = []
        for i, idx in enumerate(indices):
            if 0 <= idx < n and scores[i] >= min_similarity:
                results.append((self.values[idx], float(scores[i])))

        # Update statistics
        retrieval_time = time.time() - start_time
        self.total_queries += 1
        self.stats['total_retrievals'] += 1
        alpha = 0.1  # Exponential moving average
        self.stats['avg_retrieval_time'] = (
            alpha * retrieval_time + (1 - alpha) * self.stats['avg_retrieval_time']
        )

        return results

    def sample_for_consolidation(self, n: int = 1000, strategy: str = 'priority') -> List[Dict[str, Any]]:
        """
        Sample memories for sleep consolidation.

        Strategies:
        - 'priority': Combine recency, frequency, and surprise
        - 'random': Random sampling
        - 'recent': Most recent memories
        - 'frequent': Most accessed memories

        Args:
            n: Number of memories to sample
            strategy: Sampling strategy

        Returns:
            List of memory dictionaries
        """
        valid_count = min(self.count, self.capacity)
        n = min(n, valid_count)

        if strategy == 'priority':
            # Compute priority scores: recency + frequency + surprise
            recency = self.timestamps[:valid_count] / (self.count + 1)
            max_access = np.max(self.access_counts[:valid_count]) + 1
            frequency = self.access_counts[:valid_count] / max_access
            max_surprise = np.max(self.surprise_scores[:valid_count]) + 1
            surprise = self.surprise_scores[:valid_count] / max_surprise

            # Weighted combination
            priority = 0.3 * recency + 0.4 * frequency + 0.3 * surprise

            # Sample top N
            top_indices = np.argsort(priority)[-n:]

        elif strategy == 'random':
            top_indices = np.random.choice(valid_count, size=n, replace=False)

        elif strategy == 'recent':
            top_indices = np.argsort(self.timestamps[:valid_count])[-n:]

        elif strategy == 'frequent':
            top_indices = np.argsort(self.access_counts[:valid_count])[-n:]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return [
            {
                'embedding': self.keys[idx],
                'metadata': self.values[idx],
                'timestamp': int(self.timestamps[idx]),
                'access_count': int(self.access_counts[idx]),
                'surprise_score': float(self.surprise_scores[idx]),
            }
            for idx in top_indices
        ]

    def prune_redundant(self, similarity_threshold: float = 0.95, batch_size: int = 100) -> int:
        """
        Remove redundant memories (too similar to others).

        This is expensive (O(N²) comparisons) but only runs during sleep.

        Args:
            similarity_threshold: Cosine similarity threshold for redundancy
            batch_size: Process in batches to save memory

        Returns:
            Number of memories after pruning
        """
        valid_count = min(self.count, self.capacity)

        if valid_count == 0:
            return 0

        print(f"[Memory] Pruning redundant memories from {valid_count}...")

        keep_mask = np.ones(valid_count, dtype=bool)

        # Process in batches to avoid memory issues
        for i in range(0, valid_count, batch_size):
            end_i = min(i + batch_size, valid_count)
            batch_keys = self.keys[i:end_i]

            # Compute similarities with all other memories
            similarities = batch_keys @ self.keys[:valid_count].T

            for j in range(end_i - i):
                global_j = i + j
                if not keep_mask[global_j]:
                    continue

                # Find similar memories
                similar = np.where(similarities[j] > similarity_threshold)[0]

                # Keep the one with highest access count
                for k in similar:
                    if k != global_j and keep_mask[k]:
                        if self.access_counts[k] < self.access_counts[global_j]:
                            keep_mask[k] = False
                        elif self.access_counts[k] == self.access_counts[global_j] and k > global_j:
                            # Tie-break: keep older memory
                            keep_mask[k] = False

        # Compact memory
        new_count = np.sum(keep_mask)
        self.keys[:new_count] = self.keys[:valid_count][keep_mask]
        self.values[:new_count] = [self.values[i] for i in range(valid_count) if keep_mask[i]]
        self.timestamps[:new_count] = self.timestamps[:valid_count][keep_mask]
        self.access_counts[:new_count] = self.access_counts[:valid_count][keep_mask]
        self.surprise_scores[:new_count] = self.surprise_scores[:valid_count][keep_mask]

        # Clear old entries
        self.keys[new_count:valid_count] = 0
        self.values[new_count:valid_count] = [None] * (valid_count - new_count)
        self.timestamps[new_count:valid_count] = 0
        self.access_counts[new_count:valid_count] = 0
        self.surprise_scores[new_count:valid_count] = 0

        old_count = self.count
        self.count = new_count
        self.stats['last_prune_count'] = old_count - new_count

        # Rebuild index
        if self.use_faiss:
            self._rebuild_index()

        print(f"[Memory] Pruned {old_count - new_count} memories. New count: {new_count}")

        return new_count

    def _rebuild_index(self):
        """Rebuild FAISS index with current memories."""
        if not self.use_faiss:
            return

        n = min(self.count, self.capacity)
        if n == 0:
            return

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.keys[:n])

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        valid_count = min(self.count, self.capacity)
        usage_pct = (valid_count / self.capacity) * 100

        return {
            'count': valid_count,
            'capacity': self.capacity,
            'usage_percent': usage_pct,
            'threshold': self.threshold_tau,
            'total_additions': self.stats['total_additions'],
            'total_retrievals': self.stats['total_retrievals'],
            'avg_retrieval_time_ms': self.stats['avg_retrieval_time'] * 1000,
            'last_prune_count': self.stats['last_prune_count'],
            'faiss_enabled': self.use_faiss,
        }

    def save(self, path: str):
        """Save memory to disk."""
        valid_count = min(self.count, self.capacity)
        np.savez_compressed(
            path,
            keys=self.keys[:valid_count],
            values=np.array(self.values[:valid_count], dtype=object),
            timestamps=self.timestamps[:valid_count],
            access_counts=self.access_counts[:valid_count],
            surprise_scores=self.surprise_scores[:valid_count],
            count=self.count,
            threshold=self.threshold_tau,
            stats=self.stats,
        )
        print(f"[Memory] Saved {valid_count} memories to {path}")

    def load(self, path: str):
        """Load memory from disk."""
        data = np.load(path, allow_pickle=True)

        loaded_count = len(data['keys'])
        self.keys[:loaded_count] = data['keys']
        self.values[:loaded_count] = data['values'].tolist()
        self.timestamps[:loaded_count] = data['timestamps']
        self.access_counts[:loaded_count] = data['access_counts']
        self.surprise_scores[:loaded_count] = data['surprise_scores']
        self.count = int(data['count'])
        self.threshold_tau = float(data['threshold'])

        if 'stats' in data:
            self.stats = data['stats'].item()

        # Rebuild index
        if self.use_faiss:
            self._rebuild_index()

        print(f"[Memory] Loaded {loaded_count} memories from {path}")

    def reset(self):
        """Clear all memories (for testing)."""
        self.keys[:] = 0
        self.values[:] = [None] * self.capacity
        self.timestamps[:] = 0
        self.access_counts[:] = 0
        self.surprise_scores[:] = 0
        self.count = 0
        self.threshold_tau = 10.0

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dim)

        print("[Memory] Reset complete.")


# Backwards compatibility alias
class EpisodicMemory(OptimizedEpisodicMemory):
    """Alias for backwards compatibility."""
    pass
