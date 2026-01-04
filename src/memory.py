import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

class EpisodicMemory:
    """
    Simulated Hippocampus (eLTM).
    Stores vectors based on 'Novelty' (Prediction Error).
    Simple In-Memory Vector Store using Cosine Similarity.
    """
    def __init__(self, key: jax.random.PRNGKey, dim: int = 512, capacity: int = 10000):
        self.dim = dim
        self.capacity = capacity
        # We use numpy for storage to allow dynamic growth/cpu access easily
        self.keys = np.zeros((capacity, dim), dtype=np.float32)
        self.values = [] # Metadata list
        self.count = 0
        self.threshold_tau = 0.5 # Dynamic threshold, updateable

    def add(self, vector: np.ndarray, metadata: str, loss_val: float):
        """
        Add item if it meets novelty rule.
        Novelty Rule: Loss > Threshold.
        """
        # Auto-update threshold (Moving average)
        if loss_val > self.threshold_tau:
            # Commit to memory
            idx = self.count % self.capacity
            self.keys[idx] = vector
            # If overwriting, replace metadata
            if idx < len(self.values):
                self.values[idx] = metadata
            else:
                self.values.append(metadata)
            
            self.count += 1
            
            # Raise threshold slightly to habituate
            self.threshold_tau = 0.95 * self.threshold_tau + 0.05 * loss_val
            return True
        else:
            # Lower threshold slightly to avoid stagnation
            self.threshold_tau = 0.99 * self.threshold_tau
            return False

    def retrieve(self, query: np.ndarray, k: int = 3):
        """
        Retrieve top-k similar memories.
        """
        if self.count == 0:
            return []
            
        # Limit search to filled area
        valid_keys = self.keys[:min(self.count, self.capacity)]
        
        # Cosine Sim
        # Norms
        q_norm = query / (np.linalg.norm(query) + 1e-8)
        k_norm = valid_keys / (np.linalg.norm(valid_keys, axis=1, keepdims=True) + 1e-8)
        
        scores = np.dot(k_norm, q_norm)
        
        # Top K
        if len(scores) < k:
            k = len(scores)
            
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.values[idx], scores[idx]))
            
        return results

    def get_stats(self):
        return {
            "count": self.count,
            "threshold": self.threshold_tau
        }

    def save(self, path: str):
        np.savez(path, keys=self.keys, values=self.values, count=self.count, threshold=self.threshold_tau)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.keys = data['keys']
        self.values = data['values'].tolist()
        self.count = int(data['count'])
        self.threshold_tau = float(data['threshold'])

