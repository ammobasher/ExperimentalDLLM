import numpy as np

class EpisodicMemory:
    """
    Episodic Memory for storing high-surprise events.
    PyTorch Version: Expects inputs to be torch Tensors or Numpy arrays.
    """
    def __init__(self, dim: int, capacity: int = 1000, threshold: float = 10.0):
        self.dim = dim
        self.capacity = capacity
        # Storage
        self.keys = np.zeros((capacity, dim), dtype=np.float32)
        # Values store metadata (e.g. text/tokens)
        self.values = [None] * capacity
        
        self.count = 0
        self.threshold_tau = threshold # Dynamic threshold for surprise
        
    def add(self, vector, metadata, loss_val: float):
        """
        Add a memory if loss > threshold.
        vector: [Dim] (Numpy or Torch)
        metadata: Arbitrary (e.g. text string or tokens)
        """
        # Convert Torch -> Numpy if needed
        if hasattr(vector, 'cpu'):
            vector = vector.detach().cpu().numpy()
            
        if loss_val > self.threshold_tau:
            # Commit to memory
            idx = self.count % self.capacity
            self.keys[idx] = vector
            self.values[idx] = metadata
            self.count += 1
            
            # Dynamic Threshold Update (Slowly increase or adapt)
            # If we add, we make it harder to add next time? Or average?
            # Standard: Moving average of recent losses.
            self.threshold_tau = 0.95 * self.threshold_tau + 0.05 * loss_val
            return True
        else:
            # Decay threshold slightly to allow new things eventually
            self.threshold_tau = 0.99 * self.threshold_tau
            return False

    def retrieve(self, query_vector, k=1):
        """
        Retrieve k nearest neighbors.
        """
        if self.count == 0:
            return []
            
        # Convert Torch -> Numpy
        if hasattr(query_vector, 'cpu'):
            query_vector = query_vector.detach().cpu().numpy()
            
        # Simple Dot Product / Cosine
        # Normalize Query
        q = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Valid Memory (up to capacity or count)
        n = min(self.count, self.capacity)
        keys = self.keys[:n]
        
        # Normalize Keys
        k_norms = np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
        keys_norm = keys / k_norms
        
        # Scores
        scores = keys_norm @ q
        
        # Top-K
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((self.values[idx], scores[idx]))
            
        return results

    def save(self, path: str):
        # Convert values to object array to handle variable-length data
        values_arr = np.empty(len(self.values), dtype=object)
        values_arr[:] = self.values
        np.savez(path, keys=self.keys, values=values_arr, count=self.count, threshold=self.threshold_tau)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.keys = data['keys']
        self.values = data['values'].tolist()
        self.count = int(data['count'])
        self.threshold_tau = float(data['threshold'])

    def sample_for_consolidation(self, n=100, strategy='random'):
        """Sample memories for sleep consolidation."""
        if self.count == 0:
            return []
            
        n = min(n, self.count)
        indices = np.random.choice(self.count, n, replace=False)
        
        # Return list of memory values (the content/tokens)
        sampled = []
        for idx in indices:
            sampled.append(self.values[idx])
        return sampled

    def prune_redundant(self, similarity_threshold=0.95):
        """Prune memories that are too similar (placeholder)."""
        # Full implementation would use faiss/clustering.
        # For now, we simulate pruning by removing oldest if full.
        # Returning current count for now.
        return self.count

    def get_stats(self):
        return {
            'count': self.count,
            'capacity': self.capacity,
            'threshold': self.threshold_tau,
            'usage_percent': (self.count / self.capacity) * 100 if self.capacity > 0 else 0,
            'avg_retrieval_time_ms': 0.0 # Placeholder
        }
