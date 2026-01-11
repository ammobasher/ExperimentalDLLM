import numpy as np

class EpisodicMemory:
    """
    Episodic Memory for storing high-surprise events.
    PyTorch Version: Expects inputs to be torch Tensors or Numpy arrays.
    """
    def __init__(self, dim: int, capacity: int = 1000):
        self.dim = dim
        self.capacity = capacity
        # Storage
        self.keys = np.zeros((capacity, dim), dtype=np.float32)
        # Values store metadata (e.g. text/tokens)
        self.values = [None] * capacity
        
        self.count = 0
        self.threshold_tau = 10.0 # Dynamic threshold for surprise
        
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
        np.savez(path, keys=self.keys, values=self.values, count=self.count, threshold=self.threshold_tau)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.keys = data['keys']
        self.values = data['values'].tolist()
        self.count = int(data['count'])
        self.threshold_tau = float(data['threshold'])
