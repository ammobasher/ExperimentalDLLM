import torch
import numpy as np
import os

class CachedDataLoader:
    """
    Fast data loader from pre-cached npz files.
    Supports sharded lazy loading for large datasets (Rolling Cache).
    """
    
    def __init__(self, cache_dir, device, target_batch_size=4):
        self.device = device
        self.step = 0
        self.cache_dir = cache_dir
        self.target_batch_size = target_batch_size
        
        # --- Vision Cache (Load Fully - usually small) ---
        vision_path = os.path.join(cache_dir, "vision_cache.npz")
        self.has_vision = False
        if os.path.exists(vision_path):
            print(f">> Loading vision cache from {vision_path}...")
            try:
                data = np.load(vision_path)
                self.vision_latents = data['latents']  # [N, B, C, H, W]
                self.vision_input_ids = data['input_ids']  # [N, B, SeqLen]
                self.n_vision_steps = len(self.vision_latents)
                self.has_vision = True
                print(f"   Loaded {self.n_vision_steps} vision batches")
            except Exception as e:
                print(f"!! Failed to load vision cache: {e}")

        # --- Text Cache (Sharded Lazy Load) ---
        # 1. Look for monolithic legacy cache first
        legacy_path = os.path.join(cache_dir, "text_cache.npz")
        self.sharded_mode = False
        self.chunk_files = []
        self.current_chunk_idx = -1
        self.current_chunk_data = None
        self.current_chunk_offset = 0
        self.total_samples_approx = 0
        
        if os.path.exists(legacy_path):
            print(f">> Loading legacy text cache from {legacy_path}...")
            data = np.load(legacy_path)
            raw = data['batches'] # [N, B, SeqLen]
            self.current_chunk_data = raw.reshape(-1, raw.shape[-1]) # [N*B, SeqLen]
            self.total_samples_approx = len(self.current_chunk_data)
            print(f"   Loaded {self.total_samples_approx} samples (Legacy Mode)")
        else:
            # 2. Look for shards
            files = sorted([f for f in os.listdir(cache_dir) 
                          if f.startswith("text_cache_chunk_") and f.endswith(".npz")])
            
            if files:
                self.sharded_mode = True
                self.chunk_files = files
                print(f">> Found {len(files)} text cache shards. Lazy loading initialized.")
                # Load first chunk to start
                self._load_next_chunk()
            else:
                 raise FileNotFoundError(f"No text cache found in {cache_dir} (Checked legacy and shards)")
                 
        # Compatibility with train_episodic.py
        # Estimate based on samples / target batch size
        self.n_text_steps = self.total_samples_approx // self.target_batch_size

    @property
    def text_steps(self):
        return self.step
    
    @text_steps.setter
    def text_steps(self, value):
        self.step = value

    def _load_next_chunk(self):
        """Load the next available chunk into memory and FLATTEN it."""
        self.current_chunk_idx = (self.current_chunk_idx + 1) % len(self.chunk_files)
        filename = self.chunk_files[self.current_chunk_idx]
        path = os.path.join(self.cache_dir, filename)
        
        data = np.load(path)
        raw = data['batches'] # [N, FileBatch, SeqLen]
        self.current_chunk_data = raw.reshape(-1, raw.shape[-1]) # [N*FileBatch, SeqLen]
        self.current_chunk_offset = 0 
        
        # Update estimate (Uses first chunk size as assumption)
        if self.total_samples_approx == 0:
            self.total_samples_approx = len(self.current_chunk_data) * len(self.chunk_files)

    def get_train_batch(self):
        """Get multimodal training batch (inner loop)."""
        idx_vis = self.step % self.n_vision_steps if self.has_vision else 0
        
        if self.has_vision:
            latents = torch.from_numpy(self.vision_latents[idx_vis]).float().to(self.device)
            input_ids = torch.from_numpy(self.vision_input_ids[idx_vis]).long().to(self.device)
        else:
            latents = None
            
            # Text Fetching with Sharding
            if self.current_chunk_data is None:
                 raise RuntimeError("No text data loaded")
                 
            # Check if current chunk exhausted (or not enough for full batch)
            if self.current_chunk_offset + self.target_batch_size > len(self.current_chunk_data):
                if self.sharded_mode:
                    self._load_next_chunk()
                else:
                    self.current_chunk_offset = 0 # Loop legacy
            
            batch_slice = self.current_chunk_data[self.current_chunk_offset : self.current_chunk_offset + self.target_batch_size]
            input_ids = torch.from_numpy(batch_slice).long().to(self.device)
            self.current_chunk_offset += self.target_batch_size
        
        return input_ids, latents
    
    def get_val_batch(self):
        """Get text validation batch (outer loop)."""
        # Simple random sample from current loaded chunk for speed
        if self.current_chunk_data is None:
             return None
             
        # Pick random index in current chunk
        idx = np.random.randint(0, len(self.current_chunk_data))
        return torch.from_numpy(self.current_chunk_data[idx]).long().to(self.device)
    
    def advance(self):
        self.step += 1
