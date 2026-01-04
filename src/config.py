import jax.numpy as jnp

class Config:
    # Architecture
    # Architecture (Scaled for Phase 13: ~300M Params)
    embed_dim: int = 1024
    n_layers: int = 12 # Increased from 6
    n_heads: int = 16 # Increased from 8
    chunk_size: int = 32  # Mini-column size
    
    # Diffusion
    n_timesteps: int = 1000
    beta_min: float = 0.1
    beta_max: float = 20.0
    
    # Training
    vocab_size: int = 32000 # Default
    lr_llm: float = 1e-4
    lr_ctrl: float = 1e-5
    batch_size: int = 32
    
    # Dimensions
    # Assuming input sequence length N. N must be divisible by chunk_size.
    # For testing, we can assume a standard sequence length.
    seq_len: int = 512
