# Config is framework agnostic


class Config:
    # Architecture
    # Architecture (Scaled for Phase 13: ~254M Params)
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

    # Episodic Memory & Freezing (Phase 1)
    freeze_base: bool = False  # Set to True after pre-training
    enable_sleep: bool = False  # Enable sleep consolidation
    memory_capacity: int = 50000  # Episodic memory capacity
    memory_threshold: float = 2.0  # Initial surprise threshold


class ConfigSmall:
    """
    Optimized for 250M params - Local deployment on CPU/mobile devices.

    Parameter calculation:
    - Embeddings: 32000 × 768 = 24.6M
    - Layers: 8 × [
        Attention: 768² × 4 (Q,K,V,O) = 2.36M per layer
        FFN: 768 × 3072 × 2 = 4.72M per layer
        Total per layer: ~7.1M
      ] = 56.8M
    - Output head: 32000 × 768 = 24.6M
    - Total: ~106M params (leaving headroom for future additions)

    NOTE: This is smaller than target to allow for episodic memory overhead.
    Can scale to 250M by increasing to 10 layers or 896 dim if needed.
    """
    # Architecture (Optimized for efficiency)
    embed_dim: int = 768  # Reduced from 1024
    n_layers: int = 8     # Reduced from 12
    n_heads: int = 12     # Adjusted for 768 (64 per head)
    chunk_size: int = 32  # Keep mini-column size

    # Diffusion (Simplified for faster inference)
    n_timesteps: int = 100  # Reduced from 1000 for speed
    beta_min: float = 0.1
    beta_max: float = 10.0  # Reduced from 20

    # Training
    vocab_size: int = 50257
    lr_llm: float = 1e-4
    lr_ctrl: float = 1e-5
    lr_sleep: float = 1e-5  # Learning rate for sleep consolidation
    batch_size: int = 8
    seq_len: int = 512

    # Episodic Memory & Freezing
    freeze_base: bool = True   # Freeze after pre-training
    enable_sleep: bool = True   # Enable sleep consolidation
    memory_capacity: int = 50000  # 50K vectors (~50MB at 768 dim)
    memory_dim: int = 768  # Must match embed_dim
    memory_threshold: float = 0.5  # Tuned for quick adoption (Avg suprise ~0.85)

    # Sleep Consolidation
    sleep_trigger_threshold: float = 0.8  # Trigger when memory >80% full
    sleep_replay_samples: int = 1000  # Number of memories to replay
    sleep_epochs: int = 3  # Training epochs during sleep

    # Adaptive Computation
    enable_adaptive_depth: bool = True  # Enable early exit
    early_exit_threshold: float = 0.8  # Exit if signal > threshold


class ConfigMicro:
    """
    Ultra-compact for mobile/IoT devices - 125M params target.
    """
    embed_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    chunk_size: int = 32

    n_timesteps: int = 50
    beta_min: float = 0.1
    beta_max: float = 10.0

    vocab_size: int = 50257
    lr_llm: float = 1e-4
    lr_ctrl: float = 1e-5
    lr_sleep: float = 1e-5
    batch_size: int = 16
    seq_len: int = 256

    freeze_base: bool = True
    enable_sleep: bool = True
    memory_capacity: int = 25000  # 25K vectors (~13MB)
    memory_dim: int = 512
    memory_threshold: float = 0.5

    sleep_trigger_threshold: float = 0.8
    sleep_replay_samples: int = 500
    sleep_epochs: int = 2

    enable_adaptive_depth: bool = True
    early_exit_threshold: float = 0.85
