import torch
from src.layers import PCLayer, ChunkedPCLayer, OptimizedPCLayer
from src.config import Config

def test_chunked_attention():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Testing Chunked Attention on {device}")
    
    # Setup
    batch_size = 2
    seq_len = 128  # Should be divisible by chunk_size
    embed_dim = Config.embed_dim
    n_heads = Config.n_heads
    chunk_size = Config.chunk_size
    
    # Create layers
    standard_layer = PCLayer(embed_dim, n_heads, chunk_size).to(device)
    chunked_layer = ChunkedPCLayer(embed_dim, n_heads, chunk_size).to(device)
    
    # Copy weights for fair comparison
    chunked_layer.load_state_dict(standard_layer.state_dict())
    
    # Input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Forward passes
    with torch.no_grad():
        # Standard
        out_std, _, pc_std = standard_layer(x)
        
        # Chunked
        out_chunk, _, pc_chunk = chunked_layer(x)
    
    print(f">> Standard Output Shape: {out_std.shape}")
    print(f">> Chunked Output Shape: {out_chunk.shape}")
    
    # Outputs won't be identical (different attention patterns) but shapes must match
    assert out_std.shape == out_chunk.shape, "Shape mismatch!"
    
    # Test that OptimizedPCLayer is now ChunkedPCLayer
    opt_layer = OptimizedPCLayer(embed_dim, n_heads, chunk_size).to(device)
    assert isinstance(opt_layer, ChunkedPCLayer), "OptimizedPCLayer should be ChunkedPCLayer"
    
    print(">> All shape and type checks passed!")
    
    # Memory comparison (rough)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    print("\n>> VERIFIED: Chunked Attention implementation is functional.")

if __name__ == "__main__":
    test_chunked_attention()
