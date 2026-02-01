import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PCLayer(nn.Module):
    """
    Standard Predictive Coding Layer (PyTorch Version).
    Uses full O(N²) attention for baseline/debugging.
    """
    def __init__(self, embed_dim, n_heads, chunk_size=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.chunk_size = chunk_size or 32  # Default chunk size
        
        # Standard Attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, p_i_plus_1=None, inference=False):
        """
        x: [Batch, Seq, Dim]
        p_i_plus_1: [Batch, Seq, Dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Attention (Bottom-Up)
        resid = x
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        x = resid + self.out_proj(attn_out)
        
        # 2. FFN
        resid = x
        x = resid + self.ffn(self.norm2(x))
        
        x_next = x 
        
        # 3. Top-Down Prediction (P_i)
        p_i = x_next
        
        # 4. PC Loss Calculation (E_i)
        pc_loss = torch.tensor(0.0, device=x.device)
        
        if p_i_plus_1 is not None and not inference:
            error = x - p_i_plus_1 
            pc_loss = torch.mean(error ** 2)
            
        return x_next, p_i, pc_loss


class ChunkedPCLayer(nn.Module):
    """
    Optimized Predictive Coding Layer with Sliding Window Attention.
    
    Instead of O(N²) global attention, uses O(N * W) local attention where:
    - Each position attends to a window of size W = 2 * chunk_size
    - Causality is enforced within the window
    - Global context flows through hierarchical top-down predictions
    
    This implements the "Mini-Columnar Attention" described in the paper.
    """
    def __init__(self, embed_dim, n_heads, chunk_size=32, key=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.chunk_size = chunk_size
        self.window_size = 2 * chunk_size  # Each chunk sees itself + previous chunk
        
        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def _chunked_attention(self, q, k, v):
        """
        Sliding window attention with causal masking.
        
        q, k, v: [Batch, Heads, Seq, HeadDim]
        Returns: [Batch, Heads, Seq, HeadDim]
        """
        B, H, N, D = q.shape
        C = self.chunk_size
        
        # Pad sequence to be divisible by chunk_size
        pad_len = (C - N % C) % C
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        
        N_padded = q.shape[2]
        n_chunks = N_padded // C
        
        # Reshape into chunks: [B, H, n_chunks, C, D]
        q_chunks = q.view(B, H, n_chunks, C, D)
        k_chunks = k.view(B, H, n_chunks, C, D)
        v_chunks = v.view(B, H, n_chunks, C, D)
        
        # For each chunk, we need keys/values from current + previous chunk
        # Pad keys/values with a "previous chunk" (zeros for first chunk)
        k_padded = F.pad(k_chunks, (0, 0, 0, 0, 1, 0))  # [B, H, n_chunks+1, C, D]
        v_padded = F.pad(v_chunks, (0, 0, 0, 0, 1, 0))
        
        # Create windows: each chunk sees [prev_chunk, current_chunk]
        # k_windows[i] = concat(k[i-1], k[i]) -> shape [B, H, n_chunks, 2C, D]
        k_windows = torch.cat([k_padded[:, :, :-1], k_padded[:, :, 1:]], dim=3)
        v_windows = torch.cat([v_padded[:, :, :-1], v_padded[:, :, 1:]], dim=3)
        
        # Compute attention scores: [B, H, n_chunks, C, 2C]
        # q_chunks: [B, H, n_chunks, C, D]
        # k_windows: [B, H, n_chunks, 2C, D]
        scale = 1.0 / math.sqrt(D)
        
        # Manual matmul: q @ k.transpose for each chunk
        # q_chunks: [B, H, n_chunks, C, D] -> [B*H*n_chunks, C, D]
        # k_windows: [B, H, n_chunks, 2C, D] -> [B*H*n_chunks, 2C, D]
        BH_nc = B * H * n_chunks
        q_flat = q_chunks.reshape(BH_nc, C, D)
        k_flat = k_windows.reshape(BH_nc, 2 * C, D)
        v_flat = v_windows.reshape(BH_nc, 2 * C, D)
        
        # scores: [BH_nc, C, 2C]
        scores = torch.bmm(q_flat, k_flat.transpose(1, 2)) * scale
        
        # Create causal mask for the window
        # Position i in current chunk can attend to:
        # - All positions in previous chunk (indices 0 to C-1)
        # - Positions 0 to i in current chunk (indices C to C+i)
        causal_mask = torch.ones(C, 2 * C, dtype=torch.bool, device=q.device)
        for i in range(C):
            # Position i can see: all of prev chunk + positions 0..i of current chunk
            causal_mask[i, C + i + 1:] = False
        
        # Apply mask: [BH_nc, C, 2C]
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention: [BH_nc, C, D]
        attn_out_flat = torch.bmm(attn_weights, v_flat)
        
        # Reshape back: [B, H, N_padded, D]
        attn_out = attn_out_flat.view(B, H, n_chunks, C, D).reshape(B, H, N_padded, D)
        
        # Remove padding
        if pad_len > 0:
            attn_out = attn_out[:, :, :N]
        
        return attn_out

    def forward(self, x, p_i_plus_1=None, inference=False):
        """
        x: [Batch, Seq, Dim]
        p_i_plus_1: [Batch, Seq, Dim] - Top-down prediction from layer above
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Chunked Attention (Bottom-Up)
        resid = x
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_out = self._chunked_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        x = resid + self.out_proj(attn_out)
        
        # 2. FFN
        resid = x
        x = resid + self.ffn(self.norm2(x))
        
        x_next = x 
        
        # 3. Top-Down Prediction (P_i)
        p_i = x_next
        
        # 4. PC Loss Calculation (E_i)
        pc_loss = torch.tensor(0.0, device=x.device)
        
        if p_i_plus_1 is not None and not inference:
            error = x - p_i_plus_1 
            pc_loss = torch.mean(error ** 2)
            
        return x_next, p_i, pc_loss


# Alias for backwards compatibility
class OptimizedPCLayer(ChunkedPCLayer):
    """
    Wrapper for compatibility with existing model.py.
    Now uses actual chunked attention instead of standard attention.
    """
    pass
