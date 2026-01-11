import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLayer(nn.Module):
    """
    Standard Predictive Coding Layer (PyTorch Version).
    """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
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
        # Handle scalar loss correctly for autograd
        pc_loss = torch.tensor(0.0, device=x.device)
        
        if p_i_plus_1 is not None and not inference:
            error = x - p_i_plus_1 
            pc_loss = torch.mean(error ** 2)
            
        return x_next, p_i, pc_loss

class OptimizedPCLayer(PCLayer):
    """
    Wrapper for compatibility.
    """
    def __init__(self, embed_dim, n_heads, chunk_size=None, key=None):
        super().__init__(embed_dim, n_heads)
