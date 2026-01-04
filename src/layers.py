import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

class PCLayer(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    ffn: eqx.nn.MLP
    prediction_head: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    chunk_size: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, n_heads: int, chunk_size: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 3)
        self.chunk_size = chunk_size
        self.attention = eqx.nn.MultiheadAttention(n_heads, embed_dim, key=keys[0])
        self.ffn = eqx.nn.MLP(embed_dim, embed_dim, embed_dim, depth=1, key=keys[1], activation=jax.nn.gelu) # Simple MLP/FFN
        self.prediction_head = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

    def __call__(self, x_i: jax.Array, p_i_plus_1: Optional[jax.Array] = None, key: Optional[jax.random.PRNGKey] = None, inference: bool = False) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Forward pass for PCLayer.
        
        Args:
            x_i: Input tensor (Bottom-Up) [Seq, Dim] or [Batch, Seq, Dim]
                 Note: equinox layers usually expect unbatched input by default, so we might need vmap outside.
                 However, for simplicity in definitions, let's assume inputs are [Seq, Dim].
            p_i_plus_1: Prediction from layer above [Seq, Dim]. None for top layer.
            key: PRNG Key (needed for dropout if enabled, though we aren't using dropout yet).
            
        Returns:
            x_i_plus_1: Output for next layer (Bottom-Up).
            p_i: Prediction for previous layer (Top-Down).
            local_pc_loss: Scalar loss for this layer.
        """
        
        # 1. Error Calculation (The "Surprise")
        # E_i = X_i - P_i+1
        if p_i_plus_1 is None:
            e_i = x_i
        else:
            e_i = x_i - p_i_plus_1
            
        # 2. Localized Processing (Chunked Attention)
        # We need to handle arbitrary dimensions: [Seq, Dim] or [Batch, Seq, Dim]
        # We assume the last dimension is EmbedDim, and the second to last is SeqLen.
        input_shape = e_i.shape
        seq_len = input_shape[-2]
        dim = input_shape[-1]
        
        # Flatten leading dimensions (Batch) into one "Batch of Sequences" or just treat as is?
        # Actually, if we have chunks, we want to reshape (..., Seq, Dim) -> (..., NumChunks, ChunkSize, Dim)
        
        num_chunks = seq_len // self.chunk_size
        
        # Reshape to insert Chunk Axis
        # shape[:-2] + (NumChunks, ChunkSize, Dim)
        new_shape = input_shape[:-2] + (num_chunks, self.chunk_size, dim)
        e_i_reshaped = jnp.reshape(e_i, new_shape)
        
        # Define local attention function to vmap over chunks
        # We need to map over the new Chunk Axis (dimension -3) and potentially Batch axes.
        # But simpler: Flatten everything leading up to ChunkSize into a single "Batch" for eqx.MultiHeadAttention?
        # EQX MHA expects (Sequence, Dim) if unbatched, or (Batch, Seq, Dim) if batched=True.
        # But we are using vmap over chunks to simulate "Independent Mini-Columns".
        
        # Let's use the property that vmap maps over the leading axis.
        # We can flatten all leading dims (Batch, NumChunks) into one large batch dimension.
        # e_i_reshaped: (..., NumChunks, ChunkSize, Dim) -> (TotalChunks, ChunkSize, Dim)
        
        e_i_flat = jnp.reshape(e_i, (-1, self.chunk_size, dim))
        
        # vmap over the TotalChunks dimension
        def local_attn(chunk, k_):
             # chunk: [ChunkSize, Dim]
            return self.attention(chunk, chunk, chunk, key=k_, inference=inference)

        # Pass None for keys if no dropout
        processed_flat = jax.vmap(local_attn)(e_i_flat, None)
        
        # Reshape back to original structure
        # (TotalChunks, ChunkSize, Dim) -> (..., NumChunks, ChunkSize, Dim)
        processed_e = jnp.reshape(processed_flat, new_shape)
        
        # Collapse chunks back to sequence
        # (..., NumChunks, ChunkSize, Dim) -> (..., Seq, Dim)
        processed_e = jnp.reshape(processed_e, input_shape)
        
        # 3. Residual & Norm (Standard Transformer Steps)
        # Note: We process the ERROR signal, not the raw input.
        # Apply LayerNorm per token (vmap over sequence)
        x_residual = jax.vmap(self.norm1)(e_i + processed_e)
        
        # FFN
        # eqx.nn.MLP applies to the last dimension, so we can apply it directly to the sequence
        # vmap over sequence len or rely on broadcasting if supported. Equinox MLP is usually unbatched.
        # We use jax.vmap for safety over the sequence dimension.
        # FFN
        # Apply to last dimension. We handle arbitrary leading dims by flattening.
        x_res_flat = jnp.reshape(x_residual, (-1, input_shape[-1]))
        ffn_out_flat = jax.vmap(self.ffn)(x_res_flat)
        ffn_out = jnp.reshape(ffn_out_flat, x_residual.shape)
        
        x_i_plus_1 = jax.vmap(self.norm2)(x_residual + ffn_out)
        
        # 5. Calculate Top-Down Prediction for Layer Below (P_i)
        # Same flattening strategy for prediction head
        x_next_flat = jnp.reshape(x_i_plus_1, (-1, input_shape[-1]))
        p_i_flat = jax.vmap(self.prediction_head)(x_next_flat)
        p_i = jnp.reshape(p_i_flat, input_shape)
        
        # 5. Calculate Local PC Loss
        # Mean Squared Error of the prediction residual (e_i)
        # If p_i_plus_1 was None (Top Layer), the error is just the input magnitude. 
        # Usually typical PC models might have a different rule for top layer, but standardizing 
        # to "minimize activity" (sparsity) or just 0 is fine.
        # However, typically PC loss is only relevant if there WAS a prediction.
        # For now, we compute it on e_i always.
        local_pc_loss = jnp.mean(jnp.square(e_i))
        
        return x_i_plus_1, p_i, local_pc_loss

# Optimization temporarily disabled
OptimizedPCLayer = PCLayer
