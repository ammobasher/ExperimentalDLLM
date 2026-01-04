import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple

from src.layers import OptimizedPCLayer, PCLayer
from src.diffusion import DiffusionSDE
from src.config import Config

class PCModel(eqx.Module):
    embedding: eqx.nn.Embedding
    layers: List[PCLayer]
    output_head: eqx.nn.Linear
    visual_proj: eqx.nn.Linear # Phase 17
    n_layers: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(self, key: jax.random.PRNGKey):
        keys = jax.random.split(key, Config.n_layers + 2)
        
        self.n_layers = Config.n_layers
        self.embed_dim = Config.embed_dim
        
        # Use Config-defined vocab size (can be monkey-patched)
        vocab_size = Config.vocab_size
        
        self.embedding = eqx.nn.Embedding(vocab_size, Config.embed_dim, key=keys[0])
        
        self.layers = [
            OptimizedPCLayer(
                embed_dim=Config.embed_dim, 
                n_heads=Config.n_heads, 
                chunk_size=Config.chunk_size, 
                key=keys[i+1]
            ) 
            for i in range(Config.n_layers)
        ]
        
        self.output_head = eqx.nn.Linear(Config.embed_dim, vocab_size, key=keys[-1])
        
        # Multimodal Projection (Phase 17)
        # Assuming VAE latent channels (4)
        # We project 4 -> embed_dim
        self.visual_proj = eqx.nn.Linear(4, Config.embed_dim, key=keys[0]) # Reuse key[0] or new? ideally new. split again.

    def __call__(self, input_ids: jax.Array = None, t: jax.Array = None, inputs_embeds: jax.Array = None, 
                 visual_latents: jax.Array = None, inference: bool = False) -> Tuple[jax.Array, jax.Array]:
        """
        Dual-Pass Forward.
        Args:
            input_ids: [Batch, Seq] or [Seq]. Tokens.
            t: timestep (scalar).
            inputs_embeds: [Batch, Seq, Dim]. Noisy embeddings for Diffusion.
                           If provided, 'input_ids' is ignored for embedding (but maybe needed for something else?).
                           Actually, if inputs_embeds is passed, we skip self.embedding.
        
        Returns:
            logits: [Batch, Seq, Vocab]
            total_pc_loss: Scalar
        """
        # 1. Prediction / Embedding
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            # If input_ids has batch dim, we vmap (handled outside).
            # Here input_ids is [Seq]. Embedding expects scalar. So we vmap over Seq.
            x = jax.vmap(self.embedding)(input_ids)
        else:
            raise ValueError("Must provide either input_ids or inputs_embeds")
        
        # Multimodal Injection (Phase 17)
        if visual_latents is not None:
            # visual_latents: [Channels, H, W]
            # Flatten to [Seq_Vis, Channels]
            # We assume channels is first dim here (as per convention) or last? 
            # VAE output is usually (Batch, C, H, W) or (Batch, H, W, C).
            # Let's assume (C, H, W) since we unbatched.
            
            # Permute C to last: (C, H, W) -> (H, W, C)
            vis = jnp.transpose(visual_latents, (1, 2, 0))
            # Flatten spatial: (H, W, C) -> (H*W, C)
            vis = jnp.reshape(vis, (-1, vis.shape[-1]))
            # Project: (Seq_Vis, C) -> (Seq_Vis, EmbedDim)
            vis_emb = jax.vmap(self.visual_proj)(vis)
            
            # Concatenate: [Vis, Text]
            # Warning: This changes sequence length. 
            # If we are using vmap over chunks later, seq_len must be divisible by chunk_size.
            # We might need to pad or trim.
            # For now, we trust the caller to ensure shapes or we crop.
            x = jnp.concatenate([vis_emb, x], axis=0)
            
        # Pad to multiple of chunk_size (1024)
        seq_len = x.shape[0]
        chunk_size = Config.chunk_size
        remainder = seq_len % chunk_size
        if remainder != 0:
            pad_len = chunk_size - remainder
            # Pad with zeros
            padding = jnp.zeros((pad_len, x.shape[-1]))
            x = jnp.concatenate([x, padding], axis=0)
            
        # --- PHASE 1: BOTTOM-UP ---
        # Run layers to generate X_i_plus_1 and P_i (predictions for below)
        
        layer_inputs = [] # Stores inputs to each layer (X_i)
        preds = []        # Stores predictions output by each layer (P_i)
        
        current_x = x
        
        for layer in self.layers:
            layer_inputs.append(current_x)
            
            # Initial pass: No top-down info yet
            x_next, p_i, _ = layer(current_x, p_i_plus_1=None, inference=inference)
            
            preds.append(p_i)
            current_x = x_next
            
        final_features = current_x
        
        # --- PHASE 2: TOP-DOWN (Loss Calculation) ---
        # Calculates E_i = X_i - P_i+1
        # P_i+1 comes from the layer above.
        
        total_pc_loss = 0.0
        
        for i in range(self.n_layers):
            x_i = layer_inputs[i]
            
            # Find P_i+1
            if i == self.n_layers - 1:
                # Top layer has no prediction from above
                p_from_above = None
            else:
                # Layer i+1 produced preds[i+1] which IS p_from_above (prediction for Layer i)
                p_from_above = preds[i+1]
                
            # Re-run layer partially to get loss (or full run, cheap with checkpointing)
            # We call the layer again with the correct prediction to calculate E_i and its loss
            _, _, loss = self.layers[i](x_i, p_from_above, inference=inference)
            
            total_pc_loss = total_pc_loss + loss
            
        # Final Output
        logits = jax.vmap(self.output_head)(final_features)
        
        return logits, total_pc_loss
