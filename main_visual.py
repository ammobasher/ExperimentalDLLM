import jax
import jax.numpy as jnp
import numpy as np
import time

from src.config import Config
from src.world_model import LatentWorldModel
from src.meta_trainer import MetaTrainer
# Reuse trainer logic but adapt for visual latents

def main_visual():
    print("==================================================")
    print("    PROJECT SYNAPSE: VISUAL WORLD MODEL (V1)      ")
    print("==================================================")
    
    key = jax.random.PRNGKey(999)
    model_key, data_key = jax.random.split(key)
    
    # 1. Initialize World Model
    print("\n[System] Loading Visual Cortex (VAE) + Neocortex...")
    world_model = LatentWorldModel(model_key)
    
    # 2. Simulate Visual Data (MineRL-like)
    # Shape: [Batch, T, H, W, C]
    # Let's say we have 64x64 images
    print("[Data] Generating Dummy MineRL Trajectories...")
    # Latent shape from SD is 4 channels, H/8.
    # If Input is 64x64 -> Latent is 8x8.
    # Seq Len = Config.seq_len ?
    # Seq Len must be multiple of chunk_size (32)
    seq_len = 64
    latent_h, latent_w = 8, 8
    latent_dim = 4
    
    # Flattened Latent Dim = 8*8*4 = 256. 
    # Our Config uses 512 embed dim. We need a projection or reshape.
    # For prototype: We assume Latents fit into Model Dims or we project.
    # Current PCModel expects `inputs_embeds` of shape [Batch, Seq, Config.embed_dim].
    # We will simulate ALREADY ENCODED latents for simplicity.
    
    # Batch of 4 videos, 16 frames each.
    # Project 256 -> 512 (padding)
    simulated_latents = jax.random.normal(data_key, (4, seq_len, 256))
    simulated_latents = jnp.pad(simulated_latents, ((0,0), (0,0), (0, 256))) # Pad to 512
    
    print(f"visual_latents shape: {simulated_latents.shape}")
    
    # 3. Running Neocortex on Visual Latents
    print("\n[Simulation] Dreaming future frames...")
    
    # We use the Neocortex (Attributes of world_model)
    # Normally we use trainer, here we just do a forward pass to prove compatibility.
    
    batch_latents = simulated_latents
    t = jnp.array(0.1) # Denoising step
    
    start = time.time()
    # Forward Pass
    logits, pc_loss = jax.vmap(lambda x: world_model.neocortex(inputs_embeds=x, t=t))(batch_latents)
    end = time.time()
    
    print(f"Dream complete in {end-start:.3f}s")
    print(f"Output Logic Shape: {logits.shape}")
    print(f"PC Loss: {jnp.mean(pc_loss):.4f}")
    
    print("\n>> Visual World Model Integration Successful (Latent Dynamics).")

if __name__ == "__main__":
    main_visual()
