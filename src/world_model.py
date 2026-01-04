import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from src.model import PCModel
from src.config import Config


# Optional Import (Now Mandatory for Phase 12)
try:
    from diffusers import FlaxAutoencoderKL
except ImportError:
    print("Error: diffusers not installed. Visual Cortex will fail.")
    FlaxAutoencoderKL = None

class LatentWorldModel(eqx.Module):
    """
    World Model Adapter.
    Visual Cortex (V1): Frozen Pretrained VAE (Standard Diffusion).
    Neocortex: Trainable PCModel (Predicts Latent Dynamics).
    """
    vae: object = eqx.field(static=True) # Frozen module definition
    vae_params: dict # Frozen weights (PyTree)
    neocortex: PCModel
    scaling_factor: float = eqx.field(static=True)
    
    def __init__(self, key: jax.random.PRNGKey):
        # Load VAE from HuggingFace
        # Using CompVis/stable-diffusion-v1-4 as verified in Phase 12
        model_id = "CompVis/stable-diffusion-v1-4"
        print(f"Loading Visual Cortex (VAE) from {model_id}...")
        try:
            self.vae, self.vae_params = FlaxAutoencoderKL.from_pretrained(
                model_id, 
                subfolder="vae", 
                revision="flax"
            )
            # Default scaling factor known for SD v1.x
            self.scaling_factor = 0.18215 
            print("Visual Cortex Activated.")
        except Exception as e:
            raise RuntimeError(f"Failed to activate Visual Cortex: {e}")
            
        self.neocortex = PCModel(key)

    def encode(self, image_batch, key: jax.random.PRNGKey):
        """
        Input: [Batch, H, W, 3] (0-1 float)
        Output: [Batch, Seq, Dim] (Latents)
        """
        # Ensure NCHW for VAE?
        # FlaxAutoencoderKL usually expects NHWC input for 'sample'?
        # Let's check: encode(self, sample) -> sample = jnp.transpose(sample, (0, 2, 3, 1))
        # This implies it EXPECTS NCHW and converts to NHWC?
        # NO. Flax usually works in NHWC.
        # If `encode` transposes `(0,2,3,1)`, it means input was NCHW (0,1,2,3).
        # Diffusers JAX usually mimics PyTorch NCHW input convention?
        # Let's assume input image_batch is NHWC (standard JAX/TF).
        # If we pass NHWC, and it transposes, it becomes ... wrong?
        # Wait, step 1440 output for `encode`:
        # sample = jnp.transpose(sample, (0, 2, 3, 1))
        # This explicitly converts Channel-First to Channel-Last.
        # So it EXPECTS Channel-First (NCHW).
        
        # image_batch is commonly [B, H, W, C] (NHWC) in JAX data loaders.
        # So we must transpose it to NCHW before passing to VAE.
        
        inp = jnp.transpose(image_batch, (0, 3, 1, 2)) # NHWC -> NCHW
        
        # Run VAE Encoder
        # posterior is FlaxAutoencoderKLOutput
        posterior = self.vae.apply(
            {"params": self.vae_params}, 
            inp, 
            method=self.vae.encode
        )
        
        # Sample from posterior
        # We need a key for sampling.
        latents = posterior.latent_dist.sample(key)
        
        # Scale latents
        latents = latents * self.scaling_factor
        
        # Shape is [B, 4, H/8, W/8] (NCHW) or [B, H/8, W/8, 4] (NHWC)?
        # VAE output is usually NCHW (Channels First) because of the transpose in encode/decode?
        # Let's check `encode`: returns `latent_dist`. `latent_dist` uses `moments`.
        # `quant_conv` is `nn.Conv`. Flax key logic: NCHW or NHWC?
        # Flax `nn.Conv` defaults to NHWC (features last).
        # But `encode` transposed input to NHWC (if expected NCHW).
        # So internal computation is NHWC.
        # `quant_conv` output is NHWC.
        # So `latents` is [B, H', W', C].
        
        # Flatten spatial dims to sequence
        # [B, H', W', C] -> [B, H'*W', C]
        B, H_prime, W_prime, C_prime = latents.shape
        latents_flat = latents.reshape((B, H_prime * W_prime, C_prime))
        
        return latents_flat

    def decode(self, latents_flat, ref_shape=None):
        """
        Input: [Batch, Seq, Dim]
        Output: [Batch, H, W, 3]
        """
        # latents_flat is [B, L, C]. We need to unflatten.
        # We assume square image for now or use ref_shape.
        # If 512x512 image -> 64x64 latent. L=4096.
        # C=4.
        
        B, L, C = latents_flat.shape
        S = int(np.sqrt(L)) # Assuming square
        
        # Reshape to [B, H', W', C] (NHWC)
        latents = latents_flat.reshape((B, S, S, C))
        
        # Unscale
        latents = latents / self.scaling_factor
        
        # VAE decode expects NCHW latents??
        # `decode(self, latents)`:
        # if latents.shape[-1] != self.config.latent_channels:
        #     latents = jnp.transpose(latents, (0, 2, 3, 1))
        # If we pass NHWC (C is last). latent_channels is 4.
        # If shape[-1] (4) == 4, it proceeds.
        # So it Expects NHWC.
        
        output = self.vae.apply(
            {"params": self.vae_params},
            latents,
            method=self.vae.decode
        )
        
        images = output.sample # [B, H, W, 3] or [B, 3, H, W]?
        # `decode` implementation:
        # hidden_states = self.decoder(...) -> NHWC
        # hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2)) -> NCHW
        # return FlaxDecoderOutput(sample=hidden_states)
        
        # So output is NCHW.
        # Convert to NHWC for standard visualization
        images = jnp.transpose(images, (0, 2, 3, 1))
        
        return images

    def __call__(self, latents, t):
        """
        Predict next latent state.
        Args:
            latents: [Batch, Seq, Dim] (Noisy or Clean?)
            t: timestep
        """
        return self.neocortex(inputs_embeds=latents, t=t)
