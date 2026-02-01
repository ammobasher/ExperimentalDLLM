import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import jax
import jax.numpy as jnp
import sys
print(f"Python Version: {sys.version}")

try:
    from diffusers import FlaxAutoencoderKL
    print("Success: Imported FlaxAutoencoderKL")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    model_id = "CompVis/stable-diffusion-v1-4"
    
    model_id = "CompVis/stable-diffusion-v1-4"
    import inspect
    from flax.linen import Conv
    print(f"Conv Signature: {inspect.signature(Conv)}")
    print(f"Loading VAE from {model_id} (revision='flax')...")
    vae, params = FlaxAutoencoderKL.from_pretrained(model_id, subfolder="vae", revision="flax")
    print("Success: Loaded VAE Weights")
    
except Exception as e:
    import traceback
    traceback.print_exc()
