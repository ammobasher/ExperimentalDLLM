
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from src.world_model import LatentWorldModel

def main():
    print("Initializing Visual Cortex...")
    key = jax.random.PRNGKey(0)
    world_model = LatentWorldModel(key)
    
    # Create a random RGB image [1, 512, 512, 3]
    # Use a gradient or pattern to verify reconstruction structure
    H, W = 512, 512
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(x, y)
    
    img_r = xv
    img_g = yv
    img_b = np.abs(np.sin(xv * 10) * np.cos(yv * 10))
    
    img = np.stack([img_r, img_g, img_b], axis=-1)
    img = img[None, ...] # Batch dim
    img = jnp.array(img, dtype=jnp.float32)
    
    print(f"Input Image Shape: {img.shape}")
    
    # Debug: Print top-level keys
    print("VAE Params Keys (Top Level):", world_model.vae_params.keys())
    if 'encoder' in world_model.vae_params:
        print("Encoder Params Keys:", world_model.vae_params['encoder'].keys())
        if 'conv_in' in world_model.vae_params['encoder']:
             print("Encoder conv_in Keys:", world_model.vae_params['encoder']['conv_in'].keys())
    
    # Flatten checks?
    flat_params = jax.tree_util.tree_leaves(world_model.vae_params)
    print(f"Total params count: {len(flat_params)}")
    
    # Encode
    print("Encoding...")
    key, subkey = jax.random.split(key)
    try:
        latents = world_model.encode(img, subkey)
        print(f"Latents Shape: {latents.shape}") # Expect [1, 4096, 4]
        
        # Decode
        print("Decoding...")
        recon = world_model.decode(latents)
        print(f"Reconstruction Shape: {recon.shape}") # Expect [1, 512, 512, 3]
        
        # Save images
        img_np = np.array(img[0] * 255).astype(np.uint8)
        recon_np = np.array(jnp.clip(recon[0], 0, 1) * 255).astype(np.uint8)
        
        Image.fromarray(img_np).save("visual_test_input.png")
        Image.fromarray(recon_np).save("visual_test_recon.png")
        print("Saved visual_test_input.png and visual_test_recon.png")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
