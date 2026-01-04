import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from src.config import Config
from src.model import PCModel

def test():
    print("--- Multimodal Test ---")
    key = jax.random.PRNGKey(0)
    
    # Init Model
    model = PCModel(key)
    print("Model Initialized.")
    
    # Dummy Text
    text_len = 50
    input_ids = jnp.arange(text_len) # [50]
    
    # Dummy Visual Latents (Channels=4, H=8, W=8 -> 64 tokens)
    # Total = 114 tokens. Should pad to 1024.
    visual_latents = jnp.ones((4, 8, 8)) 
    
    print(f"Text Len: {text_len}")
    print(f"Visual Shape: {visual_latents.shape} -> {8*8} tokens")
    print(f"Total raw: {text_len + 64}")
    print(f"Chunk Size: {Config.chunk_size}")
    
    # Forward
    # Note: inputs_embeds is None, so it uses input_ids
    # But we map over batch usually? model __call__ expects single sample (unbatched)?
    # "Here input_ids is [Seq]. Embedding expects scalar. So we vmap over Seq." -> implies Unbatched.
    
    logits, loss = model(input_ids=input_ids, visual_latents=visual_latents)
    
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Loss: {loss}")
    
    expected_len = ((text_len + 64 - 1) // Config.chunk_size + 1) * Config.chunk_size
    # Wait, remainder logic in code:
    # if rem != 0: pad = chunk - rem. Total = len + (chunk - rem) = chunk * ceil(len/chunk).
    # So if 114 < 1024, it becomes 1024.
    
    if logits.shape[0] == 1024: # Assuming chunk_size=1024
        print("SUCCESS: Padding worked.")
    else:
        print(f"FAILURE: Shape {logits.shape} != 1024")

if __name__ == "__main__":
    test()
