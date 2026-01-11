import torch
from src.multimodal_adapter import MultimodalAdapter

def test_adapter():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Testing on {device}")
    
    try:
        adapter = MultimodalAdapter(batch_size=2, image_size=128, device=device)
        print(">> Adapter Initialized.")
        
        batch = adapter.get_batch(device)
        latents = batch['visual_latents']
        
        print(f">> Latent Shape: {latents.shape}")
        
        # Expected: [2, 4, 16, 16] (128 / 8 = 16)
        assert latents.shape == (2, 4, 16, 16), f"Expected (2,4,16,16), got {latents.shape}"
        assert not torch.isnan(latents).any(), "Latents contain NaNs"
        
        print(">> VERIFIED: Real VAE + Image download works.")
        
    except Exception as e:
        print(f"!! TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_adapter()
