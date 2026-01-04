from src.multimodal_adapter import MultimodalAdapter
import numpy as np

def test():
    print("Testing Multimodal Adapter...")
    adapter = MultimodalAdapter(batch_size=2, image_size=256)
    
    imgs, tokens = adapter.get_batch()
    
    print(f"Images Shape: {imgs.shape}") # Expect [2, 256, 256, 3]
    print(f"Tokens Shape: {tokens.shape}") # Expect [2, 64]
    
    print(f"Image Range: {np.min(imgs)} - {np.max(imgs)}")
    print(f"Sample Text Token 0: {tokens[0, 0]}")
    
    if imgs.shape == (2, 256, 256, 3):
        print("SUCCESS: Data Loaded Correctly.")
    else:
        print("FAILURE: Shape Mismatch.")

if __name__ == "__main__":
    test()
