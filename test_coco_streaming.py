import torch
from src.multimodal_adapter import MultimodalAdapter
from src.text_adapter import TextAdapter

def test_coco():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Testing COCO Streaming on {device}")
    
    try:
        # Initialize with Streaming=True
        adapter = MultimodalAdapter(batch_size=2, image_size=128, device=device, streaming=True)
        print(">> Adapter Initialized (Streaming Mode).")
        
        # Fetch Batch
        print(">> Fetching real batch from COCO...")
        batch = adapter.get_batch(device)
        
        latents = batch['visual_latents']
        input_ids = batch['input_ids']
        
        print(f">> Latent Shape: {latents.shape}") # Expect [2, 4, 16, 16]
        print(f">> Input IDs Shape: {input_ids.shape}") # Expect [2, 64]
        
        # Decode Caption
        tokenizer = TextAdapter(split="test").tokenizer
        captions = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        print("\n>> Sample Captions:")
        for i, cap in enumerate(captions):
            print(f"   [{i}] {cap}")
            
        assert latents.shape == (2, 4, 16, 16)
        print("\n>> VERIFIED: COCO Streaming works.")
        
    except Exception as e:
        print(f"!! TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coco()
