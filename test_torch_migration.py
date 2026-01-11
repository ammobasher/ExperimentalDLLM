import torch
from src.model import PCModel
from src.config import Config

def test_torch():
    print("Testing PyTorch Migration...")
    
    # Check Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">> MPS (Metal) Available! 🚀")
    else:
        device = torch.device("cpu")
        print(">> MPS Not found. Using CPU.")
        
    model = PCModel().to(device)
    Config.vocab_size = 32000
    
    # Dummy Input
    B, Seq = 2, 64
    input_ids = torch.randint(0, 1000, (B, Seq)).to(device)
    t = torch.tensor(0.5, device=device)
    
    # Forward Pass
    logits, pc_loss = model(input_ids=input_ids, t=t)
    
    print(f"Logits Shape: {logits.shape}") # [2, 64, 32000]
    print(f"PC Loss: {pc_loss.item()}")
    
    if logits.shape == (B, Seq, Config.vocab_size):
        print("SUCCESS: Model Forward Pass works.")
    else:
        print("FAILURE: Shape Mismatch.")
        
if __name__ == "__main__":
    test_torch()
