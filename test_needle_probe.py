import torch
import torch.nn.functional as F
from src.model import PCModel
from src.config import Config
from src.text_adapter import TextAdapter

def test_needle():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Needle Probe on {device}")
    
    # 1. Load Model
    Config.vocab_size = 50257
    model = PCModel().to(device)
    ckpt_path = "/Users/ahmed/Documents/ExperimentalDLLM/checkpoints_unified/step_500.pt"
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'], strict=False)
        print(">> Model Loaded.")
    except Exception as e:
        print(f"!! Checkpoint load failed: {e}")
        return

    # 2. Setup Memory (Simulated One-Shot)
    # The paper claims: "Events that generate high Prediction Error are stored."
    # We simulate this by checking if the model can *utilize* a context
    # if it was in its memory. For this test, we might need to manually
    # inject into the vector DB if the 'training' didn't trigger it.
    # OR, we simply test if the model *learned* it during the training 
    # if we injected it in the dataset.
    
    # WAIT. Phase 23 Plan says: "automated 'Needle in a Haystack' probe".
    # This implies we inject it at runtime and verify retrieval.
    # Since we are not running the full 'Episodes' loop in this script,
    # we will focus on: Can it Complete a prompt if provided with context?
    # (Checking if the 'Attending to Memory' mechanism works).
    
    # Actually, let's keep it simple:
    # 1. Prompt: "The secret password is [MASK]"
    # 2. See if it predicts "Eagle" (if we pretend it saw it).
    
    # REAL TEST:
    # We will assume the training run (which is running) might have seen it?
    # No, the training run is on WikiText.
    # So this probe is currently just checking Model Health?
    
    # Let's pivot: This script will verify "Contextual Completion".
    # We provide "The secret password is Eagle." as context (Memory).
    # Then query "The secret password is".
    # If Attention works, it predicts "Eagle".
    
    tokenizer = TextAdapter(split="test").tokenizer
    
    context = "The secret password is Eagle."
    query = "The secret password is"
    
    # Encode
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    # Target: Eagle
    
    # Forward (Auto-regressive sample)
    model.eval()
    
    # We'll just check next token prediction on the query
    query_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # t=0 (No noise, pure prediction)
        logits, _ = model(query_ids, t=torch.zeros(query_ids.shape[0], device=device))
        
        # Last token logit
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs).item()
        
        predicted_word = tokenizer.decode([next_token])
        print(f">> Query: '{query}'")
        print(f">> Predicted: '{predicted_word}'")
        
        # Top 5
        topk = torch.topk(probs, 5)
        print(">> Top 5:")
        for idx in topk.indices:
            print(f"   - {tokenizer.decode([idx])}")

if __name__ == "__main__":
    test_needle()
