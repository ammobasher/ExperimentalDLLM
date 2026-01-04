import jax
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from src.text_adapter import TextAdapter
from src.config import Config
import torch.nn.functional as F

def benchmark_gpt2_baseline(n_batches=50):
    print("==================================================")
    print("    BENCHMARK BASELINE: GPT-2 SMALL (124M)        ")
    print("    Dataset: WikiText-2 (Same as Synapse)         ")
    print("==================================================")
    
    # 1. Load Model (PyTorch)
    print("[System] Loading GPT-2 Small...")
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    
    # 2. Text Adapter (Same data source)
    adapter = TextAdapter(seq_len=Config.seq_len, batch_size=4)
    
    total_nll = 0.0
    total_tokens = 0
    
    print(f"\n[Evaluation] Running {n_batches} batches...")
    
    with torch.no_grad():
        for i in range(n_batches):
            # Get data (numpy) -> convert to torch
            batch_np = adapter.get_batch()
            input_ids = torch.tensor(batch_np, dtype=torch.long).to(device)
            target_ids = input_ids.clone()
            
            # Forward
            outputs = model(input_ids, labels=target_ids)
            
            # GPT-2 returns mean loss by default
            # We want sum to be aggregating correctly like our metric script
            # But let's just trust the mean for simple PPL
            loss = outputs.loss # NLL
            
            total_nll += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            
            if i % 10 == 0:
                print(f"Batch {i}: Loss {loss.item():.4f}")

    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    bpd = avg_nll / np.log(2)
    
    print("-" * 30)
    print(f"BASELINE RESULTS (GPT-2 Small)")
    print(f"Perplexity (PPL): {ppl:.4f}")
    print(f"Bits Per Dim (BPD): {bpd:.4f}")
    print(f"Avg NLL:          {avg_nll:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    benchmark_gpt2_baseline()
