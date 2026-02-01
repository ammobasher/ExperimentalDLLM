import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
import numpy as np
import os
import argparse
from tqdm import tqdm

from src.model import PCModel
from src.config import Config, ConfigSmall, ConfigMicro
from src.text_adapter import TextAdapter

def calculate_ppl(model, test_loader, device, is_baseline=False, vocab_size=50257):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {'Baseline' if is_baseline else 'Synapse'}"):
             # batch is usually list of dicts or just tokens
             # TextAdapter handles this, but here we might use raw dataset
             # Let's assume input_ids tensor
             input_ids = batch.to(device)
             
             if is_baseline:
                 # Standard GPT-2
                 outputs = model(input_ids, labels=input_ids)
                 loss = outputs.loss
             else:
                 # Synapse PCModel
                 # We need to simulate the "denoising" or just raw generation?
                 # PCModel returns logits from forward pass (Dual Pass)
                 # inputs_embeds or input_ids
                 # For PPL, we provide input_ids and set t=0? 
                 # Wait, PCModel generation is diffusion-based.
                 # The 'Direct' PPL is only valid if the model acts as a standard causal predictor.
                 # Our PCModel outputs `logits` which are the "Denoised x_0" estimate.
                 # If we pass t=0 (clean), it should behave like an autoencoder/predictor.
                 # Let's use t=0.1 (low noise) or t=0.0 depending on training task.
                 # Actually, standard PPL is P(x_t | x_<t).
                 # Our model is trained to minimize PC Loss + CE Loss.
                 # So evaluating CE Loss on input_ids works as a proxy for PPL.
                 
                 # Inputs
                 t = torch.zeros(input_ids.shape[0], device=device) # t=0 (No noise)
                 logits, pc_loss = model(input_ids=input_ids, t=t)
                 
                 # Calculate CE manually (With Causal Shift)
                 # logits[i] is probability of x[i] given x[0..i] (if we use t=0 and it sees itself).
                 # BUT for Next Token Prediction PPL, we want P(x[i+1] | x[0..i]).
                 # So we compare logits[i] with input[i+1].
                 shift_logits = logits[..., :-1, :].contiguous()
                 shift_labels = input_ids[..., 1:].contiguous()
                 
                 loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                 
             total_loss += loss.item()
             total_steps += 1
             
    avg_loss = total_loss / total_steps
    ppl = np.exp(avg_loss)
    return avg_loss, ppl

def benchmark_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synapse_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default='base', choices=['base', 'small', 'micro'], help="Model configuration to use")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Benchmarking on {device}")
    
    # 1. Prepare Data (WikiText-2 Test)
    # We use TextAdapter logic but specifically for WikiText
    print(">> Loading WikiText-2 Test Set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    adapter = TextAdapter(dataset_name="wikitext", split="test") # Init adapter to get tokenizer
    
    # Select Config based on argument
    if args.config == 'small':
        CurrentConfig = ConfigSmall
    elif args.config == 'micro':
        CurrentConfig = ConfigMicro
    else:
        CurrentConfig = Config

    CurrentConfig.vocab_size = adapter.vocab_size # Update Config BEFORE model init
    
    # Preprocess
    encodings = adapter.tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    # Sliding window or chunks
    seq_len = CurrentConfig.seq_len
    input_ids = encodings.input_ids
    
    # Create batches
    # Truncate to multiple of seq_len
    n_tokens = input_ids.size(1)
    n_batches = n_tokens // seq_len
    input_ids = input_ids[:, :n_batches * seq_len]
    input_ids = input_ids.view(-1, seq_len) # [N_Batches, Seq_Len]
    
    # DataLoader
    test_loader = DataLoader(input_ids, batch_size=4, shuffle=False)
    
    results = {}
    
    # 2. Evaluate Synapse
    if args.synapse_ckpt and os.path.exists(args.synapse_ckpt):
        print(f">> Loading Synapse Model ({args.config}) from {args.synapse_ckpt}...")
        synapse_model = PCModel(config=CurrentConfig).to(device)
        ckpt = torch.load(args.synapse_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            print(">> Detected Unified Checkpoint format.")
            state_dict = ckpt['model_state']
        else:
            state_dict = ckpt
            
        synapse_model.load_state_dict(state_dict, strict=False)
        
        loss, ppl = calculate_ppl(synapse_model, test_loader, device, is_baseline=False, vocab_size=adapter.vocab_size)
        results['Synapse'] = {'Loss': loss, 'PPL': ppl}
        del synapse_model
    else:
        print(f"!! Skipping Synapse (Checkpoint {args.synapse_ckpt} not found)")

    # 3. Evaluate Baseline
    if args.baseline_ckpt and os.path.exists(args.baseline_ckpt):
        print(f">> Loading Baseline GPT-2 from {args.baseline_ckpt}...")
        config = GPT2Config(vocab_size=adapter.vocab_size, n_positions=CurrentConfig.seq_len, n_embd=CurrentConfig.embed_dim, n_layer=CurrentConfig.n_layers, n_head=CurrentConfig.n_heads)
        baseline_model = GPT2LMHeadModel(config).to(device)
        baseline_model.load_state_dict(torch.load(args.baseline_ckpt, map_location=device))
        
        loss, ppl = calculate_ppl(baseline_model, test_loader, device, is_baseline=True, vocab_size=adapter.vocab_size)
        results['Baseline'] = {'Loss': loss, 'PPL': ppl}
        del baseline_model
    else:
        print(f"!! Skipping Baseline (Checkpoint {args.baseline_ckpt} not found)")
        
    print("\n" + "="*40)
    print("       BENCHMARK RESULTS (WIKITEXT-2)")
    print("="*40)
    print(f"{'Model':<15} | {'Loss':<10} | {'Perplexity':<10}")
    print("-" * 40)
    for name, data in results.items():
        print(f"{name:<15} | {data['Loss']:.4f}     | {data['PPL']:.2f}")
    print("="*40)

if __name__ == "__main__":
    benchmark_all()
