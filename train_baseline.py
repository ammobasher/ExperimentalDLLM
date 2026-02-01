import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import time
from transformers import GPT2Config, GPT2LMHeadModel
from torch.amp import autocast, GradScaler

from src.config import Config
from src.text_adapter import TextAdapter

def train_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4) # Match Synapse phys batch
    parser.add_argument("--grad_accum", type=int, default=8) # Match Synapse effective batch (32)
    parser.add_argument("--use_amp", action="store_true", default=True) # Enabled by default for MPS
    parser.add_argument("--dataset", type=str, default="wikitext") # Match Synapse dataset
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_baseline")
    args = parser.parse_args()
    
    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">> Using Apple Metal acceleration (MPS) 🚀")
    else:
        device = torch.device("cpu")
        print(">> Using CPU 🐢")
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 2. Data Loader
    adapter = TextAdapter(seq_len=Config.seq_len, batch_size=args.batch_size, dataset_name=args.dataset)
    vocab_size = adapter.vocab_size
    print(f">> Loaded Tokenizer. Vocab Size: {vocab_size}")

    # 3. Model Setup (Control Group: Standard GPT-2)
    # Match Synapse Config: 12 Layers, 1024 Hidden, 16 Heads
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=Config.seq_len,
        n_embd=Config.embed_dim,
        n_layer=Config.n_layers,
        n_head=Config.n_heads,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    model = GPT2LMHeadModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f">> Baseline Model Initialized ({param_count/1e6:.1f}M Params)")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("mps") if (args.use_amp and device.type == "mps") else GradScaler("cpu")
    
    # 4. Training Loop
    print(f"Starting Baseline Training for {args.steps} steps...")
    print(f"Configs: Phys Batch={args.batch_size} | Grad Accum={args.grad_accum} | Effective Batch={args.batch_size * args.grad_accum}")
    
    model.train()
    start_time = time.time()
    accum_loss = 0.0
    
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        
        # Inner loop for gradient accumulation
        current_accum_loss = 0.0
        for _ in range(args.grad_accum):
            input_ids_np = adapter.get_batch()
            input_ids = torch.from_numpy(np.array(input_ids_np)).long().to(device)
            
            # Forward with AMP
            with autocast(device_type=device.type, enabled=args.use_amp):
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss / args.grad_accum
            
            # Backward
            scaler.scale(loss).backward()
            current_accum_loss += loss.item() * args.grad_accum
            
        # Step
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}] Baseline Loss: {current_accum_loss:.4f} | PPL: {np.exp(min(current_accum_loss, 20)):.2f} | Time: {elapsed:.2f}s")
            start_time = time.time()
            
        # Checkpoint
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"baseline_step_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f">> Saved Baseline Checkpoint: {ckpt_path}")
            
    print(">> Baseline training complete.")

if __name__ == "__main__":
    train_baseline()
