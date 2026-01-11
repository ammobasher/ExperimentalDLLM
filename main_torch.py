import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import time

from src.model import PCModel
from src.diffusion import DiffusionSDE
from src.config import Config
from src.memory import EpisodicMemory
from src.text_adapter import TextAdapter

def main_torch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--sleep_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_torch")
    args = parser.parse_args()
    
    # 1. Hardware Setup (Apple Metal)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">> Using Apple Metal acceleration (MPS) 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(">> Using NVIDIA CUDA acceleration 🚀")
    else:
        device = torch.device("cpu")
        print(">> Using CPU (Slow) 🐢")
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 2. Data Loader (Init first to get correct vocab size)
    adapter = TextAdapter(seq_len=Config.seq_len, batch_size=args.batch_size, dataset_name=args.dataset)
    vocab_size = adapter.vocab_size
    Config.vocab_size = vocab_size # Patch Config
    print(f">> Loaded Tokenizer. Vocab Size: {vocab_size}")

    # 3. Model & Optimizer
    model = PCModel().to(device)
    print(f">> Model Initialized ({sum(p.numel() for p in model.parameters())/1e6:.1f}M Params)")

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr_llm)
    
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    memory = EpisodicMemory(dim=Config.embed_dim)

    
    # 4. Training Loop
    print(f"Starting Training for {args.steps} steps...")
    model.train()
    
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        # A. Fetch Data
        # adapter.get_batch() returns numpy array [B, Seq]
        input_ids_np = adapter.get_batch()
        input_ids = torch.from_numpy(np.array(input_ids_np)).long().to(device)
        
        # B. Forward Pass (Diffusion Training)
        # Sample t
        t = torch.rand(args.batch_size, device=device) # [0, 1]
        
        optimizer.zero_grad()
        
        # Embed X_0
        x_0 = model.embedding(input_ids) # [B, Seq, Dim]
        
        # Add Noise
        drift, diffusion = sde.sde(x_0, t) # drift usually applied in ode, here we need Marginal
        mean, std = sde.marginal_prob(x_0, t)
        noise = torch.randn_like(x_0)
        x_t = mean + (std * noise)
        
        # Run Model
        # inputs_embeds arg in Model overrides input_ids
        logits, pc_loss = model(inputs_embeds=x_t, t=t)
        # Model returns Denoised Estimate (Logits -> X_0_hat) or Score?
        # In current design, model predicts X_0 via logits.
        
        # Diffusion Loss
        # We need to compute Loss(x_0, x_0_hat)
        # Since outputs are logits, we use CrossEntropy against Input Ids?
        # Or MSE against embeddings? 
        # PCModel philosophy: Standard Cross Entropy on final output handles the "Reconstruction".
        # But for diffusion, we usually want MSE on noise or x_0.
        # Let's stick to Cross Entropy on Tokens for the "LLM" aspect.
        # Loss = CE(logits, input_ids) + PC_Loss
        
        loss_ce = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), input_ids.reshape(-1))
        
        total_loss = loss_ce + pc_loss
        
        # C. Backward
        total_loss.backward()
        optimizer.step()
        
        # D. Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}] Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f} | PC: {pc_loss.item():.4f}) | Time: {elapsed:.2f}s")
            start_time = time.time()
            
        # E. Sleep Cycle (Replay)
        if step % args.sleep_every == 0 and memory.count > 0:
            print(f">> [Sleep] Consolidating {memory.count} Memories (Step {step})...")
            # Replay Loop
            sleep_steps = min(memory.count, 100) # Replay up to 100 memories
            
            # Create a localized optimizer for sleep? Or use main one?
            # Standard: Use main optimizer (plasticity).
            
            # Retrieve 'all' memories (or random batch)
            # memory.values stores the tokens [Seq]
            # We reconstruct batches from memory.
            
            # Simplified Replay: Just iterate
            for _ in range(5): # 5 batches of sleep
                # Sample random indices
                indices = np.random.randint(0, memory.count, size=args.batch_size)
                
                # Fetch tokens from memory values
                batch_tokens = []
                valid_batch = False
                for idx in indices:
                    val = memory.values[idx % memory.capacity]
                    if val is not None:
                        batch_tokens.append(val)
                
                if len(batch_tokens) > 0:
                    # Pad/Stack
                    # Assumes stored tokens are truncated to seq_len
                    # We need to ensure they match expected shape
                    max_len = Config.seq_len
                    padded_batch = np.zeros((len(batch_tokens), max_len), dtype=np.int64)
                    for i, seq in enumerate(batch_tokens):
                        sl = min(len(seq), max_len)
                        padded_batch[i, :sl] = seq[:sl]
                        
                    sleep_input = torch.from_numpy(padded_batch).long().to(device)
                    
                    # Train Step (Sleep)
                    optimizer.zero_grad()
                    x_0_s = model.embedding(sleep_input)
                    t_s = torch.rand(len(batch_tokens), device=device)
                    
                    # Noise
                    mean_s, std_s = sde.marginal_prob(x_0_s, t_s)
                    x_t_s = mean_s + (std_s * torch.randn_like(x_0_s))
                    
                    logits_s, pc_loss_s = model(inputs_embeds=x_t_s, t=t_s)
                    loss_ce_s = nn.CrossEntropyLoss()(logits_s.reshape(-1, vocab_size), sleep_input.reshape(-1))
                    
                    loss_sleep = loss_ce_s + pc_loss_s
                    loss_sleep.backward()
                    optimizer.step()
            print(">> [Sleep] Cycle Complete. Memories integrated.")

        # F. Memory Population (Surprise-Based)
        # Check per-sample loss to find 'surprising' data
        # We re-calculate CE with reduction='none' for this batch to get per-sample info
        with torch.no_grad():
            per_sample_loss = nn.CrossEntropyLoss(reduction='none')(logits.reshape(-1, vocab_size), input_ids.reshape(-1))
            per_sample_loss = per_sample_loss.view(args.batch_size, -1).mean(dim=1) # [Batch]
            
            # Add to memory if high loss
            for i in range(args.batch_size):
                loss_val = per_sample_loss[i].item()
                # Threshold check handled inside memory.add
                # Vector key: Mean of x_0 (clean embedding)
                key_vec = x_0[i].mean(dim=0).cpu().numpy() # [Dim]
                token_data = input_ids_np[i] # Metadata
                
                memory.add(key_vec, token_data, loss_val)

        
        # G. Save Checkpoint
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f">> Checkpoint saved to {ckpt_path}")
            
import sys

if __name__ == "__main__":
    main_torch()
    print(">> Done. Exiting.")
    sys.exit(0)
