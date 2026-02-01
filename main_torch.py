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
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", help="Use Mixed Precision (AMP)")
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
    
    # 3.5 AMP Setup
    scaler = None
    if args.use_amp:
        # Note: torch.amp.GradScaler is more robust for cross-device
        scaler = torch.amp.GradScaler("mps") if device.type == "mps" else torch.amp.GradScaler("cuda")
        print(f">> Mixed Precision (AMP) Enabled for {device.type.upper()}")

    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    memory = EpisodicMemory(dim=Config.embed_dim)

    
    # 4. Training Loop
    start_step = 1
    if args.resume:
        # Find latest checkpoint
        files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")]
        if len(files) > 0:
            # Sort by step number
            files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            latest_ckpt = files[-1]
            ckpt_path = os.path.join(args.checkpoint_dir, latest_ckpt)
            
            print(f">> Resuming from checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            
            start_step = int(latest_ckpt.split("_")[1].split(".")[0]) + 1
            print(f">> Start step adjusted to {start_step}")
        else:
            print(">> No checkpoints found to resume from. Starting from scratch.")

    print(f"Starting Training for {args.steps} steps (from {start_step})...")
    model.train()
    
    start_time = time.time()
    
    optimizer.zero_grad()
    
    for step in range(start_step, args.steps + 1):
        # A. Fetch Data
        input_ids_np = adapter.get_batch()
        input_ids = torch.from_numpy(np.array(input_ids_np)).long().to(device)
        
        # B. Forward Pass (Diffusion Training)
        t = torch.rand(args.batch_size, device=device) # [0, 1]
        
        # Use AMP context if enabled
        with torch.amp.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=args.use_amp):
            # Embed X_0
            x_0 = model.embedding(input_ids) # [B, Seq, Dim]
            
            # Add Noise
            mean, std = sde.marginal_prob(x_0, t)
            noise = torch.randn_like(x_0)
            x_t = mean + (std * noise)
            
            # Run Model
            logits, pc_loss = model(inputs_embeds=x_t, t=t)
            
            # Loss Calculation
            loss_ce = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), input_ids.reshape(-1))
            total_loss = (loss_ce + pc_loss) / args.grad_accum
        
        # C. Backward
        if scaler:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
            
        if step % args.grad_accum == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # D. Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}] Loss (Accum): {total_loss.item() * args.grad_accum:.4f} (CE: {loss_ce.item():.4f} | PC: {pc_loss.item():.4f}) | Time: {elapsed:.2f}s")
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
