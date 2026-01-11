import torch
import numpy as np
import argparse
import os
import time

from src.config import Config
from src.text_adapter import TextAdapter
from src.meta_trainer_torch import MetaTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    # 1. Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Project Synapse: Meta-Training (Phase 20) on {device}")

    # 2. Data
    # Meta-Training requires a Train set and a Val set
    train_adapter = TextAdapter(batch_size=args.batch_size, split="train")
    val_adapter = TextAdapter(batch_size=args.batch_size, split="test")
    
    # 3. Trainer
    trainer = MetaTrainer(device)
    
    # 4. Loop
    prev_beta = torch.tensor([1.0], device=device) # Initial beta guess
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        # Fetch batches
        train_ids = torch.from_numpy(train_adapter.get_batch()).long().to(device)
        val_ids = torch.from_numpy(val_adapter.get_batch()).long().to(device)
        
        # Meta Step
        metrics = trainer.train_step(train_ids, val_ids, prev_beta)
        
        # Update rolling beta
        prev_beta = torch.tensor([metrics['beta']], device=device)
        
        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}] Total: {metrics['loss_total']:.4f} | "
                  f"CE: {metrics['loss_ce']:.4f} | PC: {metrics['loss_pc']:.4f} | "
                  f"Beta: {metrics['beta']:.4f} | Time: {elapsed:.2f}s")
            start_time = time.time()
            
        # Checkpointing
        if step % args.save_every == 0:
            ckpt_dir = "checkpoints_meta"
            os.makedirs(ckpt_dir, exist_ok=True)
            
            torch.save({
                'model_state': trainer.model.state_dict(),
                'ctrl_state': trainer.controller.state_dict(),
                'step': step
            }, os.path.join(ckpt_dir, f"meta_step_{step}.pt"))
            print(f">> Checkpoint saved at step {step}")

if __name__ == "__main__":
    main()
