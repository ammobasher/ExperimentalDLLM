import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import os
import time
import math
import numpy as np

from src.config import Config
from src.meta_trainer_torch import MetaTrainer
from src.memory import EpisodicMemory
from src.sleep import run_sleep_cycle


class CachedDataLoader:
    """Fast data loader from pre-cached npz files."""
    
    def __init__(self, cache_dir, device):
        self.device = device
        self.step = 0
        
        # Load text cache
        text_path = os.path.join(cache_dir, "text_cache.npz")
        if os.path.exists(text_path):
            print(f">> Loading text cache from {text_path}...")
            data = np.load(text_path)
            self.text_batches = data['batches']  # [N, B, SeqLen]
            self.n_text_steps = len(self.text_batches)
            print(f"   Loaded {self.n_text_steps} text batches")
        else:
            raise FileNotFoundError(f"Text cache not found: {text_path}")
        
        # Load vision cache
        vision_path = os.path.join(cache_dir, "vision_cache.npz")
        if os.path.exists(vision_path):
            print(f">> Loading vision cache from {vision_path}...")
            data = np.load(vision_path)
            self.vision_latents = data['latents']  # [N, B, C, H, W]
            self.vision_input_ids = data['input_ids']  # [N, B, SeqLen]
            self.n_vision_steps = len(self.vision_latents)
            self.has_vision = True
            print(f"   Loaded {self.n_vision_steps} vision batches")
        else:
            print(f"   Vision cache not found, using text-only mode")
            self.has_vision = False
    
    def get_train_batch(self):
        """Get multimodal training batch (inner loop)."""
        idx = self.step % self.n_vision_steps if self.has_vision else 0
        
        if self.has_vision:
            latents = torch.from_numpy(self.vision_latents[idx]).float().to(self.device)
            input_ids = torch.from_numpy(self.vision_input_ids[idx]).long().to(self.device)
        else:
            latents = None
            input_ids = torch.from_numpy(self.text_batches[self.step % self.n_text_steps]).long().to(self.device)
        
        return input_ids, latents
    
    def get_val_batch(self):
        """Get text validation batch (outer loop)."""
        # Use different indices for validation
        idx = (self.step + self.n_text_steps // 2) % self.n_text_steps
        return torch.from_numpy(self.text_batches[idx]).long().to(self.device)
    
    def advance(self):
        self.step += 1


def main_unified():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500000,
                       help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=10000,
                       help="Run evaluation every N steps (0=disabled)")
    parser.add_argument("--sleep_every", type=int, default=5000, 
                       help="Steps between sleep cycles (0=disabled)")
    parser.add_argument("--meta_update_every", type=int, default=10, 
                       help="Steps between BLO controller updates")
    parser.add_argument("--cache_dir", type=str, default="cached_data", 
                       help="Directory with cached data")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from latest checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000,
                       help="LR warmup steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                       help="Minimum learning rate at end of training")
    args = parser.parse_args()

    # 1. Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Phase 25: Optimized Unified Training on {device}")
    
    # 2. Config
    Config.vocab_size = 50257 
    print(f">> Config Vocab Size: {Config.vocab_size}")
    print(f">> Meta Update Frequency: Every {args.meta_update_every} steps")
    print(f">> Sleep Cycle Frequency: Every {args.sleep_every} steps")

    # 3. Load Cached Data
    data_loader = CachedDataLoader(args.cache_dir, device)

    # 4. Initialize System
    trainer = MetaTrainer(device)
    
    # 5. Setup LR Scheduler (Warmup + Cosine Decay)
    def get_lr(step):
        """Warmup + Cosine decay schedule."""
        if step < args.warmup_steps:
            # Linear warmup
            return args.lr * (step / args.warmup_steps) 
        else:
            # Cosine decay
            progress = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
            return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    
    def update_lr(optimizer, step):
        """Update optimizer learning rate."""
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    # 6. Initialize Episodic Memory
    memory = EpisodicMemory(dim=Config.embed_dim, capacity=5000)  # Larger capacity for long runs
    print(f">> Episodic Memory initialized (Capacity: {memory.capacity})")
    
    # 6. Load Pre-trained Weights
    ckpt_path = "checkpoints_torch/step_50000.pt"
    if os.path.exists(ckpt_path):
        print(f">> Loading Pre-trained Synapse Core...")
        state_dict = torch.load(ckpt_path, map_location=device)
        try:
            trainer.model.load_state_dict(state_dict, strict=False)
            print(">> Synapse Brain Loaded.")
        except Exception as e:
            print(f">> Warning: Load failed: {e}")
            
    # 7. Resume from checkpoint if requested
    start_step = 1
    if args.resume:
        # Find latest checkpoint in checkpoints_unified
        ckpt_files = [f for f in os.listdir("checkpoints_unified") if f.startswith("step_") and f.endswith(".pt")]
        if ckpt_files:
            # Sort by step number
            ckpt_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            latest = ckpt_files[-1]
            resume_path = f"checkpoints_unified/{latest}"
            print(f">> Resuming from {resume_path}...")
            ckpt = torch.load(resume_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                trainer.model.load_state_dict(ckpt['model_state'], strict=False)
                if 'ctrl_state' in ckpt:
                    trainer.controller.load_state_dict(ckpt['ctrl_state'])
                start_step = ckpt.get('step', 0) + 1
                if 'memory_count' in ckpt:
                    print(f"   Previous memory count: {ckpt['memory_count']}")
            print(f">> Resuming from step {start_step}")
    
    # 9. Training Loop
    prev_beta = torch.tensor([1.0], device=device)
    start_time = time.time()
    best_val_loss = float('inf')
    
    print("\n>> Starting Synapse Unified Training...")
    print("   [Inner Loop]: Vision-Grounded Text (Cached)")
    print("   [Outer Loop]: General Language Modeling (Cached)")
    print(f"   [BLO]: Controller updates every {args.meta_update_every} steps")
    print(f"   [LR]: Warmup {args.warmup_steps} steps, then cosine decay to {args.min_lr}")
    if args.resume:
        print(f"   [Resume]: Starting from step {start_step}")
    
    for step in range(start_step, args.steps + 1):
        # Set data loader to correct position
        data_loader.step = step - 1
        
        # A. Update Learning Rate
        current_lr = update_lr(trainer.optimizer_llm, step)
        
        # B. Fetch Cached Data
        train_ids, vis_latents = data_loader.get_train_batch()
        val_ids = data_loader.get_val_batch()
        data_loader.advance()
        
        # B. Training Step
        # Only do full BLO update every N steps
        if step % args.meta_update_every == 0:
            # Full step with controller update
            metrics = trainer.train_step(
                train_batch=train_ids, 
                val_batch=val_ids, 
                prev_beta=prev_beta, 
                visual_latents=vis_latents
            )
        else:
            # Fast step: No controller update
            metrics = trainer.simple_step(
                train_batch=train_ids,
                prev_beta=prev_beta.item(),
                visual_latents=vis_latents
            )
        
        prev_beta = torch.tensor([metrics['beta']], device=device)
        
        # C. Memory Storage
        if metrics['loss_pc'] > memory.threshold_tau:
            with torch.no_grad():
                embed = trainer.model(train_ids, return_embeds=True).mean(dim=1)
            stored = memory.add(
                embed[0].cpu().numpy(),
                train_ids[0].cpu().tolist(),
                metrics['loss_pc']
            )
            if stored and step % 100 == 0:
                print(f"  [Memory] Stored @ Step {step} (Count: {memory.count})")
        
        # D. Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = 100 / elapsed
            eta_hours = (args.steps - step) / steps_per_sec / 3600
            print(f"[Step {step}] Loss: {metrics['loss_total']:.4f} "
                  f"| CE: {metrics['loss_ce']:.4f} | PC: {metrics['loss_pc']:.4f} "
                  f"| β: {metrics['beta']:.3f} | LR: {current_lr:.2e} | Mem: {memory.count} "
                  f"| {steps_per_sec:.1f} steps/s | ETA: {eta_hours:.1f}h")
            start_time = time.time()
        
        # E. Sleep Cycle
        if args.sleep_every > 0 and step % args.sleep_every == 0 and memory.count >= 10:
            run_sleep_cycle(
                model=trainer.model,
                optimizer=trainer.optimizer_llm,
                memory=memory,
                device=device,
                n_steps=50
            )
            
        # F. Checkpoint
        if step % args.save_every == 0:
            os.makedirs("checkpoints_unified", exist_ok=True)
            path = f"checkpoints_unified/step_{step}.pt"
            torch.save({
                'model_state': trainer.model.state_dict(),
                'ctrl_state': trainer.controller.state_dict(),
                'memory_count': memory.count,
                'step': step,
                'lr': current_lr
            }, path)
            print(f">> Checkpoint saved: {path}")
        
        # G. Evaluation Hook
        if args.eval_every > 0 and step % args.eval_every == 0 and step > 0:
            try:
                from src.metrics import evaluate_generation_quality
                from transformers import GPT2TokenizerFast
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
                
                eval_results = evaluate_generation_quality(
                    trainer.model, tokenizer, device, n_samples=10
                )
                print(f"  [Eval @ {step}] Distinct-1: {eval_results['distinct_1']:.3f} | "
                      f"Distinct-2: {eval_results['distinct_2']:.3f} | "
                      f"Repetition: {eval_results['avg_repetition']:.3f}")
            except Exception as e:
                print(f"  [Eval Error]: {e}")
    
    # Save final memory
    memory.save("checkpoints_unified/memory_final.npz")
    print(">> Training Complete. Final memory saved.")


if __name__ == "__main__":
    main_unified()
