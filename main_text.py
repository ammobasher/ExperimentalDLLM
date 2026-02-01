import jax
import jax.numpy as jnp
import numpy as np
import time
import equinox as eqx

from src.config import Config
from src.meta_trainer import MetaTrainer
from src.text_adapter import TextAdapter

import jax
import jax.numpy as jnp
import numpy as np
import time
import equinox as eqx
import argparse
import os
import shutil

from src.config import Config
from src.meta_trainer import MetaTrainer
from src.text_adapter import TextAdapter
from src.evaluate_metrics import calculate_metrics

def save_checkpoint(path, model, controller, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    eqx.tree_serialise_leaves(path, (model, controller, step))
    print(f"   >> 💾 Checkpoint saved: {path}")

def main_text():
    parser = argparse.ArgumentParser(description="Train Diffusion LLM on WikiText")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate metrics every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help="Batch size")
    args = parser.parse_args()

    print("==================================================")
    print(f"    PROJECT SYNAPSE: REAL TEXT DIFFUSION LLM     ")
    print(f"    Steps: {args.steps} | Batch: {args.batch_size}")
    print("==================================================")
    
    key = jax.random.PRNGKey(42)
    key, t_key = jax.random.split(key)
    
    # 1. Initialize Adapter
    print("\n[System] Initializing Text Adapter (WikiText + GPT2)...")
    adapter = TextAdapter(seq_len=Config.seq_len, batch_size=args.batch_size)
    vocab_size = adapter.vocab_size
    print(f"Vocab Size: {vocab_size}")
    
    # 2. Initialize Trainer
    Config.vocab_size = vocab_size # Patch vocab size
    trainer = MetaTrainer(key) 
    
    print("[System] Model Initialized.")
    
    # 3. Training Loop
    last_beta = jnp.array(1.0)
    
    # Setup Checkpoints
    if os.path.exists(args.checkpoint_dir):
        print(f"Warning: Checkpoint dir '{args.checkpoint_dir}' exists.")
    
    print(f"\n[Training] Starting {args.steps} steps...")
    
    try:
        for step in range(1, args.steps + 1):
            # Get Batch
            batch_tokens = adapter.get_batch()
            val_tokens = batch_tokens # Reuse for speed in demo
            
            start_t = time.time()
            
            # Train Step
            (
                new_model, new_ctrl, s_llm, s_ctrl, metrics, current_beta
            ) = trainer.train_step(
                trainer.model, trainer.controller, 
                trainer.state_llm, trainer.state_ctrl,
                batch_tokens, val_tokens, t_key, last_beta
            )
            dt = time.time() - start_t
            
            # Update
            trainer.model = new_model
            trainer.controller = new_ctrl
            trainer.state_llm = s_llm
            trainer.state_ctrl = s_ctrl
            last_beta = current_beta
            
            # Log
            if step % 1 == 0:
                print(f"[Step {step}/{args.steps}] Loss: {metrics['loss_inner']:.4f} | Beta: {metrics['beta']:.4f} | Time: {dt:.3f}s")
            
            # Text Sample
            if step % 50 == 0:
                 print(f"   Sample: \"{adapter.decode(batch_tokens[0])[:50]}...\"")

            # Evaluation
            if step % args.eval_every == 0:
                print(f"\n[Step {step}] Running Evaluation...")
                ppl, bpd = calculate_metrics(trainer, adapter, n_batches=2) # Short eval
                print(f"   >> 📊 PPL: {ppl:.2f} | BPD: {bpd:.2f}")

            # Checkpoint
            if step % args.save_every == 0:
                ckpt_path = f"{args.checkpoint_dir}/step_{step}.eqx"
                save_checkpoint(ckpt_path, trainer.model, trainer.controller, step)

    except KeyboardInterrupt:
        print("\n\n!! Training Interrupted. Saving Emergency Checkpoint...")
        save_checkpoint(f"{args.checkpoint_dir}/emergency_stop.eqx", trainer.model, trainer.controller, step)

    print("\n>> Training Complete.")

if __name__ == "__main__":
    main_text()
