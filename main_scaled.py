import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import os
import time
import numpy as np

from src.config import Config
from src.meta_trainer import MetaTrainer
from src.text_adapter import TextAdapter
from src.evaluate_metrics import calculate_metrics
from src.sharding_utils import create_mesh_sharding, replicate_state, shard_batch
from src.checkpoint_manager import CheckpointManager
from src.memory import EpisodicMemory
from src.sleep import run_sleep_cycle

def main_scaled():
    parser = argparse.ArgumentParser(description="Scaled Training (300M+ Params)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32) # Larger batch for multi-device
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_scaled")
    parser.add_argument("--dataset", type=str, default="openwebtext", choices=["openwebtext", "wikitext"], help="Dataset to use")
    parser.add_argument("--sleep_every", type=int, default=5000, help="Steps between sleep cycles")
    args = parser.parse_args()

    print("==================================================")
    print(f"    PROJECT SYNAPSE: SCALE-UP (PHASE 11)         ")
    print(f"    Mode: Data Parallel Sharding                 ")
    print("==================================================")
    
    # 1. Setup JAX Mesh
    sharding, n_devices = create_mesh_sharding()
    print(f"[System] Sharding across {n_devices} devices.")
    
    # 2. Config for Scaling
    # If 300M target, we overdrive Config
    # Config.n_layers = 12
    # Config.n_heads = 16
    # Config.embed_dim = 1024
    # print(f"[Config] Scaling Up: 12 Layers, 1024 Dim, 16 Heads")
    
    key = jax.random.PRNGKey(42)
    key, t_key = jax.random.split(key)
    
    # 3. Adapter
    dataset_name = args.dataset
    adapter = TextAdapter(seq_len=Config.seq_len, batch_size=args.batch_size, dataset_name=dataset_name)
    Config.vocab_size = adapter.vocab_size
    
    # 4. Initialize Trainer & Replicate
    print("[System] Initializing & Replicating Model...")
    trainer = MetaTrainer(key)
    
    # Replicate weights to all devices
    trainer.model = replicate_state(trainer.model)
    trainer.controller = replicate_state(trainer.controller)
    trainer.state_llm = replicate_state(trainer.state_llm)
    trainer.state_ctrl = replicate_state(trainer.state_ctrl)
    
    # 5. Checkpoint Manager
    ckpt_manager = CheckpointManager(args.checkpoint_dir)
    
    # 6. Episodic Memory (Hippocampus)
    # Hosted on CPU/Main process usually
    memory = EpisodicMemory(key, dim=Config.embed_dim)
    
    # 7. Loop
    print(f"\n[Training] Starting {args.steps} steps...")
    
    last_beta = jnp.array(1.0) # On host, let JAX broadcast it or we replicate it
    # last_beta = replicate_state(last_beta)

    for step in range(1, args.steps + 1):
        # 1. Get Batch & Shard It
        try:
            batch_cpu = adapter.get_batch()
        except StopIteration:
            print("Dataset exhausted.")
            break
            
        batch_sharded = shard_batch(batch_cpu, sharding) 
        # batch_sharded = jnp.array(batch_cpu) # Use vanilla JAX array 
        
        start_t = time.time()
        
        # 2. Step (JAX handles Pjit automatically if inputs are sharded correctly)
        (
            trainer.model, trainer.controller, 
            trainer.state_llm, trainer.state_ctrl, 
            metrics, last_beta
        ) = trainer.train_step(
            trainer.model, trainer.controller, 
            trainer.state_llm, trainer.state_ctrl,
            batch_sharded, batch_sharded, t_key, last_beta
        )
        
        # Block for timing accuracy
        jax.block_until_ready(metrics['loss_inner'])
        dt = time.time() - start_t
        
        # 3. Log (Mean across devices?)
        # metrics are sharded. We want to pull them to host.
        loss_val = metrics['loss_inner'].mean() # average over replicas if needed, or already scalar
        beta_val = metrics['beta'].mean()
        
        # --- Cognitive: Memory & Sleep ---
        # 1. Populate Memory
        # pc_loss_batch is sharded. We fetch it to CPU.
        pc_losses = jax.device_get(metrics['pc_loss_batch']) # [Batch]
        mem_keys = jax.device_get(metrics['memory_keys']) # [Batch, Dim]
        # batch_cpu is [Batch, Seq]
        
        # We iterate over the batch on CPU
        # Note: batch_sharded is usually same content as batch_cpu just on device
        # keys need not be precise for now, but let's try to get them from x_0 if possible.
        
        for i, val in enumerate(pc_losses):
            val = float(val)
            # Add to memory if high error
            
            # Real Key from Model
            key_vector = mem_keys[i]
            
            # Store the TOKENS as metadata
            meta_tokens = batch_cpu[i]
            memory.add(key_vector, meta_tokens, val)

        # 2. Sleep Trigger
        if step % args.sleep_every == 0 and memory.count > 10: # Lowered threshold for testing
             print(f"\n[Cognitive] Triggering Sleep Cycle at step {step}...")
             
             # Need to ensure model is in correct format (it is sharded eqx module)
             # run_sleep_cycle expects (model, optimizer, opt_state, memory, key)
             # It likely runs on Single Device or Host.
             # We pass the sharded model. JAX might handle it or we unshard.
             # For safety in this "Awakening" phase, let's assume JAX handles it 
             # or we might hit Sharding mismatch.
             
             # For now, we trust run_sleep_cycle's @eqx.filter_jit
             
             trainer.model, trainer.state_llm, sleep_loss = run_sleep_cycle(
                 trainer.model, trainer.opt_llm, trainer.state_llm, memory, t_key
             )
             
             # Save model and memory after sleep
             save_items = {
                 'model': trainer.model,
                 'controller': trainer.controller,
                 'state_llm': trainer.state_llm,
                 'state_ctrl': trainer.state_ctrl
             }
             save_items_filtered = eqx.filter(save_items, eqx.is_array)
             ckpt_manager.save(step, save_items_filtered)
             print(f"   >> 💾 Orbax Snapshot queued: step_{step}")
             
             # Save Memory (Numpy)
             memory_path = os.path.join(args.checkpoint_dir, f"memory_step_{step}.npz")
             memory.save(memory_path)
             print(f"   >> 🧠 Episodic Memory Saved: {memory.count} entries")

             # Re-replicate ensures consistency across mesh after sleep
             trainer.model = replicate_state(trainer.model)
             trainer.state_llm = replicate_state(trainer.state_llm)
             
             print(f"[Cognitive] Woke up. Sleep Loss: {sleep_loss:.4f}")

        if step % 1 == 0:
            print(f"[Step {step}] Loss: {loss_val:.4f} | Beta: {beta_val:.4f} | Time: {dt:.3f}s")

        # 4. Checkpoint (Orbax handles sharded arrays!)
        if step % args.save_every == 0:
            # We must filter out non-array types (like functions in Equinox modules)
            # to prevent Orbax from trying to serialize them.
            save_items = {
                'model': trainer.model,
                'controller': trainer.controller
            }
            save_items_filtered = eqx.filter(save_items, eqx.is_array)
            ckpt_manager.save(step, save_items_filtered)
            
            # Save Memory
            memory.save(os.path.join(args.checkpoint_dir, f"memory_step_{step}.npz"))

    print(">> Scaled Training Complete.")

if __name__ == "__main__":
    main_scaled()
