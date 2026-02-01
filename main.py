import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time

from src.config import Config
from src.meta_trainer import MetaTrainer
from src.memory import EpisodicMemory
from src.sleep import run_sleep_cycle

def main():
    print("==================================================")
    print("       PROJECT SYNAPSE: SYNTHETIC NEOCORTEX       ")
    print("==================================================")
    
    key = jax.random.PRNGKey(1337)
    
    # 1. Initialize System
    print("\n[System] Initializing Components...")
    trainer = MetaTrainer(key)
    memory = EpisodicMemory(key, dim=Config.embed_dim, capacity=1000)
    
    print(f"Model: PCModel ({Config.n_layers} layers)")
    print(f"Controller: BetaController (Neuromodulator)")
    print(f"Memory: eLTM (Capacity {memory.capacity})")
    
    # 2. Simulation Loop (Days & Nights)
    n_days = 2
    steps_per_day = 5
    
    # Dummy Data Generator
    def get_batch(k):
        # Return discrete tokens [Batch, Seq]
        return jax.random.randint(k, (Config.batch_size, Config.seq_len), 0, 32000)

    global_step = 0
    
    for day in range(1, n_days + 1):
        print(f"\n☀️  DAY {day} STARTED")
        print("-" * 30)
        
        # --- WAKING PHASE (Meta-Learning) ---
        for step in range(steps_per_day):
            key, k1, k2, k3 = jax.random.split(key, 4)
            train_batch = get_batch(k1)
            val_batch = get_batch(k2) # In reality, use separate val set
            
            # Retrieve from Memory? (Not implemented in training loop yet, but conceptual)
            # context = memory.retrieve(...)
            # train_batch = append(context, train_batch)
            
            # BLO Step
            # Note: We need to pass the CURRENT beta state? 
            # In trainer.train_step, beta_val is computed from controller. 
            # We pass a 'prev_beta' as input feature.
            # Let's keep track of last beta.
            if global_step == 0:
                last_beta = jnp.array(1.0)
            else:
                last_beta = current_beta_val # From prev step
                
            start_t = time.time()
            (
                new_model, 
                new_ctrl, 
                new_s_llm, 
                new_s_ctrl, 
                metrics, 
                current_beta_val
            ) = trainer.train_step(
                trainer.model, trainer.controller, 
                trainer.state_llm, trainer.state_ctrl,
                train_batch, val_batch, k3, last_beta
            )
            dt = time.time() - start_t
            
            # Update State
            trainer.model = new_model
            trainer.controller = new_ctrl
            trainer.state_llm = new_s_llm
            trainer.state_ctrl = new_s_ctrl
            
            # Log
            print(f"[Step {global_step}] Loss: {metrics['loss_inner']:.4f} | Beta: {metrics['beta']:.4f} | Meta-Loss: {metrics['loss_meta']:.4f} | Time: {dt:.3f}s")
            
            # --- MEMORY TRIGGER ---
            # Check novelty of this batch (using PC Loss Mean)
            # In real system, check per sample. Here aggregate.
            batch_loss_val = float(metrics['pc_inner'])
            
            # We treat the text itself (metadata) as something to save?
            # Save the first sequence of the batch as a representative memory
            # Vector: We need the latent vector. 
            # Ideally trainer returns latent, but for prototype we use random vec or re-embed.
            # Let's just use a dummy vector for the prototype key.
            # Real implementation: Extract latent x_t or x_final from model.
            dummy_key = np.random.randn(Config.embed_dim).astype(np.float32)
            
            saved = memory.add(dummy_key, train_batch[0], batch_loss_val)
            if saved:
                print(f"   >> 🧠 Event Saved to Hippocampus (Loss {batch_loss_val:.2f} > Threshold)")
            
            global_step += 1
            
        # --- SLEEPING PHASE (Consolidation) ---
        print(f"\n🌙 NIGHT {day} (Consolidating...)")
        
        trainer.model, trainer.state_llm, sleep_loss = run_sleep_cycle(
            trainer.model, 
            trainer.opt_llm, # Helper to get gradients, but run_sleep_cycle assumes 'optimizer' object not state
            trainer.state_llm, 
            memory, 
            key, 
            n_steps=3 # Short sleep
        )
        
        print(f"   >> 💤 Sleep Complete. Consolidation Loss: {sleep_loss:.4f}")

    print("\n==================================================")
    print("          SYSTEM SHUTDOWN: SUCCESSS               ")
    print("==================================================")

if __name__ == "__main__":
    main()
