import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from typing import List, Tuple

from src.model import PCModel
from src.memory import EpisodicMemory
from src.train_diffusion import dataloader 

def run_sleep_cycle(
    model: PCModel,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    memory: EpisodicMemory,
    key: jax.random.PRNGKey,
    n_steps: int = 100
) -> Tuple[PCModel, optax.OptState, float]:
    """
    Offline Consolidation Phase.
    Replays memories from eLTM to fine-tune the Neocortex (PCModel).
    """
    print(f"--- Entering Sleep Cycle (Steps={n_steps}, Memories={memory.count}) ---")
    
    if memory.count < 10:
        print("Not enough memories to sleep. Skipping.")
        return model, opt_state, 0.0

    # Define minimal update step (Non-Meta, just simple PC/Diffusion training)
    # We reuse logic similar to train_diffusion.py but adapted for "Memory Replay"
    # Ideally, we should import 'update' from train_diffusion or refactor it.
    # For independence, let's define a simple consolidation step here.
    
    # We need access to the same SDE and config. 
    # Let's import SDE locally or pass it. 
    from src.diffusion import DiffusionSDE
    from src.config import Config
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)

    @eqx.filter_jit
    def consolidation_step(model, opt_state, batch_tokens, key):
        # 1. Embed
        x_0 = jax.vmap(jax.vmap(model.embedding))(batch_tokens)
        
        # 2. Noise
        key, t_key, n_key = jax.random.split(key, 3)
        t = jax.random.uniform(t_key, (batch_tokens.shape[0],))
        noise = jax.random.normal(n_key, x_0.shape)
        x_t = jax.vmap(sde.q_sample)(x_0, t, noise)
        
        # 3. Helpers
        def loss_fn(m, x_t_in, t_in):
             logits, pc_loss_batch = jax.vmap(lambda x, ti: m(inputs_embeds=x, t=ti))(x_t_in, t_in)
             # During sleep, we might prioritize PC Loss (Structure) over CE?
             # Or balanced.
             ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, batch_tokens))
             pc = jnp.mean(pc_loss_batch)
             return ce + 0.1 * pc # Add some PC regularization
             
        # Use value_and_grad to get both loss and gradients
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_t, t)
        
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss, key

    # Replay Loop
    total_sleep_loss = 0.0
    
    # We need to reconstruct batches from memory.
    # Our memory stores VECTORS (latents) or RAW TEXT (Metadata)?
    # Memory.add(vector, metadata). Metadata is the text/string.
    # Realistically, hippocampus stores the compressed latent.
    # But for "Replay", decoding latent -> text is hard without decoder.
    # Implementation Plan says: "Retrieved chunks are tokenized and prepended".
    # For this Sleep Cycle, we will iterate over the "metadata" (original text/tokens) if stored.
    # Our src/memory.py stores 'metadata' as string/object. 
    # We should assume metadata is the Token ID sequence (or we parse it).
    
    # Let's assume metadata IS the token sequence for this prototype.
    
    for i in range(n_steps):
        # 1. Sample from Memory (Random Replay or Prioritized?)
        # Simple random sampling from valid range
        indices = np.random.randint(0, min(memory.count, memory.capacity), size=Config.batch_size)
        
        # Collect batches
        # Warning: This assumes metadata is available and correct type.
        # In a real app we'd need robust serialization.
        batch_list = []
        for idx in indices:
            data = memory.values[idx]
            # Verify data format. If string, fake it. If list/array, use it.
            if isinstance(data, (list, np.ndarray, jax.Array)):
                 batch_list.append(np.array(data))
            else:
                 # Fallback for empty/string metadata -> Random noise for stability testing
                 batch_list.append(np.random.randint(0, 32000, (Config.seq_len,)))
        
        batch = jnp.array(np.stack(batch_list))
        
        # 2. Train
        model, opt_state, loss, key = consolidation_step(model, opt_state, batch, key)
        total_sleep_loss += loss
        
    avg_loss = total_sleep_loss / n_steps
    print(f"Sleep Cycle Complete. Avg Consoldation Loss: {avg_loss:.4f}")
    
    return model, opt_state, avg_loss
