import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from src.config import Config
from src.model import PCModel
from src.diffusion import DiffusionSDE

def dataloader(key, batch_size=32, seq_len=64, vocab_size=32000):
    """Infinite generator for dummy training data."""
    while True:
        key, subkey = jax.random.split(key)
        yield jax.random.randint(subkey, (batch_size, seq_len), 0, vocab_size)

def main():
    print("--- Initializing Synapse Training Loop ---")
    
    # 1. Setup
    key = jax.random.PRNGKey(42)
    key, model_key, loader_key = jax.random.split(key, 3)
    
    model = PCModel(model_key)
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    optimizer = optax.adamw(Config.lr_llm)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Enable JIT for the update step
    @eqx.filter_jit
    def train_step(model, opt_state, batch_tokens, key):
        # 1. Prepare Data (Embed X0)
        # We need model.embedding to get X0. 
        # batch_tokens is [Batch, Seq]. Embedding expects scalar.
        # We need to map over Batch AND Seq.
        x_0 = jax.vmap(jax.vmap(model.embedding))(batch_tokens)
        
        # 2. Sample Noise
        key, t_key, noise_key = jax.random.split(key, 3)
        batch_size = batch_tokens.shape[0]
        
        # Sample t uniform [0, T] or discrete [0, 1000]
        # Diffrax/Diffusion usually continuous [0, 1].
        t = jax.random.uniform(t_key, (batch_size,)) # [Batch]
        
        # Sample noise
        noise = jax.random.normal(noise_key, x_0.shape)
        
        # 3. Diffuse
        # vmap sde.q_sample over batch
        # q_sample(x0, t, noise)
        x_t = jax.vmap(sde.q_sample)(x_0, t, noise)
        
        # 4. Model Forward
        # vmap model over batch.
        # model(inputs_embeds=x_t, t=t) (t is handled by vmap if passed as array?)
        # model call signature: (input_ids, t, inputs_embeds).
        # We want to map inputs_embeds and t over batch dimensions.
        # We pass None for input_ids.
        
        def forward_single(xt_i, t_i):
            return model(input_ids=None, inputs_embeds=xt_i, t=t_i)
            
        logits, pc_loss_batch = jax.vmap(forward_single)(x_t, t)
        
        # 5. Loss Calculation
        # Reconstruction Loss (Cross Entropy)
        # logits: [Batch, Seq, Vocab] -> Flatten to [Batch*Seq, Vocab]
        # targets: batch_tokens -> Flatten to [Batch*Seq]
        
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch_tokens)
        loss_ce = jnp.mean(loss_ce)
        
        # PC Loss
        # Currently Beta = 0 (Pure Diffusion Training Phase)
        # But we can calculate it for monitoring
        loss_pc = jnp.mean(pc_loss_batch)
        
        total_loss = loss_ce + 0.0 * loss_pc # Beta=0
        
        # Return (loss, aux)
        # aux can be any pytree
        return total_loss, ((loss_ce, loss_pc), key)

    # Gradient transformation
    @eqx.filter_jit
    def update(model, opt_state, batch_tokens, key):
        # Use filter_grad to differentiate only w.r.t arrays
        grads, ((loss_ce, loss_pc), next_key) = eqx.filter_grad(train_step, has_aux=True)(model, opt_state, batch_tokens, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        # Reconstruct total loss for logging if needed? Or just use ce partials.
        # We can re-calculate total scalar from partials since Beta=0
        total_loss = loss_ce # + 0
        
        return model, opt_state, total_loss, (loss_ce, loss_pc), next_key

    # Loop
    dl = dataloader(loader_key)
    print("Starting training steps...")
    
    for step in range(10): # Run 10 steps to verify stability
        batch = next(dl)
        model, opt_state, loss, (ce, pc), key = update(model, opt_state, batch, key)
        print(f"Step {step}: Total Loss={loss:.4f}, CE={ce:.4f}, PC={pc:.4f}")
        
    print("Training loop verified successfully.")

if __name__ == "__main__":
    main()
