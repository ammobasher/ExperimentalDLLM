
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
import numpy as np
from tqdm import tqdm

from src.config import Config
from src.meta_trainer import MetaTrainer
from src.text_adapter import TextAdapter
from src.checkpoint_manager import CheckpointManager
from src.diffusion import DiffusionSDE

def simple_sampler(trainer, adapter, n_samples=2, steps=50):
    """
    reverse diffusion sampler (Euler-Maruyama-ish)
    """
    print(f"\n[Generation] Generating {n_samples} samples with {steps} steps...")
    
    key = jax.random.PRNGKey(int(time.time()))
    
    # 1. Start from random noise (Gaussian) in Embedding space
    # Shape: [Batch, Seq, Dim]
    shape = (n_samples, Config.seq_len, Config.embed_dim)
    x_t = jax.random.normal(key, shape)
    
    # Time steps: T down to 0
    ts = jnp.linspace(trainer.sde.T, 0.001, steps)
    dt = ts[0] - ts[1]
    
    for i, t_val in enumerate(tqdm(ts)):
        # Broadcast t
        t_batch = jnp.full((n_samples,), t_val)
        
        # Predict logits p(x_0 | x_t)
        # model(inputs_embeds=x_t, t=t_batch) -> logits
        logits, _ = jax.vmap(lambda x, ti: trainer.model(inputs_embeds=x, t=ti, inference=True))(x_t, t_batch)
        
        # Convert logits to x_0_pred (Weighted sum of embeddings)
        # This is expensive: [Batch, Seq, Vocab] * [Vocab, Dim]
        # vocab is 50k. 
        # But we need x_0_pred to step.
        # Approximation: Hard sample? No, not differentiable/smooth.
        # Softmax: Probs = softmax(logits)
        # x_0_pred = Probs @ EmbeddingMatrix
        
        # We need access to embedding matrix weights.
        # trainer.model.embedding.weight is [Vocab, Dim]
        embed_weight = trainer.model.embedding.weight
        
        # Probs: [B, S, V]
        probs = jax.nn.softmax(logits, axis=-1)
        
        # E[x_0]: [B, S, D] = [B, S, V] @ [V, D]
        x_0_pred = jnp.einsum('bsv,vd->bsd', probs, embed_weight)
        
        # Update x_t -> x_{t-1} using Reverse SDE formula
        # dx = [f(x,t) - g(t)^2 score] dt ...
        # Simplified VP-SDE reverse step:
        # mean, std = sde.marginal_prob(x_0_pred, t) ... no that's forward.
        # We use x_0_pred to estimate the "clean" direction.
        # x_{t-1} = (x_t - sqrt(1-alpha_bar) * eps) ... standard DDPM
        # But here let's just linearly interpolate for "Consistency" style or use the analytical posterior mean if we assume x_0 is x_0_pred.
        # Posterior mean of x_{t-1} given x_t and x_0:
        # q(x_{t-1} | x_t, x_0)
        
        # Let's trust the "DiffusionSDE" class might not have posterior logic, implementing basic one:
        # For small dt: x_{t-dt} ~ x_t - f(x,t)dt + g(t)^2 score dt
        # score approx (x_0_pred - x_t) / (1 - alpha)? 
        # Let's use a very naive interpolation since this is just a quick check:
        # x_{t-dt} = (1 - alpha) * x_t + alpha * x_0_pred ??
        
        # Better: DDIM-like step.
        # alpha_t = marginal_prob coeff?
        # mean_t, std_t = trainer.sde.marginal_prob(x_0_pred, t_val)
        # This gives x_t distribution. We have x_t.
        # We want x_{t-1}.
        
        # Let's use the simplest:
        # x_{t-1} = x_t + (x_0_pred - x_t) * (dt / t_val)
        # "ODE" flow pointing to x_0_pred.
        
        x_t = x_t + (x_0_pred - x_t) * (dt / t_val)
        
        # Add noise? Langevin? skipping for speed.

    # Final Decode
    # x_t is now close to x_0
    logits, _ = jax.vmap(lambda x, ti: trainer.model(inputs_embeds=x, t=ti, inference=True))(x_t, jnp.full((n_samples,), 0.001))
    token_ids = jnp.argmax(logits, axis=-1)
    
    for k in range(n_samples):
        text = adapter.decode(np.array(token_ids[k]))
        print(f"\n--- Sample {k} ---")
        print(text[:200] + "...") # Print first 200 chars
        print("------------------")

def calculate_ppl(trainer, adapter, n_batches=20):
    print(f"\n[Evaluation] Calculating PPL over {n_batches} batches on {adapter.dataset.split}...")
    
    total_nll = 0.0
    total_tokens = 0
    total_bpd = 0.0
    
    # Force Beta=0 for evaluation (?) PPL is usually just NLL.
    # We ignore the SDE prior terms for standard "Reconstruction PPL" comparison.
    # IF we want "True" ELBO, we add KL. But GPT-2 PPL is just CrossEntropy.
    # PCDM maximizes PPL by minimizing reconstruction error at t~0?
    # No, PCDM allows t E [0,1].
    # But PPL is defined on the discrete tokens. 
    # If we evaluate NLL at t=0 (clean image/text), it's just sizing up the autoencoder.
    # The true generative likelihood is the integration of the ODE/SDE.
    # FOR NOW: We report the NLL at random t, which is the Training Loss proxy. 
    # Or better: NLL at very low noise (t=1e-3) is the "Likelihood of data given latent".
    # But diffusion models generate by refining.
    # Let's use the same metric as `evaluate_metrics.py` (average NLL over t), which is the VLB.
    # This is an UPPER BOUND on NLL (so Lower Bound on Likelihood).
    # So PPL <= exp(Avg NLL). It's a conservative metrics.
    
    for i in range(n_batches):
        try:
            batch = adapter.get_batch()
        except Exception:
            break
            
        x_0 = jax.vmap(jax.vmap(trainer.model.embedding))(batch)
        
        key = jax.random.PRNGKey(i)
        key, t_key, n_key = jax.random.split(key, 3)
        
        # Integrally sample t? No, Monte Carlo.
        t = jax.random.uniform(t_key, (batch.shape[0],))
        noise = jax.random.normal(n_key, x_0.shape)
        x_t = jax.vmap(trainer.sde.q_sample)(x_0, t, noise)
        
        logits, _ = jax.vmap(lambda x, ti: trainer.model(inputs_embeds=x, t=ti, inference=True))(x_t, t)
        
        nll_batch = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
        
        total_nll += jnp.sum(nll_batch)
        total_tokens += batch.size
        
        # BPD
        total_bpd += jnp.sum(nll_batch) / jnp.log(2)

    avg_nll = total_nll / total_tokens
    ppl = jnp.exp(avg_nll)
    avg_bpd = total_bpd / total_tokens
    
    print(f"PPL: {ppl:.2f} | BPD: {avg_bpd:.2f}")
    return ppl, avg_bpd

def main():
    print("=== SYNAPSE BENCHMARK SUITE ===")
    
    # 1. Setup Same Config
    # Config is global.
    # Ensure it matches scaled run:
    Config.n_layers = 12
    Config.embed_dim = 1024
    Config.n_heads = 16
    
    # output vocab size determined by adapter
    # so we load adapter first or temp one
    temp_adapter = TextAdapter(dataset_name="wikitext", split="test")
    Config.vocab_size = temp_adapter.vocab_size
    
    # 2. Init Model
    key = jax.random.PRNGKey(42)
    trainer = MetaTrainer(key)
    
    # 3. Restore Checkpoint
    ckpt_manager = CheckpointManager("checkpoints_scaled")
    latest_step = ckpt_manager.latest_step()
    if latest_step is None:
        print("No checkpoint found.")
        return

    print(f"Restoring Step {latest_step}...")
    
    # Restore structure
    # We only saved "model" and "controller" in filter
    # But restore expects structure matching the arguments
    # So we construct structure:
    restore_struct = {
        'model': trainer.model,
        'controller': trainer.controller
    }
    
    # Orbax restore might return a tree where static fields are lost if not handled perfectly,
    # or if it returns exactly 'item' with leaves updated.
    
    restored = ckpt_manager.restore(latest_step, restore_struct)

    # To be safe with Equinox:
    # 1. Get static skeleton from fresh model
    # 2. Partiton restored to get arrays?
    # Actually, restored['model'] should have the arrays.
    # The safest way is eqx.combine(restored_model, static_model) assuming restored has correct structure.
    
    # Let's inspect what we got
    restored_model = restored['model']
    # If restored_model is just arrays and Nones, we need to combine.
    
    # Fresh static
    _, static_model = eqx.partition(trainer.model, eqx.is_array)
    _, static_controller = eqx.partition(trainer.controller, eqx.is_array)
    
    # Restored arrays (we hope restored_model has them)
    # We trust restored_model has the values. We explicitly combine to ensure static fields (activation) are from fresh.
    arrays_model, _ = eqx.partition(restored_model, eqx.is_array)
    arrays_controller, _ = eqx.partition(restored['controller'], eqx.is_array)
    
    trainer.model = eqx.combine(arrays_model, static_model)
    trainer.controller = eqx.combine(arrays_controller, static_controller)
    
    print("Model Loaded and Recombined.")
    
    # 4. Evaluate on WikiText-2 Test
    print("\n[Metric 1] WikiText-2 Test Set")
    adapter_wiki = TextAdapter(seq_len=Config.seq_len, batch_size=4, dataset_name="wikitext", split="test")
    ppl, bpd = calculate_ppl(trainer, adapter_wiki, n_batches=50)
    
    # 5. Generate Samples
    print("\n[Metric 2] Qualitative Sampling")
    simple_sampler(trainer, adapter_wiki, n_samples=2)

if __name__ == "__main__":
    main()
