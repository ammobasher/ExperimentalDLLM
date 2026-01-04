import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import numpy as np

from src.config import Config
from src.meta_trainer import MetaTrainer
from src.text_adapter import TextAdapter

def calculate_metrics(trainer, adapter, n_batches=10):
    print(f"\n[Evaluation] Calculating metrics over {n_batches} batches...")
    
    total_nll = 0.0
    total_tokens = 0
    total_bpd = 0.0
    
    # We need to set beta=0 for pure likelihood evaluation (standard Diffusion ELBO)
    # OR we evaluate the joint loss.
    # Standard LLM metric is Perplexity (PPL) based on NLL.
    # For Diffusion, NLL is the Variational Lower Bound (ELBO).
    
    # We will use the 'inner_loss_fn' logic from trainer but purely for evaluation.
    # We need to access the diffusion SDE to calculate the correct ELBO term
    # ELBO = Reconstruction (CE) + KL (Prior Matching).
    # In our trained model, 'loss_ce' is the Reconstruction term.
    # PC Loss corresponds implicitly to the structure learning.
    
    # For standard PPL comparison: exp(NLL / N_tokens)
    
    for _ in range(n_batches):
        batch = adapter.get_batch()
        # [Batch, Seq]
        
        # 1. Embed X0
        x_0 = jax.vmap(jax.vmap(trainer.model.embedding))(batch)
        
        # 2. Diffusion Forward (Sample t uniformly)
        # To get accurate VB, we should ideally integrate over all t, 
        # but Monte Carlo sampling over batch is standard approximation.
        key = jax.random.PRNGKey(int(time.time()*1000))
        key, t_key, n_key = jax.random.split(key, 3)
        
        t = jax.random.uniform(t_key, (batch.shape[0],))
        noise = jax.random.normal(n_key, x_0.shape)
        x_t = jax.vmap(trainer.sde.q_sample)(x_0, t, noise)
        
        # 3. Model Forward
        # We enforce Beta=0 to see pure "Generative" capability regarding tokens
        logits, _ = jax.vmap(lambda x, ti: trainer.model(inputs_embeds=x, t=ti))(x_t, t)
        
        # 4. NLL (Cross Entropy)
        # This is the "Reconstruction" part of the ELBO
        # Weighting by SDE term is complex, but standardized CE is a good proxy for PPL here.
        nll_batch = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
        
        total_nll += jnp.sum(nll_batch)
        total_tokens += batch.size
        
        # Bits Per Dimension (BPD) = NLL / (pixels * ln(2)) or tokens * ln(2)
        # BPD = NLL / (SeqLen * Ln(2))
        total_bpd += jnp.sum(nll_batch) / jnp.log(2)

    avg_nll = total_nll / total_tokens
    ppl = jnp.exp(avg_nll)
    avg_bpd = total_bpd / total_tokens
    
    print("-" * 30)
    print(f"Perplexity (PPL): {ppl:.4f}")
    print(f"Bits Per Dim (BPD): {avg_bpd:.4f}")
    print(f"Avg NLL:          {avg_nll:.4f}")
    print("-" * 30)
    
    return ppl, avg_bpd

import time
import optax # Need optax for loss

if __name__ == "__main__":
    # Standalone run
    key = jax.random.PRNGKey(42)
    trainer = MetaTrainer(key)
    
    # Ideally load checkpoint here
    # trainer.model = eqx.tree_deserialise_leaves("checkpoints/ste_1000.eqx", trainer.model)
    
    adapter = TextAdapter(seq_len=Config.seq_len, batch_size=4)
    # Fix vocab
    Config.vocab_size = adapter.vocab_size
    # Re-init model to match vocab if needed (or load checkpoint)
    # For demo, we just init fresh random model to show script working
    trainer = MetaTrainer(key)
    
    calculate_metrics(trainer, adapter)
