import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.meta_trainer import MetaTrainer
from src.config import Config

def verify_meta_learning():
    print("--- Verifying Meta-Learning (BLO) ---")
    key = jax.random.PRNGKey(42)
    trainer = MetaTrainer(key)
    
    # Create dummy batch
    batch_size = 4
    seq_len = 32
    vocab_size = 32000
    
    key, t_key, v_key = jax.random.split(key, 3)
    train_batch = jax.random.randint(t_key, (batch_size, seq_len), 0, vocab_size)
    val_batch = jax.random.randint(v_key, (batch_size, seq_len), 0, vocab_size)
    
    start_beta = 1.0 # arbitrary
    
    print("Executing Meta-Step...")
    # Trace the step
    new_model, new_ctrl, s_llm, s_ctrl, metrics, beta_val = trainer.train_step(
        trainer.model,
        trainer.controller,
        trainer.state_llm,
        trainer.state_ctrl,
        train_batch,
        val_batch,
        key,
        jnp.array(start_beta)
    )
    
    print(f"Beta Output: {beta_val}")
    print(f"Inner Loss: {metrics['loss_inner']}")
    print(f"Meta Loss: {metrics['loss_meta']}")
    
    # Check if controller changed (implies gradients flowed)
    # We can check parameters difference
    
    diff = 0.0
    leaves_old = jax.tree_util.tree_leaves(eqx.filter(trainer.controller, eqx.is_array))
    leaves_new = jax.tree_util.tree_leaves(eqx.filter(new_ctrl, eqx.is_array))
    
    for o, n in zip(leaves_old, leaves_new):
        diff += jnp.sum(jnp.abs(o - n))
        
    print(f"Controller Parameter Update Norm: {diff}")
    
    assert diff > 0.0, "Controller parameters did not change! Meta-gradients are zero or disconnected."
    assert not jnp.isnan(diff), "Controller update is NaN"
    assert beta_val > 0.0, "Beta must be positive (Softplus)"
    
    print("Meta-Learning Gradients Verified Successfully!\n")

if __name__ == "__main__":
    verify_meta_learning()
