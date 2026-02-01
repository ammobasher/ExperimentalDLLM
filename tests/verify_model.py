import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.model import PCModel

def verify_pc_model():
    print("--- Verifying PCModel ---")
    key = jax.random.PRNGKey(0)
    
    # Initialize model
    model = PCModel(key)
    
    # Dummy input: Batch of sequences
    # Batch size 2, Seq len 64
    batch_size = 2
    seq_len = 64
    
    # Random integers for tokens
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, 32000)
    
    # Forward pass
    # PCModel.__call__ expects unbatched or handled via vmap?
    # Our implementation:
    # x = self.embedding(input_ids)
    # embedding Vmaps automatically if input has extra dims? No, usually embedding takes [Seq] or [Batch, Seq] depending on implementation.
    # eqx.nn.Embedding expects integer array.
    # But layer logic: "current_x = x". "layer(current_x)".
    # PCLayer expects [Seq, Dim].
    # If input is [Batch, Seq], PCLayer needs vmap.
    # PCModel currently does NOT vmap layers internally over batch.
    # So we should vmap the model forward pass itself.
    
    model_forward = jax.vmap(model)
    
    logits, total_pc_loss_batch = model_forward(input_ids)
    
    print(f"Logits Shape: {logits.shape}")
    print(f"Loss Shape: {total_pc_loss_batch.shape}")
    
    assert logits.shape == (batch_size, seq_len, 32000), "Logits shape mismatch"
    assert total_pc_loss_batch.shape == (batch_size,), "Loss shape mismatch"
    
    print("Forward Pass Successful.")
    
    # Verify Gradients
    print("--- Verifying Gradients ---")
    
    def loss_fn(m, x):
        logits, pc_loss = jax.vmap(m)(x)
        return jnp.mean(pc_loss) # Validate PC Loss propagates gradients
        
    params, static = eqx.partition(model, eqx.is_array)
    
    def wrapped_loss(p, x):
        m = eqx.combine(p, static)
        return loss_fn(m, x)
        
    grads = jax.grad(wrapped_loss)(params, input_ids)
    
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads) if g is not None))
    
    print(f"Gradient Norm: {grad_norm}")
    assert not jnp.isnan(grad_norm), "Gradient is NaN"
    assert grad_norm > 0.0, "Gradient is Zero (Graph Disconnected)"
    
    print("PCModel Gradients Verified.\n")

if __name__ == "__main__":
    verify_pc_model()
