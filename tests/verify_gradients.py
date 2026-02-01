import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import sys
import os

# Create a minimal config for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.layers import OptimizedPCLayer, PCLayer
from src.diffusion import DiffusionSDE

def verify_diffusion():
    print("--- Verifying Diffusion SDE ---")
    sde = DiffusionSDE(beta_min=0.1, beta_max=20.0, T=1.0)
    
    # Test t=0 (Should have std ~ 0, mean ~ x_0)
    x_0 = jnp.array([1.0, -1.0])
    mean_0, std_0 = sde.marginal_prob(x_0, jnp.array(0.0))
    print(f"t=0: Mean={mean_0}, Std={std_0}")
    
    assert jnp.allclose(mean_0, x_0, atol=1e-5), "t=0 mean mismatch"
    assert jnp.allclose(std_0, 0.0, atol=1e-5), "t=0 std mismatch"
    
    # Test t=1 (Should have high std, mean ~ 0)
    mean_1, std_1 = sde.marginal_prob(x_0, jnp.array(1.0))
    print(f"t=1: Mean={mean_1}, Std={std_1}")
    
    # Variance Preserving implies marginal variance approaches 1.
    # At t=large, mean -> 0, std -> 1
    assert jnp.all(std_1 > 0.9), "t=1 std too low for VP SDE"
    print("Diffusion SDE Verified.\n")

def verify_layer_gradients():
    print("--- Verifying PCLayer Gradients ---")
    key = jax.random.PRNGKey(42)
    layer = OptimizedPCLayer(embed_dim=16, n_heads=2, chunk_size=4, key=key)
    
    # Dummy Input
    x_i = jax.random.normal(key, (8, 16)) # Seq=8, Dim=16
    
    def loss_fn(layer_instance, x):
        # layer_instance is 'l', the combined model with tracers
        out_x, out_p, pc_loss = layer_instance(x)
        return jnp.mean(out_x) + pc_loss

    # Partition
    params, static = eqx.partition(layer, eqx.is_array)
    
    def wrapped_loss(p, x):
        l = eqx.combine(p, static)
        return loss_fn(l, x)
        
    grads = jax.grad(wrapped_loss)(params, x_i)
    
    # Check if gradients are non-zero and not NaN
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grad_leaves if g is not None))
    
    print(f"Optimized Gradient Norm: {grad_norm}")
    assert not jnp.isnan(grad_norm), "Gradient is NaN"
    # assert grad_norm > 0.0, "Gradient is Zero" # Commenting out to allow full run
    
    print("--- Verifying Base PCLayer Gradients ---")
    base_layer = PCLayer(embed_dim=16, n_heads=2, chunk_size=4, key=key)
    # Copy params from optimized layer to allow direct comparison (optional, but keep simple)
    
    b_params, b_static = eqx.partition(base_layer, eqx.is_array)
    def b_wrapped_loss(p, x):
        l = eqx.combine(p, b_static)
        out_x, _, pc_loss = l(x)
        return jnp.mean(out_x) + pc_loss
        
    b_grads = jax.grad(b_wrapped_loss)(b_params, x_i)
    b_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(b_grads) if g is not None))
    print(f"Base PC Layer Gradient Norm: {b_grad_norm}")
    
    if b_grad_norm > 0.0 and grad_norm == 0.0:
        print("FAIL: Base layer has gradients, but Optimized layer does not. Checkpointing issue.")
        assert False, "Optimized Layer Gradient Zero"
    elif b_grad_norm == 0.0:
        print("FAIL: Base layer has zero gradients. Logic issue.")
        assert False, "Base Layer Gradient Zero"
        
    print("PCLayer Gradients Verified.\n")

if __name__ == "__main__":
    verify_diffusion()
    verify_layer_gradients()
    print("All Phase 1 Verification Tests Passed!")
