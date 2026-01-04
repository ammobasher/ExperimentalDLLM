import jax
import jax.numpy as jnp
import equinox as eqx

class BetaController(eqx.Module):
    """
    Neuromodulatory Controller (Meta-Learner).
    Observes system state and outputs scalar 'beta' (PC Loss Weight).
    """
    mlp: eqx.nn.MLP
    
    def __init__(self, key: jax.random.PRNGKey):
        # Input: [timestep_t, current_pc_loss, previous_beta] (Dim=3)
        # Output: [beta] (Dim=1)
        self.mlp = eqx.nn.MLP(
            in_size=3,
            out_size=1,
            width_size=32, # Small, lightweight controller
            depth=2,       # 2 Hidden layers
            activation=jax.nn.silu,
            key=key
        )

    def __call__(self, t: jax.Array, pc_loss: jax.Array, prev_beta: jax.Array) -> jax.Array:
        """
        Args:
            t: Scalar or [1]. Diffusion timestep (normalized 0-1 usually preferred).
            pc_loss: Scalar or [1]. Current PC error signal from model.
            prev_beta: Scalar or [1]. Previous beta value.
            
        Returns:
            beta: Scalar > 0.
        """
        # Ensure inputs are 1D arrays for MLP
        inputs = jnp.stack([t, pc_loss, prev_beta])
        
        raw_out = self.mlp(inputs)
        
        # Enforce positivity via Softplus
        beta = jax.nn.softplus(raw_out[0])
        
        # Optional: Add small epsilon to prevent beta=0 or explosion
        # And maybe clamp?
        # Let's keep it simple: beta + 0.01
        return beta + 0.01
