import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple

class DiffusionSDE(eqx.Module):
    beta_min: float
    beta_max: float
    T: float = 1.0
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def marginal_prob(self, x_0: jax.Array, t: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Calculates the mean and standard deviation for the transition p(x_t | x_0).
        Based on Variance Preserving (VP) SDE.
        
        Args:
            x_0: Clean input data.
            t: Timestep (0 to T).
            
        Returns:
            mean: Mean of x_t.
            std: Standard deviation of x_t.
        """
        # Integral of beta(s)ds from 0 to t
        # beta(t) = beta_min + t * (beta_max - beta_min)
        # Integral = beta_min * t + 0.5 * t^2 * (beta_max - beta_min)
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        
        mean = jnp.exp(log_mean_coeff) * x_0
        std = jnp.sqrt(1. - jnp.exp(2. * log_mean_coeff))
        return mean, std

    def q_sample(self, x_0: jax.Array, t: jax.Array, noise: jax.Array) -> jax.Array:
        """
        Adds noise to the clean input x_0 at timestep t.
        x_t = mean + std * noise
        """
        mean, std = self.marginal_prob(x_0, t)
        # mean and std are scalars (or broadcastable), x_0/noise are tensors
        return mean + std * noise

    def get_beta_t(self, t: jax.Array) -> jax.Array:
        """
        Returns the instantaneous noise level beta(t).
        Useful for the Neuromodulator (BetaController).
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)
