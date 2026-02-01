import torch
import math

class DiffusionSDE:
    """
    Variance Preserving SDE (PyTorch).
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.T = 1.0

    def marginal_prob(self, x_0, t):
        """
        Compute p_0t(x(t) | x(0))
        mean = x(0) * exp(-0.5 * integral_0^t beta(s) ds)
        std = sqrt(1 - exp(-integral_0^t beta(s) ds))
        """
        # Integral of linear beta: 0.5 * t^2 * (beta_max - beta_min) + t * beta_min
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        
        # Broadcast to x_0 shape
        # x_0: [Batch, Seq, Dim]
        # t: [Batch] or Scalar
        if isinstance(t, torch.Tensor):
            mean_coeff = mean_coeff.view(-1, 1, 1) # [Batch, 1, 1]
            std = std.view(-1, 1, 1)
            
        return mean_coeff * x_0, std

    def prior_sampling(self, shape, device):
        """
        Sample from N(0, I)
        """
        return torch.randn(shape, device=device)
        
    def sde(self, x, t):
        """
        Drift and Diffusion coefficients.
        drift = -0.5 * beta(t) * x
        diffusion = sqrt(beta(t))
        """
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t.view(-1, 1, 1) * x
        diffusion = torch.sqrt(beta_t).view(-1, 1, 1)
        return drift, diffusion
