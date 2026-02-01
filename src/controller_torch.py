import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaController(nn.Module):
    """
    Neuromodulatory Controller (Meta-Learner) in PyTorch.
    Observes system state and outputs scalar 'beta' (PC Loss Weight).
    """
    def __init__(self):
        super().__init__()
        # Input size: 3 ([t, current_pc_loss, previous_beta])
        # Output size: 1 ([beta])
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights for stability
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, pc_loss, prev_beta):
        """
        Args:
            t: Tensor [Batch] or [1]. Diffusion timestep.
            pc_loss: Tensor [Batch] or [1]. Current PC error.
            prev_beta: Tensor [Batch] or [1].
            
        Returns:
            beta: Tensor [Batch] or [1].
        """
        # Ensure all inputs are tensors and have similar shapes
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)
        if not isinstance(pc_loss, torch.Tensor):
            pc_loss = torch.tensor([pc_loss], dtype=torch.float32)
        if not isinstance(prev_beta, torch.Tensor):
            prev_beta = torch.tensor([prev_beta], dtype=torch.float32)
            
        # Reshape to [B, 1] if needed
        if t.dim() == 0: t = t.unsqueeze(0)
        if pc_loss.dim() == 0: pc_loss = pc_loss.unsqueeze(0)
        if prev_beta.dim() == 0: prev_beta = prev_beta.unsqueeze(0)
        
        # Concatenate inputs
        inputs = torch.cat([t.view(-1, 1), pc_loss.view(-1, 1), prev_beta.view(-1, 1)], dim=-1)
        
        raw_out = self.mlp(inputs)
        
        # Enforce positivity via Softplus
        beta = F.softplus(raw_out) + 0.01
        
        return beta
