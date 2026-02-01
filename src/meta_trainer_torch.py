import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from src.config import Config
from src.model import PCModel
from src.controller_torch import BetaController
from src.diffusion import DiffusionSDE


class MetaTrainer:
    """
    Memory-Efficient Bi-Level Optimizer (Meta-Trainer).
    
    Unlike the torch.func version, this uses a simpler approach:
    - Inner loop: Train model with current beta
    - Outer loop: Update controller based on validation performance
    - No meta-gradients through inner loop (saves 50%+ memory)
    
    This is an approximation of true BLO but works on limited memory devices.
    """
    def __init__(self, device: torch.device):
        self.device = device
        
        # Models
        self.model = PCModel().to(device)
        self.controller = BetaController().to(device)
        self.sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
        
        # Optimizers
        self.optimizer_ctrl = optim.Adam(self.controller.parameters(), lr=Config.lr_ctrl)
        self.optimizer_llm = optim.AdamW(self.model.parameters(), lr=Config.lr_llm)
        
        # Running statistics for controller
        self.running_val_loss = None
        self.ema_alpha = 0.1
        
        print(f">> MetaTrainer (Memory-Efficient) Initialized on {device}")

    def _compute_loss(self, input_ids, t_batch, visual_latents=None):
        """Compute forward pass and losses."""
        # Get embeddings
        x_0 = self.model(input_ids, return_embeds=True)
        
        # Add noise
        mean, std = self.sde.marginal_prob(x_0, t_batch)
        x_t = mean + std * torch.randn_like(x_0)
        
        # Forward
        logits, pc_loss = self.model(inputs_embeds=x_t, t=t_batch, visual_latents=visual_latents)
        
        # Slice logits for text if visual latents present
        if visual_latents is not None:
            seq_txt = input_ids.shape[1]
            logits = logits[:, -seq_txt:, :]
        
        # Causal LM loss (shifted)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        ce_loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, Config.vocab_size),
            shift_labels.view(-1)
        )
        
        return ce_loss, pc_loss.mean()

    def train_step(self, train_batch: torch.Tensor, val_batch: torch.Tensor, 
                   prev_beta: torch.Tensor, visual_latents: torch.Tensor = None):
        """
        Memory-efficient training step.
        
        1. Get beta from controller (no meta-gradient)
        2. Train model on training batch
        3. Evaluate on validation batch
        4. Update controller based on validation loss trend
        """
        self.model.train()
        self.controller.train()
        
        batch_size = train_batch.shape[0]
        t_batch = torch.rand(batch_size, device=self.device)
        
        # --- 1. Get Beta from Controller (detached) ---
        with torch.no_grad():
            # Compute current PC loss for controller input
            _, pc_loss_sample = self._compute_loss(train_batch, t_batch, visual_latents)
            t_mean = t_batch.mean()
            beta_val = self.controller(t_mean, pc_loss_sample.detach(), prev_beta)
            beta_val = beta_val.detach()
        
        # --- 2. Train Model (Inner Loop) ---
        self.optimizer_llm.zero_grad()
        
        ce_loss, pc_loss = self._compute_loss(train_batch, t_batch, visual_latents)
        total_loss = ce_loss + beta_val.squeeze() * pc_loss
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer_llm.step()
        
        # --- 3. Evaluate on Validation (Outer Loop Signal) ---
        with torch.no_grad():
            t_val = torch.rand(val_batch.shape[0], device=self.device)
            val_ce, val_pc = self._compute_loss(val_batch, t_val, visual_latents=None)
            val_loss = val_ce.item()
        
        # --- 4. Update Controller (Simple Gradient) ---
        # Instead of meta-gradient, we update controller to minimize validation loss
        # by treating it as a direct signal
        
        if self.running_val_loss is None:
            self.running_val_loss = val_loss
        else:
            # Compute improvement signal
            improvement = self.running_val_loss - val_loss
            
            # Update controller to encourage beta values that led to improvement
            self.optimizer_ctrl.zero_grad()
            
            # Recompute beta with gradients
            t_mean_grad = t_batch.mean().detach()
            pc_for_ctrl = pc_loss.detach()
            beta_with_grad = self.controller(t_mean_grad, pc_for_ctrl, prev_beta)
            
            # Simple loss: if improvement > 0, reinforce current beta
            # if improvement < 0, adjust beta
            ctrl_loss = -improvement * beta_with_grad.squeeze()
            ctrl_loss.backward()
            
            self.optimizer_ctrl.step()
            
            # Update EMA
            self.running_val_loss = (1 - self.ema_alpha) * self.running_val_loss + self.ema_alpha * val_loss
        
        return {
            "loss_total": total_loss.item(),
            "loss_ce": ce_loss.item(),
            "loss_pc": pc_loss.item(),
            "beta": beta_val.item(),
            "val_loss": val_loss
        }
    
    def simple_step(self, train_batch: torch.Tensor, prev_beta: float, 
                    visual_latents: torch.Tensor = None):
        """
        Ultra-simple training step without any controller update.
        Used for fast steps between full BLO updates.
        """
        self.model.train()
        self.optimizer_llm.zero_grad()
        
        t_batch = torch.rand(train_batch.shape[0], device=self.device)
        
        ce_loss, pc_loss = self._compute_loss(train_batch, t_batch, visual_latents)
        total_loss = ce_loss + prev_beta * pc_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer_llm.step()
        
        return {
            "loss_total": total_loss.item(),
            "loss_ce": ce_loss.item(),
            "loss_pc": pc_loss.item(),
            "beta": prev_beta
        }
