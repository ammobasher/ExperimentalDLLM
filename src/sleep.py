import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

from src.model import PCModel
from src.memory import EpisodicMemory
from src.diffusion import DiffusionSDE
from src.config import Config


def run_sleep_cycle(
    model: PCModel,
    optimizer: optim.Optimizer,
    memory: EpisodicMemory,
    device: torch.device,
    n_steps: int = 100
) -> Tuple[float, int]:
    """
    Offline Consolidation Phase (PyTorch Version).
    Replays memories from Episodic Memory to fine-tune the Neocortex (PCModel).
    
    Args:
        model: The PCModel to consolidate.
        optimizer: Optimizer for the model.
        memory: EpisodicMemory containing stored experiences.
        device: Torch device.
        n_steps: Number of consolidation steps to run.
    
    Returns:
        (avg_loss, memories_replayed): Tuple of average consolidation loss and count.
    """
    print(f"--- Entering Sleep Cycle (Steps={n_steps}, Memories={memory.count}) ---")
    
    if memory.count < 10:
        print("Not enough memories to sleep. Skipping.")
        return 0.0, 0
    
    # Setup
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    model.train()
    
    total_sleep_loss = 0.0
    memories_replayed = 0
    
    for step in range(n_steps):
        # 1. Sample batch from Memory (Random Replay)
        n_valid = min(memory.count, memory.capacity)
        indices = np.random.randint(0, n_valid, size=min(Config.batch_size, n_valid))
        
        # Collect batches from memory
        batch_list = []
        for idx in indices:
            data = memory.values[idx]
            # Verify data format
            if isinstance(data, (list, np.ndarray)):
                tokens = np.array(data)
                # Truncate or pad to seq_len
                if len(tokens) > Config.seq_len:
                    tokens = tokens[:Config.seq_len]
                elif len(tokens) < Config.seq_len:
                    tokens = np.pad(tokens, (0, Config.seq_len - len(tokens)), mode='constant')
                batch_list.append(tokens)
            else:
                # Fallback: Random tokens for stability testing
                batch_list.append(np.random.randint(0, Config.vocab_size, (Config.seq_len,)))
        
        if len(batch_list) == 0:
            continue
            
        batch = torch.tensor(np.stack(batch_list), dtype=torch.long, device=device)
        memories_replayed += len(batch_list)
        
        # 2. Forward Pass with Diffusion
        t = torch.rand(batch.shape[0], device=device)
        
        # Embed
        x_0 = model(batch, return_embeds=True)
        
        # Add noise
        mean, std = sde.marginal_prob(x_0, t)
        noise = torch.randn_like(x_0)
        x_t = mean + std * noise
        
        # Forward through model
        logits, pc_loss = model(inputs_embeds=x_t, t=t)
        
        # 3. Calculate Loss
        # During sleep, we emphasize structure (PC Loss) over exact reconstruction
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        
        ce_loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, Config.vocab_size), 
            shift_labels.view(-1)
        )
        
        # Sleep consolidation emphasizes structural learning
        total_loss = ce_loss + 0.5 * pc_loss.mean()
        
        # 4. Backward and Step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_sleep_loss += total_loss.item()
        
        if (step + 1) % 20 == 0:
            print(f"  [Sleep Step {step + 1}/{n_steps}] Loss: {total_loss.item():.4f}")
    
    avg_loss = total_sleep_loss / max(n_steps, 1)
    print(f"--- Sleep Cycle Complete. Avg Loss: {avg_loss:.4f}, Replayed: {memories_replayed} ---")
    
    return avg_loss, memories_replayed
