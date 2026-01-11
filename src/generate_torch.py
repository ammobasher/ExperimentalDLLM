import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
import math

from src.model import PCModel
from src.diffusion import DiffusionSDE
from src.config import Config
from src.text_adapter import TextAdapter
from src.memory import EpisodicMemory


def ddpm_sample(model, sde, x_t, prompt_embeds, mask, steps=50, temperature=1.0, 
                corrector_steps=0, device="cpu"):
    """
    True DDPM Reverse Sampling with optional Predictor-Corrector.
    
    Args:
        model: PCModel
        sde: DiffusionSDE
        x_t: Initial noise [1, N, D]
        prompt_embeds: Embedded prompt [1, N, D] 
        mask: Binary mask for prompt positions [N]
        steps: Number of reverse diffusion steps
        temperature: Sampling temperature
        corrector_steps: Langevin corrector steps per predictor step (0 = disabled)
        device: Torch device
    
    Returns:
        x_0: Denoised embeddings [1, N, D]
    """
    dt = 1.0 / steps
    x = x_t.clone()
    
    for i in range(steps):
        t_val = 1.0 - (i * dt)
        t_next = t_val - dt
        t = torch.full((1,), t_val, device=device)
        
        # === PREDICTOR (Reverse SDE Step) ===
        with torch.no_grad():
            # 1. Get model prediction (score/denoised estimate)
            logits, _ = model(inputs_embeds=x, t=t)
            probs = F.softmax(logits / temperature, dim=-1)
            
            # Soft projection to embedding space
            x_0_hat = torch.matmul(probs, model.embedding.weight)  # [1, N, D]
            
            # 2. Compute SDE coefficients
            drift, diffusion = sde.sde(x, t)
            
            # 3. Score function approximation
            # score ≈ -(x - mean_coeff * x_0) / sigma^2
            mean_coeff, sigma = sde.marginal_prob(torch.ones_like(x), t)
            score = -(x - mean_coeff * x_0_hat) / (sigma ** 2 + 1e-8)
            
            # 4. Reverse SDE step (Euler-Maruyama)
            # dx = [f(x,t) - g(t)^2 * score] dt + g(t) dw
            x_mean = x - (drift - diffusion ** 2 * score) * dt
            noise = torch.randn_like(x) * math.sqrt(dt) if t_next > 0 else 0
            x = x_mean + diffusion * noise
        
        # === CORRECTOR (Langevin MCMC) ===
        for _ in range(corrector_steps):
            with torch.no_grad():
                t_corr = torch.full((1,), t_next, device=device)
                logits_c, _ = model(inputs_embeds=x, t=t_corr)
                probs_c = F.softmax(logits_c / temperature, dim=-1)
                x_0_c = torch.matmul(probs_c, model.embedding.weight)
                
                mean_c, sigma_c = sde.marginal_prob(torch.ones_like(x), t_corr)
                score_c = -(x - mean_c * x_0_c) / (sigma_c ** 2 + 1e-8)
                
                # Langevin step
                step_size = 0.1 * (sigma_c ** 2)
                x = x + step_size * score_c + math.sqrt(2 * step_size) * torch.randn_like(x)
        
        # === GUIDANCE (Condition on Prompt) ===
        # For masked (prompt) positions, interpolate toward prompt embeddings
        if prompt_embeds is not None and mask is not None:
            m = mask.view(1, -1, 1)
            mean_gt, _ = sde.marginal_prob(prompt_embeds, torch.tensor([t_next], device=device))
            x = m * mean_gt + (1 - m) * x
    
    return x


def generate_torch(prompt, steps=50, temperature=1.0, device="cpu", 
                   checkpoint="checkpoints_torch/step_50000.pt", 
                   method="ddpm", corrector_steps=0, memory_path=None):
    """
    Generate text using trained Synapse model.
    
    Args:
        prompt: Text prompt to continue
        steps: Number of diffusion steps
        temperature: Sampling temperature
        device: Torch device
        checkpoint: Path to model checkpoint
        method: "ddpm" (proper sampling) or "fast" (original approximation)
        corrector_steps: Langevin corrector steps (0 = disabled)
        memory_path: Optional path to memory.npz for retrieval augmentation
    """
    print(f">> Loading PyTorch Generator")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   Method: {method} | Steps: {steps} | Temp: {temperature}")
    
    # 1. Setup
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    adapter = TextAdapter()
    vocab_size = adapter.vocab_size
    Config.vocab_size = vocab_size
    
    # 2. Model
    model = PCModel().to(device)
    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        if isinstance(state_dict, dict) and 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)
        print(f">> Weights loaded from {checkpoint}")
    else:
        print(f"!! Checkpoint {checkpoint} not found. Using random weights.")
    
    model.eval()
    
    # 3. Optional Memory Retrieval
    retrieved_ctx = ""
    if memory_path and os.path.exists(memory_path):
        print(f">> Loading Episodic Memory from {memory_path}")
        memory = EpisodicMemory(dim=Config.embed_dim)
        memory.load(memory_path)
        
        # Create query embedding
        prompt_tokens = adapter.tokenizer.encode(prompt)
        query_ids = torch.tensor(prompt_tokens[:Config.seq_len]).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embed = model(query_ids, return_embeds=True).mean(dim=1)  # [1, D]
        
        # Retrieve
        results = memory.retrieve(query_embed[0].cpu().numpy(), k=1)
        if results:
            retrieved_tokens, score = results[0]
            if retrieved_tokens:
                retrieved_ctx = adapter.tokenizer.decode(retrieved_tokens[:32])
                print(f">> Retrieved Memory (score={score:.2f}): '{retrieved_ctx[:50]}...'")
    
    # Prepend retrieved context to prompt
    full_prompt = retrieved_ctx + " " + prompt if retrieved_ctx else prompt
    
    # 4. Tokenize Prompt
    tokens = adapter.tokenizer.encode(full_prompt)
    N = Config.seq_len
    if len(tokens) > N: 
        tokens = tokens[:N]
    
    # Pad and Mask
    mask = torch.zeros(N, device=device)
    mask[:len(tokens)] = 1.0
    padded_tokens = tokens + [0] * (N - len(tokens))
    input_ids = torch.tensor(padded_tokens).long().to(device)
    
    # Embed Prompt
    with torch.no_grad():
        prompt_embeds = model.embedding(input_ids).unsqueeze(0)  # [1, N, Dim]
    
    # 5. Initial Noise
    x_t = torch.randn(1, N, Config.embed_dim, device=device)
    
    # 6. Sampling
    print(f">> Sampling {steps} steps (Prompt: '{prompt[:50]}...')...")
    start_gen = time.time()
    
    if method == "ddpm":
        x_0 = ddpm_sample(
            model, sde, x_t, prompt_embeds, mask,
            steps=steps, temperature=temperature,
            corrector_steps=corrector_steps, device=device
        )
    else:
        # Fast/Original approximation
        x_0 = x_t
        dt = 1.0 / steps
        for i in range(steps):
            t_val = 1.0 - (i * dt)
            t = torch.full((1,), t_val, device=device)
            
            with torch.no_grad():
                logits, _ = model(inputs_embeds=x_0, t=t)
                probs = F.softmax(logits / temperature, dim=-1)
                x_0_hat = torch.matmul(probs, model.embedding.weight)
                
                mean_pred, std_pred = sde.marginal_prob(x_0_hat, t - dt)
                x_pred = mean_pred + std_pred * torch.randn_like(x_0)
                
                mean_gt, std_gt = sde.marginal_prob(prompt_embeds, t - dt)
                x_gt = mean_gt + std_gt * torch.randn_like(x_0)
                
                m = mask.view(1, N, 1)
                x_0 = x_gt * m + x_pred * (1 - m)
    
    # 7. Final Decode
    with torch.no_grad():
        final_logits, _ = model(inputs_embeds=x_0, t=torch.zeros(1, device=device))
        final_tokens = final_logits[0].argmax(dim=-1).cpu().numpy().tolist()
    
    # Decode
    decoded = adapter.tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    elapsed = time.time() - start_gen
    print(f">> Generation Complete ({elapsed:.2f}s)")
    print("-" * 60)
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {decoded}")
    print("-" * 60)
    
    return decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The artificial intelligence")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--method", type=str, default="ddpm", choices=["ddpm", "fast"])
    parser.add_argument("--corrector", type=int, default=0, help="Langevin corrector steps")
    parser.add_argument("--ckpt", type=str, default="checkpoints_torch/step_50000.pt")
    parser.add_argument("--memory", type=str, default=None, help="Path to memory.npz")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    generate_torch(
        args.prompt, 
        steps=args.steps, 
        temperature=args.temp, 
        device=device, 
        checkpoint=args.ckpt,
        method=args.method,
        corrector_steps=args.corrector,
        memory_path=args.memory
    )
