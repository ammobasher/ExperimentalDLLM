import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time

from src.config import Config
from src.model import PCModel
from src.diffusion import DiffusionSDE
from src.multimodal_adapter import MultimodalAdapter

def main_visual():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()
    
    # 1. Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">> Phase 17: Vision-Language Training on {device}")
    
    # 2. Components
    # Update Config Vocab Size to match GPT-2 (used in checkpoint)
    Config.vocab_size = 50257
    print(f">> Config Vocab Size set to: {Config.vocab_size}")

    # Initialize MultimodalAdapter with SMALL latents
    adapter = MultimodalAdapter(batch_size=args.batch_size, image_size=8)
    
    model = PCModel().to(device)
    
    # Init Weights (Optional: Load Step 50k)
    ckpt_path = "checkpoints_torch/step_50000.pt"
    if os.path.exists(ckpt_path):
        print(f">> Loading Pre-trained Synapse Core (Step 50k)...")
        state_dict = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=False)
            print(">> Weights loaded successfully (Strict=False for visual_proj).")
        except Exception as e:
            print(f">> Warning: Load failed: {e}")

    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Training Loop
    model.train()
    start_time = time.time()
    
    print(">> Starting Vision-Language Loop...")
    
    for step in range(1, args.steps + 1):
        # A. Get Batch
        batch = adapter.get_batch(device)
        vis = batch['visual_latents'] # [B, 4, 8, 8] -> 64 tokens
        input_ids = batch['input_ids'] # [B, Seq]
        
        # B. Forward Pass
        t = torch.rand(args.batch_size, device=device)
        
        # 1. Embed Text
        text_emb = model.embedding(input_ids) # [B, Seq_Txt, Dim]
        
        # 2. Add Noise to Text (Conditioned Generation)
        mean, std = sde.marginal_prob(text_emb, t)
        text_noisy = mean + std * torch.randn_like(text_emb)
        
        # 3. Model Forward (Inject Vision)
        optimizer.zero_grad()
        # forward() handles correct concatenation: [Visual, Text]
        # returns logits: [B, Seq_Total, Vocab] where Seq_Total = Seq_Vis + Seq_Txt
        logits, pc_loss = model(inputs_embeds=text_noisy, t=t, visual_latents=vis)
        
        # C. Loss Calculation
        # We need to slice the logits to align with text labels
        # The visual tokens are prepended, so the LAST N tokens correspond to text
        seq_txt_len = input_ids.shape[1]
        text_logits = logits[:, -seq_txt_len:, :] # [B, Seq_Txt, Vocab]
        
        loss_ce = nn.CrossEntropyLoss()(text_logits.reshape(-1, Config.vocab_size), input_ids.reshape(-1))
        
        # Total Loss (PC Loss is already hierarchical and includes vision prop if connected)
        total_loss = loss_ce + pc_loss
        
        total_loss.backward()
        optimizer.step()
        
        # D. Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}] Total: {total_loss.item():.4f} (CE: {loss_ce.item():.4f} | PC: {pc_loss.item():.4f}) | Time: {elapsed:.2f}s")
            start_time = time.time()
            
        # E. Save
        if step % args.save_every == 0:
            os.makedirs("checkpoints_visual", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints_visual/step_{step}.pt")
            print(f">> Checkpoint saved at step {step}")

if __name__ == "__main__":
    main_visual()
