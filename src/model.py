import torch
import torch.nn as nn
from typing import List, Tuple

from src.layers import OptimizedPCLayer, PCLayer
from src.config import Config

class PCModel(nn.Module):
    def __init__(self, key=None):
        super().__init__()
        # Ignore key, use torch.manual_seed if needed externally
        
        self.n_layers = Config.n_layers
        self.embed_dim = Config.embed_dim
        vocab_size = Config.vocab_size
        
        self.embedding = nn.Embedding(vocab_size, Config.embed_dim)
        
        # Stack layers
        # Usage of ModuleList to register parameters
        self.layers = nn.ModuleList([
            OptimizedPCLayer(
                embed_dim=Config.embed_dim, 
                n_heads=Config.n_heads, 
                chunk_size=Config.chunk_size
            ) 
            for _ in range(Config.n_layers)
        ])
        
        self.output_head = nn.Linear(Config.embed_dim, vocab_size)
        
        # Multimodal Projection (Phase 17)
        self.visual_proj = nn.Linear(4, Config.embed_dim) # VAE 4 -> Dim

    def forward(self, input_ids: torch.Tensor = None, t: float = None, inputs_embeds: torch.Tensor = None, 
                visual_latents: torch.Tensor = None, inference: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-Pass Forward.
        """
        device = self.embedding.weight.device
        
        # 1. Prediction / Embedding
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = self.embedding(input_ids)
        else:
            raise ValueError("Must provide either input_ids or inputs_embeds")
        
        # Multimodal Injection (Phase 17)
        if visual_latents is not None:
            # visual_latents: [Batch, C, H, W]
            # PyTorch VAEs are usually (B, C, H, W)
            # Permute to (B, H, W, C)
            vis = visual_latents.permute(0, 2, 3, 1) 
            B, H, W, C = vis.shape
            vis = vis.reshape(B, H * W, C) # [B, Seq_Vis, C]
            
            # Project
            vis_emb = self.visual_proj(vis)
            
            # Concatenate
            x = torch.cat([vis_emb, x], dim=1)
            
        # Pad to multiple of chunk_size
        seq_len = x.shape[1]
        chunk_size = Config.chunk_size
        remainder = seq_len % chunk_size
        if remainder != 0:
            pad_len = chunk_size - remainder
            padding = torch.zeros((x.shape[0], pad_len, x.shape[-1]), device=device)
            x = torch.cat([x, padding], dim=1)
            
        # --- PHASE 1: BOTTOM-UP ---
        layer_inputs = [] 
        preds = []       
        
        current_x = x
        
        for layer in self.layers:
            layer_inputs.append(current_x)
            x_next, p_i, _ = layer(current_x, p_i_plus_1=None, inference=inference)
            preds.append(p_i)
            current_x = x_next
            
        final_features = current_x
        
        # --- PHASE 2: TOP-DOWN (Loss Calculation) ---
        total_pc_loss = torch.tensor(0.0, device=device)
        
        for i in range(self.n_layers):
            x_i = layer_inputs[i]
            
            if i == self.n_layers - 1:
                p_from_above = None
            else:
                p_from_above = preds[i+1]
                
            _, _, loss = self.layers[i](x_i, p_from_above, inference=inference)
            total_pc_loss = total_pc_loss + loss
            
        # Final Output
        logits = self.output_head(final_features)
        
        return logits, total_pc_loss
