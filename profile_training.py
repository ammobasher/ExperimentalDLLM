"""
Training Profiler - Identify Performance Bottlenecks
"""
import torch
import time
import numpy as np

from src.config import Config
from src.model import PCModel
from src.layers import ChunkedPCLayer, PCLayer
from src.meta_trainer_torch import MetaTrainer
from src.multimodal_adapter import MultimodalAdapter
from src.text_adapter import TextAdapter
from src.memory import EpisodicMemory

Config.vocab_size = 50257

def profile_component(name, fn, n_iters=5):
    """Run a function multiple times and report timing."""
    times = []
    for i in range(n_iters):
        start = time.time()
        fn()
        elapsed = time.time() - start
        times.append(elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"[{name}] Avg: {avg:.3f}s ± {std:.3f}s per call")
    return avg

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 64
    embed_dim = Config.embed_dim
    
    # --- 1. Test ChunkedPCLayer vs Standard PCLayer ---
    print("\n[1] LAYER COMPARISON")
    
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Standard layer
    std_layer = PCLayer(embed_dim, Config.n_heads).to(device)
    def run_std():
        with torch.no_grad():
            std_layer(x)
    t_std = profile_component("PCLayer (Standard O(N²))", run_std)
    
    # Chunked layer
    chunk_layer = ChunkedPCLayer(embed_dim, Config.n_heads, Config.chunk_size).to(device)
    def run_chunk():
        with torch.no_grad():
            chunk_layer(x)
    t_chunk = profile_component("ChunkedPCLayer (O(N*W))", run_chunk)
    
    if t_chunk > t_std * 2:
        print("  ⚠️ ChunkedPCLayer is SLOWER than standard - possible bug!")
    
    # --- 2. Test Data Loading ---
    print("\n[2] DATA LOADING")
    
    # Streaming
    print("  Initializing streaming adapter...")
    start = time.time()
    try:
        mm_streaming = MultimodalAdapter(batch_size=batch_size, image_size=128, device=device, streaming=True)
        init_time = time.time() - start
        print(f"  Streaming adapter init: {init_time:.2f}s")
        
        def fetch_streaming():
            mm_streaming.get_batch(device)
        t_stream = profile_component("Streaming get_batch()", fetch_streaming, n_iters=3)
    except Exception as e:
        print(f"  ⚠️ Streaming failed: {e}")
        t_stream = float('inf')
    
    # Cached
    print("  Initializing cached adapter...")
    start = time.time()
    mm_cached = MultimodalAdapter(batch_size=batch_size, image_size=128, device=device, streaming=False)
    init_time = time.time() - start
    print(f"  Cached adapter init: {init_time:.2f}s")
    
    def fetch_cached():
        mm_cached.get_batch(device)
    t_cached = profile_component("Cached get_batch()", fetch_cached, n_iters=3)
    
    if t_stream > t_cached * 10:
        print("  ⚠️ Streaming is 10x+ slower than cached - network bottleneck!")
    
    # --- 3. Test Full Model Forward ---
    print("\n[3] FULL MODEL FORWARD")
    
    model = PCModel().to(device)
    input_ids = torch.randint(0, Config.vocab_size, (batch_size, seq_len), device=device)
    
    def run_forward():
        with torch.no_grad():
            model(input_ids=input_ids, t=torch.zeros(1, device=device))
    
    profile_component("PCModel.forward()", run_forward)
    
    # --- 4. Test MetaTrainer Step ---
    print("\n[4] META-TRAINER STEP (BLO)")
    
    trainer = MetaTrainer(device)
    train_batch = torch.randint(0, Config.vocab_size, (batch_size, seq_len), device=device)
    val_batch = torch.randint(0, Config.vocab_size, (batch_size, seq_len), device=device)
    prev_beta = torch.tensor([1.0], device=device)
    vis_latents = torch.randn(batch_size, 4, 16, 16, device=device)
    
    def run_train_step():
        trainer.train_step(train_batch, val_batch, prev_beta, vis_latents)
    
    profile_component("MetaTrainer.train_step()", run_train_step, n_iters=3)
    
    # --- 5. Test Memory Operations ---
    print("\n[5] MEMORY OPERATIONS")
    
    memory = EpisodicMemory(dim=embed_dim, capacity=100)
    test_vec = np.random.randn(embed_dim).astype(np.float32)
    
    def run_memory_add():
        memory.add(test_vec, [1, 2, 3], loss_val=100.0)
    
    profile_component("EpisodicMemory.add()", run_memory_add, n_iters=100)
    
    # Add some items for retrieval test
    for i in range(50):
        memory.add(np.random.randn(embed_dim).astype(np.float32), [i], 100.0)
    
    def run_memory_retrieve():
        memory.retrieve(test_vec, k=3)
    
    profile_component("EpisodicMemory.retrieve()", run_memory_retrieve, n_iters=100)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    if t_stream > 1.0:
        print("🔴 CRITICAL: Streaming data is slow (>1s per batch)")
        print("   FIX: Use cached mode (streaming=False)")
    
    if t_chunk > t_std * 1.5:
        print("🟡 WARNING: ChunkedPCLayer is slower than standard")
        print("   FIX: Review _chunked_attention implementation")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
