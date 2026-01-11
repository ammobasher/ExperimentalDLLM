"""
Synapse Benchmark Suite
========================
Comprehensive evaluation of the Synapse model across 5 metrics:
1. Perplexity (WikiText-2)
2. Memory Recall (Needle-in-Haystack)
3. Visual Grounding (CIFAR-10 Caption Match)
4. Generation Quality (Sample coherence)
5. Speed (Tokens/second)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
import json
from datetime import datetime

from src.model import PCModel
from src.diffusion import DiffusionSDE
from src.config import Config
from src.text_adapter import TextAdapter
from src.multimodal_adapter import MultimodalAdapter
from src.memory import EpisodicMemory
from src.metrics import distinct_n, self_bleu, repetition_ratio, evaluate_generation_quality


def benchmark_perplexity(model, adapter, sde, device, n_batches=50):
    """
    Calculate perplexity on WikiText-2 test set.
    Uses strict causal LM (shifted labels).
    """
    print("\n[1/5] Benchmarking Perplexity...")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    test_adapter = TextAdapter(batch_size=8, split="test")
    
    with torch.no_grad():
        for i in range(n_batches):
            batch_np = test_adapter.get_batch()
            input_ids = torch.from_numpy(batch_np).long().to(device)
            
            # Forward at t=0 (clean)
            logits, _ = model(input_ids=input_ids, t=torch.zeros(1, device=device))
            
            # Shifted labels (Causal LM)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, Config.vocab_size),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    ppl = np.exp(min(avg_loss, 20))  # Cap to avoid overflow
    
    print(f"   Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
    return {"perplexity": ppl, "cross_entropy_loss": avg_loss}


def benchmark_memory_recall(model, device, n_trials=10):
    """
    Needle-in-Haystack test for episodic memory.
    Injects a random code, stores in memory, then retrieves.
    """
    print("\n[2/5] Benchmarking Memory Recall...")
    model.eval()
    
    memory = EpisodicMemory(dim=Config.embed_dim, capacity=100)
    adapter = TextAdapter()
    
    correct = 0
    
    for trial in range(n_trials):
        # Generate random "needle"
        code = f"SECRET-{np.random.randint(1000, 9999)}"
        text = f"The secret code is {code}. Remember this."
        
        # Tokenize and embed
        tokens = adapter.tokenizer.encode(text)[:Config.seq_len]
        tokens = tokens + [0] * (Config.seq_len - len(tokens))
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embed = model(input_ids, return_embeds=True).mean(dim=1)
        
        # Store with high "surprise"
        memory.add(embed[0].cpu().numpy(), tokens, loss_val=100.0)
        
        # Create query
        query_text = "What is the secret code?"
        query_tokens = adapter.tokenizer.encode(query_text)[:Config.seq_len]
        query_tokens = query_tokens + [0] * (Config.seq_len - len(query_tokens))
        query_ids = torch.tensor(query_tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_embed = model(query_ids, return_embeds=True).mean(dim=1)
        
        # Retrieve
        results = memory.retrieve(query_embed[0].cpu().numpy(), k=1)
        
        if results:
            retrieved_tokens, score = results[0]
            retrieved_text = adapter.tokenizer.decode(retrieved_tokens)
            if code in retrieved_text:
                correct += 1
    
    accuracy = correct / n_trials
    print(f"   Recall Accuracy: {accuracy:.1%} ({correct}/{n_trials})")
    return {"memory_recall_accuracy": accuracy, "trials": n_trials}


def benchmark_visual_grounding(model, device, n_batches=20):
    """
    Test if model associates images with correct class labels.
    Uses CIFAR-10 streaming with synthetic captions.
    """
    print("\n[3/5] Benchmarking Visual Grounding...")
    model.eval()
    
    try:
        mm_adapter = MultimodalAdapter(batch_size=4, image_size=128, device=device, streaming=True)
    except:
        print("   SKIPPED: Could not load streaming adapter.")
        return {"visual_grounding_accuracy": None, "skipped": True}
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(n_batches):
            batch = mm_adapter.get_batch(device)
            vis_latents = batch['visual_latents']
            input_ids = batch['input_ids']
            
            # Forward with visual conditioning
            logits, _ = model(input_ids=input_ids, visual_latents=vis_latents, t=torch.zeros(1, device=device))
            
            # Check if predicted tokens match input (caption)
            preds = logits.argmax(dim=-1)
            
            # Simple metric: Do the first 5 tokens match?
            for j in range(input_ids.shape[0]):
                if torch.equal(preds[j, :5], input_ids[j, :5]):
                    correct += 1
                total += 1
    
    accuracy = correct / max(total, 1)
    print(f"   Caption Match Accuracy: {accuracy:.1%}")
    return {"visual_grounding_accuracy": accuracy, "samples": total}


def benchmark_speed(model, device, n_iters=100):
    """
    Measure forward pass speed (tokens/second).
    """
    print("\n[4/5] Benchmarking Speed...")
    model.eval()
    
    batch_size = 4
    seq_len = Config.seq_len
    
    input_ids = torch.randint(0, Config.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids=input_ids, t=torch.zeros(1, device=device))
    
    # Timed run
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(input_ids=input_ids, t=torch.zeros(1, device=device))
    
    elapsed = time.time() - start
    total_tokens = batch_size * seq_len * n_iters
    tokens_per_sec = total_tokens / elapsed
    
    print(f"   Throughput: {tokens_per_sec:.0f} tokens/sec ({elapsed:.2f}s for {n_iters} iters)")
    return {"tokens_per_second": tokens_per_sec, "batch_size": batch_size, "seq_len": seq_len}


def benchmark_generation(model, adapter, sde, device, n_samples=5):
    """
    Generate samples and score coherence (basic length/repetition check).
    """
    print("\n[5/5] Benchmarking Generation Quality...")
    model.eval()
    
    prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "Scientists have discovered that",
        "The most important thing about",
        "Once upon a time there was"
    ][:n_samples]
    
    results = []
    
    for prompt in prompts:
        tokens = adapter.tokenizer.encode(prompt)[:64]
        tokens = tokens + [0] * (Config.seq_len - len(tokens))
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        
        # Simple generation: forward pass at t=0
        with torch.no_grad():
            logits, _ = model(input_ids=input_ids, t=torch.zeros(1, device=device))
            gen_tokens = logits[0].argmax(dim=-1).cpu().tolist()
        
        output = adapter.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        # Basic quality metrics
        unique_tokens = len(set(gen_tokens))
        repetition_ratio = 1 - (unique_tokens / len(gen_tokens))
        
        results.append({
            "prompt": prompt,
            "output_preview": output[:100],
            "repetition_ratio": repetition_ratio
        })
    
    avg_rep = np.mean([r["repetition_ratio"] for r in results])
    print(f"   Avg Repetition Ratio: {avg_rep:.2%} (lower is better)")
    
    return {"samples": results, "avg_repetition_ratio": avg_rep}


def run_benchmark_suite(checkpoint, device, output_path=None):
    """
    Run full benchmark suite and output results.
    """
    print("=" * 60)
    print("SYNAPSE BENCHMARK SUITE")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Setup
    Config.vocab_size = 50257
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    adapter = TextAdapter()
    
    # Load model
    model = PCModel().to(device)
    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        if isinstance(state_dict, dict) and 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        print(f">> Model loaded from {checkpoint}")
    else:
        print(f"!! Checkpoint not found: {checkpoint}")
    
    # Run benchmarks
    results = {
        "checkpoint": checkpoint,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vocab_size": Config.vocab_size,
            "embed_dim": Config.embed_dim,
            "n_layers": Config.n_layers,
            "n_heads": Config.n_heads
        }
    }
    
    results["perplexity"] = benchmark_perplexity(model, adapter, sde, device)
    results["memory"] = benchmark_memory_recall(model, device)
    results["vision"] = benchmark_visual_grounding(model, device)
    results["speed"] = benchmark_speed(model, device)
    results["generation"] = benchmark_generation(model, adapter, sde, device)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"| Metric                | Value           |")
    print(f"|----------------------|-----------------|")
    print(f"| Perplexity           | {results['perplexity']['perplexity']:.2f}           |")
    print(f"| Memory Recall        | {results['memory']['memory_recall_accuracy']:.1%}           |")
    if results['vision'].get('visual_grounding_accuracy') is not None:
        print(f"| Visual Grounding     | {results['vision']['visual_grounding_accuracy']:.1%}           |")
    print(f"| Speed (tok/s)        | {results['speed']['tokens_per_second']:.0f}         |")
    print(f"| Repetition Ratio     | {results['generation']['avg_repetition_ratio']:.1%}           |")
    print("=" * 60)
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n>> Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints_unified/step_500.pt")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    run_benchmark_suite(args.ckpt, device, args.output)
