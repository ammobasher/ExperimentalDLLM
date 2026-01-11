"""
Pre-Cache Training Data for Large-Scale Training
=================================================
Creates cached data files to eliminate streaming overhead.
Supports:
- WikiText-2 (small, ~100MB)
- SlimPajama (large, 627B tokens from HuggingFace streaming)
"""
import torch
import numpy as np
import os
import time
import argparse
from tqdm import tqdm

from src.config import Config

# Try to import dependencies
try:
    from transformers import GPT2TokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets not available")

try:
    from diffusers import AutoencoderKL
    from torchvision import transforms
    from PIL import Image
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
    print("Warning: Vision dependencies not available")

Config.vocab_size = 50257


def get_tokenizer():
    """Get GPT-2 tokenizer."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def cache_wikitext_data(n_steps, batch_size, output_dir, seq_len=512):
    """Cache WikiText-2 text batches (original method)."""
    print(f"\n[TEXT] Caching WikiText-2 ({n_steps} batches)...")
    
    from src.text_adapter import TextAdapter
    adapter = TextAdapter(batch_size=batch_size, split="train")
    
    all_batches = []
    for i in tqdm(range(n_steps), desc="Caching WikiText"):
        batch = adapter.get_batch()  # [B, SeqLen]
        all_batches.append(batch)
    
    # Stack and save
    all_batches = np.stack(all_batches)  # [N, B, SeqLen]
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "text_cache.npz")
    np.savez_compressed(path, batches=all_batches)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"   Saved {all_batches.shape} to {path} ({size_mb:.1f} MB)")
    return path


def cache_slimpajama_data(n_steps, batch_size, output_dir, seq_len=512, 
                          chunk_size=10000, resume=False):
    """
    Stream and cache SlimPajama data from HuggingFace.
    
    Args:
        n_steps: Number of batches to cache
        batch_size: Samples per batch
        output_dir: Directory to save cached data
        seq_len: Sequence length for tokenization
        chunk_size: Save checkpoint every N steps (for large caching runs)
        resume: Resume from last checkpoint if exists
    """
    print(f"\n[TEXT] Caching SlimPajama ({n_steps} batches)...")
    print(f"   Dataset: cerebras/SlimPajama-627B")
    print(f"   Batch Size: {batch_size}")
    print(f"   Seq Length: {seq_len}")
    print(f"   Checkpoint Every: {chunk_size} steps")
    
    if not HAS_DATASETS or not HAS_TRANSFORMERS:
        raise RuntimeError("datasets and transformers packages required")
    
    tokenizer = get_tokenizer()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for resume
    start_chunk = 0
    all_batches = []
    
    if resume:
        # Find existing chunks
        existing_chunks = [f for f in os.listdir(output_dir) 
                         if f.startswith("text_cache_chunk_") and f.endswith(".npz")]
        if existing_chunks:
            start_chunk = len(existing_chunks)
            print(f"   Resuming from chunk {start_chunk}")
    
    # Stream SlimPajama
    print("   Loading SlimPajama streaming dataset...")
    ds = load_dataset(
        "cerebras/SlimPajama-627B", 
        split="train", 
        streaming=True,
        trust_remote_code=True
    )
    ds_iter = iter(ds)
    
    # Skip to resume point
    if start_chunk > 0:
        skip_samples = start_chunk * chunk_size * batch_size
        print(f"   Skipping {skip_samples} samples for resume...")
        for _ in tqdm(range(skip_samples), desc="Skipping"):
            try:
                next(ds_iter)
            except StopIteration:
                ds_iter = iter(ds)
                next(ds_iter)
    
    current_chunk_batches = []
    samples_in_batch = []
    
    step = start_chunk * chunk_size
    target_steps = n_steps
    
    pbar = tqdm(total=target_steps - step, desc="Caching SlimPajama", initial=0)
    
    while step < target_steps:
        # Collect batch_size samples
        while len(samples_in_batch) < batch_size:
            try:
                sample = next(ds_iter)
                text = sample.get("text", "")
                
                # Skip empty texts
                if not text or len(text.strip()) < 50:
                    continue
                
                # Tokenize
                tokens = tokenizer.encode(
                    text,
                    max_length=seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None
                )
                
                # Ensure correct length
                if len(tokens) < seq_len:
                    tokens = tokens + [tokenizer.pad_token_id] * (seq_len - len(tokens))
                tokens = tokens[:seq_len]
                
                samples_in_batch.append(tokens)
                
            except StopIteration:
                # Restart dataset if exhausted
                print("\n   Dataset exhausted, restarting...")
                ds = load_dataset(
                    "cerebras/SlimPajama-627B",
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
                ds_iter = iter(ds)
        
        # Create batch
        batch = np.array(samples_in_batch[:batch_size], dtype=np.int64)
        samples_in_batch = samples_in_batch[batch_size:]
        
        current_chunk_batches.append(batch)
        step += 1
        pbar.update(1)
        
        # Save checkpoint
        if len(current_chunk_batches) >= chunk_size:
            chunk_idx = step // chunk_size
            chunk_data = np.stack(current_chunk_batches)
            chunk_path = os.path.join(output_dir, f"text_cache_chunk_{chunk_idx:04d}.npz")
            np.savez_compressed(chunk_path, batches=chunk_data)
            size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            print(f"\n   Saved chunk {chunk_idx}: {chunk_data.shape} ({size_mb:.1f} MB)")
            current_chunk_batches = []
    
    pbar.close()
    
    # Save remaining batches
    if current_chunk_batches:
        chunk_idx = (step // chunk_size) + 1
        chunk_data = np.stack(current_chunk_batches)
        chunk_path = os.path.join(output_dir, f"text_cache_chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(chunk_path, batches=chunk_data)
        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        print(f"   Saved final chunk {chunk_idx}: {chunk_data.shape} ({size_mb:.1f} MB)")
    
    # Merge all chunks into single file (optional, for compatibility)
    merge_chunks(output_dir)
    
    return os.path.join(output_dir, "text_cache.npz")


def merge_chunks(output_dir):
    """Merge chunk files into single text_cache.npz for compatibility."""
    chunk_files = sorted([f for f in os.listdir(output_dir) 
                         if f.startswith("text_cache_chunk_") and f.endswith(".npz")])
    
    if not chunk_files:
        return
    
    print(f"\n   Merging {len(chunk_files)} chunks...")
    all_batches = []
    
    for chunk_file in tqdm(chunk_files, desc="Merging"):
        chunk_path = os.path.join(output_dir, chunk_file)
        data = np.load(chunk_path)
        all_batches.append(data['batches'])
    
    merged = np.concatenate(all_batches, axis=0)
    merged_path = os.path.join(output_dir, "text_cache.npz")
    np.savez_compressed(merged_path, batches=merged)
    
    size_mb = os.path.getsize(merged_path) / (1024 * 1024)
    print(f"   Merged {merged.shape} to {merged_path} ({size_mb:.1f} MB)")


def cache_vision_data(n_steps, batch_size, output_dir, device):
    """Cache CIFAR-10 images as VAE latents."""
    if not HAS_VISION:
        print("\n[VISION] Skipping (dependencies missing)")
        return None
    
    print(f"\n[VISION] Caching CIFAR-10 VAE Latents ({n_steps} batches)...")
    
    # Load VAE
    print("   Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device).eval()
    
    # Load CIFAR-10
    print("   Loading CIFAR-10 dataset...")
    ds = load_dataset("cifar10", split="train", streaming=False)
    
    # CIFAR-10 class names for captions
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    tokenizer = get_tokenizer()
    
    all_latents = []
    all_input_ids = []
    
    n_samples = len(ds)
    idx = 0
    
    for step in tqdm(range(n_steps), desc="Caching vision"):
        batch_latents = []
        batch_ids = []
        
        for b in range(batch_size):
            sample = ds[idx % n_samples]
            idx += 1
            
            # Image -> Latent
            img = sample['img']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.sample()
                latent = latent * 0.18215  # Scaling factor
            
            batch_latents.append(latent.cpu().numpy())
            
            # Caption
            label = sample['label']
            caption = f"A photo of a {class_names[label]}"
            tokens = tokenizer.encode(caption, max_length=Config.seq_len, 
                                       padding='max_length', truncation=True)
            batch_ids.append(tokens)
        
        all_latents.append(np.concatenate(batch_latents, axis=0))
        all_input_ids.append(np.array(batch_ids))
    
    # Stack
    all_latents = np.stack(all_latents)  # [N, B, C, H, W]
    all_input_ids = np.stack(all_input_ids)  # [N, B, SeqLen]
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "vision_cache.npz")
    np.savez_compressed(path, latents=all_latents, input_ids=all_input_ids)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"   Saved latents {all_latents.shape}, ids {all_input_ids.shape} to {path} ({size_mb:.1f} MB)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Cache training data for Synapse")
    parser.add_argument("--steps", type=int, default=50000, 
                       help="Number of batches to cache")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Samples per batch")
    parser.add_argument("--output_dir", type=str, default="cached_data",
                       help="Output directory for cached data")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["wikitext", "slimpajama"],
                       help="Dataset to cache")
    parser.add_argument("--seq_len", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--skip_vision", action="store_true",
                       help="Skip vision data caching")
    parser.add_argument("--chunk_size", type=int, default=10000,
                       help="Checkpoint every N steps (for SlimPajama)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA CACHING FOR SYNAPSE TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Steps: {args.steps}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Seq Length: {args.seq_len}")
    print(f"Output Dir: {args.output_dir}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    start = time.time()
    
    # Cache text data
    if args.dataset == "wikitext":
        text_path = cache_wikitext_data(
            args.steps, args.batch_size, args.output_dir, args.seq_len
        )
    else:  # slimpajama
        text_path = cache_slimpajama_data(
            args.steps, args.batch_size, args.output_dir, args.seq_len,
            chunk_size=args.chunk_size, resume=args.resume
        )
    
    # Cache vision data
    vision_path = None
    if not args.skip_vision:
        vision_path = cache_vision_data(
            args.steps, args.batch_size, args.output_dir, device
        )
    
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"CACHING COMPLETE ({elapsed/60:.1f} minutes)")
    print("=" * 60)
    print(f"Text Cache: {text_path}")
    print(f"Vision Cache: {vision_path}")
    
    # Print usage instructions
    print("\n" + "-" * 60)
    print("USAGE:")
    print(f"  python main_unified.py --cache_dir {args.output_dir} --steps {args.steps}")
    print("-" * 60)


if __name__ == "__main__":
    main()
