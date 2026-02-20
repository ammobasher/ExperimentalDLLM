"""
Streaming Data Loader for HuggingFace Datasets.
Drop-in replacement for CachedDataLoader that streams
directly from HuggingFace, eliminating the caching step.

Provides the same interface:
  - get_train_batch() -> (input_ids, latents)
  - advance()
  - text_steps property
"""

import torch
import numpy as np
from transformers import GPT2TokenizerFast
from datasets import load_dataset


class StreamingDataLoader:
    """
    Streams training data directly from HuggingFace.
    No pre-caching required — tokenizes on-the-fly.

    Compatible with train_episodic.py (same API as CachedDataLoader).
    """

    def __init__(self, dataset_name, device, target_batch_size=8,
                 seq_len=512, skip_samples=0):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g. 'HuggingFaceFW/fineweb-edu')
            device: torch device ('cuda', 'mps', 'cpu')
            target_batch_size: Number of samples per batch
            seq_len: Sequence length for tokenization
            skip_samples: Number of samples to skip (for resuming)
        """
        self.device = device
        self.target_batch_size = target_batch_size
        self.seq_len = seq_len
        self.step = 0
        self.has_vision = False

        # Tokenizer
        print(">> Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Stream dataset
        print(f">> Loading {dataset_name} (streaming)...")
        try:
            self.dataset = load_dataset(dataset_name, split="train", streaming=True)
        except Exception:
            self.dataset = load_dataset(dataset_name, name="default",
                                        split="train", streaming=True)
        self.ds_iter = iter(self.dataset)

        # Skip samples if resuming
        if skip_samples > 0:
            print(f">> Skipping {skip_samples} samples for resume...")
            for _ in range(skip_samples):
                try:
                    next(self.ds_iter)
                except StopIteration:
                    self.ds_iter = iter(self.dataset)
                    next(self.ds_iter)
            print(f"   ✓ Skipped {skip_samples} samples")

        # Pre-fill a small buffer for smoother batching
        self._buffer = []
        self._buffer_size = target_batch_size * 10  # Keep 10 batches ahead
        self._fill_buffer()

        # Compatibility
        self.n_text_steps = 999999999  # Effectively infinite

        print(f"✓ StreamingDataLoader ready (batch_size={target_batch_size}, seq_len={seq_len})")

    def _tokenize_sample(self, text):
        """Tokenize a single text sample to fixed length."""
        tokens = self.tokenizer.encode(
            text,
            max_length=self.seq_len,
            truncation=True,
        )
        # Pad if needed
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
        return tokens[:self.seq_len]

    def _get_next_sample(self):
        """Get next valid tokenized sample from the stream."""
        while True:
            try:
                sample = next(self.ds_iter)
            except StopIteration:
                # Restart stream
                print("\n>> Dataset stream exhausted, restarting...")
                self.ds_iter = iter(self.dataset)
                sample = next(self.ds_iter)

            text = sample.get("text", "")
            if not text or len(text.strip()) < 50:
                continue  # Skip empty/tiny texts

            return self._tokenize_sample(text)

    def _fill_buffer(self):
        """Fill the internal buffer with tokenized samples."""
        while len(self._buffer) < self._buffer_size:
            self._buffer.append(self._get_next_sample())

    def get_train_batch(self):
        """Get training batch. Same API as CachedDataLoader."""
        # Ensure buffer has enough samples
        if len(self._buffer) < self.target_batch_size:
            self._fill_buffer()

        # Take a batch from buffer
        batch_tokens = self._buffer[:self.target_batch_size]
        self._buffer = self._buffer[self.target_batch_size:]

        # Convert to tensor
        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=self.device)

        # Refill buffer in background (non-blocking, just top up)
        if len(self._buffer) < self._buffer_size // 2:
            self._fill_buffer()

        return input_ids, None  # No vision latents

    def get_val_batch(self):
        """Get a validation batch (uses buffer samples)."""
        if len(self._buffer) > 0:
            idx = np.random.randint(0, len(self._buffer))
            return torch.tensor(self._buffer[idx], dtype=torch.long, device=self.device)
        return None

    @property
    def text_steps(self):
        return self.step

    @text_steps.setter
    def text_steps(self, value):
        self.step = value

    def advance(self):
        self.step += 1
