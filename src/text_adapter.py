import numpy as np
# import jax - REMOVED for PyTorch Migration
# import jax.numpy as jnp - REMOVED

import numpy as np
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import random
import time

class TextAdapter:
    """
    Handles streaming text data, tokenization, and batching 
    for the Diffusion LLM.
    """
    def __init__(self, seq_len=64, batch_size=4, dataset_name="wikitext", split="train"):
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        # Load Tokenizer
        print(f"[TextAdapter] Loading GPT-2 Tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        
        # Load Dataset
        print(f"[TextAdapter] Loading {dataset_name} Dataset ({split})...")
        try:
            if dataset_name == "wikitext":
                self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
            elif dataset_name == "openwebtext":
                # OpenWebText usually only has train
                self.dataset = load_dataset("openwebtext", split="train", streaming=True)
            else:
                 self.dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            self.iterator = iter(self.dataset)
        except Exception as e:
            print(f"Warning: Could not load {dataset_name} ({e}). Using dummy text.")
            self.dataset = None

    def get_batch(self):
        """
        Returns: [Batch, Seq] integer array of tokens.
        """
        texts = []
        
        while len(texts) < self.batch_size:
            if self.dataset is None:
                # Dummy mode
                return np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
                
            try:
                # Resilience: Retry loop for network glitches
                retries = 0
                max_retries = 12 # ~1 hour of backoff total
                
                while True:
                    try:
                        item = next(self.iterator)
                        break # Success
                    except StopIteration:
                        raise StopIteration # Propagate up to reset logic
                    except Exception as e:
                        retries += 1
                        if retries > max_retries:
                            print(f"!! [TextAdapter] Critical: Max retries exhausted ({e}). Crashing.")
                            raise e
                        
                        sleep_time = min(2 ** retries, 300) # Cap at 5 mins
                        print(f"!! [TextAdapter] Fetch Error: {e}. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)

                text = item['text'].strip()
                if len(text) > 20: # Skip empty/short lines
                    texts.append(text)
            except StopIteration:
                # Reset
                self.iterator = iter(self.dataset)
        
        # Tokenize
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=self.seq_len, 
            return_tensors="np"
        )
        
        return np.array(encodings['input_ids'])

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
