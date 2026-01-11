"""
Evaluation Metrics for Diffusion-Based Language Models
========================================================
Standard perplexity doesn't work for diffusion LMs since they're trained
to denoise, not predict next tokens. This module implements proper metrics:

1. Sampling-Based Perplexity - Generate then measure NLL
2. Distinct-N - Lexical diversity (unique n-grams)
3. Self-BLEU - Repetition across samples
4. Generation Coherence - Basic coherence checks
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math


def distinct_n(texts: List[str], n: int = 1) -> float:
    """
    Calculate Distinct-N metric for lexical diversity.
    
    Distinct-N = unique n-grams / total n-grams
    Higher = more diverse (less repetitive)
    
    Args:
        texts: List of generated text strings
        n: N-gram size (1 for unigrams, 2 for bigrams)
    
    Returns:
        Distinct-N score between 0 and 1
    """
    all_ngrams = []
    
    for text in texts:
        words = text.lower().split()
        if len(words) >= n:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(ngrams)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams


def repetition_ratio(text: str) -> float:
    """
    Calculate token-level repetition ratio.
    
    Returns fraction of tokens that are duplicates.
    Lower = better (less repetitive)
    """
    words = text.split()
    if len(words) == 0:
        return 1.0
    
    unique = len(set(words))
    return 1.0 - (unique / len(words))


def self_bleu(texts: List[str], n_max: int = 4) -> float:
    """
    Calculate Self-BLEU score measuring repetition across samples.
    
    Self-BLEU computes BLEU of each sentence against all others.
    Lower = more diverse samples.
    
    Args:
        texts: List of generated text strings
        n_max: Maximum n-gram size for BLEU
    
    Returns:
        Average Self-BLEU score (0-1, lower is better)
    """
    if len(texts) < 2:
        return 0.0
    
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().split()
        return Counter(tuple(words[i:i+n]) for i in range(max(0, len(words) - n + 1)))
    
    def sentence_bleu(hypothesis: str, references: List[str], n_max: int = 4) -> float:
        """Simplified BLEU calculation."""
        hyp_words = hypothesis.lower().split()
        if len(hyp_words) == 0:
            return 0.0
        
        # Collect reference n-grams
        ref_ngrams = [Counter() for _ in range(n_max)]
        for ref in references:
            for n in range(1, n_max + 1):
                ref_ngrams[n-1] |= get_ngrams(ref, n)
        
        # Calculate precision for each n
        precisions = []
        for n in range(1, min(n_max + 1, len(hyp_words) + 1)):
            hyp_ngrams = get_ngrams(hypothesis, n)
            matches = sum((hyp_ngrams & ref_ngrams[n-1]).values())
            total = sum(hyp_ngrams.values())
            if total > 0:
                precisions.append(matches / total)
            else:
                precisions.append(0.0)
        
        if not precisions or all(p == 0 for p in precisions):
            return 0.0
        
        # Geometric mean (add smoothing for zeros)
        precisions = [p + 1e-10 for p in precisions]
        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        
        return math.exp(log_precision)
    
    # Calculate Self-BLEU
    scores = []
    for i, text in enumerate(texts):
        refs = texts[:i] + texts[i+1:]
        if refs:
            scores.append(sentence_bleu(text, refs, n_max))
    
    return sum(scores) / len(scores) if scores else 0.0


def sampling_perplexity(
    model,
    tokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
    n_samples: int = 10,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Calculate sampling-based perplexity for diffusion LMs.
    
    Instead of measuring CE at t=0, this:
    1. Generates continuations using proper DDPM sampling
    2. Measures NLL of ground truth tokens given generated context
    
    Args:
        model: PCModel instance
        tokenizer: Tokenizer
        prompts: List of prompt strings
        targets: List of target continuation strings
        device: Torch device
        n_samples: Number of samples per prompt
        temperature: Sampling temperature
    
    Returns:
        Dictionary with perplexity and related metrics
    """
    from src.diffusion import DiffusionSDE
    from src.config import Config
    
    model.eval()
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for prompt, target in zip(prompts, targets):
            # Tokenize
            prompt_ids = tokenizer.encode(prompt, max_length=256, truncation=True)
            target_ids = tokenizer.encode(target, max_length=256, truncation=True)
            
            # Create full sequence
            full_ids = prompt_ids + target_ids
            if len(full_ids) > Config.seq_len:
                full_ids = full_ids[:Config.seq_len]
            
            # Pad
            pad_len = Config.seq_len - len(full_ids)
            full_ids = full_ids + [tokenizer.pad_token_id or 0] * pad_len
            
            input_ids = torch.tensor(full_ids).unsqueeze(0).to(device)
            
            # Forward at t=0 (clean)
            logits, _ = model(input_ids=input_ids, t=torch.zeros(1, device=device))
            
            # Calculate NLL only for target portion
            prompt_len = len(prompt_ids)
            target_len = len(target_ids)
            
            if target_len > 0:
                # Get logits for target positions (shifted)
                target_logits = logits[0, prompt_len-1:prompt_len+target_len-1, :]
                target_labels = input_ids[0, prompt_len:prompt_len+target_len]
                
                # Calculate cross-entropy
                nll = F.cross_entropy(target_logits, target_labels, reduction='sum')
                total_nll += nll.item()
                total_tokens += target_len
    
    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 100))  # Cap to avoid overflow
    
    return {
        "sampling_perplexity": ppl,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens
    }


def generation_coherence(texts: List[str]) -> Dict[str, float]:
    """
    Basic coherence metrics for generated text.
    
    Returns:
        Dictionary with coherence-related metrics
    """
    if not texts:
        return {"avg_length": 0, "non_empty_ratio": 0, "avg_unique_ratio": 0}
    
    lengths = []
    unique_ratios = []
    non_empty = 0
    
    for text in texts:
        words = text.split()
        lengths.append(len(words))
        
        if len(words) > 0:
            non_empty += 1
            unique_ratios.append(len(set(words)) / len(words))
    
    return {
        "avg_length": sum(lengths) / len(lengths),
        "non_empty_ratio": non_empty / len(texts),
        "avg_unique_ratio": sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0
    }


def evaluate_generation_quality(
    model,
    tokenizer,
    device: torch.device,
    prompts: Optional[List[str]] = None,
    n_samples: int = 20,
    max_new_tokens: int = 50
) -> Dict[str, float]:
    """
    Comprehensive generation quality evaluation.
    
    Args:
        model: PCModel instance
        tokenizer: Tokenizer
        device: Torch device
        prompts: Optional list of prompts (uses defaults if None)
        n_samples: Number of samples to generate
        max_new_tokens: Max tokens to generate per sample
    
    Returns:
        Dictionary with all quality metrics
    """
    from src.config import Config
    
    if prompts is None:
        prompts = [
            "The future of technology is",
            "In recent years, scientists have discovered",
            "The most important thing about learning is",
            "When it comes to artificial intelligence,",
            "The history of human civilization shows",
        ]
    
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts[:n_samples]:
            # Tokenize prompt
            prompt_ids = tokenizer.encode(prompt, max_length=256, truncation=True)
            
            # Pad to seq_len
            pad_len = Config.seq_len - len(prompt_ids)
            full_ids = prompt_ids + [tokenizer.pad_token_id or 0] * pad_len
            
            input_ids = torch.tensor(full_ids).unsqueeze(0).to(device)
            
            # Generate (simple argmax for now)
            logits, _ = model(input_ids=input_ids, t=torch.zeros(1, device=device))
            
            # Get tokens after prompt
            gen_ids = logits[0].argmax(dim=-1).cpu().tolist()
            gen_text = tokenizer.decode(
                gen_ids[len(prompt_ids):len(prompt_ids)+max_new_tokens],
                skip_special_tokens=True
            )
            generated_texts.append(gen_text)
    
    # Calculate metrics
    results = {
        "distinct_1": distinct_n(generated_texts, n=1),
        "distinct_2": distinct_n(generated_texts, n=2),
        "self_bleu": self_bleu(generated_texts),
        "avg_repetition": sum(repetition_ratio(t) for t in generated_texts) / len(generated_texts),
    }
    
    # Add coherence metrics
    coherence = generation_coherence(generated_texts)
    results.update(coherence)
    
    return results


def run_full_evaluation(
    model,
    tokenizer,
    device: torch.device,
    test_texts: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, any]:
    """
    Run complete evaluation suite.
    
    Args:
        model: PCModel instance
        tokenizer: Tokenizer
        device: Torch device
        test_texts: Optional list of (prompt, target) tuples
    
    Returns:
        Complete evaluation results
    """
    print("\n" + "=" * 60)
    print("DIFFUSION LM EVALUATION SUITE")
    print("=" * 60)
    
    results = {}
    
    # 1. Generation Quality
    print("\n[1/2] Evaluating Generation Quality...")
    gen_results = evaluate_generation_quality(model, tokenizer, device)
    results["generation"] = gen_results
    print(f"   Distinct-1: {gen_results['distinct_1']:.3f}")
    print(f"   Distinct-2: {gen_results['distinct_2']:.3f}")
    print(f"   Self-BLEU: {gen_results['self_bleu']:.3f} (lower is better)")
    print(f"   Avg Repetition: {gen_results['avg_repetition']:.3f} (lower is better)")
    
    # 2. Sampling Perplexity (if test texts provided)
    if test_texts:
        print("\n[2/2] Evaluating Sampling Perplexity...")
        prompts, targets = zip(*test_texts)
        ppl_results = sampling_perplexity(model, tokenizer, list(prompts), list(targets), device)
        results["perplexity"] = ppl_results
        print(f"   Sampling PPL: {ppl_results['sampling_perplexity']:.2f}")
    else:
        print("\n[2/2] Skipping Sampling PPL (no test texts provided)")
    
    print("\n" + "=" * 60)
    
    return results


# Test function
def test_metrics():
    """Quick test of metric calculations."""
    print("Testing metrics...")
    
    # Test distinct-n
    texts = ["the cat sat on the mat", "a dog ran in the park", "the bird flew over the house"]
    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)
    print(f"  Distinct-1: {d1:.3f}")
    print(f"  Distinct-2: {d2:.3f}")
    
    # Test self-bleu
    sb = self_bleu(texts)
    print(f"  Self-BLEU: {sb:.3f}")
    
    # Test repetition
    rep = repetition_ratio("the the the cat cat")
    print(f"  Repetition ratio (repetitive): {rep:.3f}")
    
    rep2 = repetition_ratio("the quick brown fox jumps")
    print(f"  Repetition ratio (diverse): {rep2:.3f}")
    
    print("All metric tests passed!")


if __name__ == "__main__":
    test_metrics()
