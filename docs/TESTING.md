# Testing, Models & Evaluation

## 1. Test Strategy

We employ a three-tiered testing strategy:

### 1.1 Unit Tests (Automated)
Located in `tests/`, these verify the mathematical correctness of individual components.
*   **Gradient Verification (`verify_gradients.py`):** Ensures that gradients flow correctly through the `DiffusionSDE` and `PCLayer`, specifically checking that discrete operations don't break the computational graph.
*   **Meta-Gradient Verification (`verify_meta.py`):** Crucial test that checks if the "Outer Loop" (Controller) receives gradients from the "Inner Loop" (Model) through the `jax.grad` trace.
*   **Memory Verification (`verify_memory.py`):** Tests the Vector DB (Add, Retrieve) and the Sleep Cycle logic (Consolidation gain).

### 1.2 Integration Tests (Simulation)
Scripts that run end-to-end scenarios to verify system stability.
*   **System Simulation (`main.py`):** Runs a full "Day/Night" cycle with synthetic data to ensure all components interact without crashing.
*   **Visual Integration (`main_visual.py`):** Verifies the `LatentWorldModel` adapter can process 3D tensor inputs (visual latents).

### 1.3 Production Training (`main_text.py`)
The *Real World* test. Runs the model on actual language data (WikiText) for extended periods to verify:
*   Loss convergence (not NaN).
*   Checkpoint creation/saving.
*   Throughput (Steps/Sec).

**How to run all tests:**
```bash
./run_tests.sh
```

## 2. Models Selected

### 2.1 The "Neocortex" Model (`PCModel`)
*   **Type:** 6-Layer Hierarchical Transformer (Custom JAX implementation).
*   **Dimensions:** 512 Embedding Dim, 8 Heads, 32 Chunk Size.
*   **Justification:** Chosen for its ability to unify Top-Down prediction (hallucination/planning) with Bottom-Up perception in a single differentiable stack.

### 2.2 The "Visual Cortex" (`FlaxAutoencoderKL`)
*   **Source:** Stability AI (Stable Diffusion v1.4 VAE).
*   **Role:** Compresses $512 \times 512$ RGB images into $64 \times 64 \times 4$ latent approximations.
*   **Status:** Currently mocked in code due to `diffusers`/Python 3.14 compatibility, but architecture is ready for drop-in replacement.

### 2.3 The "Language Adapter"
*   **Tokenizer:** GPT-2 Tokenizer (`transformers`).
*   **Vocab:** 50,257 tokens (BPE).
*   **Justification:** Standard, robust tokenizer that handles open-domain English text well.

## 3. Training Datasets

### 3.1 WikiText-2 (Language)
*   **Source:** HuggingFace Datasets.
*   **Content:** High-quality Wikipedia articles.
*   **Size:** ~2M tokens.
*   **Role:** Used for Phase 7/8 "Real Text" training. Ideal for proving capability without requiring massive compute clusters.

### 3.2 MineRL (Visual / Agents)
*   **Source:** Minecraft Gameplay Trajectories (Simulated in Phase 6).
*   **Role:** Used to test the "World Model" capabilities. The logic assumes `(Batch, Time, H, W, C)` inputs typical of RL environments.

## 4. Evaluation Metrics

To measure success, run `src/evaluate_metrics.py`.

### 4.1 Perplexity (PPL)
*   **Definition:** $e^{\text{NLL}}$. How well does the model predict the next token?
*   **Target:** < 50 for coherent English.
*   **Current Baseline (Random):** ~67,000.

### 4.2 Bits-Per-Dimension (BPD)
*   **Definition:** Normalized NLL commonly used in Density Estimation / Diffusion papers.
*   **Formula:** $\frac{\text{NLL}}{\text{SeqLen} \times \ln(2)}$.

### 4.3 Beta Dynamics (System Health)
*   **Definition:** The variance of the `beta` parameter over time.
*   **Interpretation:**
    *   **High Variance:** Healthy. The brain is modulating attention based on surprise.
    *   **Flatline:** Unhealthy. The Meta-Learner has collapsed to a local minimum (always ignore PC loss or always prioritize it).
