# Architecture & Implementation

## 1. The Synthetic Neocortex (`PCModel`)
**Location:** `src/model.py`, `src/layers.py`

The core of Project Synapse is the `PCModel`, a hierarchical Predictive Coding Transformer. Unlike standard LLMs which are purely autoregressive, `PCModel` is trained to act as a **Denoising Autoencoder** with explicit top-down predictions.

### 1.1 `PCLayer` (The Micro-Circuit)
Each layer implements a "Canonical Micro-Circuit" inspired by biological cortex columns.
*   **Chunked Attention:** Input sequence is reshaping into `(Batch, N_Chunks, Chunk_Size, Dim)`. Attention is applied *locally* within chunks to simulate mini-columnar processing.
*   **Dual Inputs:** Accepts `current_x` (Bottom-Up input from layer below) and `prediction_from_above` (Top-Down expectation).
*   **Output:** Generates `x_next` (feature mapping for layer above) and `prediction` (top-down signal for layer below).

### 1.2 Dual-Pass Forward
The model executes two distinct passes for every timestep $t$:
1.  **Bottom-Up (Perception):** Information flows from Embedding $\to$ Layer 1 $\to$ Layer $N$. This builds the "belief".
2.  **Top-Down (Prediction):** Information flows from Layer $N \to$ Layer 1. Each layer predicts the activity of the layer below it.
3.  **Error Calculation:** $E_l = X_l - P_{l+1}$. The "Predictive Coding Loss" is the magnitude of this error.

## 2. Diffusion Engine (`DiffusionSDE`)
**Location:** `src/diffusion.py`

We model text generation as a **Variance Preserving (VP) SDE**.
*   **Continuous Embeddings:** Tokens are embedded into continuous space $x_0$.
*   **Forward Process (Noise):** $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$. We add Gaussian noise to text embeddings.
*   **Reverse Process (Denoise):** The `PCModel` predicts $x_0$ (clean embedding) from $x_t$.
*   **Rounding:** The final denoised vector is mapped to the nearest discrete token via `output_head`.

## 3. Neuromodulation (`BetaController`)
**Location:** `src/controller.py`, `src/meta_trainer.py`

A key innovation is the **Meta-Learner** that adjusts the learning objective dynamically.
*   **The Problem:** Standard Predictive Coding has a fixed trade-off between "Global Task Loss" (CE) and "Local Consistency Loss" (PC).
*   **The Solution:** A small MLP (`BetaController`) observes the system state (Timestep, Current Error) and outputs a scalar $\beta$.
*   **Bi-Level Optimization (BLO):**
    *   **Inner Loop:** Update `PCModel` weights to minimize $L = L_{CE} + \beta \cdot L_{PC}$.
    *   **Outer Loop:** Update `BetaController` weights to minimize $L_{Validation}$ (using higher-order gradients via `jax.grad`).

## 4. Episodic Memory (`Hippocampus`)
**Location:** `src/memory.py`, `src/sleep.py`

*   **Vector Store:** An in-memory similarity search (cosine distance) stores latent vectors of events.
*   **Novelty Trigger:** Events are strictly filtered. Only inputs where the `PC Loss > Threshold` are saved. This creates a "Surprise-based" memory.
*   **Sleep Cycle:** A dedicated training phase (e.g., "Night") where the model halts intake and re-trains on batches retrieved from the Memory. This consolidates episodic data into semantic weights.

## 5. Adapters
*   **Visual (`src/world_model.py`):** Wraps a Pretrained VAE (Stable Diffusion) to feed visual latents into the Neocortex.
*   **Text (`src/text_adapter.py`):** Wraps HuggingFace Tokenizers and Datasets (WikiText) to feed token embeddings.
