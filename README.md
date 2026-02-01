# Project Synapse: Synthetic Neocortex

**Project Synapse** is an experimental "Diffusion Large Language Model" (DLLM) designed to mimic the architecture of the human neocortex. It moves beyond standard Transformers by integrating **Predictive Coding (PC)**, **Bi-Level Optimization (BLO)** for neuromodulation, and an **Episodic Memory (eLTM)** system.

Built with **JAX** and **Equinox** for high-performance research.

## 🌟 Key Features

### 1. The Synthetic Neocortex (`PCModel`)
*   **Hierarchical Predictive Coding:** A 6-layer model where each layer predicts the state of the layer below (Top-Down) while processing errors (Bottom-Up).
*   **Dual-Pass Forward:** Explicit separate passes for *Perception* (Embedding $\to$ Top Layer) and *Prediction* (Top Layer $\to$ Reconstruction).
*   **Diffusion Decoding:** Treats text generation as denoising a continuous latent space over time ($t=1.0 \to t=0$).

### 2. Neuromodulation (`BetaController`)
*   **Dynamic Attention:** A Meta-Controller network (3-layer MLP) that dynamically adjusts the "PC Loss Weight" ($\beta$) based on the current timestep and error signal.
*   **Bi-Level Optimization:** The Controller is trained via meta-gradients to optimize the *validation performance* of the Neocortex (using `jax.grad` through the inner update loop).

### 3. Episodic Memory (`Hippocampus`)
*   **Novelty Trigger:** High-surprise events (loss > threshold) are automatically saved to an external Vector Database (eLTM).
*   **Sleep Cycles:** Offline consolidation phase where the model "dreams" (replays) memories to fine-tune weights, preventing catastrophic forgetting.

## 🚀 Quick Start

### 1. Installation
```bash
# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
# (Optional) Install JAX with TPU/CUDA support if available
```

### 2. Verify Installation
Run the automated test suite to check Gradients, Memory, and Diffusion baselines:
```bash
./run_tests.sh
```

### 3. Training on Real Text
Train the Diffusion LLM on **WikiText-2** (Auto-downloaded):
```bash
./venv/bin/python3 main_text.py --steps 1000 --batch_size 8 --save_every 100
```
*   **Metrics:** Watch for `Loss` (NLL) decreasing.
*   **Checkpoints:** Saved to `checkpoints/`.

### 4. Visual World Model (Experimental)
Simulate a visual neocortex predicting latent dynamics (Mock Minecraft):
```bash
./venv/bin/python3 main_visual.py
```

## 📚 Documentation
For detailed technical info, see the `docs/` folder:
*   [Architecture & Implementation Details](docs/ARCHITECTURE.md)
*   [Test Strategy, Models & Metrics](docs/TESTING.md)

## 📁 Repository Structure
*   `src/model.py`: Core `PCModel` (Dual-Pass Transformer).
*   `src/layers.py`: `PCLayer` with Chunked Attention.
*   `src/controller.py`: `BetaController` (Neuromodulator).
*   `src/meta_trainer.py`: Bi-Level Optimization Logic.
*   `src/memory.py`: eLTM (Episodic Memory).
*   `src/text_adapter.py`: WikiText & GPT-2 Tokenizer.
