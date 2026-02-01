# Implementation Roadmap: Episodic-Centric Small LLM
## Practical Steps to Transform Current Codebase

---

## OVERVIEW

This roadmap details the specific code changes needed to transform the current experimental DLLM into a novel, locally-trainable small language model focused on **episodic-centric personalization**.

**Goal**: Create a 250M parameter model that personalizes through memory updates instead of weight updates.

---

## PHASE 1: MODEL COMPRESSION & OPTIMIZATION (Week 1-2)

### 1.1 Optimize Model Size

**Current**: 254M params (12 layers × 1024 dim)
**Target**: 250M params optimized for efficiency

**Changes to `src/config.py`**:
```python
class ConfigSmall:
    # Optimized for 250M params
    embed_dim: int = 768  # Reduced from 1024
    n_layers: int = 8     # Reduced from 12
    n_heads: int = 12     # Adjusted for 768 (64 per head)
    chunk_size: int = 32  # Keep mini-column size

    # Diffusion (simplified for faster inference)
    n_timesteps: int = 100  # Reduced from 1000 for speed
    beta_min: float = 0.1
    beta_max: float = 10.0  # Reduced from 20

    # Training
    vocab_size: int = 32000
    lr_llm: float = 1e-4
    lr_ctrl: float = 1e-5
    batch_size: int = 32
    seq_len: int = 512

    # NEW: Freezing config
    freeze_base: bool = True  # Freeze after pre-training
    enable_sleep: bool = True  # Allow sleep consolidation
```

**Calculation**:
- Embeddings: 32000 × 768 = 24.6M
- Layers: 8 × (768² × 4 [QKV+FFN]) ≈ 188M
- Total: ~213M params (leaving room for episodic memory)

### 1.2 Implement Model Freezing

**New file: `src/freezing.py`**:
```python
import equinox as eqx
import jax

def freeze_model(model):
    """
    Freeze all model parameters (no gradients computed).
    """
    def make_static(layer):
        return eqx.tree_at(
            lambda l: l,
            layer,
            replace_fn=lambda x: jax.lax.stop_gradient(x)
        )

    return jax.tree_map(make_static, model)

def unfreeze_model(model):
    """
    Unfreeze model for sleep consolidation.
    """
    # Simply return model without stop_gradient
    return model

def is_frozen(model):
    """
    Check if model is frozen.
    """
    # Check if gradients are stopped
    return hasattr(model, '_frozen') and model._frozen
```

**Modify `src/model.py`**:
```python
class PCModel(nn.Module):
    def __init__(self, config, frozen=False):
        super().__init__()
        self.config = config
        self._frozen = frozen
        # ... rest of init

    def freeze(self):
        """Mark model as frozen"""
        self._frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze for sleep consolidation"""
        self._frozen = False
        for param in self.parameters():
            param.requires_grad = True
```

---

## PHASE 2: EPISODIC MEMORY ENHANCEMENTS (Week 3-4)

### 2.1 Optimize Memory Retrieval Speed

**Current**: Basic numpy cosine similarity
**Target**: <10ms retrieval on 50K vectors

**Modify `src/memory.py`**:
```python
import numpy as np
import faiss  # For fast similarity search

class OptimizedEpisodicMemory:
    """
    High-performance episodic memory with FAISS indexing.
    """
    def __init__(self, dim: int, capacity: int = 50000):
        self.dim = dim
        self.capacity = capacity

        # Storage
        self.keys = np.zeros((capacity, dim), dtype=np.float32)
        self.values = [None] * capacity
        self.timestamps = np.zeros(capacity, dtype=np.int64)
        self.access_counts = np.zeros(capacity, dtype=np.int32)

        # FAISS index for fast retrieval
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

        self.count = 0
        self.threshold_tau = 10.0

    def add(self, vector, metadata, surprise_score: float):
        """
        Add with optimized indexing.
        """
        if hasattr(vector, 'cpu'):
            vector = vector.detach().cpu().numpy()

        # Normalize for cosine similarity
        vector = vector / (np.linalg.norm(vector) + 1e-8)

        if surprise_score > self.threshold_tau:
            idx = self.count % self.capacity

            # Store
            self.keys[idx] = vector
            self.values[idx] = metadata
            self.timestamps[idx] = self.count
            self.access_counts[idx] = 0

            # Update FAISS index (rebuild periodically for efficiency)
            if self.count % 100 == 0:
                n = min(self.count, self.capacity)
                self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(self.keys[:n])

            self.count += 1
            self.threshold_tau = 0.95 * self.threshold_tau + 0.05 * surprise_score
            return True
        else:
            self.threshold_tau = 0.99 * self.threshold_tau
            return False

    def retrieve(self, query_vector, k=5):
        """
        Fast retrieval using FAISS (<10ms for 50K vectors).
        """
        if self.count == 0:
            return []

        if hasattr(query_vector, 'cpu'):
            query_vector = query_vector.detach().cpu().numpy()

        # Normalize
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        # FAISS search
        n = min(self.count, self.capacity)
        k = min(k, n)
        scores, indices = self.index.search(query_vector, k)

        # Update access counts (for sleep consolidation priority)
        for idx in indices[0]:
            self.access_counts[idx] += 1

        results = []
        for i, idx in enumerate(indices[0]):
            results.append((self.values[idx], scores[0][i]))

        return results

    def sample_for_consolidation(self, n=1000):
        """
        Sample memories for sleep consolidation.
        Priority: high access count + high surprise + recent.
        """
        valid_count = min(self.count, self.capacity)

        # Compute priority scores
        recency = self.timestamps[:valid_count] / (self.count + 1)
        frequency = self.access_counts[:valid_count] / (np.max(self.access_counts[:valid_count]) + 1)
        priority = 0.4 * recency + 0.6 * frequency

        # Sample top N
        top_indices = np.argsort(priority)[-n:]

        return [
            {
                'embedding': self.keys[idx],
                'metadata': self.values[idx],
                'priority': priority[idx]
            }
            for idx in top_indices
        ]

    def prune_redundant(self, similarity_threshold=0.95):
        """
        Remove redundant memories (too similar to others).
        """
        valid_count = min(self.count, self.capacity)
        keep_mask = np.ones(valid_count, dtype=bool)

        # Compute pairwise similarities (expensive, only during sleep)
        similarities = self.keys[:valid_count] @ self.keys[:valid_count].T

        for i in range(valid_count):
            if not keep_mask[i]:
                continue

            # Find similar memories
            similar = np.where(similarities[i] > similarity_threshold)[0]

            # Keep the one with highest access count
            for j in similar:
                if j != i and self.access_counts[j] < self.access_counts[i]:
                    keep_mask[j] = False

        # Compact memory
        new_keys = self.keys[:valid_count][keep_mask]
        new_values = [self.values[i] for i in range(valid_count) if keep_mask[i]]
        new_timestamps = self.timestamps[:valid_count][keep_mask]
        new_access_counts = self.access_counts[:valid_count][keep_mask]

        self.keys[:len(new_keys)] = new_keys
        self.values[:len(new_values)] = new_values
        self.timestamps[:len(new_timestamps)] = new_timestamps
        self.access_counts[:len(new_access_counts)] = new_access_counts
        self.count = len(new_keys)

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(new_keys)

        return len(new_keys)
```

### 2.2 Memory-Augmented Generation

**New file: `src/memory_generate.py`**:
```python
def generate_with_memory(model, memory, prompt, max_length=100, k_memories=5):
    """
    Generate text using frozen model + episodic memory augmentation.

    1. Encode prompt
    2. Retrieve relevant memories
    3. Augment context with memories
    4. Generate using frozen model
    """
    # Encode prompt
    prompt_embedding = model.encode(prompt)  # [embed_dim]

    # Retrieve memories
    memories = memory.retrieve(prompt_embedding, k=k_memories)

    # Build augmented context
    context_parts = []
    for mem_text, score in memories:
        if score > 0.5:  # Only use relevant memories
            context_parts.append(f"[Memory: {mem_text}]")
    context_parts.append(prompt)
    full_context = "\n".join(context_parts)

    # Generate (model is frozen, no gradient updates)
    with torch.no_grad():
        output = model.generate(
            full_context,
            max_length=max_length,
            temperature=0.7
        )

    return output
```

---

## PHASE 3: SLEEP-BASED CONSOLIDATION (Week 5-6)

### 3.1 Implement Sleep Cycle

**New file: `src/sleep.py`**:
```python
import torch
import torch.nn as nn
from src.memory import OptimizedEpisodicMemory
from src.model import PCModel

class SleepConsolidation:
    """
    Implements sleep-based memory consolidation.

    Process:
    1. Sample high-priority memories
    2. Unfreeze model temporarily
    3. Replay memories with PC-guided learning
    4. Prune redundant memories
    5. Re-freeze model
    """
    def __init__(self, model: PCModel, memory: OptimizedEpisodicMemory, config):
        self.model = model
        self.memory = memory
        self.config = config

    def should_sleep(self):
        """
        Trigger sleep when memory >80% full.
        """
        usage = self.memory.count / self.memory.capacity
        return usage > 0.8

    def consolidate(self, n_replay=1000, n_epochs=3):
        """
        Main consolidation algorithm.
        """
        print(f"[Sleep] Starting consolidation...")
        print(f"[Sleep] Memory usage: {self.memory.count}/{self.memory.capacity}")

        # 1. Sample memories for replay
        memories = self.memory.sample_for_consolidation(n=n_replay)
        print(f"[Sleep] Sampled {len(memories)} memories for replay")

        # 2. Unfreeze model
        self.model.unfreeze()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # 3. Replay memories (multiple epochs)
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(0, len(memories), self.config.batch_size):
                batch = memories[i:i+self.config.batch_size]

                # Extract embeddings
                embeddings = torch.tensor(
                    [m['embedding'] for m in batch],
                    dtype=torch.float32
                )

                # Forward pass with PC loss
                logits, pc_loss = self.model(embeddings, return_pc_loss=True)

                # Loss: reconstruction + PC consistency
                # (No CE loss since we're replaying embeddings, not tokens)
                loss = pc_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(memories) // self.config.batch_size)
            print(f"[Sleep] Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

        # 4. Prune redundant memories
        old_count = self.memory.count
        new_count = self.memory.prune_redundant(similarity_threshold=0.95)
        print(f"[Sleep] Pruned {old_count - new_count} redundant memories")

        # 5. Re-freeze model
        self.model.freeze()
        print(f"[Sleep] Consolidation complete. Model re-frozen.")

        return {
            'memories_replayed': len(memories),
            'memories_pruned': old_count - new_count,
            'final_memory_count': new_count
        }
```

### 3.2 Integrate into Training Loop

**Modify `main_text.py`**:
```python
from src.sleep import SleepConsolidation

def train_with_sleep(model, memory, data_loader, config):
    """
    Training loop with episodic memory and sleep consolidation.
    """
    # Freeze model after pre-training
    if config.freeze_base:
        model.freeze()
        print("[Train] Model frozen. Personalization via episodic memory only.")

    # Initialize sleep consolidation
    sleep = SleepConsolidation(model, memory, config)

    for step, batch in enumerate(data_loader):
        # 1. Forward pass (frozen model)
        with torch.no_grad():
            logits, pc_loss = model(batch)

        # 2. Add high-surprise events to memory
        surprise_score = pc_loss.item()
        if surprise_score > memory.threshold_tau:
            # Extract embedding (mean pool over sequence)
            embedding = logits.mean(dim=1).cpu().numpy()
            memory.add(embedding, batch['text'], surprise_score)

        # 3. Check if sleep needed
        if config.enable_sleep and sleep.should_sleep():
            print(f"\n[Train] Memory threshold reached. Initiating sleep...")
            stats = sleep.consolidate()
            print(f"[Train] Sleep complete: {stats}\n")

        # 4. Logging
        if step % 100 == 0:
            print(f"Step {step}, PC Loss: {pc_loss.item():.4f}, "
                  f"Memory: {memory.count}/{memory.capacity}")
```

---

## PHASE 4: ADAPTIVE COMPUTATION (Week 7-8)

### 4.1 Dynamic Depth via Beta Controller

**Modify `src/controller.py`**:
```python
class AdaptiveController(eqx.Module):
    """
    Extended beta controller for adaptive computation.

    Outputs:
    - beta: Loss weighting (original)
    - early_exit_signal: Whether to stop computation early
    """
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=3,     # [timestep, pc_loss, prev_beta]
            out_size=2,    # [beta, exit_signal]
            width_size=32,
            depth=2,
            key=key
        )

    def __call__(self, t, pc_loss, prev_beta):
        inputs = jnp.array([t, pc_loss, prev_beta])
        outputs = self.mlp(inputs)

        beta = jax.nn.softplus(outputs[0]) + 0.01
        exit_signal = jax.nn.sigmoid(outputs[1])  # 0-1 probability

        return beta, exit_signal
```

**Modify `src/model.py`** to support early exit:
```python
class PCModel(nn.Module):
    def forward(self, x, adaptive=False, exit_threshold=0.8):
        """
        Forward with optional adaptive depth.
        """
        outputs = []
        pc_losses = []

        for i, layer in enumerate(self.layers):
            x_out, pc_loss = layer(x)
            pc_losses.append(pc_loss)

            if adaptive and i > 2:  # Don't exit too early
                # Use controller to decide if we can stop
                beta, exit_signal = self.controller(
                    t=i / len(self.layers),
                    pc_loss=pc_loss.item(),
                    prev_beta=beta if i > 0 else 1.0
                )

                if exit_signal > exit_threshold:
                    # Confident enough, stop here
                    return self.output_head(x_out), sum(pc_losses)

            x = x_out

        return self.output_head(x), sum(pc_losses)
```

---

## PHASE 5: EVALUATION & BENCHMARKING (Week 9-10)

### 5.1 Personalization Benchmark

**New file: `benchmarks/personalization_test.py`**:
```python
def test_instant_adaptation(model, memory):
    """
    Test: Can model instantly adapt after single interaction?
    """
    # Baseline: Ask question without context
    q1 = "What is my favorite color?"
    a1_before = model.generate(q1)  # Expected: "I don't know"

    # Provide context (single interaction)
    context = "My favorite color is blue."
    embedding = model.encode(context)
    memory.add(embedding, context, surprise_score=100.0)  # Force add

    # Test: Ask same question with memory
    a1_after = generate_with_memory(model, memory, q1)

    # Evaluate: Should mention "blue"
    success = "blue" in a1_after.lower()
    print(f"Instant Adaptation: {'✓' if success else '✗'}")
    print(f"Before: {a1_before}")
    print(f"After: {a1_after}")

    return success

def test_zero_forgetting(model, memory, test_set):
    """
    Test: Does model forget pre-training knowledge after many interactions?
    """
    # Measure baseline performance on MMLU
    baseline_acc = evaluate_mmlu(model, test_set)
    print(f"Baseline MMLU: {baseline_acc:.2%}")

    # Simulate 10,000 user interactions (add to memory)
    for i in range(10000):
        fake_interaction = f"User interaction {i}"
        embedding = model.encode(fake_interaction)
        memory.add(embedding, fake_interaction, surprise_score=10.0)

    # Re-evaluate MMLU
    after_acc = evaluate_mmlu(model, test_set)
    print(f"After 10K interactions: {after_acc:.2%}")

    # Success: <2% degradation
    degradation = baseline_acc - after_acc
    success = degradation < 0.02
    print(f"Zero Forgetting: {'✓' if success else '✗'} (degradation: {degradation:.2%})")

    return success
```

### 5.2 On-Device Benchmark

**New file: `benchmarks/device_benchmark.py`**:
```python
import time
import psutil

def benchmark_device_performance(model, device='cpu'):
    """
    Measure latency and memory on target device.
    """
    model.eval()
    model.to(device)

    # Warm-up
    dummy_input = torch.randint(0, 32000, (1, 512))
    for _ in range(10):
        _ = model.generate(dummy_input, max_length=50)

    # Measure latency
    latencies = []
    for _ in range(100):
        start = time.time()
        output = model.generate(dummy_input, max_length=50)
        latencies.append(time.time() - start)

    avg_latency = np.mean(latencies)
    tokens_per_sec = 50 / avg_latency

    # Measure memory
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    print(f"Device: {device}")
    print(f"Latency: {avg_latency*1000:.1f}ms per 50 tokens")
    print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")
    print(f"Memory: {memory_mb:.1f}MB")

    return {
        'latency_ms': avg_latency * 1000,
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_mb
    }
```

---

## PHASE 6: DEPLOYMENT & DEMO (Week 11-12)

### 6.1 Export to ONNX

**New file: `export_onnx.py`**:
```python
import torch
import torch.onnx

def export_to_onnx(model, output_path="synapse_250m.onnx"):
    """
    Export frozen model to ONNX for cross-platform deployment.
    """
    model.eval()

    # Dummy input
    batch_size = 1
    seq_len = 512
    dummy_input = torch.randint(0, 32000, (batch_size, seq_len))

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14
    )

    print(f"Model exported to {output_path}")

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    print(f"ONNX model verified. Input shape: {session.get_inputs()[0].shape}")
```

### 6.2 Demo Application

**New file: `demo/local_assistant.py`**:
```python
import streamlit as st
from src.model import PCModel
from src.memory import OptimizedEpisodicMemory
from src.memory_generate import generate_with_memory

# Load model and memory
@st.cache_resource
def load_model():
    model = PCModel.load("checkpoints/synapse_250m.pt")
    model.freeze()
    memory = OptimizedEpisodicMemory(dim=768, capacity=50000)
    return model, memory

model, memory = load_model()

st.title("Synapse: Privacy-First Local Assistant")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response with memory
    response = generate_with_memory(model, memory, prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Add to episodic memory if surprising
    surprise = compute_surprise(model, prompt, response)
    if surprise > memory.threshold_tau:
        memory.add(
            model.encode(f"{prompt} -> {response}"),
            f"{prompt} -> {response}",
            surprise
        )

# Sidebar: Memory stats
st.sidebar.title("Memory Statistics")
st.sidebar.metric("Memories Stored", memory.count)
st.sidebar.metric("Capacity", f"{memory.count/memory.capacity*100:.1f}%")
if st.sidebar.button("Sleep (Consolidate)"):
    from src.sleep import SleepConsolidation
    sleep = SleepConsolidation(model, memory, config)
    stats = sleep.consolidate()
    st.sidebar.success(f"Consolidated {stats['memories_replayed']} memories!")
```

---

## CRITICAL IMPLEMENTATION NOTES

### Dependencies to Add
```bash
pip install faiss-cpu  # For fast similarity search
pip install onnxruntime  # For deployment
pip install streamlit  # For demo
```

### File Structure After Implementation
```
ExperimentalDLLM/
├── src/
│   ├── model.py (MODIFIED: add freeze/unfreeze)
│   ├── memory.py (REPLACED: OptimizedEpisodicMemory)
│   ├── config.py (MODIFIED: add ConfigSmall)
│   ├── freezing.py (NEW)
│   ├── sleep.py (NEW)
│   ├── memory_generate.py (NEW)
│   └── ...
├── benchmarks/
│   ├── personalization_test.py (NEW)
│   └── device_benchmark.py (NEW)
├── demo/
│   └── local_assistant.py (NEW)
├── docs/
│   ├── NOVEL_LOCAL_DEPLOYMENT.md (NEW)
│   └── IMPLEMENTATION_ROADMAP.md (THIS FILE)
└── main_text.py (MODIFIED: add sleep training)
```

---

## SUCCESS METRICS

After implementation, validate:

✅ **Model Size**: 250M params, <600MB total (model + memory)
✅ **Instant Adaptation**: >15% improvement on user-specific queries after single interaction
✅ **Zero Forgetting**: <2% degradation on MMLU after 10K interactions
✅ **Latency**: >50 tok/sec on CPU (M1 Mac or equivalent)
✅ **Memory Retrieval**: <10ms for 50K vectors
✅ **Sleep Consolidation**: Successfully consolidates without forgetting

---

## NEXT STEPS

1. **Start with Phase 1**: Compress model to 250M params
2. **Validate each phase**: Don't move forward until current phase works
3. **Document results**: Keep track of benchmarks at each phase
4. **Iterate**: Adjust hyperparameters based on results

**Timeline**: 12 weeks (3 months) for full implementation and validation.

**Outcome**: A genuinely novel small language model that enables privacy-first, instant personalization on local devices.
