import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory import EpisodicMemory
from src.sleep import run_sleep_cycle
from src.model import PCModel
from src.config import Config

def verify_memory_system():
    print("--- Verifying Memory & Sleep System ---")
    key = jax.random.PRNGKey(42)
    key, m_key = jax.random.split(key)
    
    # 1. Initialize Memory
    print("Initializing Episodic Memory...")
    memory = EpisodicMemory(key, dim=512, capacity=100)
    
    # 2. Test Novelty Trigger (Add Items)
    print("Testing Novelty Storage...")
    dummy_vec = np.random.randn(512).astype(np.float32)
    dummy_data = np.random.randint(0, 32000, (Config.seq_len,)) # Mock token sequence
    
    # Low Loss -> Should NOT store
    added_low = memory.add(dummy_vec, dummy_data, loss_val=0.1)
    if added_low:
        print("FAIL: Low loss triggered storage.")
    
    # High Loss -> Should store
    added_high = memory.add(dummy_vec, dummy_data, loss_val=2.0)
    if not added_high:
        print("FAIL: High loss failed to trigger storage.")
    else:
        print("Novelty Trigger working. Memory Count:", memory.count)
        
    # Fill memory a bit for Sleep Test
    for _ in range(15):
        memory.add(np.random.randn(512), dummy_data, loss_val=1.0)
        
    # 3. Test Retrieval
    print("Testing Retrieval...")
    results = memory.retrieve(dummy_vec, k=2)
    print(f"Retrieved {len(results)} items.")
    assert len(results) > 0, "Retrieval failed"
    
    # 4. Test Sleep Cycle (Consolidation)
    print("Testing Sleep Cycle (Replay)...")
    
    model = PCModel(m_key)
    optimizer = optax.adamw(Config.lr_llm)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Run sleep
    model_slept, opts, loss = run_sleep_cycle(
        model, optimizer, opt_state, memory, key, n_steps=5
    )
    
    assert loss > 0.0, "Sleep Cycle produced zero loss (did it run?)"
    print(f"Sleep Cycle Verified. Loss: {loss}")
    print("\nMemory System Verified Successfully!")

if __name__ == "__main__":
    verify_memory_system()
