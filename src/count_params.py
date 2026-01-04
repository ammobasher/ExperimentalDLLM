import jax
import jax.numpy as jnp
import equinox as eqx
from src.config import Config
from src.model import PCModel

def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

if __name__ == "__main__":
    # Use the vocab size from our text run
    Config.vocab_size = 50257 
    key = jax.random.PRNGKey(0)
    model = PCModel(key)
    
    total = count_params(model)
    print(f"Total Parameters: {total:,}")
    
    # Breakdown
    print(f"Embedding: {count_params(model.embedding):,}")
    print(f"Output Head: {count_params(model.output_head):,}")
    print(f"Layers (x{Config.n_layers}): {count_params(model.layers):,}")
