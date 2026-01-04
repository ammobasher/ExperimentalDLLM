
import jax
import jax.numpy as jnp
import equinox as eqx
from src.config import Config
from src.model import PCModel

def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

def main():
    # Force config to current scaled values if not picked up (though they should be)
    # Config.n_layers = 12
    # Config.embed_dim = 1024
    # Config.n_heads = 16
    
    print(f"Config: Layers={Config.n_layers}, Dim={Config.embed_dim}, Heads={Config.n_heads}")
    
    key = jax.random.PRNGKey(0)
    model = PCModel(key)
    
    params = count_params(model)
    print(f"Total Parameters: {params:,}")
    print(f"Total Parameters (Millions): {params/1e6:.2f}M")

if __name__ == "__main__":
    main()
