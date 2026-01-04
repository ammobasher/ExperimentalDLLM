
import jax
import jax.numpy as jnp
import equinox as eqx

def main():
    key = jax.random.PRNGKey(0)
    try:
        # Matching src/layers.py usage exactly:
        # self.ffn = eqx.nn.MLP(embed_dim, embed_dim, embed_dim, depth=1, key=keys[1], activation=jax.nn.gelu)
        dim = 32
        mlp = eqx.nn.MLP(
            dim, dim, dim, 
            depth=1, 
            key=key, 
            activation=jax.nn.gelu
        )
        print("MLP initialized successfully.")
        
        x = jax.random.normal(key, (32,))
        y = mlp(x)
        print("MLP forward pass successful.")
        print(y.shape)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
