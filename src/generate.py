import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import argparse

from src.config import Config
from src.model import PCModel
from src.controller import BetaController
from src.diffusion import DiffusionSDE
from src.text_adapter import TextAdapter

def get_score_from_model(model, x_t, t):
    """
    The PCModel predicts logits or 'denoised x0'.
    We need the score (gradient of log density).
    For VP-SDE, score ~ -(x_t - x_0_hat) / (1 - exp(...)) ?
    
    Actually, our model is trained to minimize PC Error.
    The "Top-Down" prediction is essentially the denoised estimate x_0.
    
    Let's assume the model outputs `logits`. We converting logits -> embedding E[x_0].
    """
    # 1. Forward Pass
    # x_t: [1, Seq, Dim]
    # t: scalar
    
    # Model expects inputs_embeds
    logits, _ = model(inputs_embeds=x_t[0], t=t) # returns [Seq, Vocab]
    
    # 2. Convert Logits to Projected Embedding (Softmax * EmbedMatrix)
    # This gives us the "Expected x_0"
    probs = jax.nn.softmax(logits, axis=-1) # [Seq, Vocab]
    
    # We need the embedding matrix to project back!
    # embedding layer: [Vocab, Dim]
    embed_matrix = model.embedding.weight 
    
    x_0_hat = probs @ embed_matrix # [Seq, Dim]
    
    return x_0_hat

def reverse_diffusion_step(sde, x_t, t, dt, x_0_hat):
    """
    Euler-Maruyama step for Reverse SDE.
    dx = [f(x,t) - g(t)^2 score] dt + g(t) dw
    
    Using the 'Predictor-Corrector' or simple 'Denoising' form.
    Since we have x_0_hat, we can just use the posterior q(x_t-1 | x_t, x_0_hat).
    
    Formula for VP-SDE posterior mean:
    mu = (sqrt(alpha_bar_prev) * beta_t * x_0 + sqrt(alpha_t) * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar_t)
    """
    # Simplified Euler step using x_0_hat
    # x_{t-1} approx x_0_hat + noise * sigma(t-1)
    # But we want to follow the trajectory.
    
    # Let's use standard DDIM/DDPM-like update for simplicity if SDE is complex?
    # Or just linear interpolation for prototype.
    # x_{t-1} = (1 - (t-dt)) * x_0_hat + (t-dt) * noise_direction
    
    # For VP SDE: 
    # x_t = alpha * x_0 + sigma * eps
    # x_0_hat is our best guess.
    # clean direction: x_0_hat
    # noise direction: (x_t - alpha * x_0_hat) / sigma
    
    # Let's rely on simple linear interpolation for this high-level demo
    # x_{t-1} is a mix of x_0_hat and x_t.
    # As t -> 0, weight of x_0_hat -> 1.
    
    # Calculate noise coefficients
    mean, std = sde.marginal_prob(x_0_hat, t - dt)
    return mean + std * jax.random.normal(jax.random.PRNGKey(0), x_t.shape) # Deterministic sample for demo?

def generate(prompt, temperature=1.0):
    print(f"Loading Model for Generation... (Prompt: '{prompt}', Temp: {temperature})")
    
    # 1. Setup
    key = jax.random.PRNGKey(42)
    sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
    
    # Load Adapter for Tokenizer
    adapter = TextAdapter()
    Config.vocab_size = adapter.vocab_size
    
    # Init Model & Load Weights
    model = PCModel(key)
    controller = BetaController(key) # Dummy for loading alignment
    dummy_step = 0
    
    # Strict loading usually fails with Orbax if static fields aren't perfectly aligned or if Orbax saved a slightly different structure.
    # We use the robust partition/combine method.
    
    from src.checkpoint_manager import CheckpointManager
    ckpt_manager = CheckpointManager("checkpoints_scaled")
    latest_step = ckpt_manager.latest_step()
    
    if latest_step is None:
        print("!! No checkpoint found. Using random weights.")
    else:
        print(f"Loading Checkpoint Step {latest_step}...")
        restore_struct = {
            'model': model,
            'controller': controller
        }
        restored = ckpt_manager.restore(latest_step, restore_struct)
        
        # Robust Recombination
        # 1. Get static skeleton from fresh model
        _, static_model = eqx.partition(model, eqx.is_array)
        _, static_controller = eqx.partition(controller, eqx.is_array)
        
        # 2. Get arrays from restored
        arrays_model, _ = eqx.partition(restored['model'], eqx.is_array)
        arrays_controller, _ = eqx.partition(restored['controller'], eqx.is_array)
        
        # 3. Combine
        model = eqx.combine(arrays_model, static_model)
        controller = eqx.combine(arrays_controller, static_controller)
        print(">> Weights Loaded (Robust Combine).")
    
    # Load Episodic Memory
    from src.memory import EpisodicMemory
    import os
    memory = EpisodicMemory(key, dim=Config.embed_dim)
    mem_path = os.path.join("checkpoints_scaled", f"memory_step_{latest_step}.npz")
    if os.path.exists(mem_path):
        memory.load(mem_path)
        print(f">> Loaded Episodic Memory ({memory.count} entries)")
    else:
        print(">> No memory file found. Starting empty.")

    # 2. RAG Retrieval (Phase 17)
    # Tokenize Prompt first to get Query Vector
    tokens = adapter.tokenizer.encode(prompt)
    
    # Needs dummy forward to get embedding? No, we used model.embedding vmap below.
    # Let's do it here.
    temp_ids = jnp.array(tokens)
    if len(temp_ids) > 0:
         # [Seq, Dim]
         q_embeds = jax.vmap(model.embedding)(temp_ids)
         # Mean pooling
         q_vec = jnp.mean(q_embeds, axis=0) # [Dim]
         
         # Retrieve
         # Convert to numpy for memory
         q_vec_np = np.array(q_vec)
         results = memory.retrieve(q_vec_np, k=1)
         
         if results:
             print(f"   [Memory] Retrieved: {len(results[0][0])} tokens (Score: {results[0][1]:.2f})")
             # Prepend context
             # memory.values stores 'meta_tokens' (numpy array)
             retrieved_tokens = results[0][0].tolist() 
             # Truncate if too long?
             
             # Format: Context: <retrieved> \n\n Prompt: <prompt>
             # We construct new token sequence
             # Assuming GPT-2 tokenizer, maybe we can decode-encode or just cat ids
             context_ids = retrieved_tokens + adapter.tokenizer.encode("\n\n") + tokens
             tokens = context_ids
             print("   [Memory] Context injected.")
    
    N = Config.seq_len
    if len(tokens) > N: tokens = tokens[:N]
    
    # Create mask (1 for prompt, 0 for generate)
    mask = np.zeros(N)
    mask[:len(tokens)] = 1.0
    
    # Pad tokens
    padded_tokens = tokens + [0] * (N - len(tokens))
    
    # Embed Prompt (x_0_prompt)
    prompt_ids = jnp.array(padded_tokens)
    x_0_prompt = jax.vmap(model.embedding)(prompt_ids)
    
    # 3. Start Noise
    key, n_key = jax.random.split(key)
    x_t = jax.random.normal(n_key, (1, N, Config.embed_dim))
    
    # 4. Reverse Loop
    # We step from T=1.0 down to 0.0
    steps = 50 # Faster generation
    dt = 1.0 / steps
    
    print(f"Generating ({steps} steps)...")
    
    for i in range(steps):
        t_val = 1.0 - (i * dt)
        t = jnp.array(t_val)
        
        # A. Predict x_0
        x_0_hat = get_score_from_model(model, x_t, t)
        
        # B. Denoise Step (x_t -> x_{t-1})
        # Simple update: move towards x_0_hat
        # alpha = shrinkage factor
        # x_prev = x_t + (x_0_hat - x_t) * dt * rate?
        # Let's use the explicit marginal prob re-noising (Langevin-like)
        
        # x_{t-1} using Model
        key, step_key = jax.random.split(key)
        mean_pred, std_pred = sde.marginal_prob(x_0_hat, t - dt)
        x_pred = mean_pred + std_pred * jax.random.normal(step_key, x_t.shape)
        
        # x_{t-1} using Ground Truth (for Prompt)
        # We must preserve the prompt!
        mean_gt, std_gt = sde.marginal_prob(x_0_prompt, t - dt)
        x_gt = mean_gt + std_gt * jax.random.normal(step_key, x_t.shape) # Same noise for consistency?
        
        # Compose
        # Masks need to be broadcast [Seq, 1]
        m = jnp.array(mask).reshape(1, N, 1)
        
        x_t = x_gt * m + x_pred * (1 - m)
        
        if i % 10 == 0:
            print(f"   Step {i}/{steps}...")

    # 5. Decode
    logits, _ = model(inputs_embeds=x_t[0], t=0.0)
    
    # Apply Temperature
    logits = logits / temperature
    
    # Penalize EOS (50256) slightly to encourage text
    # Or strict ban if we want to force output. 
    # Let's set it to -1e9 for now to force it.
    logits = logits.at[..., 50256].set(-1e9)

    # Sample
    gen_key, _ = jax.random.split(key)
    token_ids = jax.random.categorical(gen_key, logits, axis=-1)
    
    # Cast to list for tokenizer
    token_ids_list = np.array(token_ids).tolist()
    
    output_text = adapter.tokenizer.decode(token_ids_list, skip_special_tokens=True)
    print("-" * 30)
    print(f"RESULT:\n{output_text}")
    print("-" * 30)

    # --- PHASE 21: ONLINE LEARNING (Test-Time Memorization) ---
    if args.learn:
        # We store the interaction (Prompt + Output) as a new memory.
        # This allows the model to "remember" this conversation in future generations (if RAG picks it up).
        # And in future training (if Sleep Cycle picks it up).
        
        # 1. Embed the full interaction
        full_text = args.prompt + output_text # Approximation
        full_tokens = adapter.tokenizer.encode(full_text)
        if len(full_tokens) > 0:
            ft_ids = jnp.array(full_tokens)
            ft_embeds = jax.vmap(model.embedding)(ft_ids)
            ft_vec = jnp.mean(ft_embeds, axis=0) # [Dim]
            
            # 2. Add to Memory (Surprise = 1.0 implies 'Pay Attention')
            added = memory.add(np.array(ft_vec), np.array(full_tokens), loss_val=1.5) # High surprise for user interaction
            
            if added:
                print("[Online Learning] Interaction stored in Short-Term Memory.")
                # 3. Persist Memory
                if latest_step is not None:
                    new_mem_path = os.path.join("checkpoints_scaled", f"memory_step_{latest_step}.npz")
                    memory.save(new_mem_path)
                    print(f"[Online Learning] Memory saved to {new_mem_path}.")
            else:
                print("[Online Learning] Interaction was too similar to existing memories. Ignored.")
    else:
        print("[Online Learning] Skipped (Safety Mode). Use --learn to enable.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The artificial intelligence")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--learn", action="store_true", help="Enable online memorization of this interaction.")
    args = parser.parse_args()
    
    generate(args.prompt, temperature=args.temp)

