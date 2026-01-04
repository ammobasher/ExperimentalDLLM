import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple

from src.config import Config
from src.model import PCModel
from src.controller import BetaController
from src.diffusion import DiffusionSDE

# Meta-Training requires handling two optimization states and two models nested.

class MetaTrainer:
    def __init__(self, key: jax.random.PRNGKey):
        # Keys
        key, m_key, c_key = jax.random.split(key, 3)
        
        # Models
        self.model = PCModel(m_key)
        self.controller = BetaController(c_key)
        self.sde = DiffusionSDE(Config.beta_min, Config.beta_max, Config.n_timesteps)
        
        # Optimizers
        self.opt_llm = optax.adamw(Config.lr_llm)
        self.opt_ctrl = optax.adam(Config.lr_ctrl) # Simple Adam for controller
        
        # States
        self.state_llm = self.opt_llm.init(eqx.filter(self.model, eqx.is_array))
        self.state_ctrl = self.opt_ctrl.init(eqx.filter(self.controller, eqx.is_array))
        
    @eqx.filter_jit
    def train_step(self, 
                   model: PCModel, 
                   controller: BetaController, 
                   state_llm: optax.OptState, 
                   state_ctrl: optax.OptState, 
                   train_batch: jax.Array, 
                   val_batch: jax.Array,
                   key: jax.random.PRNGKey,
                   prev_beta: jax.Array
                   ) -> Tuple[PCModel, BetaController, optax.OptState, optax.OptState, dict, jax.Array]:
        
        # --- 1. INNER LOOP (LLM Update) ---
        # We need to compute gradients w.r.t LLM parameters, but these gradients depend on Beta,
        # which depends on Controller parameters.
        # To meta-learn, we need to trace the influence of Controller params on the *updated* LLM.
        # But efficiently: First-Order MAML approximation (Zero-order w.r.t Hessian) or Full BLO?
        # JAX handles full BLO perfectly via autodiff through the update step!
        
        key, t_key, dist_key = jax.random.split(key, 3)
        
        # Sample t and Noise for Train Batch
        batch_size = train_batch.shape[0]
        t = jax.random.uniform(t_key, (batch_size,)) # [0, 1]
        t_scalar = jnp.mean(t) # Controller sees average t for simple batching? or per sample?
        # Controller structure takes scalar t? Let's check src/controller.py
        # Yes, __call__(t, pc_loss, prev_beta).
        # We'll use scalar summary stats for the controller for now.
        
        x_0_train = jax.vmap(jax.vmap(model.embedding))(train_batch)
        noise = jax.random.normal(dist_key, x_0_train.shape)
        x_t_train = jax.vmap(self.sde.q_sample)(x_0_train, t, noise)
        
        def inner_loss_fn(m_params, c_params, x_t, t_batch):
            # Combine params
            # We treat controller as fixed here mostly, but we need gradients to flow back to it later.
            # Actually for standard inner step, we just use current controller.
            
            # Forward Model
            # m_params is just the model struct if we passed it correctly
            
            # 1. Get PC Error Signal (Noisy forward)
            logits, pc_loss_batch = jax.vmap(lambda x, ti: m_params(inputs_embeds=x, t=ti))(x_t, t_batch)
            
            pc_loss_mean = jnp.mean(pc_loss_batch)
            
            # 2. Get Beta from Controller
            # Controller inputs: t (mean), pc_error (mean), prev_beta
            beta = c_params(jnp.mean(t_batch), pc_loss_mean, prev_beta)
            
            # 3. Total Loss
            ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, train_batch))
            
            total_loss = ce_loss + beta * pc_loss_mean
            
            return total_loss, (ce_loss, pc_loss_mean, beta, pc_loss_batch)

        # Calculate Inner Gradients
        # We differentiate w.r.t MODEL params
        # But we capture Controller params in closure? No, pass strictly.
        
        (loss_inner, (ce_in, pc_in, beta_val, pc_loss_batch)), grads_llm = eqx.filter_value_and_grad(inner_loss_fn, has_aux=True)(
            model, controller, x_t_train, t
        )
        
        # Update Model (Differentiable Update!)
        updates_llm, new_state_llm = self.opt_llm.update(grads_llm, state_llm, model)
        new_model = eqx.apply_updates(model, updates_llm)
        
        # --- 2. OUTER LOOP (Controller Update) ---
        # Evaluate New Model on Validation Set
        # The performance of new_model depends on the beta (and thus controller) used in the update above.
        
        key, val_t_key, val_dist_key = jax.random.split(key, 3)
        x_0_val = jax.vmap(jax.vmap(new_model.embedding))(val_batch) # Use NEW model embedding
        val_noise = jax.random.normal(val_dist_key, x_0_val.shape)
        t_val = jax.random.uniform(val_t_key, (batch_size,))
        x_t_val = jax.vmap(self.sde.q_sample)(x_0_val, t_val, val_noise)
        
        def outer_loss_fn(c_params):
            # We must re-run the INNER update conceptually to get gradients w.r.t c_params?
            # JAX "magic": If 'new_model' was computed using 'controller' (which it was, via beta -> loss -> grads -> update),
            # then 'new_model' is a function of 'controller'.
            # So a loss on 'new_model' IS a loss on 'controller'.
            # HOWEVER: We need to ensure we use the 'controller' passed in arguments here, not closure.
            # But 'new_model' is ALREADY computed.
            # Wait. We need to define the ENTIRE inner update logic INSIDE this function if we want `grad` to trace it w.r.t c_params.
            # OR ensuring the graph is connected.
            
            # Use functional approach: Re-calculate new_model from scratch inside here?
            # Yes, that's proper meta-learning implementation.
            
            # 1. Re-compute inner loss & update using c_params
            # inner_loss_fn handles m_curr, c_params
            (_, (_, _, _, _)), grads_inner = eqx.filter_value_and_grad(inner_loss_fn, has_aux=True)(
                model, c_params, x_t_train, t
            )
            updates, _ = self.opt_llm.update(grads_inner, state_llm, model)
            model_proposal = eqx.apply_updates(model, updates)
            
            # 2. Main Validation Loss (Performance)
            # Just CE Loss (Task Performance)
            val_logits, _ = jax.vmap(lambda x, ti: model_proposal(inputs_embeds=x, t=ti))(x_t_val, t_val)
            val_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(val_logits, val_batch))
            
            return val_loss

        # Meta-Gradients
        meta_loss, grads_ctrl = eqx.filter_value_and_grad(outer_loss_fn)(controller)
        
        # Update Controller
        updates_ctrl, new_state_ctrl = self.opt_ctrl.update(grads_ctrl, state_ctrl, controller)
        new_controller = eqx.apply_updates(controller, updates_ctrl)
        
        # Return UPDATED model and UPDATED controller
        # We use the new model (already updated in inner loop logic, but technically we have 'new_model' variable)
        # Note: Optimization state for LLM also updated.
        
        metrics = {
            "loss_inner": loss_inner,
            "ce_inner": ce_in,
            "pc_inner": pc_in,
            "pc_loss_batch": pc_loss_batch, # Export for Episodic Memory Trigger
            "memory_keys": jnp.mean(x_0_train, axis=1), # Export Mean Global Vector for RAG
            "beta": beta_val,
            "loss_meta": meta_loss
        }
        
        return new_model, new_controller, new_state_llm, new_state_ctrl, metrics, beta_val
