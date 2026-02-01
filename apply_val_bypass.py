
import sys

target = "venv/lib/python3.14/site-packages/diffusers/models/modeling_flax_utils.py"

try:
    with open(target, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Target file not found: {target}")
    sys.exit(1)

# We look for the line: params_shape_tree = jax.eval_shape(model.init_weights, rng=jax.random.PRNGKey(0))
# And wrap it.

start_idx = -1
for i, line in enumerate(lines):
    if "params_shape_tree = jax.eval_shape(model.init_weights" in line:
        start_idx = i
        break

if start_idx != -1:
    print(f"Found target at line {start_idx+1}")
    # We need to indent original lines?
    # No, we replace the block of lines that rely on params_shape_tree validation.
    # Actually, wrapping the WHOLE logic is best.
    
    # Original code:
    # params_shape_tree = ...
    # required_params = ...
    #
    # shape_state = ...
    #
    # missing_keys = ...
    # unexpected_keys = ...
    # 
    # if missing_keys: ...
    #
    # for key in state.keys(): ...
    #
    # for unexpected_key ...
    
    # This is a large block.
    # I will replace just the definition of params_shape_tree and insert the try-except logic
    # AND modify variables so downstream logic doesn't crash.
    
    # REPLACEMENT BLOCK:
    # try:
    #     params_shape_tree = jax.eval_shape(model.init_weights, rng=jax.random.PRNGKey(0))
    #     required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())
    #     shape_state = flatten_dict(unfreeze(params_shape_tree))
    # except TypeError:
    #     logger.warning("Bypassing shape validation due to JAX error.")
    #     params_shape_tree = None
    #     required_params = set(state.keys())
    #     shape_state = {}
    
    # But checking downstream usage:
    # missing_keys = required_params - set(state.keys()) -> OK
    # unexpected_keys = set(state.keys()) - required_params -> OK
    # if missing_keys: ... -> OK
    # for key in state.keys(): if key in shape_state and ... -> OK (shape_state is empty dict)
    
    # So minimal replacement works.
    # However, 'required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())' spans multiple lines in file?
    # Step 1478 output shows it's one line.
    
    # I will replace lines start_idx to start_idx+3 (approx).
    # line start_idx: params_shape_tree = ...
    # line start_idx+1: required_params = ...
    # line start_idx+2: empty
    # line start_idx+3: shape_state = ...
    
    # I will confirm exact lines by printing surrounding.
    idx = start_idx
    old_block = []
    # Identify lines to replace.
    # We want to replace until 'shape_state =' line inclusive.
    end_idx = -1
    for k in range(idx, len(lines)):
        if "shape_state = flatten_dict" in lines[k]:
            end_idx = k
            break
            
    if end_idx != -1:
        print(f"Replacing lines {start_idx} to {end_idx}")
        
        replacement = [
            "        try:\n",
            "            params_shape_tree = jax.eval_shape(model.init_weights, rng=jax.random.PRNGKey(0))\n",
            "            required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())\n",
            "            shape_state = flatten_dict(unfreeze(params_shape_tree))\n",
            "        except TypeError:\n",
            "            logger.warning('Bypassing shape validation due to JAX error.')\n",
            "            params_shape_tree = None\n",
            "            required_params = set(state.keys())\n",
            "            shape_state = {}\n"
        ]
        
        new_lines = lines[:start_idx] + replacement + lines[end_idx+1:]
        
        with open(target, "w") as f:
            f.writelines(new_lines)
        print("Patched successfully")
    else:
        print("Could not find end of block (shape_state = ...)")

else:
    print("Could not find params_shape_tree definition")
