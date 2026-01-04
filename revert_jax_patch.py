
import sys

target = "venv/lib/python3.14/site-packages/jax/_src/nn/initializers.py"

try:
    with open(target, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Target file not found: {target}")
    sys.exit(1)

# Look for block starting with "try: print(f'JAX_DEBUG"
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "try: print(f'JAX_DEBUG" in line or 'try: print(f"JAX_DEBUG' in line:
        start_idx = i
    if start_idx != -1 and "shape = core.canonicalize_shape(shape)" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    print(f"Found debug block from {start_idx} to {end_idx}")
    
    # We want to keep ONLY the line at end_idx.
    # And remove lines start_idx to end_idx-1.
    
    new_lines = lines[:start_idx] + [lines[end_idx]] + lines[end_idx+1:]
    
    with open(target, "w") as f:
        f.writelines(new_lines)
    print("Reverted successfully")
else:
    print(f"Could not find debug block to revert. start={start_idx} end={end_idx}")
    # Maybe already reverted or mismatched?
    # Check if original line exists without try block?
    for line in lines:
        if "shape = core.canonicalize_shape(shape)" in line and "try:" not in line and "except" not in line:
            # print("File seems clean.")
            pass
