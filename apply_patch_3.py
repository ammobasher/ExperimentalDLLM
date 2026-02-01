
import sys

target = "venv/lib/python3.14/site-packages/jax/_src/nn/initializers.py"

try:
    with open(target, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Target file not found: {target}")
    sys.exit(1)

# Look for line starting with "try: print(f\"JAX_DEBUG" OR "shape = core.canonicalize_shape(shape)"
start_idx = -1
for i, line in enumerate(lines):
    if "try: print(f\"JAX_DEBUG" in line:
        start_idx = i
        break
    if "shape = core.canonicalize_shape(shape)" in line and "JAX_DEBUG" not in line:
        # Original line found (if revert needed or not applied)
        start_idx = i
        break

if start_idx != -1:
    print(f"Found target at line {start_idx+1}")
    
    # We replace strict line.
    # Indentation should be 4 spaces (inside init function)
    # def init(...):
    #   shape = ...
    
    # Check indentation of previous line
    indent = "    " # assumption
    if start_idx > 0:
        prev = lines[start_idx-1]
        # count leading spaces?
        # Assuming 4 spaces based on file inspection previously.
    
    replacement = [
        "    try:\n",
        "        print(f'JAX_DEBUG: init called. key_type={type(key)}, shape_type={type(shape)}, shape={shape}')\n",
        "    except Exception:\n",
        "        pass\n",
        "    shape = core.canonicalize_shape(shape)\n"
    ]
    
    # If we matched the bad line, we replace 1 line.
    # If we matched the original line, we replace 1 line.
    new_lines = lines[:start_idx] + replacement + lines[start_idx+1:]
    
    with open(target, "w") as f:
        f.writelines(new_lines)
    print("Patched successfully")
else:
    print("Could not find line to patch")
