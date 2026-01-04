
import sys
import os

target = "venv/lib/python3.14/site-packages/flax/linen/kw_only_dataclasses.py"
patch = open("flax_patch_code.py").read()

try:
    with open(target, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Target file not found: {target}")
    sys.exit(1)

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if 'if cls.__name__ == "_Conv":' in line:
        start_idx = i
    if 'transformed_cls: type[M] =' in line:
        if start_idx != -1: # Found start, looking for end
             end_idx = i
             break

if start_idx != -1 and end_idx != -1:
    print(f"Replacing lines {start_idx} to {end_idx}")
    # Patch includes transformed_cls line, so we need to ensure we replace correctly.
    # Lines[end_idx] is 'transformed_cls: ...'
    # We replace lines[start_idx:end_idx+1] with patch.
    # But wait, patch string usually ends with newline?
    # Let's ensure patch lines are list.
    
    new_lines = lines[:start_idx] + [patch + "\n"] + lines[end_idx+1:]
    
    with open(target, "w") as f:
        f.writelines(new_lines)
    print("Patched successfully")
else:
    print(f"Could not find patch target block. Start: {start_idx}, End: {end_idx}")
    sys.exit(1)
