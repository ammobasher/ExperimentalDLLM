
target = "venv/lib/python3.14/site-packages/jax/_src/nn/initializers.py"
try:
    with open(target, "r") as f:
        lines = f.readlines()
    
    # We know line 339 (0-indexed 338) has the debug print.
    # The block is try: ... except: pass
    # It likely spans 4 lines (338, 339, 340, 341).
    # And line 342 is the original code? 
    # Or did I duplicate it? "shape = ..."
    
    # Let's just filter out the lines we added.
    new_lines = []
    for line in lines:
        if "JAX_DEBUG" in line: continue
        if "try:" in line and "JAX_DEBUG" in lines[lines.index(line)+1]: continue
        if "except Exception:" in line: continue
        if "pass" in line and "except" in lines[lines.index(line)-1]: continue
        # This naive filter is risky if multiple try/except blocks exist.
        new_lines.append(line)
        
    # Better: Identify the block precisely.
    idx = -1
    for i, line in enumerate(lines):
        if "JAX_DEBUG" in line:
            idx = i
            break
            
    if idx != -1:
        # lines[idx] is the print.
        # lines[idx-1] is try:
        # lines[idx+1] is except:
        # lines[idx+2] is pass:
        # lines[idx+3] is "shape = ..." (original, kept)
        
        # We want to remove idx-1, idx, idx+1, idx+2.
        print(f"Removing lines {idx-1} to {idx+2}")
        cleaned = lines[:idx-1] + lines[idx+3:]
        
        with open(target, "w") as f:
            f.writelines(cleaned)
        print("Cleaned JAX.")
    else:
        print("Debug line not found.")

except Exception as e:
    print(e)
