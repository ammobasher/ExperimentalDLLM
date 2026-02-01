
import numpy as np
import os

path = "cached_data/text_cache_chunk_0000.npz"
if os.path.exists(path):
    data = np.load(path)
    if 'batches' in data:
        print(f"Shape of {path}: {data['batches'].shape}")
    else:
        print(f"Key 'batches' not found in {path}. Keys: {list(data.keys())}")
else:
    print(f"File {path} not found")
