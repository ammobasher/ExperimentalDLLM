#!/usr/bin/env python3
"""Fix colab_train.ipynb to use streaming mode instead of caching."""
import json

with open('colab_train.ipynb', 'r') as f:
    nb = json.load(f)

# 1. Replace the caching cell with a streaming note
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'CACHE_DIR' in src and 'cache_data.py' in src:
        cell['source'] = [
            '# No caching needed! Using --stream mode to load data directly from HuggingFace.\n',
            '# This eliminates the multi-hour caching step entirely.\n',
            'print("\\u2705 Using streaming mode \\u2014 no data caching needed!")\n',
            'print("   Data will be tokenized on-the-fly from HuggingFace FineWeb-Edu.")'
        ]
        print(f'  Fixed cell {i}: replaced caching with streaming note')
        break

# 2. Update the caching markdown cell
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'Cache Training Data' in src and cell['cell_type'] == 'markdown':
        cell['source'] = [
            '## 4. Data Loading (Streaming)\n',
            '\n',
            'No pre-caching needed! Training streams directly from HuggingFace FineWeb-Edu.\n',
            'Data is tokenized on-the-fly, eliminating the multi-hour caching step.'
        ]
        print(f'  Fixed cell {i}: updated caching markdown')
        break

# 3. Update training command to use --stream instead of --cache_dir
for i, cell in enumerate(nb['cells']):
    new_source = []
    changed = False
    for line in cell['source']:
        if "'--cache_dir', CACHE_DIR," in line:
            line = line.replace("'--cache_dir', CACHE_DIR,", "'--stream',")
            changed = True
        new_source.append(line)
    if changed:
        print(f'  Fixed cell {i}: updated training command to use --stream')
    cell['source'] = new_source

with open('colab_train.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Done - notebook updated for streaming mode')
