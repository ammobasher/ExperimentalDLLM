
import sys
import inspect
from collections.abc import Sequence
from typing import Any

# Mock class similar to _Conv
class MockConv:
    features: int
    kernel_size: int | Sequence[int]
    strides: Sequence[int] | int = 1

print(f"Python: {sys.version}")
try:
    anns = inspect.get_annotations(MockConv)
    print(f"MockConv annotations: {list(anns.keys())}")
except Exception as e:
    print(f"MockConv failed: {e}")

# Now try real Flax if available

try:
    from flax.linen import LayerNorm
    print("Found LayerNorm")
    print(f"LayerNorm attrs: {[x for x in dir(LayerNorm) if 'axes' in x]}")
    anns_ln = inspect.get_annotations(LayerNorm)
    print(f"LayerNorm annotations: {list(anns_ln.keys())}")
except Exception as e:
    print(f"LayerNorm failed: {e}")

