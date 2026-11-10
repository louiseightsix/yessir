"""Addition: can a transformer follow an in-context numeral system zero-shot?

The TransformAdder prototype: a quick, unreviewed exploration kept for reference.
See prototypes/transformadder/README.md.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
