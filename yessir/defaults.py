"""Default hyperparameters shared by training code and experiment configs.

Kept free of heavy imports (no torch) so experiment configs can be imported
without loading the training stack.
"""

DEFAULT_LR = 3e-4
