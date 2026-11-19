"""Default hyperparameters shared by training code and experiment configs.

Kept free of heavy imports (no torch) so experiment configs can be imported
without loading the training stack.
"""

DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_MIN_LR_RATIO = 0.1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EVAL_CHECKPOINTS = 20
MIN_DEFAULT_EVAL_EVERY = 2_500


def default_eval_every(max_steps: int) -> int:
    """Derive the default evaluation interval targeting 20 checkpoints, floored at 2,500 steps."""
    return max(MIN_DEFAULT_EVAL_EVERY, max_steps // DEFAULT_EVAL_CHECKPOINTS)
