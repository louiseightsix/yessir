"""Training utilities: seeding, AdamW parameter groups, and the warm-up + cosine
learning rate schedule. (Defaults: spec/06_implementation.md Section 5.)
"""

import math
import os
import random
import warnings

import torch


def seed_everything(seed: int):
    """Seeds Python and PyTorch and selects deterministic kernels where they exist.

    Gives run-to-run reproducibility on a fixed hardware/software stack; bitwise identity across
    GPU models, drivers, or PyTorch versions is not guaranteed. warn_only=True because a few CUDA
    kernels (e.g. some SDPA backward paths) and MPS operations have no deterministic implementation.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(model: torch.nn.Module, lr: float = 3e-4, weight_decay: float = 0.1):
    """AdamW with weight decay on weight matrices only (0 on norm gains and role embeddings)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "role_embed" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 1000, final_ratio: float = 0.1):
    """Linear warm-up, then cosine decay to final_ratio of the peak learning rate."""
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
