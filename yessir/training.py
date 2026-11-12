"""Minimal training utilities (`seed_everything`, `build_optimizer`, `multi_head_cross_entropy`, `train_step`, `train`)."""

import itertools
import os
import random
import warnings
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from yessir.batch import collate
from yessir.dataset import IterableDataset, LabelledExample
from yessir.model import Transformer


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch RNGs and enable deterministic PyTorch algorithms.

    Why each step is needed:
    - `random.seed(seed)` seeds Python's standard-library RNG (used when callers
      don't instantiate their own `random.Random(seed)`).
    - `torch.manual_seed(seed)` seeds PyTorch's CPU and device RNGs (used by
      `Transformer.reset_parameters` and any stochastic tensor ops).
    - `CUBLAS_WORKSPACE_CONFIG=":4096:8"` is required by cuBLAS for deterministic
      GEMMs on CUDA when `torch.use_deterministic_algorithms(True)` is active.
    - `warn_only=True` avoids crashing on the handful of backward kernels on
      MPS/CUDA that lack a deterministic implementation.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    warnings.filterwarnings(
        "ignore", message=".*does not have a deterministic implementation.*"
    )
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 3e-4,
) -> torch.optim.AdamW:
    """Create an AdamW optimiser with standard Transformer hyperparameters.

    Uses `betas=(0.9, 0.95)` and `eps=1e-8` (GPT-3 / LLaMA convention) and
    `weight_decay=0.0` (selective weight decay is added in Stage B).
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def multi_head_cross_entropy(
    logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of cross-entropy losses across parallel output heads.

    When a `MultiTaskHead` predicts `H` conditionally independent targets
    `(y_0, ..., y_{H-1})` from the final hidden state `h_last`, the joint
    log-probability is `sum_h log P(y_h | h_last)`. Minimising the negative
    joint log-likelihood is therefore the unweighted sum of each head's
    cross-entropy loss.

    Args:
        logits: Sequence of `H` tensors, where `logits[h]` has shape
            `(batch_size, num_classes_h)`.
        targets: LongTensor of shape `(batch_size, H)`, where `targets[b, h]`
            is the ground-truth class index for example `b` and head `h`.

    Returns:
        Scalar loss tensor (sum of mean cross-entropy over the batch for each head).
    """
    if not logits:
        raise ValueError("logits must contain at least one head")
    num_heads = len(logits)
    batch_size = logits[0].size(0)
    if targets.shape != (batch_size, num_heads):
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match (batch_size, num_heads) = ({batch_size}, {num_heads})"
        )

    total_loss = torch.zeros((), dtype=logits[0].dtype, device=logits[0].device)
    for h, head_logits in enumerate(logits):
        if head_logits.size(0) != batch_size:
            raise ValueError(
                f"logits[{h}] batch size {head_logits.size(0)} does not match logits[0] batch size {batch_size}"
            )
        total_loss = total_loss + F.cross_entropy(head_logits, targets[:, h])
    return total_loss


def _pack_multi_head_targets(items: Sequence[LabelledExample]) -> torch.Tensor:
    """Stack fixed-width `LabelledExample.labels` into a `(batch_size, num_heads)` LongTensor."""
    num_heads = len(items[0].labels)
    for idx, item in enumerate(items):
        if len(item.labels) != num_heads:
            raise ValueError(
                f"all LabelledExamples in a multi-head batch must have {num_heads} labels; "
                f"item {idx} has {len(item.labels)}"
            )
    return torch.tensor([item.labels for item in items], dtype=torch.long)


def train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    items: Sequence[LabelledExample],
) -> float:
    """Run one fp32 CPU forward/backward/optimiser step on a batch of `LabelledExample`s.

    Returns:
        The scalar loss value (`float`) for this step before the parameter update.
    """
    if not items:
        raise ValueError("items must not be empty")

    batch = collate([item.example for item in items])
    targets = _pack_multi_head_targets(items)

    optimizer.zero_grad()
    logits = model(
        batch.input_ids,
        batch.lengths,
        token_type_ids=batch.token_type_ids,
        position_ids=batch.position_ids,
    )
    loss = multi_head_cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train(
    model: Transformer,
    dataset: IterableDataset,
    *,
    steps: int,
    batch_size: int,
    lr: float = 3e-4,
) -> list[float]:
    """Train `model` on batches drawn from `dataset` for `steps` optimisation steps on CPU.

    Returns:
        List of per-step scalar training losses of length `steps`.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    model.train()
    optimizer = build_optimizer(model, lr=lr)
    stream = iter(dataset)

    losses: list[float] = []
    for _ in range(steps):
        items = list(itertools.islice(stream, batch_size))
        if len(items) < batch_size:
            raise ValueError(
                f"dataset exhausted early: needed {batch_size} examples, got {len(items)}"
            )
        losses.append(train_step(model, optimizer, items))
    return losses
