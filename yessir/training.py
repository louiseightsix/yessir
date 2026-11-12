"""Minimal training utilities (`seed_everything`, `build_optimizer`, `multitask_cross_entropy`, `pack_multitask_labels`, `train_step`, `train`)."""

import itertools
import os
import random
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from yessir.batch import collate
from yessir.dataset import LabelledExample
from yessir.defaults import DEFAULT_LR
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
    - `warn_only=True` warns rather than crashing on the handful of backward
      kernels on MPS/CUDA that lack a deterministic implementation.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = DEFAULT_LR,
) -> torch.optim.AdamW:
    """Create an AdamW optimiser with `betas=(0.9, 0.95)`, `eps=1e-8` (GPT-3), and `weight_decay=0.0`."""
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def multitask_cross_entropy(
    logits: Sequence[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of cross-entropy losses across `MultiTaskHead` outputs.

    When a `MultiTaskHead` predicts `H` conditionally independent labels
    `(y_0, ..., y_{H-1})` from the final hidden state `h_last`, the joint
    log-probability is `sum_h log P(y_h | h_last)`. Minimising the negative
    joint log-likelihood is therefore the unweighted sum of each head's
    cross-entropy loss.

    Args:
        logits: Sequence of `H` tensors, where `logits[h]` has shape
            `(batch_size, num_classes_h)`.
        labels: LongTensor of shape `(batch_size, H)`, where `labels[b, h]`
            is the ground-truth class index for example `b` and head `h`.

    Returns:
        Scalar loss tensor (sum of mean cross-entropy over the batch for each head).
    """
    if not logits:
        raise ValueError("logits must contain at least one head")
    num_heads = len(logits)
    batch_size = logits[0].size(0)
    if labels.shape != (batch_size, num_heads):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match (batch_size, num_heads) = ({batch_size}, {num_heads})"
        )

    total_loss = torch.zeros((), dtype=logits[0].dtype, device=logits[0].device)
    for h, head_logits in enumerate(logits):
        if head_logits.size(0) != batch_size:
            raise ValueError(
                f"logits[{h}] batch size {head_logits.size(0)} does not match logits[0] batch size {batch_size}"
            )
        total_loss = total_loss + F.cross_entropy(head_logits, labels[:, h])
    return total_loss


def pack_multitask_labels(items: Sequence[LabelledExample]) -> torch.Tensor:
    """Stack fixed-width `LabelledExample.labels` into a `(batch_size, num_heads)` LongTensor."""
    if not items:
        raise ValueError("items must not be empty")
    num_heads = len(items[0].labels)
    for idx, item in enumerate(items):
        if len(item.labels) != num_heads:
            raise ValueError(
                f"all LabelledExamples in a multitask batch must have {num_heads} labels; "
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

    model.train()
    batch = collate([item.example for item in items])
    labels = pack_multitask_labels(items)

    optimizer.zero_grad()
    logits = model(
        batch.input_ids,
        batch.lengths,
        token_type_ids=batch.token_type_ids,
        position_ids=batch.position_ids,
    )
    loss = multitask_cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    steps: int,
    batch_size: int,
    lr: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Train `model` on batches drawn from `dataset` for `steps` optimisation steps on CPU.

    Pass a persistent iterator `stream = iter(dataset)` and an existing `optimizer`
    to resume training across multiple `train(...)` calls (e.g. between evaluations)
    without restarting the data stream or resetting Adam state. Cannot specify both
    `lr` and `optimizer`.

    Returns:
        List of per-step scalar training losses of length `steps`.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if optimizer is not None and lr is not None:
        raise ValueError("cannot specify both lr and optimizer")

    if optimizer is None:
        optimizer = build_optimizer(model, lr=DEFAULT_LR if lr is None else lr)
    stream = iter(dataset)

    losses: list[float] = []
    for step_idx in range(1, steps + 1):
        items = list(itertools.islice(stream, batch_size))
        if len(items) < batch_size:
            raise ValueError(
                f"dataset exhausted early: needed {batch_size} examples, got {len(items)}"
            )
        step_loss = train_step(model, optimizer, items)
        losses.append(step_loss)
        if on_step is not None:
            on_step(step_idx, step_loss)
    return losses
