"""Minimal training utilities (`resolve_device`, `seed_everything`, `build_optimizer`, `parallel_classification_cross_entropy`, `next_token_cross_entropy`, `pack_classification_labels`, `train_step`, `train`)."""

import itertools
import os
import random
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from yessir.batch import collate, collate_labelled
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.defaults import DEFAULT_LR
from yessir.heads import SubspaceLMHead
from yessir.model import Transformer

_CONCRETE_DEVICES = ("cpu", "mps", "cuda")
_VALID_DEVICES = ("auto", *_CONCRETE_DEVICES)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a target compute device (`'auto'` / `None` -> `'cuda'` -> `'mps'` -> `'cpu'`).

    When `device` is `None` or `'auto'`, selects CUDA if available, otherwise
    macOS Metal (`mps`) if available, otherwise `'cpu'`. When a specific device is
    requested, validates that it is a supported choice (`'cpu'`, `'mps'`, `'cuda'`)
    and that its backend is available before returning.
    """
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if isinstance(device, torch.device):
        resolved = device
    else:
        try:
            resolved = torch.device(device.strip().lower())
        except RuntimeError as exc:
            choices = ", ".join(repr(d) for d in _VALID_DEVICES)
            raise ValueError(
                f"unknown device {device!r}; expected one of {choices}"
            ) from exc

    if resolved.type not in _CONCRETE_DEVICES:
        choices = ", ".join(repr(d) for d in _VALID_DEVICES)
        raise ValueError(f"unknown device {str(device)!r}; expected one of {choices}")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device requested ({str(device)!r}), but torch.cuda.is_available() is False"
        )
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            f"MPS (Metal) device requested ({str(device)!r}), but torch.backends.mps.is_available() is False"
        )
    return resolved


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


def parallel_classification_cross_entropy(
    logits: Sequence[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of cross-entropy losses across `ParallelClassificationHead` outputs.

    When a `ParallelClassificationHead` predicts `H` conditionally independent labels
    `(y_0, ..., y_{H-1})` from a hidden state `h`, the joint log-probability is
    `sum_h log P(y_h | h)`. Minimising the negative joint log-likelihood is therefore
    the unweighted sum of each head's cross-entropy loss.

    Supports both single-step classification (`logits[h]` of shape `(B, num_classes_h)`,
    `labels` of shape `(B, H)`) and sequence-level multi-position classification
    (`logits[h]` of shape `(B, T, num_classes_h)`, `labels` of shape `(B, T, H)` with
    `IGNORE_INDEX = -100` at masked prefix and padding positions).

    Args:
        logits: Sequence of `H` tensors, where `logits[h]` has shape
            `(B, num_classes_h)` or `(B, T, num_classes_h)`.
        labels: LongTensor of shape `(B, H)` or `(B, T, H)` containing class
            indices in `0..num_classes_h - 1` or `IGNORE_INDEX`.

    Returns:
        Scalar loss tensor (sum across heads of mean cross-entropy over non-ignored positions).
    """
    if not logits:
        raise ValueError("logits must contain at least one head")
    num_heads = len(logits)
    leading_shape = logits[0].shape[:-1]
    if len(leading_shape) not in (1, 2):
        raise ValueError(
            f"logits[0] must have 2 or 3 dimensions, got shape {tuple(logits[0].shape)}"
        )
    if labels.shape != (*leading_shape, num_heads):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match (*leading_shape, num_heads) = {(*leading_shape, num_heads)}"
        )

    total_loss = torch.zeros((), dtype=logits[0].dtype, device=logits[0].device)
    for h, head_logits in enumerate(logits):
        if head_logits.shape[:-1] != leading_shape:
            raise ValueError(
                f"logits[{h}] leading shape {tuple(head_logits.shape[:-1])} does not match "
                f"logits[0] leading shape {tuple(leading_shape)}"
            )
        total_loss = total_loss + F.cross_entropy(
            head_logits.reshape(-1, head_logits.size(-1)),
            labels[..., h].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    return total_loss


def next_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute masked next-token cross-entropy loss across sequence positions.

    Unlike Hugging Face causal LMs (which shift `labels` internally), `labels` here
    are already pre-shifted so `labels[b, t]` is the target token predicted by
    `logits[b, t]`. Positions where `labels == IGNORE_INDEX` (`-100`, e.g. prompt
    prefix tokens and right-padding positions) are excluded from the mean.

    Args:
        logits: FloatTensor of shape `(batch_size, seq_len, vocab_size)`.
        labels: Pre-shifted LongTensor of shape `(batch_size, seq_len)` with class
            indices in `0..vocab_size - 1` or `IGNORE_INDEX`.

    Returns:
        Scalar loss tensor averaged over all non-ignored positions in the batch.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have 3 dimensions (batch_size, seq_len, vocab_size), got shape {tuple(logits.shape)}"
        )
    batch_size, seq_len, vocab_size = logits.shape
    if labels.shape != (batch_size, seq_len):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match (batch_size, seq_len) = ({batch_size}, {seq_len})"
        )
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def pack_classification_labels(items: Sequence[LabelledExample]) -> torch.Tensor:
    """Stack fixed-width `LabelledExample.labels` into a `(batch_size, num_heads)` LongTensor."""
    if not items:
        raise ValueError("items must not be empty")
    num_heads = len(items[0].labels)
    for idx, item in enumerate(items):
        if len(item.labels) != num_heads:
            raise ValueError(
                f"all LabelledExamples in a parallel classification batch must have {num_heads} labels; "
                f"item {idx} has {len(item.labels)}"
            )
    return torch.tensor([item.labels for item in items], dtype=torch.long)


def train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    items: Sequence[LabelledExample],
) -> float:
    """Run one fp32 forward/backward/optimiser step on a batch of `LabelledExample`s.

    Moves the collated `Batch` and `labels` onto `model.device` and dispatches on
    `model.head`:
    - `SubspaceLMHead`: collates sequence-level labels with `collate_labelled`
      and computes `next_token_cross_entropy`.
    - `ParallelClassificationHead` (sequence-level, when `items[0].labels[0]` is a
      per-position tuple): collates with `collate_labelled`, projects all positions
      (`lengths=None`), and computes `parallel_classification_cross_entropy`.
    - `ParallelClassificationHead` (single-step, when `items[0].labels` is a 1D tuple
      of per-head targets): collates with `collate`, packs labels with
      `pack_classification_labels`, projects at `lengths - 1`, and computes
      `parallel_classification_cross_entropy`.

    Returns:
        The scalar loss value (`float`) for this step before the parameter update.
    """
    if not items:
        raise ValueError("items must not be empty")

    model.train()
    optimizer.zero_grad()
    device = model.device

    if isinstance(model.head, SubspaceLMHead):
        batch, labels = collate_labelled(items)
        batch = batch.to(device)
        labels = labels.to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(logits, torch.Tensor)
        loss = next_token_cross_entropy(logits, labels)
    elif isinstance(items[0].labels[0], tuple):
        batch, labels = collate_labelled(items)
        batch = batch.to(device)
        labels = labels.to(device)
        logits = model(
            batch.input_ids,
            lengths=None,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(logits, tuple)
        loss = parallel_classification_cross_entropy(logits, labels)
    else:
        batch = collate([item.example for item in items]).to(device)
        labels = pack_classification_labels(items).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(logits, tuple)
        loss = parallel_classification_cross_entropy(logits, labels)

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
    """Train `model` on batches drawn from `dataset` for `steps` optimisation steps.

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
