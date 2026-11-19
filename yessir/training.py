"""Minimal training utilities (`resolve_device`, `dtype_name`, `resolve_dtype`, `autocast_context`, `seed_everything`, `build_optimizer`, `parallel_classification_cross_entropy`, `next_token_cross_entropy`, `pack_classification_labels`, `train_step`, `train`)."""

import contextlib
import itertools
import math
import os
import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

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

_CONCRETE_DTYPES = ("float32", "bfloat16")
_VALID_DTYPES = ("auto", *_CONCRETE_DTYPES)


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


def _has_hardware_bf16(device: torch.device) -> bool:
    """Return `True` if `device` has native hardware `bfloat16` Tensor Core support.

    Passes `including_emulation=False` to `torch.cuda.is_bf16_supported` because
    PyTorch's default (`True`) counts software-emulated `bfloat16` on pre-Ampere
    GPUs such as Turing (`Tesla T4`, `SM 7.5`), which lacks `bfloat16` Tensor Cores.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if device.index is not None:
        with torch.cuda.device(device):
            return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    return bool(torch.cuda.is_bf16_supported(including_emulation=False))


def dtype_name(dtype: torch.dtype) -> str:
    """Return the canonical CLI/JSON name (`'float32'` or `'bfloat16'`) for `dtype`."""
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    choices = ", ".join(repr(d) for d in _CONCRETE_DTYPES)
    raise ValueError(f"unsupported dtype {dtype!r}; expected one of {choices}")


def resolve_dtype(
    device: str | torch.device,
    dtype: str | torch.dtype | None = None,
) -> torch.dtype:
    """Resolve the compute `torch.dtype` (`torch.bfloat16` or `torch.float32`) for `device`.

    - `'auto'` / `None`: selects `torch.bfloat16` when `device` has native hardware
      `bfloat16` support (Ampere+ CUDA GPUs such as L4, A100, H100), and
      `torch.float32` otherwise (`Tesla T4`, `mps`, `cpu`).
    - `'float32'` / `torch.float32`: forces pure `float32` (`autocast` disabled).
    - `'bfloat16'` / `torch.bfloat16`: requires hardware `bfloat16` support on
      `device` and raises `ValueError` otherwise rather than silently falling back
      to software emulation.
    """
    resolved_device = resolve_device(device)
    if dtype is None or (isinstance(dtype, str) and dtype.strip().lower() == "auto"):
        return torch.bfloat16 if _has_hardware_bf16(resolved_device) else torch.float32

    if dtype == torch.float32 or (
        isinstance(dtype, str) and dtype.strip().lower() == "float32"
    ):
        return torch.float32

    if dtype == torch.bfloat16 or (
        isinstance(dtype, str) and dtype.strip().lower() == "bfloat16"
    ):
        if not _has_hardware_bf16(resolved_device):
            raise ValueError(
                f"bfloat16 requested, but device {str(resolved_device)!r} does not have "
                f"hardware bfloat16 support; use 'auto' or 'float32'"
            )
        return torch.bfloat16

    choices = ", ".join(repr(d) for d in _VALID_DTYPES)
    raise ValueError(f"unknown dtype {dtype!r}; expected one of {choices}")


def autocast_context(
    device: str | torch.device,
    dtype: str | torch.dtype | None = None,
) -> contextlib.AbstractContextManager[Any]:
    """Return a mixed-precision `torch.autocast` context manager for `(device, dtype)`.

    Validates `(device, dtype)` via `resolve_dtype`, returning a `bfloat16`
    `torch.autocast` context when resolved to `torch.bfloat16` and a no-op
    `contextlib.nullcontext()` when resolved to `torch.float32`.
    """
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)
    if resolved_dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=resolved_device.type, dtype=torch.bfloat16)


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
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"lr must be > 0 and finite, got {lr}")
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
    """Compute the mean cross-entropy loss per head (`nats / target`) across `ParallelClassificationHead` outputs.

    When a `ParallelClassificationHead` predicts `H` conditionally independent labels
    `(y_0, ..., y_{H-1})` from a hidden state `h`, the negative joint log-likelihood is
    `sum_h CE_h`. Dividing by `H = len(logits)` normalises the scalar loss into mean
    cross-entropy per supervised target (`nats / target`), matching
    `next_token_cross_entropy` across variants while rescaling gradients by the
    constant factor `1 / H`.

    Supports both single-step classification (`logits[h]` of shape `(B, num_classes_h)`,
    `labels` of shape `(B, H)`) and per-position classification
    (`logits[h]` of shape `(B, T, num_classes_h)`, `labels` of shape `(B, T, H)` with
    `IGNORE_INDEX = -100` at masked prefix and padding positions).

    Args:
        logits: Sequence of `H` tensors, where `logits[h]` has shape
            `(B, num_classes_h)` or `(B, T, num_classes_h)`.
        labels: LongTensor of shape `(B, H)` or `(B, T, H)` containing class
            indices in `0..num_classes_h - 1` or `IGNORE_INDEX`.

    Returns:
        Scalar loss tensor (mean across the `H` heads of mean cross-entropy over non-ignored positions).
    """
    if not logits:
        raise ValueError("logits must contain at least one head")
    num_targets = len(logits)
    leading_shape = logits[0].shape[:-1]
    if len(leading_shape) not in (1, 2):
        raise ValueError(
            f"logits[0] must have 2 or 3 dimensions, got shape {tuple(logits[0].shape)}"
        )
    if labels.shape != (*leading_shape, num_targets):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match (*leading_shape, num_targets) = {(*leading_shape, num_targets)}"
        )

    # Upcast logits to float32 so log-sum-exp and head averaging do not suffer
    # 7-bit mantissa rounding when linear heads run under bfloat16 autocast.
    total_loss = torch.zeros((), dtype=torch.float32, device=logits[0].device)
    for h, head_logits in enumerate(logits):
        if head_logits.shape[:-1] != leading_shape:
            raise ValueError(
                f"logits[{h}] leading shape {tuple(head_logits.shape[:-1])} does not match "
                f"logits[0] leading shape {tuple(leading_shape)}"
            )
        total_loss = total_loss + F.cross_entropy(
            head_logits.float().reshape(-1, head_logits.size(-1)),
            labels[..., h].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    return total_loss / num_targets


def next_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute masked next-token cross-entropy loss across sequence positions.

    Unlike Hugging Face causal LMs (which shift `labels` internally), `labels` here
    are already pre-shifted so `labels[b, t]` is the target token predicted by
    `logits[b, t]`. Positions where `labels == IGNORE_INDEX` (`-100`, e.g. prompt
    prefix tokens and right-padding positions) are excluded from the mean. Logits
    are upcast to `float32` so log-sum-exp stays full precision under `bfloat16`
    autocast.

    Args:
        logits: Floating-point tensor of shape `(batch_size, seq_len, vocab_size)`.
        labels: Pre-shifted LongTensor of shape `(batch_size, seq_len)` with class
            indices in `0..vocab_size - 1` or `IGNORE_INDEX`.

    Returns:
        Scalar `float32` loss tensor averaged over all non-ignored positions in the batch.
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
        logits.float().reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def pack_classification_labels(items: Sequence[LabelledExample]) -> torch.Tensor:
    """Stack fixed-width `LabelledExample.labels` into a `(batch_size, num_targets)` LongTensor."""
    if not items:
        raise ValueError("items must not be empty")
    num_targets = len(items[0].labels)
    for idx, item in enumerate(items):
        if len(item.labels) != num_targets:
            raise ValueError(
                f"all LabelledExamples in a parallel classification batch must have {num_targets} labels; "
                f"item {idx} has {len(item.labels)}"
            )
    return torch.tensor([item.labels for item in items], dtype=torch.long)


def _train_step_with_tokens(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    items: Sequence[LabelledExample],
    *,
    resolved_dtype: torch.dtype = torch.float32,
) -> tuple[float, int]:
    """Run one optimiser step and return `(step_loss, supervised_target_count)`."""
    if not items:
        raise ValueError("items must not be empty")

    model.train()
    optimizer.zero_grad()
    device = model.device
    amp_ctx: contextlib.AbstractContextManager[Any] = (
        contextlib.nullcontext()
        if resolved_dtype == torch.float32
        else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    )

    if isinstance(model.head, SubspaceLMHead):
        batch, labels = collate_labelled(items)
        batch = batch.to(device)
        labels = labels.to(device)
        with amp_ctx:
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
        with amp_ctx:
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
        with amp_ctx:
            logits = model(
                batch.input_ids,
                batch.lengths,
                token_type_ids=batch.token_type_ids,
                position_ids=batch.position_ids,
            )
            assert isinstance(logits, tuple)
            loss = parallel_classification_cross_entropy(logits, labels)

    supervised_targets = int((labels != IGNORE_INDEX).sum().item())
    loss.backward()
    optimizer.step()
    return loss.item(), supervised_targets


def train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    items: Sequence[LabelledExample],
    *,
    dtype: str | torch.dtype | None = None,
) -> float:
    """Run one forward/backward/optimiser step on a batch of `LabelledExample`s.

    Master model weights, `AdamW` moments, and cross-entropy reductions always stay
    in `float32`; when `resolve_dtype(model.device, dtype)` selects `torch.bfloat16`
    (the `'auto'` default on `bf16`-capable CUDA GPUs), the forward pass runs under
    `torch.autocast`.

    Moves the collated `Batch` and `labels` onto `model.device` and dispatches on
    `model.head`:
    - `SubspaceLMHead`: collates per-position labels with `collate_labelled`
      and computes `next_token_cross_entropy`.
    - `ParallelClassificationHead` (per-position, when `items[0].labels[0]` is a
      per-position tuple): collates with `collate_labelled`, projects all positions
      (`lengths=None`), and computes `parallel_classification_cross_entropy`.
    - `ParallelClassificationHead` (single-step, when `items[0].labels` is a 1D tuple
      of per-head targets): collates with `collate`, packs labels with
      `pack_classification_labels`, projects at `lengths - 1`, and computes
      `parallel_classification_cross_entropy`.

    Returns:
        The scalar loss value (`float`, in `nats / target`) before the parameter update.
    """
    resolved_dtype = resolve_dtype(model.device, dtype)
    loss_val, _ = _train_step_with_tokens(
        model, optimizer, items, resolved_dtype=resolved_dtype
    )
    return loss_val


def train(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    max_steps: int,
    batch_size: int,
    lr: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    dtype: str | torch.dtype | None = None,
    on_step: Callable[[int, float], None] | None = None,
    on_step_tokens: Callable[[int, float, int], None] | None = None,
) -> list[float]:
    """Train `model` on batches drawn from `dataset` for `max_steps` optimisation steps.

    Pass a persistent iterator `stream = iter(dataset)` and an existing `optimizer`
    to resume training across multiple `train(...)` calls (e.g. between evaluations)
    without restarting the data stream or resetting Adam state. Cannot specify both
    `lr` and `optimizer`, nor both `on_step` and `on_step_tokens`.

    Returns:
        List of per-step scalar training losses of length `max_steps`.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if optimizer is not None and lr is not None:
        raise ValueError("cannot specify both lr and optimizer")
    if on_step is not None and on_step_tokens is not None:
        raise ValueError("cannot specify both on_step and on_step_tokens")

    resolved_dtype = resolve_dtype(model.device, dtype)
    if optimizer is None:
        optimizer = build_optimizer(model, lr=DEFAULT_LR if lr is None else lr)
    stream = iter(dataset)

    losses: list[float] = []
    for step_idx in range(1, max_steps + 1):
        items = list(itertools.islice(stream, batch_size))
        if len(items) < batch_size:
            raise ValueError(
                f"dataset exhausted early: needed {batch_size} examples, got {len(items)}"
            )
        step_loss, step_tokens = _train_step_with_tokens(
            model, optimizer, items, resolved_dtype=resolved_dtype
        )
        losses.append(step_loss)
        if on_step is not None:
            on_step(step_idx, step_loss)
        elif on_step_tokens is not None:
            on_step_tokens(step_idx, step_loss, step_tokens)
    return losses
