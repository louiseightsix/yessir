"""Minimal training utilities (`resolve_device`, `dtype_name`, `resolve_dtype`, `autocast_context`, `seed_everything`, `build_optimizer`, `build_lr_scheduler`, `parallel_classification_cross_entropy`, `next_token_cross_entropy`, `pack_classification_labels`, `train_step`, `train`)."""

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
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from yessir.batch import collate, collate_labelled
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.defaults import DEFAULT_LR, DEFAULT_MIN_LR_RATIO
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
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    """Create an AdamW optimiser (`betas=(0.9, 0.95)`, `eps=1e-8`) with decoupled weight decay on 2D+ matrices only.

    Splits trainable parameters by tensor rank:
    - `param.ndim >= 2` (linear projection and classification head matrices): `weight_decay`.
    - `param.ndim < 2` (`RMSNorm` gain vectors and biases): `0.0`, since decaying unit-initialised
      multiplicative gains toward zero fights the adjacent linear projections.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"lr must be > 0 and finite, got {lr}")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0 and finite, got {weight_decay}")

    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("model has no trainable parameters")

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = DEFAULT_MIN_LR_RATIO,
) -> LambdaLR:
    """Create a linear-warmup + cosine-decay learning-rate scheduler (`LambdaLR`).

    Schedule across 1-based optimisation steps `s = 1 .. max_steps` (`step = s - 1`):
    - **Linear warmup** (`s = 1 .. effective_warmup`, where
      `effective_warmup = min(warmup_steps, max(max_steps - 1, 0))`): scales peak `lr`
      by `s / effective_warmup`, reaching `1.0 * lr` on step `effective_warmup`.
    - **Cosine decay** (`s = effective_warmup + 1 .. max_steps`, or `s = 1 .. max_steps`
      when `effective_warmup == 0`): decays smoothly from `1.0 * lr` down to
      `min_lr_ratio * lr` on step `max_steps` (clamped at `min_lr_ratio * lr` thereafter).
      Setting `warmup_steps = 0` and `min_lr_ratio = 1.0` yields a constant learning rate.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if not math.isfinite(min_lr_ratio) or not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(
            f"min_lr_ratio must be in 0.0..1.0 and finite, got {min_lr_ratio}"
        )

    effective_warmup = min(warmup_steps, max(max_steps - 1, 0))

    def _lr_lambda(step: int) -> float:
        if effective_warmup > 0 and step < effective_warmup:
            return (step + 1) / effective_warmup
        if max_steps <= 1:
            return 1.0
        if effective_warmup == 0:
            progress = min(step / (max_steps - 1), 1.0)
        else:
            progress = min(
                (step + 1 - effective_warmup) / (max_steps - effective_warmup),
                1.0,
            )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, _lr_lambda)


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


def _validate_step_controls(
    *,
    batch_size: int,
    grad_accum_steps: int,
    max_grad_norm: float | None,
) -> None:
    """Validate `grad_accum_steps` divisibility and `max_grad_norm` positivity."""
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if batch_size % grad_accum_steps != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by grad_accum_steps ({grad_accum_steps})"
        )
    if max_grad_norm is not None and (
        not math.isfinite(max_grad_norm) or max_grad_norm <= 0
    ):
        raise ValueError(
            f"max_grad_norm must be > 0 and finite (or None), got {max_grad_norm}"
        )


def _count_item_targets(items: Sequence[LabelledExample]) -> int:
    """Count non-`IGNORE_INDEX` scalar targets across `items`."""
    total = 0
    for item in items:
        if item.labels and isinstance(item.labels[0], tuple):
            for pos_tuple in item.labels:
                assert isinstance(pos_tuple, tuple)
                for val in pos_tuple:
                    if val != IGNORE_INDEX:
                        total += 1
        else:
            for val in item.labels:
                if val != IGNORE_INDEX:
                    total += 1
    return total


def _forward_micro_batch(
    model: Transformer,
    items: Sequence[LabelledExample],
    *,
    device: torch.device,
    amp_ctx: contextlib.AbstractContextManager[Any],
) -> tuple[torch.Tensor, int]:
    """Collate `items`, run a forward pass under `amp_ctx`, and return `(loss, supervised_targets)`."""
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
    return loss, supervised_targets


def _train_step_with_tokens(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    items: Sequence[LabelledExample],
    *,
    resolved_dtype: torch.dtype = torch.float32,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
) -> tuple[float, int]:
    """Run one optimiser step (with optional micro-batch accumulation and grad clipping) and return `(step_loss, supervised_target_count)`."""
    if not items:
        raise ValueError("items must not be empty")
    _validate_step_controls(
        batch_size=len(items),
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    optimizer.zero_grad()
    device = model.device
    amp_ctx: contextlib.AbstractContextManager[Any] = (
        contextlib.nullcontext()
        if resolved_dtype == torch.float32
        else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    )

    if grad_accum_steps == 1:
        loss, supervised_targets = _forward_micro_batch(
            model, items, device=device, amp_ctx=amp_ctx
        )
        loss.backward()
        step_loss = loss.item()
    else:
        micro_size = len(items) // grad_accum_steps
        total_targets = _count_item_targets(items)
        step_loss = 0.0
        for start in range(0, len(items), micro_size):
            micro_items = items[start : start + micro_size]
            micro_loss, micro_targets = _forward_micro_batch(
                model, micro_items, device=device, amp_ctx=amp_ctx
            )
            # Weight each micro-batch by its share of supervised targets so
            # variable-length sequences produce the exact same mean loss and
            # gradient as `grad_accum_steps = 1`.
            weight = (
                (micro_targets / total_targets)
                if total_targets > 0
                else (1.0 / grad_accum_steps)
            )
            if micro_targets > 0 or total_targets == 0:
                (micro_loss * weight).backward()
                step_loss += micro_loss.item() * weight
        supervised_targets = total_targets

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return step_loss, supervised_targets


def train_step(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    items: Sequence[LabelledExample],
    *,
    dtype: str | torch.dtype | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
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
        model,
        optimizer,
        items,
        resolved_dtype=resolved_dtype,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
    )
    return loss_val


def train(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    max_steps: int,
    batch_size: int,
    lr: float | None = None,
    weight_decay: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
    dtype: str | torch.dtype | None = None,
    on_step: Callable[[int, float], None] | None = None,
    on_step_tokens: Callable[[int, float, int], None] | None = None,
) -> list[float]:
    """Train `model` on batches drawn from `dataset` for `max_steps` optimisation steps.

    Pass a persistent iterator `stream = iter(dataset)` and an existing `optimizer`
    (and optional `scheduler`) to resume training across multiple `train(...)` calls
    (e.g. between evaluations) without restarting the data stream or resetting Adam /
    schedule state. `scheduler.step()` runs after `on_step` / `on_step_tokens` so
    callbacks reading `optimizer.param_groups[0]['lr']` see the exact learning rate
    used on that step.

    Returns:
        List of per-step scalar training losses of length `max_steps`.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if optimizer is not None and lr is not None:
        raise ValueError("cannot specify both lr and optimizer")
    if optimizer is not None and weight_decay is not None:
        raise ValueError("cannot specify both weight_decay and optimizer")
    if scheduler is not None and optimizer is None:
        raise ValueError("cannot specify scheduler without optimizer")
    if on_step is not None and on_step_tokens is not None:
        raise ValueError("cannot specify both on_step and on_step_tokens")
    _validate_step_controls(
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
    )

    resolved_dtype = resolve_dtype(model.device, dtype)
    if optimizer is None:
        optimizer = build_optimizer(
            model,
            lr=DEFAULT_LR if lr is None else lr,
            weight_decay=0.0 if weight_decay is None else weight_decay,
        )
    stream = iter(dataset)

    losses: list[float] = []
    for step_idx in range(1, max_steps + 1):
        items = list(itertools.islice(stream, batch_size))
        if len(items) < batch_size:
            raise ValueError(
                f"dataset exhausted early: needed {batch_size} examples, got {len(items)}"
            )
        step_loss, step_tokens = _train_step_with_tokens(
            model,
            optimizer,
            items,
            resolved_dtype=resolved_dtype,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
        )
        losses.append(step_loss)
        if on_step is not None:
            on_step(step_idx, step_loss)
        elif on_step_tokens is not None:
            on_step_tokens(step_idx, step_loss, step_tokens)
        if scheduler is not None:
            scheduler.step()
    return losses
