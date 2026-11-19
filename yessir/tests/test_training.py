"""Tests for `yessir/training.py`."""

import contextlib
import itertools
import math
import random
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F

from yessir.batch import collate
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.training import (
    autocast_context,
    build_lr_scheduler,
    build_optimizer,
    dtype_name,
    next_token_cross_entropy,
    pack_classification_labels,
    parallel_classification_cross_entropy,
    resolve_device,
    resolve_dtype,
    seed_everything,
    train,
    train_step,
)
from yessir.vocab import VOCAB_SIZE


class _SyntheticDataset:
    """Deterministic toy dataset mapping a single input token `t` to `(t + 1, t + 2, 0)`."""

    def __init__(self) -> None:
        self.yielded = 0

    def __iter__(self) -> Iterator[LabelledExample]:
        self.yielded = 0
        for step in itertools.count():
            t = step % 4
            self.yielded += 1
            yield LabelledExample(
                example=Example(
                    input_ids=(1, 10 + t, 2),
                    token_type_ids=(0, 1, 3),
                    position_ids=(0, 0, 0),
                ),
                labels=(10 + ((t + 1) % 4), 10 + ((t + 2) % 4), 0),
            )


def test_seed_everything_reproducibility():
    seed_everything(123)
    py_a = [random.randint(0, 1000) for _ in range(5)]
    torch_a = torch.randn(4, 4)

    seed_everything(123)
    py_b = [random.randint(0, 1000) for _ in range(5)]
    torch_b = torch.randn(4, 4)

    assert py_a == py_b
    assert torch.equal(torch_a, torch_b)


def test_build_optimizer_hyperparameters_and_validation():
    model = torch.nn.Linear(4, 2)
    opt = build_optimizer(model, lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    group = opt.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.0

    # Decoupled weight decay applies to 2D+ matrices only, leaving 1D RMSNorm gains at 0.0.
    tx = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    opt_wd = build_optimizer(tx, lr=3e-4, weight_decay=0.1)
    assert len(opt_wd.param_groups) == 2
    decay_group, no_decay_group = opt_wd.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim < 2 for p in no_decay_group["params"])
    assert len(decay_group["params"]) + len(no_decay_group["params"]) == sum(
        1 for p in tx.parameters() if p.requires_grad
    )

    for bad_lr in (0.0, -1e-4, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="lr must be > 0"):
            build_optimizer(model, lr=bad_lr)
    for bad_wd in (-0.01, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            build_optimizer(model, weight_decay=bad_wd)

    frozen = torch.nn.Linear(2, 2, bias=False)
    frozen.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(frozen)


def test_parallel_classification_cross_entropy_matches_manual_mean():
    torch.manual_seed(0)
    logits_0 = torch.randn(3, 10, requires_grad=True)
    logits_1 = torch.randn(3, 10, requires_grad=True)
    logits_2 = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, 0], [5, 0, 3], [9, 8, 1]], dtype=torch.long)

    loss = parallel_classification_cross_entropy((logits_0, logits_1, logits_2), labels)
    expected = (
        F.cross_entropy(logits_0, labels[:, 0])
        + F.cross_entropy(logits_1, labels[:, 1])
        + F.cross_entropy(logits_2, labels[:, 2])
    ) / 3
    assert torch.allclose(loss, expected)

    loss.backward()
    assert logits_0.grad is not None
    assert logits_1.grad is not None
    assert logits_2.grad is not None

    # Also supports 3D (B, T, num_classes) per-position logits with IGNORE_INDEX
    seq_logits_0 = torch.randn(2, 3, 8, requires_grad=True)
    seq_logits_1 = torch.randn(2, 3, 4, requires_grad=True)
    seq_labels = torch.tensor(
        [
            [[IGNORE_INDEX, IGNORE_INDEX], [2, 1], [IGNORE_INDEX, IGNORE_INDEX]],
            [[IGNORE_INDEX, IGNORE_INDEX], [5, 0], [7, 3]],
        ],
        dtype=torch.long,
    )
    seq_loss = parallel_classification_cross_entropy(
        (seq_logits_0, seq_logits_1), seq_labels
    )
    expected_seq = (
        F.cross_entropy(
            torch.stack([seq_logits_0[0, 1], seq_logits_0[1, 1], seq_logits_0[1, 2]]),
            torch.tensor([2, 5, 7]),
        )
        + F.cross_entropy(
            torch.stack([seq_logits_1[0, 1], seq_logits_1[1, 1], seq_logits_1[1, 2]]),
            torch.tensor([1, 0, 3]),
        )
    ) / 2
    assert torch.allclose(seq_loss, expected_seq)


def test_parallel_classification_cross_entropy_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="at least one head"):
        parallel_classification_cross_entropy((), torch.zeros(2, 0, dtype=torch.long))

    logits = (torch.randn(2, 8), torch.randn(2, 8))
    with pytest.raises(ValueError, match="labels shape"):
        parallel_classification_cross_entropy(
            logits, torch.zeros(2, 3, dtype=torch.long)
        )

    mismatched_logits = (torch.randn(2, 8), torch.randn(3, 8))
    with pytest.raises(ValueError, match="leading shape"):
        parallel_classification_cross_entropy(
            mismatched_logits, torch.zeros(2, 2, dtype=torch.long)
        )

    with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
        parallel_classification_cross_entropy(
            (torch.randn(8),), torch.zeros(1, dtype=torch.long)
        )


def test_pack_classification_labels_shapes_and_validation():
    ex = Example(input_ids=(1, 2), token_type_ids=(0, 3), position_ids=(0, 0))
    items = [
        LabelledExample(example=ex, labels=(10, 20, 30)),
        LabelledExample(example=ex, labels=(40, 50, 60)),
    ]
    packed = pack_classification_labels(items)
    assert packed.dtype == torch.long
    assert torch.equal(packed, torch.tensor([[10, 20, 30], [40, 50, 60]]))

    with pytest.raises(ValueError, match="items must not be empty"):
        pack_classification_labels([])

    with pytest.raises(
        ValueError,
        match="all LabelledExamples in a parallel classification batch",
    ):
        pack_classification_labels(
            [
                LabelledExample(example=ex, labels=(1, 2, 3)),
                LabelledExample(example=ex, labels=(1, 2)),
            ]
        )


def test_train_step_sets_train_mode_returns_pre_update_loss_and_updates_weights():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    model.eval()
    opt = build_optimizer(model, lr=1e-2)
    items = list(itertools.islice(_SyntheticDataset(), 4))

    batch = collate([item.example for item in items])
    labels = torch.tensor([item.labels for item in items], dtype=torch.long)
    with torch.no_grad():
        pre_logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(pre_logits, tuple)
        expected_pre_loss = parallel_classification_cross_entropy(
            pre_logits, labels
        ).item()
    params_before = [p.detach().clone() for p in model.parameters()]

    step_loss = train_step(model, opt, items)

    assert model.training is True
    assert step_loss == pytest.approx(expected_pre_loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, model.parameters(), strict=True)
    )


def test_train_step_validation():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    opt = build_optimizer(model)

    with pytest.raises(ValueError, match="items must not be empty"):
        train_step(model, opt, [])

    ex = Example(input_ids=(1, 2), token_type_ids=(0, 3), position_ids=(0, 0))
    items_mismatched = [
        LabelledExample(example=ex, labels=(1, 2, 3)),
        LabelledExample(example=ex, labels=(1, 2)),
    ]
    with pytest.raises(
        ValueError,
        match="all LabelledExamples in a parallel classification batch",
    ):
        train_step(model, opt, items_mismatched)


def test_train_validation_and_early_exhaustion():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = _SyntheticDataset()

    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        train(model, dataset, max_steps=0, batch_size=4)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        train(model, dataset, max_steps=1, batch_size=0)
    with pytest.raises(ValueError, match="cannot specify both lr and optimizer"):
        train(
            model,
            dataset,
            max_steps=1,
            batch_size=4,
            lr=1e-3,
            optimizer=build_optimizer(model),
        )
    with pytest.raises(
        ValueError, match="cannot specify both on_step and on_step_tokens"
    ):
        train(
            model,
            dataset,
            max_steps=1,
            batch_size=4,
            on_step=lambda _s, _l: None,
            on_step_tokens=lambda _s, _l, _t: None,
        )

    ex = Example(input_ids=(1, 2), token_type_ids=(0, 3), position_ids=(0, 0))
    finite_dataset = [LabelledExample(example=ex, labels=(1, 2, 3))]
    with pytest.raises(ValueError, match="dataset exhausted early"):
        train(model, finite_dataset, max_steps=1, batch_size=2)


def test_train_reduces_loss_and_supports_resuming_stream_and_optimizer():
    seed_everything(42)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=2,
        n_heads=4,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )
    dataset = _SyntheticDataset()
    stream = iter(dataset)
    opt = build_optimizer(model, lr=1e-3)

    recorded_steps: list[tuple[int, float]] = []
    losses_part1 = train(
        model,
        stream,
        max_steps=5,
        batch_size=8,
        optimizer=opt,
        on_step=lambda s, loss: recorded_steps.append((s, loss)),
    )
    recorded_tokens: list[int] = []
    losses_part2 = train(
        model,
        stream,
        max_steps=5,
        batch_size=8,
        optimizer=opt,
        on_step_tokens=lambda _s, _loss, n_tok: recorded_tokens.append(n_tok),
    )
    losses = losses_part1 + losses_part2

    assert [s for s, _ in recorded_steps] == [1, 2, 3, 4, 5]
    assert [loss for _, loss in recorded_steps] == losses_part1
    assert recorded_tokens == [8 * 3] * 5

    assert len(losses) == 10
    # Initial random-guess cross-entropy per target over vocab 256 is ~ ln(256) ≈ 5.55;
    # after 10 steps loss should drop substantially.
    assert losses[-1] < losses[0] - 1.3

    first_param = next(model.parameters())
    assert int(opt.state[first_param]["step"].item()) == 10
    assert dataset.yielded == 80


def test_next_token_cross_entropy_and_subspace_lm_head_train_step():
    seed_everything(0)
    logits = torch.randn(2, 4, 16, requires_grad=True)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 3, 5, IGNORE_INDEX],
            [IGNORE_INDEX, 1, 2, 7],
        ],
        dtype=torch.long,
    )
    loss = next_token_cross_entropy(logits, labels)
    valid_logits = torch.stack(
        [logits[0, 1], logits[0, 2], logits[1, 1], logits[1, 2], logits[1, 3]]
    )
    valid_targets = torch.tensor([3, 5, 1, 2, 7], dtype=torch.long)
    assert torch.allclose(loss, F.cross_entropy(valid_logits, valid_targets))

    with pytest.raises(ValueError, match="logits must have 3 dimensions"):
        next_token_cross_entropy(torch.randn(2, 16), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="labels shape"):
        next_token_cross_entropy(logits, torch.zeros(2, 3, dtype=torch.long))

    # Verify train_step dispatches to next_token_cross_entropy for SubspaceLMHead
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    opt = build_optimizer(model, lr=1e-2)
    items = [
        LabelledExample(
            example=Example(
                input_ids=(1, 10, 11),
                token_type_ids=(0, 2, 3),
                position_ids=(0, 0, 0),
            ),
            labels=(IGNORE_INDEX, 11, 12),
        ),
        LabelledExample(
            example=Example(
                input_ids=(1, 10, 11, 12),
                token_type_ids=(0, 2, 3, 3),
                position_ids=(0, 0, 0, 1),
            ),
            labels=(IGNORE_INDEX, 11, 12, 13),
        ),
    ]
    loss1 = train_step(model, opt, items)
    loss2 = train_step(model, opt, items)
    assert loss2 < loss1


def test_resolve_device_auto_and_explicit_validation(monkeypatch: pytest.MonkeyPatch):
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device() == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA device requested"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="MPS \\(Metal\\) device requested"):
        resolve_device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device() == torch.device("mps")
    assert resolve_device("auto") == torch.device("mps")
    assert resolve_device("mps") == torch.device("mps")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == torch.device("cuda")
    assert resolve_device("auto") == torch.device("cuda")
    assert resolve_device("cuda:0") == torch.device("cuda:0")

    with pytest.raises(
        ValueError,
        match="unknown device 'gpu'; expected one of 'auto', 'cpu', 'mps', 'cuda'",
    ):
        resolve_device("gpu")
    with pytest.raises(
        ValueError,
        match="unknown device 'meta'; expected one of 'auto', 'cpu', 'mps', 'cuda'",
    ):
        resolve_device("meta")


def test_dtype_name_and_resolve_dtype_hardware_gating(monkeypatch: pytest.MonkeyPatch):
    assert dtype_name(torch.float32) == "float32"
    assert dtype_name(torch.bfloat16) == "bfloat16"
    with pytest.raises(ValueError, match="unsupported dtype"):
        dtype_name(torch.float16)

    # CPU: auto -> float32; explicit bfloat16 rejected
    assert resolve_dtype("cpu") == torch.float32
    assert resolve_dtype("cpu", "auto") == torch.float32
    assert resolve_dtype("cpu", "float32") == torch.float32
    assert resolve_dtype(torch.device("cpu"), torch.float32) == torch.float32
    with pytest.raises(ValueError, match="does not have hardware bfloat16 support"):
        resolve_dtype("cpu", "bfloat16")
    with pytest.raises(ValueError, match="does not have hardware bfloat16 support"):
        resolve_dtype("cpu", torch.bfloat16)

    # MPS: auto -> float32; explicit bfloat16 rejected
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_dtype("mps", "auto") == torch.float32
    with pytest.raises(ValueError, match="does not have hardware bfloat16 support"):
        resolve_dtype("mps", "bfloat16")

    # Pre-Ampere CUDA (e.g. Tesla T4): software emulation exists (including_emulation=True),
    # but hardware bfloat16 Tensor Cores do not (including_emulation=False).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: bool(including_emulation),
    )
    assert resolve_dtype("cuda") == torch.float32
    assert resolve_dtype("cuda", "auto") == torch.float32
    assert resolve_dtype("cuda", "float32") == torch.float32
    with pytest.raises(ValueError, match="does not have hardware bfloat16 support"):
        resolve_dtype("cuda", "bfloat16")

    # Ampere+ CUDA (e.g. L4 / A100 / H100): hardware bfloat16 supported
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: True,
    )
    assert resolve_dtype("cuda") == torch.bfloat16
    assert resolve_dtype("cuda", "auto") == torch.bfloat16
    assert resolve_dtype("cuda", " BFLOAT16 ") == torch.bfloat16
    assert resolve_dtype("cuda", torch.bfloat16) == torch.bfloat16
    assert resolve_dtype("cuda", "float32") == torch.float32
    assert resolve_dtype("cuda", torch.float32) == torch.float32

    # Unknown dtypes rejected
    for bad_dtype in ("float16", "fp8", torch.float16):
        with pytest.raises(
            ValueError,
            match="unknown dtype .*; expected one of 'auto', 'float32', 'bfloat16'",
        ):
            resolve_dtype("cpu", bad_dtype)


def test_autocast_context_returns_nullcontext_or_bfloat16_autocast(
    monkeypatch: pytest.MonkeyPatch,
):
    assert isinstance(autocast_context("cpu", "auto"), contextlib.nullcontext)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: True,
    )
    assert isinstance(autocast_context("cuda", "float32"), contextlib.nullcontext)
    ctx = autocast_context("cuda", "auto")
    assert isinstance(ctx, torch.autocast)
    assert ctx.device == "cuda"
    assert ctx.fast_dtype == torch.bfloat16


def test_cross_entropy_losses_upcast_bfloat16_logits_to_float32():
    torch.manual_seed(0)
    bf16_lm_logits = torch.randn(2, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    lm_labels = torch.tensor(
        [[IGNORE_INDEX, 3, 5, IGNORE_INDEX], [IGNORE_INDEX, 1, 2, 7]],
        dtype=torch.long,
    )
    lm_loss = next_token_cross_entropy(bf16_lm_logits, lm_labels)
    assert lm_loss.dtype == torch.float32
    expected_lm = F.cross_entropy(
        bf16_lm_logits.detach().float().reshape(-1, 16),
        lm_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    assert torch.allclose(lm_loss, expected_lm)
    lm_loss.backward()
    assert bf16_lm_logits.grad is not None
    assert bf16_lm_logits.grad.dtype == torch.bfloat16

    bf16_cls_0 = torch.randn(3, 10, dtype=torch.bfloat16, requires_grad=True)
    bf16_cls_1 = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    cls_labels = torch.tensor([[1, 0], [5, 3], [9, 1]], dtype=torch.long)
    cls_loss = parallel_classification_cross_entropy(
        (bf16_cls_0, bf16_cls_1), cls_labels
    )
    assert cls_loss.dtype == torch.float32
    expected_cls = (
        F.cross_entropy(bf16_cls_0.detach().float(), cls_labels[:, 0])
        + F.cross_entropy(bf16_cls_1.detach().float(), cls_labels[:, 1])
    ) / 2
    assert torch.allclose(cls_loss, expected_cls)
    cls_loss.backward()
    assert bf16_cls_0.grad is not None
    assert bf16_cls_1.grad is not None


def test_train_and_train_step_validate_dtype_before_consuming_data():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = _SyntheticDataset()
    items = list(itertools.islice(_SyntheticDataset(), 4))
    opt = build_optimizer(model)

    with pytest.raises(ValueError, match="does not have hardware bfloat16 support"):
        train_step(model, opt, items, dtype="bfloat16")

    with pytest.raises(ValueError, match="does not have hardware bfloat16 support"):
        train(model, dataset, max_steps=2, batch_size=4, dtype="bfloat16")
    assert dataset.yielded == 0

    losses = train(model, dataset, max_steps=2, batch_size=4, dtype="auto")
    assert len(losses) == 2
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_build_lr_scheduler_warmup_cosine_and_edge_cases():
    def _collect_lrs(
        *,
        base_lr: float,
        max_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
        run_steps: int | None = None,
    ) -> list[float]:
        layer = torch.nn.Linear(2, 2, bias=False)
        opt = build_optimizer(layer, lr=base_lr)
        sched = build_lr_scheduler(
            opt,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        lrs: list[float] = []
        n = max_steps if run_steps is None else run_steps
        for _ in range(n):
            opt.zero_grad()
            (layer.weight.sum()).backward()
            lrs.append(float(opt.param_groups[0]["lr"]))
            opt.step()
            sched.step()
        return lrs

    # 4-step linear warmup to peak 1.0, then 6-step cosine decay down to 0.1 on step 10
    lrs = _collect_lrs(
        base_lr=1.0,
        max_steps=10,
        warmup_steps=4,
        min_lr_ratio=0.1,
        run_steps=12,
    )
    assert lrs[:4] == pytest.approx([0.25, 0.50, 0.75, 1.00])
    assert lrs[4] < 1.0  # No duplicate peak step after warmup
    assert all(lrs[i] > lrs[i + 1] for i in range(3, 9))
    assert lrs[9] == pytest.approx(0.1)  # Step 10 hits exact min_lr_ratio
    assert lrs[10:] == pytest.approx([0.1, 0.1])  # Clamped past max_steps

    # Zero warmup starts at 1.0 on step 1 and decays to min_lr_ratio on step max_steps
    no_warmup = _collect_lrs(base_lr=1.0, max_steps=5, warmup_steps=0, min_lr_ratio=0.2)
    assert no_warmup[0] == pytest.approx(1.0)
    assert no_warmup[-1] == pytest.approx(0.2)

    # Constant LR recovery: warmup_steps=0, min_lr_ratio=1.0
    const_lrs = _collect_lrs(
        base_lr=3e-4, max_steps=6, warmup_steps=0, min_lr_ratio=1.0
    )
    assert const_lrs == pytest.approx([3e-4] * 6)

    # Short smoke runs where warmup_steps >= max_steps clamp cleanly
    assert _collect_lrs(
        base_lr=1.0, max_steps=1, warmup_steps=1000, min_lr_ratio=0.1
    ) == pytest.approx([1.0])
    assert _collect_lrs(
        base_lr=1.0, max_steps=2, warmup_steps=1000, min_lr_ratio=0.1
    ) == pytest.approx([1.0, 0.1])

    opt = build_optimizer(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        build_lr_scheduler(opt, max_steps=0)
    with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
        build_lr_scheduler(opt, max_steps=10, warmup_steps=-1)
    for bad_ratio in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="min_lr_ratio must be in 0.0..1.0"):
            build_lr_scheduler(opt, max_steps=10, min_lr_ratio=bad_ratio)


def test_grad_accum_steps_matches_full_batch_across_variable_length_and_cls_modes():
    # 1. SubspaceLMHead with unequal sequence lengths and unequal target counts (1, 4, 2, 5)
    lm_items = [
        LabelledExample(
            example=Example(input_ids=(1, 10, 11)),
            labels=(IGNORE_INDEX, IGNORE_INDEX, 12),
        ),
        LabelledExample(
            example=Example(input_ids=(1, 10, 11, 12, 13, 14)),
            labels=(IGNORE_INDEX, IGNORE_INDEX, 12, 13, 14, 15),
        ),
        LabelledExample(
            example=Example(input_ids=(1, 20, 21, 22)),
            labels=(IGNORE_INDEX, IGNORE_INDEX, 22, 23),
        ),
        LabelledExample(
            example=Example(input_ids=(1, 30, 31, 32, 33, 34, 35)),
            labels=(IGNORE_INDEX, IGNORE_INDEX, 32, 33, 34, 35, 36),
        ),
    ]
    seed_everything(7)
    lm_full = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        use_token_type=False,
        d_position=0,
        d_scratch=0,
    )
    seed_everything(7)
    lm_accum = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        use_token_type=False,
        d_position=0,
        d_scratch=0,
    )
    # Use SGD (linear in .grad) so unused byte coordinates with ~1e-8 BLAS rounding
    # are not magnified into +-lr steps by Adam's sign-like g / (|g| + 1e-8) on step 1.
    opt_full = torch.optim.SGD(lm_full.parameters(), lr=1e-1)
    opt_accum = torch.optim.SGD(lm_accum.parameters(), lr=1e-1)

    loss_full = train_step(
        lm_full, opt_full, lm_items, grad_accum_steps=1, max_grad_norm=0.5
    )
    toks_seen: list[int] = []
    losses_accum = train(
        lm_accum,
        lm_items,
        max_steps=1,
        batch_size=4,
        optimizer=opt_accum,
        grad_accum_steps=2,
        max_grad_norm=0.5,
        on_step_tokens=lambda _s, _l, n: toks_seen.append(n),
    )
    assert losses_accum[0] == pytest.approx(loss_full, abs=1e-6)
    assert toks_seen == [12]
    # Also verify global gradient norm was clipped to max_grad_norm = 0.5
    total_grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(p.grad)
                for p in lm_accum.parameters()
                if p.grad is not None
            ]
        )
    ).item()
    assert total_grad_norm == pytest.approx(0.5, rel=1e-5)
    for p_f, p_a in zip(lm_full.parameters(), lm_accum.parameters(), strict=True):
        assert torch.allclose(p_f, p_a, atol=1e-6)

    # 2. Per-position ParallelClassificationHead with unequal supervised counts
    per_pos_items = [
        LabelledExample(
            example=Example(
                input_ids=(1, 10, 2),
                token_type_ids=(0, 1, 3),
                position_ids=(0, 0, 0),
            ),
            labels=(
                (IGNORE_INDEX, IGNORE_INDEX),
                (IGNORE_INDEX, IGNORE_INDEX),
                (11, 4),
            ),
        ),
        LabelledExample(
            example=Example(
                input_ids=(1, 10, 2, 11, 12),
                token_type_ids=(0, 1, 3, 3, 3),
                position_ids=(0, 0, 0, 1, 2),
            ),
            labels=(
                (IGNORE_INDEX, IGNORE_INDEX),
                (IGNORE_INDEX, IGNORE_INDEX),
                (11, 3),
                (12, 3),
                (13, 4),
            ),
        ),
    ]
    seed_everything(9)
    cls_full = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    seed_everything(9)
    cls_accum = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    loss_cls_f = train_step(
        cls_full,
        torch.optim.SGD(cls_full.parameters(), lr=1e-1),
        per_pos_items,
        grad_accum_steps=1,
    )
    loss_cls_a = train_step(
        cls_accum,
        torch.optim.SGD(cls_accum.parameters(), lr=1e-1),
        per_pos_items,
        grad_accum_steps=2,
    )
    assert loss_cls_a == pytest.approx(loss_cls_f, abs=1e-6)
    for p_f, p_a in zip(cls_full.parameters(), cls_accum.parameters(), strict=True):
        assert torch.allclose(p_f, p_a, atol=1e-6)


def test_chunked_train_with_scheduler_and_validation_errors():
    # Chunked train(..., optimizer=opt, scheduler=sched) matches unchunked run
    seed_everything(11)
    m_single = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    opt_single = build_optimizer(m_single, lr=1e-3, weight_decay=0.1)
    sched_single = build_lr_scheduler(
        opt_single, max_steps=6, warmup_steps=2, min_lr_ratio=0.1
    )
    lrs_single: list[float] = []
    losses_single = train(
        m_single,
        iter(_SyntheticDataset()),
        max_steps=6,
        batch_size=4,
        optimizer=opt_single,
        scheduler=sched_single,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        on_step=lambda _s, _l: lrs_single.append(
            float(opt_single.param_groups[0]["lr"])
        ),
    )

    seed_everything(11)
    m_chunked = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    opt_chunked = build_optimizer(m_chunked, lr=1e-3, weight_decay=0.1)
    sched_chunked = build_lr_scheduler(
        opt_chunked, max_steps=6, warmup_steps=2, min_lr_ratio=0.1
    )
    stream = iter(_SyntheticDataset())
    lrs_chunked: list[float] = []
    losses_c1 = train(
        m_chunked,
        stream,
        max_steps=3,
        batch_size=4,
        optimizer=opt_chunked,
        scheduler=sched_chunked,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        on_step=lambda _s, _l: lrs_chunked.append(
            float(opt_chunked.param_groups[0]["lr"])
        ),
    )
    losses_c2 = train(
        m_chunked,
        stream,
        max_steps=3,
        batch_size=4,
        optimizer=opt_chunked,
        scheduler=sched_chunked,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        on_step=lambda _s, _l: lrs_chunked.append(
            float(opt_chunked.param_groups[0]["lr"])
        ),
    )

    assert losses_c1 + losses_c2 == pytest.approx(losses_single)
    assert lrs_chunked == pytest.approx(lrs_single)
    # Step 1 uses 0.5 * lr, Step 2 uses 1.0 * lr, Step 6 uses 0.1 * lr
    assert lrs_single[0] == pytest.approx(5e-4)
    assert lrs_single[1] == pytest.approx(1e-3)
    assert lrs_single[5] == pytest.approx(1e-4)
    for p_s, p_c in zip(m_single.parameters(), m_chunked.parameters(), strict=True):
        assert torch.equal(p_s, p_c)

    # Validation checks
    ds = _SyntheticDataset()
    with pytest.raises(
        ValueError, match="cannot specify both weight_decay and optimizer"
    ):
        train(
            m_single,
            ds,
            max_steps=1,
            batch_size=4,
            weight_decay=0.1,
            optimizer=opt_single,
        )
    with pytest.raises(ValueError, match="cannot specify scheduler without optimizer"):
        train(m_single, ds, max_steps=1, batch_size=4, scheduler=sched_single)
    with pytest.raises(ValueError, match="grad_accum_steps must be >= 1"):
        train(m_single, ds, max_steps=1, batch_size=4, grad_accum_steps=0)
    with pytest.raises(ValueError, match="must be divisible by grad_accum_steps"):
        train(m_single, ds, max_steps=1, batch_size=6, grad_accum_steps=4)
    for bad_norm in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="max_grad_norm must be > 0"):
            train(m_single, ds, max_steps=1, batch_size=4, max_grad_norm=bad_norm)
