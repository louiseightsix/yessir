"""Tests for `yessir/training.py`."""

import itertools
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
    build_optimizer,
    next_token_cross_entropy,
    pack_classification_labels,
    parallel_classification_cross_entropy,
    resolve_device,
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

    with pytest.raises(ValueError, match="lr must be > 0"):
        build_optimizer(model, lr=0.0)
    with pytest.raises(ValueError, match="lr must be > 0"):
        build_optimizer(model, lr=-1e-4)


def test_parallel_classification_cross_entropy_matches_manual_sum():
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
    )
    assert torch.allclose(loss, expected)

    loss.backward()
    assert logits_0.grad is not None
    assert logits_1.grad is not None
    assert logits_2.grad is not None


def test_parallel_classification_cross_entropy_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="at least one head"):
        parallel_classification_cross_entropy((), torch.zeros(2, 0, dtype=torch.long))

    logits = (torch.randn(2, 8), torch.randn(2, 8))
    with pytest.raises(ValueError, match="labels shape"):
        parallel_classification_cross_entropy(
            logits, torch.zeros(2, 3, dtype=torch.long)
        )

    mismatched_logits = (torch.randn(2, 8), torch.randn(3, 8))
    with pytest.raises(ValueError, match="batch size"):
        parallel_classification_cross_entropy(
            mismatched_logits, torch.zeros(2, 2, dtype=torch.long)
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

    with pytest.raises(ValueError, match="steps must be >= 1"):
        train(model, dataset, steps=0, batch_size=4)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        train(model, dataset, steps=1, batch_size=0)
    with pytest.raises(ValueError, match="cannot specify both lr and optimizer"):
        train(
            model,
            dataset,
            steps=1,
            batch_size=4,
            lr=1e-3,
            optimizer=build_optimizer(model),
        )

    ex = Example(input_ids=(1, 2), token_type_ids=(0, 3), position_ids=(0, 0))
    finite_dataset = [LabelledExample(example=ex, labels=(1, 2, 3))]
    with pytest.raises(ValueError, match="dataset exhausted early"):
        train(model, finite_dataset, steps=1, batch_size=2)


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
        steps=5,
        batch_size=8,
        optimizer=opt,
        on_step=lambda s, loss: recorded_steps.append((s, loss)),
    )
    losses_part2 = train(model, stream, steps=5, batch_size=8, optimizer=opt)
    losses = losses_part1 + losses_part2

    assert [s for s, _ in recorded_steps] == [1, 2, 3, 4, 5]
    assert [loss for _, loss in recorded_steps] == losses_part1

    assert len(losses) == 10
    # Initial random-guess cross-entropy over 3 heads of size 256 is ~ 3 * ln(256) ≈ 16.6;
    # after 10 steps loss should drop substantially.
    assert losses[-1] < losses[0] - 4.0

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
