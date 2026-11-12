"""Tests for `yessir/training.py`."""

import itertools
import random
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F

from yessir.batch import collate
from yessir.dataset import LabelledExample
from yessir.example import Example
from yessir.model import Transformer
from yessir.training import (
    build_optimizer,
    multitask_cross_entropy,
    pack_multitask_targets,
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


def test_multitask_cross_entropy_matches_manual_sum():
    torch.manual_seed(0)
    logits_0 = torch.randn(3, 10, requires_grad=True)
    logits_1 = torch.randn(3, 10, requires_grad=True)
    logits_2 = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([[1, 2, 0], [5, 0, 3], [9, 8, 1]], dtype=torch.long)

    loss = multitask_cross_entropy((logits_0, logits_1, logits_2), targets)
    expected = (
        F.cross_entropy(logits_0, targets[:, 0])
        + F.cross_entropy(logits_1, targets[:, 1])
        + F.cross_entropy(logits_2, targets[:, 2])
    )
    assert torch.allclose(loss, expected)

    loss.backward()
    assert logits_0.grad is not None
    assert logits_1.grad is not None
    assert logits_2.grad is not None


def test_multitask_cross_entropy_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="at least one head"):
        multitask_cross_entropy((), torch.zeros(2, 0, dtype=torch.long))

    logits = (torch.randn(2, 8), torch.randn(2, 8))
    with pytest.raises(ValueError, match="targets shape"):
        multitask_cross_entropy(logits, torch.zeros(2, 3, dtype=torch.long))

    mismatched_logits = (torch.randn(2, 8), torch.randn(3, 8))
    with pytest.raises(ValueError, match="batch size"):
        multitask_cross_entropy(mismatched_logits, torch.zeros(2, 2, dtype=torch.long))


def test_pack_multitask_targets_shapes_and_validation():
    ex = Example(input_ids=(1, 2), token_type_ids=(0, 3), position_ids=(0, 0))
    items = [
        LabelledExample(example=ex, labels=(10, 20, 30)),
        LabelledExample(example=ex, labels=(40, 50, 60)),
    ]
    packed = pack_multitask_targets(items)
    assert packed.dtype == torch.long
    assert torch.equal(packed, torch.tensor([[10, 20, 30], [40, 50, 60]]))

    with pytest.raises(ValueError, match="items must not be empty"):
        pack_multitask_targets([])

    with pytest.raises(ValueError, match="all LabelledExamples in a multitask batch"):
        pack_multitask_targets(
            [
                LabelledExample(example=ex, labels=(1, 2, 3)),
                LabelledExample(example=ex, labels=(1, 2)),
            ]
        )


def test_train_step_sets_train_mode_returns_pre_update_loss_and_updates_weights():
    seed_everything(0)
    model = Transformer(
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
    targets = torch.tensor([item.labels for item in items], dtype=torch.long)
    with torch.no_grad():
        pre_logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        expected_pre_loss = multitask_cross_entropy(pre_logits, targets).item()
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
    with pytest.raises(ValueError, match="all LabelledExamples in a multitask batch"):
        train_step(model, opt, items_mismatched)


def test_train_validation_and_early_exhaustion():
    seed_everything(0)
    model = Transformer(
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
