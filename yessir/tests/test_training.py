"""Tests for `yessir/training.py`."""

import random

import pytest
import torch
import torch.nn.functional as F

from experiments.sir_yes_sir.dataset import SirYesSirDataset
from yessir.dataset import LabelledExample
from yessir.example import Example
from yessir.model import Transformer
from yessir.training import (
    build_optimizer,
    multi_head_cross_entropy,
    seed_everything,
    train,
    train_step,
)
from yessir.vocab import VOCAB_SIZE


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


def test_multi_head_cross_entropy_matches_manual_sum():
    torch.manual_seed(0)
    logits_0 = torch.randn(3, 10, requires_grad=True)
    logits_1 = torch.randn(3, 10, requires_grad=True)
    logits_2 = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([[1, 2, 0], [5, 0, 3], [9, 8, 1]], dtype=torch.long)

    loss = multi_head_cross_entropy((logits_0, logits_1, logits_2), targets)
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


def test_multi_head_cross_entropy_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="at least one head"):
        multi_head_cross_entropy((), torch.zeros(2, 0, dtype=torch.long))

    logits = (torch.randn(2, 8), torch.randn(2, 8))
    with pytest.raises(ValueError, match="targets shape"):
        multi_head_cross_entropy(logits, torch.zeros(2, 3, dtype=torch.long))

    mismatched_logits = (torch.randn(2, 8), torch.randn(3, 8))
    with pytest.raises(ValueError, match="batch size"):
        multi_head_cross_entropy(mismatched_logits, torch.zeros(2, 2, dtype=torch.long))


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
    with pytest.raises(ValueError, match="all LabelledExamples in a multi-head batch"):
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
    dataset = SirYesSirDataset(alphabet_size=3, x_length=2, seed=0)

    with pytest.raises(ValueError, match="steps must be >= 1"):
        train(model, dataset, steps=0, batch_size=4)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        train(model, dataset, steps=1, batch_size=0)

    ex = Example(input_ids=(1, 2), token_type_ids=(0, 3), position_ids=(0, 0))
    finite_dataset = [LabelledExample(example=ex, labels=(1, 2, 3))]
    with pytest.raises(ValueError, match="dataset exhausted early"):
        train(model, finite_dataset, steps=1, batch_size=2)


def test_train_reduces_loss_on_sir_yes_sir_stream():
    seed_everything(42)
    model = Transformer(
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=2,
        n_heads=4,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )
    dataset = SirYesSirDataset(alphabet_size=3, x_length=2, seed=42)

    losses = train(model, dataset, steps=15, batch_size=16, lr=1e-3)
    assert len(losses) == 15
    # Initial random-guess cross-entropy over 3 heads of size 256 is ~ 3 * ln(256) ≈ 16.6;
    # after 15 steps loss should drop substantially.
    assert losses[-1] < losses[0] - 4.0
