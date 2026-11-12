"""Tests for `MultiTaskHead`."""

import pytest
import torch

from yessir.heads import MultiTaskHead
from yessir.vocab import VOCAB_SIZE


def test_multitask_head_shapes_and_bias_free():
    torch.manual_seed(0)
    head = MultiTaskHead(d_model=16, num_classes=(VOCAB_SIZE, VOCAB_SIZE, 2))
    assert len(head.heads) == 3
    for linear in head.heads:
        assert linear.bias is None

    h = torch.randn(4, 7, 16)
    lengths = torch.tensor([3, 7, 1, 5], dtype=torch.long)

    logits_a, logits_b, logits_c = head(h, lengths)
    assert logits_a.shape == (4, VOCAB_SIZE)
    assert logits_b.shape == (4, VOCAB_SIZE)
    assert logits_c.shape == (4, 2)


def test_multitask_head_reads_at_lengths_minus_one_and_ignores_padding():
    torch.manual_seed(0)
    head = MultiTaskHead(d_model=8, num_classes=(10, 2))

    h1 = torch.randn(2, 5, 8)
    lengths = torch.tensor([2, 5], dtype=torch.long)
    out1 = head(h1, lengths)

    # Corrupt all positions except index `lengths[b] - 1` (index 1 for b=0, index 4 for b=1)
    h2 = torch.randn(2, 5, 8)
    h2[0, 1] = h1[0, 1]
    h2[1, 4] = h1[1, 4]
    out2 = head(h2, lengths)

    for logits1, logits2 in zip(out1, out2, strict=True):
        assert torch.allclose(logits1, logits2)


def test_multitask_head_gradient_flows_only_through_selected_positions():
    torch.manual_seed(0)
    head = MultiTaskHead(d_model=8, num_classes=(4, 2))
    h = torch.randn(2, 4, 8, requires_grad=True)
    lengths = torch.tensor([2, 4], dtype=torch.long)

    logits_a, logits_b = head(h, lengths)
    (logits_a.sum() + logits_b.sum()).backward()

    assert h.grad is not None
    # Batch 0 has length 2 -> only index 1 receives gradient
    assert torch.count_nonzero(h.grad[0, 1]) > 0
    assert torch.count_nonzero(h.grad[0, 0]) == 0
    assert torch.count_nonzero(h.grad[0, 2:]) == 0

    # Batch 1 has length 4 -> only index 3 receives gradient
    assert torch.count_nonzero(h.grad[1, 3]) > 0
    assert torch.count_nonzero(h.grad[1, :3]) == 0


@pytest.mark.parametrize(
    ("d_model", "num_classes", "match"),
    [
        (0, (10,), "d_model must be > 0"),
        (-4, (10,), "d_model must be > 0"),
        (8, (), "num_classes must contain at least one head"),
        (8, (10, 0), r"num_classes\[1\] must be > 0"),
        (8, (-2,), r"num_classes\[0\] must be > 0"),
    ],
)
def test_multitask_head_rejects_invalid_init_args(d_model, num_classes, match):
    with pytest.raises(ValueError, match=match):
        MultiTaskHead(d_model=d_model, num_classes=num_classes)


@pytest.mark.parametrize(
    "bad_lengths",
    [
        torch.tensor([3], dtype=torch.long),
        torch.tensor([3, 4, 5], dtype=torch.long),
        torch.tensor([[3, 4]], dtype=torch.long),
    ],
)
def test_multitask_head_rejects_mismatched_lengths_shape(bad_lengths):
    head = MultiTaskHead(d_model=8, num_classes=(4,))
    h = torch.randn(2, 5, 8)
    with pytest.raises(ValueError, match="lengths shape"):
        head(h, bad_lengths)
