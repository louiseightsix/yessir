"""Tests for `Transformer`."""

import math

import pytest
import torch

from yessir.batch import collate
from yessir.blocks import TransformerBlock
from yessir.example import Example
from yessir.model import Transformer
from yessir.vocab import VOCAB_SIZE


def test_transformer_default_dimensions_and_forward_shapes():
    torch.manual_seed(0)
    model = Transformer(num_classes=(VOCAB_SIZE, VOCAB_SIZE, 2))
    assert model.d_model == 512
    assert len(model.layers) == 6

    ex1 = Example(
        input_ids=(1, 2, 3, 4),
        token_type_ids=(0, 1, 2, 3),
        position_ids=(0, 0, 0, 0),
    )
    ex2 = Example(
        input_ids=(5, 6, 7, 8, 9, 10),
        token_type_ids=(0, 1, 1, 2, 2, 3),
        position_ids=(0, 0, 1, 0, 1, 0),
    )
    batch = collate([ex1, ex2])

    logits_a, logits_b, logits_c = model(
        batch.input_ids,
        batch.lengths,
        token_type_ids=batch.token_type_ids,
        position_ids=batch.position_ids,
    )
    assert logits_a.shape == (2, VOCAB_SIZE)
    assert logits_b.shape == (2, VOCAB_SIZE)
    assert logits_c.shape == (2, 2)


def test_transformer_end_to_end_right_padding_invariance():
    torch.manual_seed(0)
    model = Transformer(
        num_classes=(16, 2),
        n_layers=2,
        n_heads=4,
        d_ffn=64,
        d_position=8,
        d_scratch=16,
    )

    short_ex = Example(
        input_ids=(10, 20, 30),
        token_type_ids=(0, 1, 3),
        position_ids=(0, 0, 0),
    )
    long_ex = Example(
        input_ids=(10, 20, 30, 40, 50, 60),
        token_type_ids=(0, 1, 1, 2, 2, 3),
        position_ids=(0, 0, 1, 0, 1, 0),
    )

    solo_batch = collate([short_ex])
    padded_batch = collate([short_ex, long_ex])

    solo_logits = model(
        solo_batch.input_ids,
        solo_batch.lengths,
        token_type_ids=solo_batch.token_type_ids,
        position_ids=solo_batch.position_ids,
    )
    padded_logits = model(
        padded_batch.input_ids,
        padded_batch.lengths,
        token_type_ids=padded_batch.token_type_ids,
        position_ids=padded_batch.position_ids,
    )

    for solo, padded in zip(solo_logits, padded_logits, strict=True):
        assert torch.allclose(solo[0], padded[0], atol=1e-5)


def test_transformer_reset_parameters_residual_scaling_and_norm_reset():
    torch.manual_seed(0)
    n_layers = 8
    model = Transformer(
        num_classes=(VOCAB_SIZE,),
        n_layers=n_layers,
        n_heads=8,
        d_ffn=1376,
    )

    expected_base_std = 0.02
    expected_residual_std = 0.02 / math.sqrt(2 * n_layers)

    layer0 = model.layers[0]
    assert isinstance(layer0, TransformerBlock)
    assert abs(layer0.attn.q_proj.weight.std().item() - expected_base_std) < 0.002
    assert abs(layer0.ffn.w_gate.weight.std().item() - expected_base_std) < 0.002
    assert (
        abs(layer0.attn.out_proj.weight.std().item() - expected_residual_std) < 0.0005
    )
    assert abs(layer0.ffn.w_down.weight.std().item() - expected_residual_std) < 0.0005

    # Perturb RMSNorm weights and verify reset_parameters restores them to 1.0
    with torch.no_grad():
        layer0.attn_norm.weight.fill_(3.5)
        model.head.norm.weight.fill_(0.25)
    model.reset_parameters()
    assert torch.equal(layer0.attn_norm.weight, torch.ones(model.d_model))
    assert torch.equal(model.head.norm.weight, torch.ones(model.d_model))


def test_transformer_gradient_flow_to_all_parameters():
    torch.manual_seed(0)
    model = Transformer(
        num_classes=(8, 2),
        n_layers=2,
        n_heads=4,
        d_ffn=32,
        d_position=8,
        d_scratch=0,
    )
    batch = collate(
        [
            Example(
                input_ids=(1, 2, 3), token_type_ids=(0, 1, 3), position_ids=(0, 0, 0)
            ),
            Example(input_ids=(4, 5), token_type_ids=(0, 3), position_ids=(0, 0)),
        ]
    )

    logits_a, logits_b = model(
        batch.input_ids,
        batch.lengths,
        token_type_ids=batch.token_type_ids,
        position_ids=batch.position_ids,
    )
    (logits_a.sum() + logits_b.sum()).backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"
        assert torch.count_nonzero(param.grad) > 0, f"zero gradient for {name}"


@pytest.mark.parametrize("bad_n_layers", [0, -2])
def test_transformer_rejects_non_positive_n_layers(bad_n_layers):
    with pytest.raises(ValueError, match="n_layers must be > 0"):
        Transformer(num_classes=(4,), n_layers=bad_n_layers)
