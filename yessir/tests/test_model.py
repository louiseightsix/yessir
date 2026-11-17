"""Tests for `Transformer`."""

import math

import pytest
import torch

from yessir.batch import collate, collate_left_padded
from yessir.blocks import TransformerBlock
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.vocab import VOCAB_SIZE


def test_transformer_default_dimensions_and_forward_shapes():
    torch.manual_seed(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
    )
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
    assert logits_c.shape == (2, VOCAB_SIZE)


def test_transformer_end_to_end_right_padding_invariance():
    torch.manual_seed(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(16, 2),
        n_layers=2,
        n_heads=4,
        d_ff=64,
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
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE,),
        n_layers=n_layers,
    )

    expected_base_std = 0.02
    expected_residual_std = 0.02 / math.sqrt(2 * n_layers)

    layer0 = model.layers[0]
    assert isinstance(layer0, TransformerBlock)
    assert abs(layer0.attn.q_proj.weight.std().item() - expected_base_std) < 0.002
    assert abs(layer0.ffn.gate_proj.weight.std().item() - expected_base_std) < 0.002
    assert (
        abs(layer0.attn.out_proj.weight.std().item() - expected_residual_std) < 0.0005
    )
    assert (
        abs(layer0.ffn.down_proj.weight.std().item() - expected_residual_std) < 0.0005
    )

    # Perturb RMSNorm weights and verify reset_parameters restores them to 1.0
    # and re-initialises residual projections rather than scaling them a second time.
    assert layer0.attn_norm.weight is not None
    assert model.head.norm.weight is not None
    with torch.no_grad():
        layer0.attn_norm.weight.fill_(3.5)
        model.head.norm.weight.fill_(0.25)
    model.reset_parameters()
    assert torch.equal(layer0.attn_norm.weight, torch.ones(model.d_model))
    assert torch.equal(model.head.norm.weight, torch.ones(model.d_model))
    assert (
        abs(layer0.attn.out_proj.weight.std().item() - expected_residual_std) < 0.0005
    )


def test_transformer_with_subspace_lm_head_shapes_and_causal_invariance():
    torch.manual_seed(0)
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )
    assert model.head.norm.weight is None

    short_ex = Example(
        input_ids=(10, 20, 30),
        token_type_ids=(0, 1, 3),
        position_ids=(0, 0, 0),
    )
    long_ex = Example(
        input_ids=(10, 20, 30, 40, 50),
        token_type_ids=(0, 1, 3, 3, 3),
        position_ids=(0, 0, 0, 1, 2),
    )
    solo_batch = collate([short_ex])
    padded_batch = collate([short_ex, long_ex])

    solo_logits = model(
        solo_batch.input_ids,
        token_type_ids=solo_batch.token_type_ids,
        position_ids=solo_batch.position_ids,
    )
    padded_logits = model(
        padded_batch.input_ids,
        padded_batch.lengths,
        token_type_ids=padded_batch.token_type_ids,
        position_ids=padded_batch.position_ids,
    )
    assert solo_logits.shape == (1, 3, VOCAB_SIZE)
    assert padded_logits.shape == (2, 5, VOCAB_SIZE)
    assert torch.allclose(solo_logits[0], padded_logits[0, :3], atol=1e-5)


def test_transformer_gradient_flow_to_all_parameters():
    torch.manual_seed(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(8, 2),
        n_layers=2,
        n_heads=4,
        d_ff=32,
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
        Transformer(
            head_type=HeadType.PARALLEL_CLASSIFICATION,
            num_classes=(4,),
            n_layers=bad_n_layers,
        )


def test_transformer_validates_head_type_and_num_classes():
    with pytest.raises(
        ValueError,
        match="num_classes is required when head_type is 'parallel_classification'",
    ):
        Transformer(head_type=HeadType.PARALLEL_CLASSIFICATION, num_classes=None)

    with pytest.raises(
        ValueError, match="num_classes must be None when head_type is 'subspace_lm'"
    ):
        Transformer(head_type=HeadType.SUBSPACE_LM, num_classes=(4,))


def test_transformer_device_property_tracks_module_device():
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    assert model.device == torch.device("cpu")
    model.to("meta")
    assert model.device.type == "meta"


@pytest.mark.parametrize(
    ("use_token_type", "d_position"),
    [
        (True, 8),  # token_type_pe
        (True, 0),  # token_type_nope
        (False, 0),  # nope
    ],
)
@torch.no_grad()
def test_transformer_left_padded_kv_cache_rollout_matches_single_example_uncached(
    use_token_type: bool,
    d_position: int,
):
    torch.manual_seed(11)
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=3,
        n_heads=4,
        d_ff=64,
        use_token_type=use_token_type,
        d_position=d_position,
        d_scratch=16,
    )
    model.eval()

    raw_prompts = [
        ((1, 10, 20), (0, 1, 3), (0, 0, 0)),
        ((1, 11, 12, 13, 21, 30), (0, 1, 1, 2, 2, 3), (0, 0, 1, 0, 1, 0)),
        ((1, 42), (0, 3), (0, 0)),
    ]
    examples = [
        Example(
            input_ids=ids,
            token_type_ids=tts if use_token_type else None,
            position_ids=pos if d_position > 0 else None,
        )
        for ids, tts, pos in raw_prompts
    ]

    prefill = collate_left_padded(examples)
    decode_steps = 5
    cache = model.init_kv_cache(
        max_seq_len=int(prefill.input_ids.size(1)) + decode_steps
    )

    # Also verify uncached left-padded forward matches single-example prefixes.
    uncached_prefill_logits = model(
        prefill.input_ids,
        token_type_ids=prefill.token_type_ids,
        position_ids=prefill.position_ids,
        key_padding_mask=prefill.key_padding_mask,
    )
    assert isinstance(uncached_prefill_logits, torch.Tensor)
    assert not torch.isnan(uncached_prefill_logits).any()

    logits = model(
        prefill.input_ids,
        token_type_ids=prefill.token_type_ids,
        position_ids=prefill.position_ids,
        key_padding_mask=prefill.key_padding_mask,
        kv_cache=cache,
    )
    assert isinstance(logits, torch.Tensor)
    assert torch.allclose(logits, uncached_prefill_logits, atol=1e-5)

    cached_step_logits: list[torch.Tensor] = [logits[:, -1, :]]
    cached_tokens: list[list[int]] = [[] for _ in examples]
    next_tokens = logits[:, -1, :].argmax(dim=-1)
    for b_idx, tok in enumerate(next_tokens.tolist()):
        cached_tokens[b_idx].append(int(tok))

    for step_idx in range(1, decode_steps):
        step_input = next_tokens.unsqueeze(1)
        step_tt = (
            torch.full((len(examples), 1), 4 + (step_idx % 2), dtype=torch.long)
            if use_token_type
            else None
        )
        step_pos = (
            torch.full((len(examples), 1), step_idx, dtype=torch.long)
            if d_position > 0
            else None
        )
        step_out = model(
            step_input,
            token_type_ids=step_tt,
            position_ids=step_pos,
            kv_cache=cache,
        )
        assert isinstance(step_out, torch.Tensor)
        last_logits = step_out[:, -1, :]
        cached_step_logits.append(last_logits)
        next_tokens = last_logits.argmax(dim=-1)
        for b_idx, tok in enumerate(next_tokens.tolist()):
            cached_tokens[b_idx].append(int(tok))

    # Compare against unpadded, uncached batch_size=1 autoregressive execution.
    for b_idx, (ids, tts, pos) in enumerate(raw_prompts):
        cur_ids = list(ids)
        cur_tts = list(tts)
        cur_pos = list(pos)
        solo_tokens: list[int] = []
        for step_idx in range(decode_steps):
            ex = Example(
                input_ids=tuple(cur_ids),
                token_type_ids=tuple(cur_tts) if use_token_type else None,
                position_ids=tuple(cur_pos) if d_position > 0 else None,
            )
            solo_batch = collate([ex])
            solo_out = model(
                solo_batch.input_ids,
                token_type_ids=solo_batch.token_type_ids,
                position_ids=solo_batch.position_ids,
            )
            assert isinstance(solo_out, torch.Tensor)
            assert torch.allclose(
                solo_out[0, -1, :], cached_step_logits[step_idx][b_idx], atol=1e-5
            )
            tok = int(solo_out[0, -1, :].argmax(dim=-1).item())
            solo_tokens.append(tok)
            cur_ids.append(tok)
            cur_tts.append(4 + ((step_idx + 1) % 2))
            cur_pos.append(step_idx + 1)

        assert solo_tokens == cached_tokens[b_idx]


def test_transformer_kv_cache_rejects_grad_enabled_and_bad_key_mask():
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        use_token_type=False,
        d_position=0,
        d_scratch=0,
    )
    batch = collate([Example(input_ids=(1, 2, 3))])
    cache = model.init_kv_cache(max_seq_len=8)

    with pytest.raises(RuntimeError, match="KVCache is inference-only"):
        model(batch.input_ids, kv_cache=cache)

    with torch.no_grad(), pytest.raises(ValueError, match="key_padding_mask"):
        model(batch.input_ids, key_padding_mask=torch.ones((1, 2), dtype=torch.bool))
