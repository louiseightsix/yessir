"""Tests for `RMSNorm`, `SwiGLUFeedForward`, `MultiHeadSelfAttention`, and `TransformerBlock`."""

import pytest
import torch

from yessir.blocks import (
    KVCache,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLUFeedForward,
    TransformerBlock,
    build_causal_attention_mask,
)


def test_rmsnorm_unit_rms_zero_preservation_and_gain():
    norm = RMSNorm(d_model=4)
    x = torch.tensor([[[3.0, 0.0, -3.0, 0.0], [1.0, 2.0, 3.0, 4.0]]])

    out = norm(x)
    assert out.shape == (1, 2, 4)
    assert out.dtype == torch.float32

    # Unit RMS when weight == 1
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(1, 2), atol=1e-5)

    # Zero coordinates stay exact zero (no mean shift or bias)
    assert out[0, 0, 1] == 0.0
    assert out[0, 0, 3] == 0.0

    # Learned gain scales each coordinate independently
    assert norm.weight is not None
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 5.0, 0.5, 1.0]))
    scaled = norm(x)
    assert torch.allclose(scaled, out * norm.weight)

    # Gain-free RMSNorm has no learnable parameters and produces unit RMS
    norm_no_gain = RMSNorm(d_model=4, gain=False)
    assert norm_no_gain.weight is None
    assert list(norm_no_gain.parameters()) == []
    assert torch.allclose(norm_no_gain(x), out)


def test_rmsnorm_preserves_bfloat16_dtype():
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    # 1. float32 weights with bfloat16 input
    norm_fp32 = RMSNorm(d_model=8)
    out_mixed = norm_fp32(x)
    assert out_mixed.dtype == torch.bfloat16
    assert out_mixed.shape == (2, 3, 8)

    # 2. Full bfloat16 module cast
    norm_bf16 = RMSNorm(d_model=8).to(torch.bfloat16)
    out_bf16 = norm_bf16(x)
    assert out_bf16.dtype == torch.bfloat16
    assert out_bf16.shape == (2, 3, 8)


@pytest.mark.parametrize("bad_d_model", [0, -4])
def test_rmsnorm_rejects_non_positive_d_model(bad_d_model):
    with pytest.raises(ValueError, match="d_model must be > 0"):
        RMSNorm(d_model=bad_d_model)


@pytest.mark.parametrize("bad_eps", [0.0, -1e-6])
def test_rmsnorm_rejects_non_positive_eps(bad_eps):
    with pytest.raises(ValueError, match="eps must be > 0"):
        RMSNorm(d_model=8, eps=bad_eps)


def test_swiglu_shape_bias_free_and_position_independence():
    torch.manual_seed(0)
    ffn = SwiGLUFeedForward(d_model=16, d_ff=32)
    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None

    # Zero input produces exact zero output (no biases)
    zeros = torch.zeros(1, 3, 16)
    assert torch.count_nonzero(ffn(zeros)) == 0

    # Changing positions 1..3 does not affect position 0
    x1 = torch.randn(2, 4, 16)
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn(2, 3, 16)
    out1 = ffn(x1)
    out2 = ffn(x2)
    assert out1.shape == (2, 4, 16)
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :])


@pytest.mark.parametrize(("d_model", "d_ff"), [(0, 16), (-1, 16), (16, 0), (16, -8)])
def test_swiglu_rejects_non_positive_dimensions(d_model, d_ff):
    with pytest.raises(ValueError, match="must be > 0"):
        SwiGLUFeedForward(d_model=d_model, d_ff=d_ff)


def test_attention_shape_bias_free_and_causal_masking():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(d_model=16, n_heads=4)
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.out_proj.bias is None

    x = torch.randn(2, 5, 16)
    out = attn(x)
    assert out.shape == (2, 5, 16)

    # Right-padding / causal invariance: prefix [0..2] gives identical outputs
    # whether run alone or followed by arbitrary trailing tokens at [3..4].
    prefix_out = attn(x[:, :3, :])
    assert torch.allclose(out[:, :3, :], prefix_out, atol=1e-6)


@pytest.mark.parametrize(
    ("d_model", "n_heads", "match"),
    [
        (0, 4, "d_model must be > 0"),
        (16, 0, "n_heads must be > 0"),
        (16, 3, "divisible by n_heads"),
    ],
)
def test_attention_rejects_invalid_head_configuration(d_model, n_heads, match):
    with pytest.raises(ValueError, match=match):
        MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)


def test_transformer_block_causal_invariance_and_gradient_flow():
    torch.manual_seed(0)
    block = TransformerBlock(d_model=16, n_heads=4, d_ff=32)

    x = torch.randn(2, 5, 16, requires_grad=True)
    out = block(x)
    assert out.shape == (2, 5, 16)

    # Prefix causal invariance through the full block
    with torch.no_grad():
        prefix_out = block(x[:, :3, :])
    assert torch.allclose(out[:, :3, :], prefix_out, atol=1e-6)

    # Backward pass populates gradients on input and every parameter
    out.sum().backward()
    assert x.grad is not None
    for name, param in block.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"


def test_build_causal_attention_mask_protects_left_pad_diagonal_and_blocks_pad_keys():
    key_mask = torch.tensor(
        [
            [False, False, True, True],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    mask = build_causal_attention_mask(key_mask, start_pos=0)
    assert mask.shape == (2, 1, 4, 4)
    # Row 0: left-PAD queries 0 and 1 attend only to their own diagonal;
    # real queries 2 and 3 attend only to real causal keys >= 2.
    assert torch.equal(
        mask[0, 0],
        torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    # Decode slice starting at position 2: shape (2, 1, 2, 4).
    decode_mask = build_causal_attention_mask(key_mask, start_pos=2)
    assert decode_mask.shape == (2, 1, 2, 4)
    assert torch.equal(decode_mask[:, 0, :, :], mask[:, 0, 2:, :])

    with pytest.raises(ValueError, match="2D bool tensor"):
        build_causal_attention_mask(torch.ones((2, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="start_pos"):
        build_causal_attention_mask(key_mask, start_pos=5)


@torch.no_grad()
def test_kv_cache_incremental_block_decoding_and_validation():
    torch.manual_seed(7)
    block = TransformerBlock(d_model=16, n_heads=4, d_ff=32)
    block.eval()

    x = torch.randn(2, 6, 16)
    full_out = block(x)

    # Incremental prefill (3 tokens) + 1-token step + 2-token chunk step.
    cache = KVCache(n_layers=1, max_seq_len=6)
    assert cache.seq_len == 0

    m0 = cache.prepare_step(2, 3, device=x.device)
    assert m0 is None
    out0 = block(x[:, :3, :], attn_mask=m0, kv_cache=cache, layer_idx=0)
    cache.commit()
    assert cache.seq_len == 3

    m1 = cache.prepare_step(2, 1, device=x.device)
    assert m1 is None
    out1 = block(x[:, 3:4, :], attn_mask=m1, kv_cache=cache, layer_idx=0)
    cache.commit()
    assert cache.seq_len == 4

    m2 = cache.prepare_step(2, 2, device=x.device)
    assert m2 is not None and m2.shape == (1, 1, 2, 6)
    out2 = block(x[:, 4:6, :], attn_mask=m2, kv_cache=cache, layer_idx=0)
    cache.commit()
    assert cache.seq_len == 6

    cached_out = torch.cat([out0, out1, out2], dim=1)
    assert torch.allclose(cached_out, full_out, atol=1e-5)

    # Validation checks on KVCache
    with pytest.raises(ValueError, match="KVCache overflow"):
        cache.prepare_step(2, 1, device=x.device)
    with pytest.raises(ValueError, match="batch_size"):
        cache.prepare_step(3, 1, device=x.device)
    with pytest.raises(ValueError, match="n_layers must be > 0"):
        KVCache(n_layers=0, max_seq_len=4)
    with pytest.raises(ValueError, match="max_seq_len must be > 0"):
        KVCache(n_layers=1, max_seq_len=0)

    fresh = KVCache(n_layers=1, max_seq_len=4)
    dummy_kv = torch.zeros(1, 4, 1, 4)
    with pytest.raises(RuntimeError, match="prepare_step must be called"):
        fresh.update(0, dummy_kv, dummy_kv)
    fresh.prepare_step(1, 1, device=x.device)
    with pytest.raises(IndexError, match="layer_idx"):
        fresh.update(1, dummy_kv, dummy_kv)
