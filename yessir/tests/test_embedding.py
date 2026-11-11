"""Tests for `SubspaceEmbedding`."""

import pytest
import torch

from yessir.embedding import SubspaceEmbedding
from yessir.vocab import VOCAB_SIZE


def test_default_dimensions_and_parameter_free():
    embed = SubspaceEmbedding()
    assert embed.d_model == 512
    assert list(embed.parameters()) == []


def test_slices_and_unit_norms_in_full_mode():
    embed = SubspaceEmbedding(use_token_type=True, d_position=32, d_scratch=208)
    input_ids = torch.tensor([[0, 65, 255], [10, 20, 30]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 1, 3], [0, 2, 15]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2], [0, 5, 100]], dtype=torch.long)

    out = embed(input_ids, token_type_ids, position_ids)
    assert out.shape == (2, 3, 512)
    assert out.dtype == torch.float32

    symbol_slice = out[:, :, :VOCAB_SIZE]
    type_slice = out[:, :, VOCAB_SIZE : VOCAB_SIZE + 16]
    pos_slice = out[:, :, VOCAB_SIZE + 16 : VOCAB_SIZE + 16 + 32]
    scratch_slice = out[:, :, VOCAB_SIZE + 16 + 32 :]

    # 1. Symbol slice: exact one-hot and unit norm
    assert symbol_slice[0, 1, 65] == 1.0
    assert torch.allclose(symbol_slice.norm(dim=-1), torch.ones(2, 3))

    # 2. Token-type slice: exact one-hot and unit norm
    assert type_slice[0, 2, 3] == 1.0
    assert type_slice[1, 2, 15] == 1.0
    assert torch.allclose(type_slice.norm(dim=-1), torch.ones(2, 3))

    # 3. Position slice: unit L2 norm at every position, and distinct positions differ
    assert torch.allclose(pos_slice.norm(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert not torch.allclose(pos_slice[0, 0], pos_slice[0, 1])
    assert torch.allclose(pos_slice[0, 0], pos_slice[1, 0])  # both have position_id=0

    # 4. Scratchpad slice: all zeros
    assert scratch_slice.shape == (2, 3, 208)
    assert torch.count_nonzero(scratch_slice) == 0


def test_omitting_position_and_token_type_slices():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

    # Token-type only (no position slice)
    embed_type_only = SubspaceEmbedding(
        use_token_type=True, d_position=0, d_scratch=240
    )
    assert embed_type_only.d_model == 512
    out_type = embed_type_only(input_ids, token_type_ids, None)
    assert out_type.shape == (1, 3, 512)
    assert torch.count_nonzero(out_type[:, :, VOCAB_SIZE + 16 :]) == 0

    # Pure NoPE with zero scratchpad (256-d one-hot only)
    embed_nope = SubspaceEmbedding(use_token_type=False, d_position=0, d_scratch=0)
    assert embed_nope.d_model == VOCAB_SIZE
    out_nope = embed_nope(input_ids)
    assert out_nope.shape == (1, 3, VOCAB_SIZE)
    assert torch.allclose(out_nope.norm(dim=-1), torch.ones(1, 3))


@pytest.mark.parametrize("bad_d_pos", [-2, -1, 1, 3, 33])
def test_rejects_invalid_d_position(bad_d_pos):
    with pytest.raises(ValueError, match="d_position must be a non-negative even int"):
        SubspaceEmbedding(d_position=bad_d_pos)


def test_rejects_negative_d_scratch():
    with pytest.raises(ValueError, match="d_scratch must be >= 0"):
        SubspaceEmbedding(d_scratch=-1)


def test_rejects_mismatched_forward_arguments():
    ids = torch.tensor([[0, 1]], dtype=torch.long)

    embed_full = SubspaceEmbedding(use_token_type=True, d_position=32)
    with pytest.raises(ValueError, match="token_type_ids is required"):
        embed_full(ids, token_type_ids=None, position_ids=ids)
    with pytest.raises(ValueError, match="position_ids is required"):
        embed_full(ids, token_type_ids=ids, position_ids=None)

    embed_nope = SubspaceEmbedding(use_token_type=False, d_position=0)
    with pytest.raises(ValueError, match="token_type_ids must be None"):
        embed_nope(ids, token_type_ids=ids)
    with pytest.raises(ValueError, match="position_ids must be None"):
        embed_nope(ids, position_ids=ids)


@pytest.mark.parametrize("bad_type_id", [-1, 16, 20])
def test_rejects_out_of_range_token_type_ids(bad_type_id):
    embed = SubspaceEmbedding(use_token_type=True, d_position=0)
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    bad_types = torch.tensor([[0, bad_type_id]], dtype=torch.long)
    with pytest.raises(ValueError, match="token_type_ids must be in 0..15"):
        embed(input_ids, token_type_ids=bad_types)


def test_to_bfloat16_updates_output_dtype():
    embed = SubspaceEmbedding().to(torch.bfloat16)
    ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out = embed(ids, ids, ids)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 3, 512)
