"""Tests for `Batch` and `collate`."""

import pytest
import torch

from yessir.batch import Batch, collate
from yessir.example import Example
from yessir.special_tokens import SpecialToken


def test_collate_right_pads_input_ids_token_type_and_position_ids():
    ex1 = Example(
        input_ids=(SpecialToken.BOS, 10),
        token_type_ids=(0, 1),
        position_ids=(0, 0),
    )
    ex2 = Example(
        input_ids=(SpecialToken.BOS, 10, 11, 12),
        token_type_ids=(0, 1, 1, 2),
        position_ids=(0, 0, 1, 0),
    )

    batch = collate([ex1, ex2])
    assert isinstance(batch, Batch)
    assert batch.input_ids.shape == (2, 4)
    assert batch.lengths.shape == (2,)
    assert batch.token_type_ids is not None and batch.token_type_ids.shape == (2, 4)
    assert batch.position_ids is not None and batch.position_ids.shape == (2, 4)

    pad = SpecialToken.PAD
    assert torch.equal(
        batch.input_ids,
        torch.tensor(
            [
                [SpecialToken.BOS, 10, pad, pad],
                [SpecialToken.BOS, 10, 11, 12],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(batch.lengths, torch.tensor([2, 4], dtype=torch.long))
    assert torch.equal(
        batch.token_type_ids,
        torch.tensor([[0, 1, 0, 0], [0, 1, 1, 2]], dtype=torch.long),
    )
    assert torch.equal(
        batch.position_ids,
        torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.long),
    )


def test_collate_with_input_ids_only_leaves_ids_none():
    ex1 = Example(input_ids=(SpecialToken.BOS, 1))
    ex2 = Example(input_ids=(SpecialToken.BOS, 1, 2))

    batch = collate([ex1, ex2])
    assert batch.token_type_ids is None
    assert batch.position_ids is None
    assert torch.equal(batch.lengths, torch.tensor([2, 3], dtype=torch.long))


def test_collate_rejects_empty_sequence():
    with pytest.raises(ValueError, match="cannot collate an empty sequence"):
        collate([])


def test_collate_rejects_inconsistent_token_type_ids():
    ex_with = Example(input_ids=(1, 2), token_type_ids=(0, 1))
    ex_without = Example(input_ids=(1, 2), token_type_ids=None)
    for pair in ([ex_with, ex_without], [ex_without, ex_with]):
        with pytest.raises(ValueError, match="token_type_ids"):
            collate(pair)


def test_collate_rejects_inconsistent_position_ids():
    ex_with = Example(input_ids=(1, 2), position_ids=(0, 1))
    ex_without = Example(input_ids=(1, 2), position_ids=None)
    for pair in ([ex_with, ex_without], [ex_without, ex_with]):
        with pytest.raises(ValueError, match="position_ids"):
            collate(pair)
