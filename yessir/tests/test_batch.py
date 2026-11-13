"""Tests for `Batch` and `collate`."""

import pytest
import torch

from yessir.batch import Batch, collate, collate_labelled
from yessir.dataset import IGNORE_INDEX, LabelledExample
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


def test_collate_labelled_right_pads_labels_with_ignore_index():
    item1 = LabelledExample(
        example=Example(input_ids=(SpecialToken.BOS, 10)),
        labels=(IGNORE_INDEX, 20),
    )
    item2 = LabelledExample(
        example=Example(input_ids=(SpecialToken.BOS, 10, 20, 30)),
        labels=(IGNORE_INDEX, 20, 30, int(SpecialToken.FLAG_FINISHED)),
    )

    batch, labels = collate_labelled([item1, item2])
    assert batch.input_ids.shape == (2, 4)
    assert labels.shape == (2, 4)
    assert torch.equal(
        labels,
        torch.tensor(
            [
                [IGNORE_INDEX, 20, IGNORE_INDEX, IGNORE_INDEX],
                [IGNORE_INDEX, 20, 30, int(SpecialToken.FLAG_FINISHED)],
            ],
            dtype=torch.long,
        ),
    )

    with pytest.raises(ValueError, match="cannot collate an empty sequence"):
        collate_labelled([])

    bad_item = LabelledExample(
        example=Example(input_ids=(1, 2, 3)),
        labels=(10, 20),
    )
    with pytest.raises(ValueError, match="has 2 labels for example of length 3"):
        collate_labelled([bad_item])


def test_collate_labelled_2d_per_position_head_labels():
    item1 = LabelledExample(
        example=Example(input_ids=(SpecialToken.BOS, 10)),
        labels=((IGNORE_INDEX, IGNORE_INDEX), (20, int(SpecialToken.FLAG_FINISHED))),
    )
    item2 = LabelledExample(
        example=Example(input_ids=(SpecialToken.BOS, 10, 20)),
        labels=(
            (IGNORE_INDEX, IGNORE_INDEX),
            (20, int(SpecialToken.FLAG_CONTINUE)),
            (30, int(SpecialToken.FLAG_FINISHED)),
        ),
    )
    batch, labels = collate_labelled([item1, item2])
    assert batch.input_ids.shape == (2, 3)
    assert labels.shape == (2, 3, 2)
    assert torch.equal(
        labels[0],
        torch.tensor(
            [
                [IGNORE_INDEX, IGNORE_INDEX],
                [20, int(SpecialToken.FLAG_FINISHED)],
                [IGNORE_INDEX, IGNORE_INDEX],
            ],
            dtype=torch.long,
        ),
    )

    item_1d = LabelledExample(
        example=Example(input_ids=(SpecialToken.BOS, 10)),
        labels=(IGNORE_INDEX, 20),
    )
    with pytest.raises(ValueError, match="same label dimensionality"):
        collate_labelled([item1, item_1d])
    with pytest.raises(ValueError, match="same label dimensionality"):
        collate_labelled([item_1d, item1])


def test_batch_to_moves_present_tensors_and_preserves_none():
    full = collate(
        [
            Example(input_ids=(1, 2), token_type_ids=(0, 1), position_ids=(0, 0)),
        ]
    )
    moved_full = full.to("meta")
    assert moved_full.input_ids.device.type == "meta"
    assert moved_full.lengths.device.type == "meta"
    assert (
        moved_full.token_type_ids is not None
        and moved_full.token_type_ids.device.type == "meta"
    )
    assert (
        moved_full.position_ids is not None
        and moved_full.position_ids.device.type == "meta"
    )

    minimal = collate([Example(input_ids=(1, 2))])
    moved_minimal = minimal.to(torch.device("meta"))
    assert moved_minimal.input_ids.device.type == "meta"
    assert moved_minimal.lengths.device.type == "meta"
    assert moved_minimal.token_type_ids is None
    assert moved_minimal.position_ids is None
