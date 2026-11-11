"""Tests for `Example`."""

from dataclasses import FrozenInstanceError

import pytest

from yessir.example import Example
from yessir.special_tokens import SpecialToken


def test_example_with_all_coordinates():
    ex = Example(
        input_ids=(SpecialToken.BOS, 0x30, 0x31),
        token_type_ids=(0, 1, 1),
        position_ids=(0, 0, 1),
    )
    assert len(ex) == 3
    assert ex.input_ids == (SpecialToken.BOS, 0x30, 0x31)
    assert ex.token_type_ids == (0, 1, 1)
    assert ex.position_ids == (0, 0, 1)


def test_example_without_side_channels():
    ex = Example(input_ids=(SpecialToken.BOS, 0x30))
    assert len(ex) == 2
    assert ex.token_type_ids is None
    assert ex.position_ids is None


def test_example_is_immutable_and_hashable():
    ex = Example(
        input_ids=(SpecialToken.BOS, 0x30),
        token_type_ids=(0, 1),
        position_ids=(0, 0),
    )
    with pytest.raises(FrozenInstanceError):
        ex.input_ids = (0,)  # ty: ignore[invalid-assignment]

    assert hash(ex) == hash(
        Example(
            input_ids=(SpecialToken.BOS, 0x30),
            token_type_ids=(0, 1),
            position_ids=(0, 0),
        )
    )


def test_rejects_non_tuple_sequences():
    with pytest.raises(TypeError, match="input_ids must be a tuple"):
        Example(input_ids=[0, 1])  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="token_type_ids must be a tuple"):
        Example(input_ids=(0, 1), token_type_ids=[0, 1])  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="position_ids must be a tuple"):
        Example(input_ids=(0, 1), position_ids=[0, 1])  # ty: ignore[invalid-argument-type]


def test_rejects_empty_input_ids():
    with pytest.raises(ValueError, match="input_ids must not be empty"):
        Example(input_ids=())


@pytest.mark.parametrize("bad_token", [-1, 256, 1.5, "a", True, False])
def test_rejects_invalid_token_ids(bad_token):
    with pytest.raises(ValueError, match="not a valid token id"):
        Example(input_ids=(SpecialToken.BOS, bad_token))


def test_rejects_mismatched_coordinate_lengths():
    with pytest.raises(ValueError, match="token_type_ids length"):
        Example(input_ids=(0, 1, 2), token_type_ids=(0, 1))

    with pytest.raises(ValueError, match="position_ids length"):
        Example(input_ids=(0, 1, 2), position_ids=(0, 1, 2, 3))


@pytest.mark.parametrize("bad_type_id", [-1, 16, 100, 1.5, "0", True, False])
def test_rejects_invalid_token_type_ids(bad_type_id):
    with pytest.raises(ValueError, match=r"token_type_ids must be ints in 0\.\.15"):
        Example(input_ids=(0, 1), token_type_ids=(0, bad_type_id))


@pytest.mark.parametrize("bad_pos", [-1, 1.5, "0", True, False])
def test_rejects_invalid_position_ids(bad_pos):
    with pytest.raises(ValueError, match="position_ids must contain non-negative ints"):
        Example(input_ids=(0, 1), position_ids=(bad_pos, 0))
