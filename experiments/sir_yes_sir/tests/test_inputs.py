"""Tests for `sir_yes_sir` shared model input construction (`inputs.py`)."""

import pytest

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    TokenType,
    build_example,
)
from experiments.sir_yes_sir.variants.ar.inputs import build_ar_labelled_example
from experiments.sir_yes_sir.variants.cls.inputs import build_cls_labelled_example
from yessir.dataset import IGNORE_INDEX
from yessir.special_tokens import SpecialToken
from yessir.vocab import TOKEN_TYPE_RANGE


def test_token_types_are_in_valid_range():
    for token_type in TokenType:
        assert token_type in TOKEN_TYPE_RANGE, f"{token_type.name} = {token_type}"


def test_build_example_layout_and_position_coupling():
    """Verify exact layout and position coupling for Example 1 (`alphabet = '0123'`, `x = '130'`)."""
    alphabet = b"0123"
    x = b"130"

    # k = 0 (empty output): [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP]
    ex0 = build_example(alphabet, x)
    assert ex0.input_ids == (
        SpecialToken.BOS,
        ord("0"),
        ord("1"),
        ord("2"),
        ord("3"),
        SpecialToken.SEP,
        ord("1"),
        ord("3"),
        ord("0"),
        SpecialToken.SEP,
    )
    assert ex0.token_type_ids == (
        TokenType.SPECIAL,
        TokenType.ALPHABET,
        TokenType.ALPHABET,
        TokenType.ALPHABET,
        TokenType.ALPHABET,
        TokenType.SPECIAL,
        TokenType.INPUT,
        TokenType.INPUT,
        TokenType.INPUT,
        TokenType.SPECIAL,
    )
    assert ex0.position_ids == (
        0,
        0,
        1,
        2,
        3,
        0,
        0,
        1,
        2,
        0,
    )

    # k = 2 (output = '20'): [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 0
    # Output position_ids start at 1 (o_0 at 1, o_1 at 2) so the query token
    # at step k (second [SEP] at k=0, o_{k-1} at k>=1) sits at position_id = k.
    ex2 = build_example(alphabet, x, output=b"20")
    assert ex2.input_ids == ex0.input_ids + (ord("2"), ord("0"))
    assert ex2.token_type_ids == ex0.token_type_ids + (
        TokenType.OUTPUT,
        TokenType.OUTPUT,
    )
    assert ex2.position_ids == ex0.position_ids + (1, 2)


def test_embedding_modes_attach_expected_ids():
    alphabet = b"0123"
    x = b"130"
    output = b"20"

    ex_pe = build_example(
        alphabet, x, output, embedding_mode=EmbeddingMode.TOKEN_TYPE_PE
    )
    assert ex_pe.token_type_ids is not None
    assert ex_pe.position_ids is not None

    ex_type_only = build_example(
        alphabet, x, output, embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE
    )
    assert ex_type_only.input_ids == ex_pe.input_ids
    assert ex_type_only.token_type_ids == ex_pe.token_type_ids
    assert ex_type_only.position_ids is None

    ex_nope = build_example(alphabet, x, output, embedding_mode=EmbeddingMode.NOPE)
    assert ex_nope.input_ids == ex_pe.input_ids
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_rejects_invalid_alphabet():
    with pytest.raises(ValueError, match="alphabet size must be between 2"):
        build_example(b"0", b"0")

    with pytest.raises(ValueError, match="duplicates"):
        build_example(b"001", b"0")

    with pytest.raises(ValueError, match="is not a symbol"):
        build_example([SpecialToken.BOS, SpecialToken.SEP], [SpecialToken.BOS])


def test_rejects_empty_x():
    with pytest.raises(ValueError, match="at least 1 symbol"):
        build_example(b"01", b"")


def test_rejects_input_symbol_not_in_alphabet():
    with pytest.raises(ValueError, match="input symbol .* not found in alphabet"):
        build_example(b"0123", b"190")


def test_rejects_unknown_embedding_mode():
    with pytest.raises(ValueError, match="unknown embedding_mode"):
        build_example(b"01", b"0", embedding_mode="invalid")  # ty: ignore[invalid-argument-type]


def test_build_ar_labelled_example_layout():
    """Verify `build_ar_labelled_example` for Example 1 (`alphabet = '0123'`, `x = '130'`)."""
    alphabet = b"0123"
    x = b"130"

    labelled = build_ar_labelled_example(alphabet, x)
    prefix_len = 1 + len(alphabet) + 1 + len(x) + 1  # [BOS] 0123 [SEP] 130 [SEP] = 10
    expected_outputs = (ord("2"), ord("0"), ord("1"))
    expected_targets = (*expected_outputs, int(SpecialToken.EOS))

    assert len(labelled.example) == prefix_len + len(x)
    assert len(labelled.labels) == len(labelled.example)
    assert labelled.example.input_ids[prefix_len:] == expected_outputs
    assert labelled.example.position_ids is not None
    # Query at second [SEP] is position_id 0; o_0, o_1, o_2 sit at position_ids 1, 2, 3
    assert labelled.example.position_ids[prefix_len - 1 :] == (0, 1, 2, 3)
    # Prefix before the second [SEP] is masked with IGNORE_INDEX; from second [SEP] onward predicts (o_0..o_{L-1}, EOS)
    assert labelled.labels[: prefix_len - 1] == (IGNORE_INDEX,) * (prefix_len - 1)
    assert labelled.labels[prefix_len - 1 :] == expected_targets


def test_build_cls_labelled_example_layout():
    """Verify `build_cls_labelled_example` for Example 1 (`alphabet = '0123'`, `x = '130'`)."""
    alphabet = b"0123"
    x = b"130"

    labelled = build_cls_labelled_example(alphabet, x)
    prefix_len = 1 + len(alphabet) + 1 + len(x) + 1  # [BOS] 0123 [SEP] 130 [SEP] = 10
    expected_steps = (
        (ord("2"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), int(SpecialToken.FLAG_FINISHED)),
    )

    assert len(labelled.example) == prefix_len + len(x) - 1
    assert len(labelled.labels) == len(labelled.example)
    # Input sequence appends (o_0, ..., o_{L-2}) = ('2', '0')
    assert labelled.example.input_ids[prefix_len:] == (ord("2"), ord("0"))
    assert labelled.example.position_ids is not None
    # Query positions are second [SEP] at 0, o_0 at 1, o_1 at 2 (matching x_0, x_1, x_2)
    assert labelled.example.position_ids[prefix_len - 1 :] == (0, 1, 2)
    assert labelled.labels[: prefix_len - 1] == ((IGNORE_INDEX, IGNORE_INDEX),) * (
        prefix_len - 1
    )
    assert labelled.labels[prefix_len - 1 :] == expected_steps
