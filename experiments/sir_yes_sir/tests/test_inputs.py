"""Tests for `sir_yes_sir` model input construction (`inputs.py`)."""

import pytest

from experiments.sir_yes_sir.inputs import EmbeddingMode, TokenType, build_sm_example
from yessir.special_tokens import SpecialToken


def test_example_1_step_0_and_step_1_layout():
    """Verify exact tracks for Example 1 (`alphabet = '0123'`, `x = '130'`)."""
    alphabet = b"0123"
    x = b"130"

    # k = 0 (empty output_history): [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] [S]
    ex0 = build_sm_example(alphabet, x)
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
        SpecialToken.STEP,
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
        TokenType.OUT,
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
        0,
    )

    # k = 1 (output_history = '2'): [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 [S]
    ex1 = build_sm_example(alphabet, x, output_history=b"2")
    assert ex1.input_ids[-2:] == (ord("2"), SpecialToken.STEP)
    assert ex1.token_type_ids is not None
    assert ex1.token_type_ids[-2:] == (TokenType.OUT, TokenType.OUT)
    assert ex1.position_ids is not None
    assert ex1.position_ids[-2:] == (0, 1)


def test_embedding_modes_attach_expected_tracks():
    alphabet = b"0123"
    x = b"130"
    history = b"20"

    ex_pe = build_sm_example(
        alphabet, x, history, embedding_mode=EmbeddingMode.TOKEN_TYPE_PE
    )
    assert ex_pe.token_type_ids is not None
    assert ex_pe.position_ids is not None

    ex_type_only = build_sm_example(
        alphabet, x, history, embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE
    )
    assert ex_type_only.input_ids == ex_pe.input_ids
    assert ex_type_only.token_type_ids == ex_pe.token_type_ids
    assert ex_type_only.position_ids is None

    ex_nope = build_sm_example(alphabet, x, history, embedding_mode=EmbeddingMode.NOPE)
    assert ex_nope.input_ids == ex_pe.input_ids
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_rejects_invalid_alphabet():
    with pytest.raises(ValueError, match="alphabet size must be between 2"):
        build_sm_example(b"0", b"0")

    with pytest.raises(ValueError, match="duplicates"):
        build_sm_example(b"001", b"0")

    with pytest.raises(ValueError, match="is not a symbol"):
        build_sm_example([SpecialToken.BOS, SpecialToken.SEP], [SpecialToken.BOS])


def test_rejects_empty_x():
    with pytest.raises(ValueError, match="at least 1 symbol"):
        build_sm_example(b"01", b"")


def test_rejects_output_history_too_long_for_x():
    with pytest.raises(IndexError, match="step index k=3"):
        build_sm_example(b"0123", b"130", output_history=b"201")


def test_rejects_symbols_not_in_alphabet():
    with pytest.raises(ValueError, match="input symbol .* not found in alphabet"):
        build_sm_example(b"0123", b"190")

    with pytest.raises(ValueError, match="output symbol .* not found in alphabet"):
        build_sm_example(b"0123", b"130", output_history=b"9")


def test_rejects_unknown_embedding_mode():
    with pytest.raises(ValueError, match="unknown embedding_mode"):
        build_sm_example(b"01", b"0", embedding_mode="invalid")  # ty: ignore[invalid-argument-type]
