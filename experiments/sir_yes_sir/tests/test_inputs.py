"""Tests for `sir_yes_sir` model input construction (`inputs.py`)."""

import pytest

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    TokenType,
)
from experiments.sir_yes_sir.variants.ar.inputs import (
    StepOrder,
    build_ar_example,
    build_ar_labelled_example,
    build_ar_trace,
)
from experiments.sir_yes_sir.variants.sm_mh.inputs import build_sm_example
from yessir.dataset import IGNORE_INDEX
from yessir.special_tokens import SpecialToken
from yessir.vocab import TOKEN_TYPE_RANGE


def test_token_types_are_in_valid_range():
    for token_type in TokenType:
        assert token_type in TOKEN_TYPE_RANGE, f"{token_type.name} = {token_type}"


def test_example_1_step_0_and_step_1_layout():
    """Verify exact layout for Example 1 (`alphabet = '0123'`, `x = '130'`)."""
    alphabet = b"0123"
    x = b"130"

    # k = 0 (empty output_history): [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] [STEP]
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
        TokenType.OUTPUT,
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

    # k = 1 (output_history = '2'): [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 [STEP]
    ex1 = build_sm_example(alphabet, x, output_history=b"2")
    assert ex1.input_ids[-2:] == (ord("2"), SpecialToken.STEP)
    assert ex1.token_type_ids is not None
    assert ex1.token_type_ids[-2:] == (TokenType.OUTPUT, TokenType.OUTPUT)
    assert ex1.position_ids is not None
    assert ex1.position_ids[-2:] == (0, 1)


def test_embedding_modes_attach_expected_ids():
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
    with pytest.raises(ValueError, match="step index k=3"):
        build_sm_example(b"0123", b"130", output_history=b"201")


def test_rejects_symbols_not_in_alphabet():
    with pytest.raises(ValueError, match="input symbol .* not found in alphabet"):
        build_sm_example(b"0123", b"190")

    with pytest.raises(ValueError, match="output symbol .* not found in alphabet"):
        build_sm_example(b"0123", b"130", output_history=b"9")


def test_rejects_unknown_embedding_mode():
    with pytest.raises(ValueError, match="unknown embedding_mode"):
        build_sm_example(b"01", b"0", embedding_mode="invalid")  # ty: ignore[invalid-argument-type]


def test_ar_example_empty_and_with_trace():
    alphabet = b"0123"
    x = b"130"

    # Start of generation (empty trace): ends at second [SEP]
    ex_start = build_ar_example(alphabet, x)
    assert ex_start.input_ids == (
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
    assert ex_start.token_type_ids == (
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
    assert ex_start.position_ids == (0, 0, 1, 2, 3, 0, 0, 1, 2, 0)

    # With a trace appended in TokenType.OUTPUT
    trace = (ord("2"), ord("0"), SpecialToken.FLAG_CONTINUE)
    ex_trace = build_ar_example(alphabet, x, trace=trace)
    assert ex_trace.input_ids == ex_start.input_ids + trace
    assert ex_trace.token_type_ids == ex_start.token_type_ids + (
        TokenType.OUTPUT,
        TokenType.OUTPUT,
        TokenType.OUTPUT,
    )
    assert ex_trace.position_ids == ex_start.position_ids + (0, 1, 2)


def test_build_ar_trace_and_labelled_example_out_first_and_wrap_first():
    alphabet = b"0123"
    x = b"130"

    trace_out = build_ar_trace(alphabet, x, step_order=StepOrder.OUT_FIRST)
    assert trace_out == (
        ord("2"),
        ord("0"),
        int(SpecialToken.FLAG_CONTINUE),
        ord("0"),
        ord("1"),
        int(SpecialToken.FLAG_CONTINUE),
        ord("1"),
        ord("0"),
        int(SpecialToken.FLAG_FINISHED),
    )

    trace_wrap = build_ar_trace(alphabet, x, step_order=StepOrder.WRAP_FIRST)
    assert trace_wrap == (
        ord("0"),
        ord("2"),
        int(SpecialToken.FLAG_CONTINUE),
        ord("1"),
        ord("0"),
        int(SpecialToken.FLAG_CONTINUE),
        ord("0"),
        ord("1"),
        int(SpecialToken.FLAG_FINISHED),
    )

    labelled = build_ar_labelled_example(alphabet, x, step_order=StepOrder.OUT_FIRST)
    prefix_len = 1 + len(alphabet) + 1 + len(x) + 1  # [BOS] 0123 [SEP] 130 [SEP] = 10
    assert len(labelled.example) == prefix_len + len(trace_out) - 1
    assert len(labelled.labels) == len(labelled.example)
    # Prefix before the second [SEP] is masked with IGNORE_INDEX; from second [SEP] onward predicts trace_out
    assert labelled.labels[: prefix_len - 1] == (IGNORE_INDEX,) * (prefix_len - 1)
    assert labelled.labels[prefix_len - 1 :] == trace_out
    assert labelled.example.input_ids[prefix_len:] == trace_out[:-1]
