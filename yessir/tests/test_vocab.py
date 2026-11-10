"""Tests for vocabulary partitioning, token enums, and create_vocab."""

import random
from enum import IntEnum

import pytest

from yessir.vocab import (
    ExperimentVocab,
    LiteralTokenSets,
    SharedControlToken,
    create_vocab,
)


class DummyExperimentControl(IntEnum):
    """Test control tokens allocated in the experiment control range (0xF8..0xFF)."""

    TRIGGER = 0xF8
    FLAG_A = 0xF9
    FLAG_B = 0xFA


class InvalidExperimentControl(IntEnum):
    """Test control token outside the experiment control range."""

    BAD_TOKEN = 0xF0  # In shared control range


def test_literal_token_sets():
    # ALL must cover all 240 literal bytes exactly
    assert len(LiteralTokenSets.ALL) == 240
    assert min(LiteralTokenSets.ALL) == 0x00
    assert max(LiteralTokenSets.ALL) == 0xEF
    assert len(set(LiteralTokenSets.ALL)) == 240

    # DECIMAL and HEX must contain unique tokens within the literal range
    assert LiteralTokenSets.DECIMAL == b"0123456789"
    assert len(LiteralTokenSets.DECIMAL) == 10

    assert LiteralTokenSets.HEX == b"0123456789ABCDEF"
    assert len(LiteralTokenSets.HEX) == 16
    assert LiteralTokenSets.HEX[:10] == LiteralTokenSets.DECIMAL


def test_shared_control_tokens():
    # 7 defined shared tokens in 0xF0..0xF7
    assert len(SharedControlToken) == 7
    for tok in SharedControlToken:
        assert 0xF0 <= tok <= 0xF7

    assert SharedControlToken.DELIM == 0xF0
    assert SharedControlToken.PAD == 0xF1
    assert SharedControlToken.BOS == 0xF2
    assert SharedControlToken.EOS == 0xF3
    assert SharedControlToken.QUERY_STEP == 0xF4
    assert SharedControlToken.NOTIFY_CONTINUE == 0xF5
    assert SharedControlToken.NOTIFY_FINISHED == 0xF6


def test_create_vocab_exact_order():
    vocab = create_vocab(LiteralTokenSets.DECIMAL[:4])

    assert isinstance(vocab, ExperimentVocab)
    assert vocab.literals == tuple(b"0123")
    assert vocab.alphabet_size == 4
    assert vocab.model_vocab_size == 256

    # Control tokens should include all 4 shared control tokens by default
    assert vocab.control_tokens == tuple(int(tok) for tok in SharedControlToken)


def test_create_vocab_with_experiment_controls():
    vocab = create_vocab(
        LiteralTokenSets.HEX[:8],
        experiment_controls=DummyExperimentControl,
    )

    assert vocab.alphabet_size == 8
    # Shared controls (4) + Dummy experiment controls (3) = 7
    expected_controls = tuple(int(tok) for tok in SharedControlToken) + (
        DummyExperimentControl.TRIGGER,
        DummyExperimentControl.FLAG_A,
        DummyExperimentControl.FLAG_B,
    )
    assert vocab.control_tokens == expected_controls


def test_create_vocab_permute_all():
    rng = random.Random(42)
    original = LiteralTokenSets.DECIMAL
    vocab = create_vocab(original, rng=rng)

    assert vocab.alphabet_size == len(original)
    assert set(vocab.literals) == set(original)
    # With seed 42, the shuffle should differ from original order
    assert vocab.literals != tuple(original)

    # Determinism: same seed produces identical order
    vocab2 = create_vocab(original, rng=random.Random(42))
    assert vocab.literals == vocab2.literals


def test_create_vocab_sample_subset():
    rng = random.Random(99)
    vocab = create_vocab(LiteralTokenSets.ALL, sample_size=10, rng=rng)

    assert vocab.alphabet_size == 10
    assert len(set(vocab.literals)) == 10
    assert all(tok in LiteralTokenSets.ALL for tok in vocab.literals)

    # Determinism
    vocab2 = create_vocab(LiteralTokenSets.ALL, sample_size=10, rng=random.Random(99))
    assert vocab.literals == vocab2.literals


def test_create_vocab_requires_rng_for_sampling():
    with pytest.raises(ValueError, match="explicit rng .* must be provided"):
        create_vocab(LiteralTokenSets.ALL, sample_size=5)


def test_create_vocab_rejects_non_literal():
    # 0xF0 is SharedControlToken.DELIM, not a literal
    with pytest.raises(ValueError, match="not within literal range"):
        create_vocab([0x01, 0xF0, 0x02])


def test_create_vocab_rejects_duplicate_literals():
    with pytest.raises(ValueError, match="duplicate tokens"):
        create_vocab([0x01, 0x02, 0x01])


def test_create_vocab_rejects_invalid_sample_size():
    rng = random.Random(0)
    # sample_size < 2
    with pytest.raises(ValueError, match="2 <= sample_size <= 10"):
        create_vocab(LiteralTokenSets.DECIMAL, sample_size=1, rng=rng)

    # sample_size > len(literals)
    with pytest.raises(ValueError, match="2 <= sample_size <= 10"):
        create_vocab(LiteralTokenSets.DECIMAL, sample_size=11, rng=rng)


def test_create_vocab_rejects_invalid_experiment_control():
    # 0xF0 is in the shared range, not the experiment control range (0xF8..0xFF)
    with pytest.raises(ValueError, match="not within experiment control range"):
        create_vocab(
            LiteralTokenSets.DECIMAL[:2],
            experiment_controls=InvalidExperimentControl,
        )
