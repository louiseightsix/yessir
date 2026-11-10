"""Tests for vocabulary partitioning, token enums, and create_vocab."""

import random
from enum import IntEnum

import pytest

from yessir.vocab import (
    EXPERIMENT_SPECIAL_RANGE,
    SHARED_SPECIAL_RANGE,
    SPECIAL_RANGE,
    SYMBOL_RANGE,
    VOCAB_SIZE,
    ExperimentVocab,
    SpecialToken,
    SymbolSets,
    create_vocab,
)


class DummyExperimentSpecial(IntEnum):
    """Test special tokens allocated in the experiment special range (0xF8..0xFF)."""

    TRIGGER = 0xF8
    FLAG_A = 0xF9
    FLAG_B = 0xFA


class InvalidExperimentSpecial(IntEnum):
    """Test special token outside the experiment special range."""

    BAD_TOKEN = 0xF0  # In shared special range


def test_vocab_partition_and_ranges():
    assert VOCAB_SIZE == 256

    # 1. Symbol range
    assert len(SYMBOL_RANGE) == 240
    assert min(SYMBOL_RANGE) == 0x00
    assert max(SYMBOL_RANGE) == 0xEF

    # 2. Special ranges
    assert len(SPECIAL_RANGE) == 16
    assert min(SPECIAL_RANGE) == 0xF0
    assert max(SPECIAL_RANGE) == 0xFF

    assert len(SHARED_SPECIAL_RANGE) == 8
    assert min(SHARED_SPECIAL_RANGE) == 0xF0
    assert max(SHARED_SPECIAL_RANGE) == 0xF7

    assert len(EXPERIMENT_SPECIAL_RANGE) == 8
    assert min(EXPERIMENT_SPECIAL_RANGE) == 0xF8
    assert max(EXPERIMENT_SPECIAL_RANGE) == 0xFF

    # Disjointness and full coverage of 0x00..0xFF
    all_bytes = (
        set(SYMBOL_RANGE) | set(SHARED_SPECIAL_RANGE) | set(EXPERIMENT_SPECIAL_RANGE)
    )
    assert len(all_bytes) == 256
    assert all_bytes == set(range(256))


def test_symbol_sets():
    # ALL must cover all 240 symbol bytes exactly
    assert len(SymbolSets.ALL) == 240
    assert min(SymbolSets.ALL) == 0x00
    assert max(SymbolSets.ALL) == 0xEF
    assert len(set(SymbolSets.ALL)) == 240

    # DECIMAL and HEX must contain unique tokens within the symbol range
    assert SymbolSets.DECIMAL == b"0123456789"
    assert len(SymbolSets.DECIMAL) == 10

    assert SymbolSets.HEX == b"0123456789ABCDEF"
    assert len(SymbolSets.HEX) == 16
    assert SymbolSets.HEX[:10] == SymbolSets.DECIMAL


def test_special_tokens():
    # 7 defined shared special tokens in 0xF0..0xF7
    assert len(SpecialToken) == 7
    for tok in SpecialToken:
        assert tok in SHARED_SPECIAL_RANGE

    assert SpecialToken.DELIM == 0xF0
    assert SpecialToken.PAD == 0xF1
    assert SpecialToken.BOS == 0xF2
    assert SpecialToken.EOS == 0xF3
    assert SpecialToken.QUERY_STEP == 0xF4
    assert SpecialToken.NOTIFY_CONTINUE == 0xF5
    assert SpecialToken.NOTIFY_FINISHED == 0xF6


def test_create_vocab_exact_order():
    vocab = create_vocab(SymbolSets.DECIMAL[:4])

    assert isinstance(vocab, ExperimentVocab)
    assert vocab.symbols == tuple(b"0123")
    assert vocab.alphabet_size == 4
    assert vocab.vocab_size == 256

    # Special tokens should include all 7 shared special tokens by default
    assert vocab.special_tokens == tuple(int(tok) for tok in SpecialToken)


def test_create_vocab_with_experiment_specials():
    vocab = create_vocab(
        SymbolSets.HEX[:8],
        experiment_specials=DummyExperimentSpecial,
    )

    assert vocab.alphabet_size == 8
    # Shared specials (7) + Dummy experiment specials (3) = 10
    expected_specials = tuple(int(tok) for tok in SpecialToken) + (
        DummyExperimentSpecial.TRIGGER,
        DummyExperimentSpecial.FLAG_A,
        DummyExperimentSpecial.FLAG_B,
    )
    assert vocab.special_tokens == expected_specials


def test_create_vocab_permute_all():
    rng = random.Random(42)
    original = SymbolSets.DECIMAL
    vocab = create_vocab(original, rng=rng)

    assert vocab.alphabet_size == len(original)
    assert set(vocab.symbols) == set(original)
    # With seed 42, the shuffle should differ from original order
    assert vocab.symbols != tuple(original)

    # Determinism: same seed produces identical order
    vocab2 = create_vocab(original, rng=random.Random(42))
    assert vocab.symbols == vocab2.symbols


def test_create_vocab_sample_subset():
    rng = random.Random(99)
    vocab = create_vocab(SymbolSets.ALL, sample_size=10, rng=rng)

    assert vocab.alphabet_size == 10
    assert len(set(vocab.symbols)) == 10
    assert all(tok in SymbolSets.ALL for tok in vocab.symbols)

    # Determinism
    vocab2 = create_vocab(SymbolSets.ALL, sample_size=10, rng=random.Random(99))
    assert vocab.symbols == vocab2.symbols


def test_create_vocab_requires_rng_for_sampling():
    with pytest.raises(ValueError, match="explicit rng .* must be provided"):
        create_vocab(SymbolSets.ALL, sample_size=5)


def test_create_vocab_rejects_non_symbol():
    # 0xF0 is SpecialToken.DELIM, not a symbol
    with pytest.raises(ValueError, match="not within symbol range"):
        create_vocab([0x01, 0xF0, 0x02])


def test_create_vocab_rejects_duplicate_symbols():
    with pytest.raises(ValueError, match="duplicate tokens"):
        create_vocab([0x01, 0x02, 0x01])


def test_create_vocab_rejects_invalid_sample_size():
    rng = random.Random(0)
    # sample_size < 2
    with pytest.raises(ValueError, match="2 <= sample_size <= 10"):
        create_vocab(SymbolSets.DECIMAL, sample_size=1, rng=rng)

    # sample_size > len(symbols)
    with pytest.raises(ValueError, match="2 <= sample_size <= 10"):
        create_vocab(SymbolSets.DECIMAL, sample_size=11, rng=rng)


def test_create_vocab_rejects_invalid_experiment_special():
    # 0xF0 is in the shared range, not the experiment special range (0xF8..0xFF)
    with pytest.raises(ValueError, match="not within experiment special range"):
        create_vocab(
            SymbolSets.DECIMAL[:2],
            experiment_specials=InvalidExperimentSpecial,
        )
