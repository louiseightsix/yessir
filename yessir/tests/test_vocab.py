"""Tests for the vocabulary size and how it is split into ranges."""

from yessir.vocab import (
    EXPERIMENT_SPECIAL_RANGE,
    MAX_TOKEN_TYPES,
    SHARED_SPECIAL_RANGE,
    SPECIAL_RANGE,
    SYMBOL_RANGE,
    TOKEN_TYPE_RANGE,
    VOCAB_SIZE,
)


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

    # 3. Token type range
    assert MAX_TOKEN_TYPES == 16
    assert list(TOKEN_TYPE_RANGE) == list(range(16))
