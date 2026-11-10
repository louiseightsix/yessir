"""Tests for the named symbol sets and create_alphabet."""

import random

import pytest

from yessir.alphabet import SymbolSet, create_alphabet
from yessir.vocab import SYMBOL_RANGE


def test_all_is_the_whole_symbol_pool():
    assert list(SymbolSet.ALL) == list(SYMBOL_RANGE)


@pytest.mark.parametrize("symbols", [SymbolSet.DECIMAL, SymbolSet.HEX])
def test_named_sets_are_valid_alphabets(symbols):
    # create_alphabet checks the symbols are distinct and in the pool.
    assert create_alphabet(symbols) == tuple(symbols)


def test_keeps_the_given_order_without_rng():
    assert create_alphabet(SymbolSet.DECIMAL[:4]) == tuple(b"0123")


def test_shuffles_all_symbols_with_rng():
    alphabet = create_alphabet(SymbolSet.DECIMAL, rng=random.Random(42))
    assert sorted(alphabet) == sorted(SymbolSet.DECIMAL)
    assert alphabet != tuple(SymbolSet.DECIMAL)


def test_samples_distinct_symbols_from_the_pool():
    alphabet = create_alphabet(SymbolSet.ALL, sample_size=10, rng=random.Random(99))
    assert len(set(alphabet)) == 10
    assert all(symbol in SYMBOL_RANGE for symbol in alphabet)


@pytest.mark.parametrize("sample_size", [None, 5])
def test_same_seed_gives_the_same_alphabet(sample_size):
    def make(seed):
        return create_alphabet(
            SymbolSet.HEX, sample_size=sample_size, rng=random.Random(seed)
        )

    assert make(7) == make(7)


def test_sampling_needs_an_explicit_rng():
    with pytest.raises(ValueError, match="explicit rng"):
        create_alphabet(SymbolSet.ALL, sample_size=5)


@pytest.mark.parametrize("bad", [0xF0, -1, 1.0, "0"])
def test_rejects_anything_that_is_not_a_symbol(bad):
    with pytest.raises(ValueError, match="is not a symbol"):
        create_alphabet([0x01, bad, 0x02])


def test_rejects_duplicates_whichever_symbols_are_drawn():
    # A duplicate must be caught even when the sample might not include it.
    for seed in range(20):
        with pytest.raises(ValueError, match="duplicates"):
            create_alphabet([1, 1, 2, 3], sample_size=2, rng=random.Random(seed))


@pytest.mark.parametrize("symbols", [[], [0x05]])
def test_rejects_fewer_than_two_symbols_without_sampling(symbols):
    with pytest.raises(ValueError, match="alphabet size must be between 2"):
        create_alphabet(symbols)


@pytest.mark.parametrize("sample_size", [0, 1, 11])
def test_rejects_sample_size_out_of_range(sample_size):
    with pytest.raises(ValueError, match="alphabet size must be between 2 and 10"):
        create_alphabet(
            SymbolSet.DECIMAL, sample_size=sample_size, rng=random.Random(0)
        )


@pytest.mark.parametrize("sample_size", [2, 10])
def test_accepts_sample_size_at_the_limits(sample_size):
    alphabet = create_alphabet(
        SymbolSet.DECIMAL, sample_size=sample_size, rng=random.Random(0)
    )
    assert len(alphabet) == sample_size
