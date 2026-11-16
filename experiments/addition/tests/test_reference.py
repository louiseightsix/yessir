"""Unit tests for the `addition` ground-truth reference oracle (`reference.py`)."""

import itertools
import random

import pytest

from experiments.addition.reference import ColumnStep, reference_add, reference_step
from yessir.alphabet import SymbolSet, create_alphabet
from yessir.special_tokens import SpecialToken
from yessir.vocab import TokenSequence


def _digits_to_int(digits: TokenSequence, alphabet: TokenSequence) -> int:
    """Decode an `LSD -> MSD` digit sequence in `alphabet` to a Python `int`."""
    rank = {sym: idx for idx, sym in enumerate(alphabet)}
    base = len(alphabet)
    value = 0
    for power, sym in enumerate(digits):
        value += rank[sym] * (base**power)
    return value


def test_spec_example_1_base_10_with_carry_overflow():
    """Example 1 from `spec/task.md`: 987 ('789') + 25 ('52') = 1012 ('2101') in base 10."""
    alphabet = SymbolSet.DECIMAL
    a = b"789"  # 987 (LSD -> MSD)
    b = b"52"  # 25  (LSD -> MSD)

    steps = reference_add(alphabet, a, b)
    assert steps == (
        ColumnStep(
            k=0,
            carry_in=ord("0"),
            sum_digit=ord("2"),
            carry_out=ord("1"),
            flag=SpecialToken.FLAG_CONTINUE,
        ),
        ColumnStep(
            k=1,
            carry_in=ord("1"),
            sum_digit=ord("1"),
            carry_out=ord("1"),
            flag=SpecialToken.FLAG_CONTINUE,
        ),
        ColumnStep(
            k=2,
            carry_in=ord("1"),
            sum_digit=ord("0"),
            carry_out=ord("1"),
            flag=SpecialToken.FLAG_CONTINUE,
        ),
        ColumnStep(
            k=3,
            carry_in=ord("1"),
            sum_digit=ord("1"),
            carry_out=ord("0"),
            flag=SpecialToken.FLAG_FINISHED,
        ),
    )

    # Individual `reference_step` calls match every step in `steps`.
    for step in steps:
        assert reference_step(alphabet, a, b, step.k, step.carry_in) == (
            step.sum_digit,
            step.carry_out,
            step.flag,
        )


def test_spec_example_2_arbitrary_symbols_base_3_no_overflow():
    """Example 2 from `spec/task.md`: alphabet=['q','#','7'], 5 ('7#') + 1 ('#') = 6 ('q7')."""
    alphabet = [ord("q"), ord("#"), ord("7")]
    a = [ord("7"), ord("#")]
    b = [ord("#")]

    steps = reference_add(alphabet, a, b)
    assert steps == (
        ColumnStep(
            k=0,
            carry_in=ord("q"),
            sum_digit=ord("q"),
            carry_out=ord("#"),
            flag=SpecialToken.FLAG_CONTINUE,
        ),
        ColumnStep(
            k=1,
            carry_in=ord("#"),
            sum_digit=ord("7"),
            carry_out=ord("q"),
            flag=SpecialToken.FLAG_FINISHED,
        ),
    )


def test_single_digit_zero_plus_zero_halts_immediately():
    """Single-digit 0 + 0 halts on k=0 with K = 0 (1 step)."""
    alphabet = b"ab"  # 'a' = 0, 'b' = 1
    steps = reference_add(alphabet, b"a", b"a")
    assert len(steps) == 1
    assert steps[0] == ColumnStep(
        k=0,
        carry_in=ord("a"),
        sum_digit=ord("a"),
        carry_out=ord("a"),
        flag=SpecialToken.FLAG_FINISHED,
    )


def test_single_digit_overflow_runs_second_column():
    """Single-digit 1 + 1 in base 2 overflows to k=1 (K = 1, 2 steps)."""
    alphabet = b"01"
    steps = reference_add(alphabet, b"1", b"1")
    assert [s.sum_digit for s in steps] == [ord("0"), ord("1")]
    assert [s.flag for s in steps] == [
        SpecialToken.FLAG_CONTINUE,
        SpecialToken.FLAG_FINISHED,
    ]


def test_msd_zeros_are_preserved_until_max_operand_length():
    """Leading zeros at the MSD end (`s_0`) still step through all `max(L_a, L_b)` columns."""
    alphabet = SymbolSet.DECIMAL
    # a = '500' (L_a = 3, value 5 with two MSD zeros), b = '2' (L_b = 1, value 2)
    steps = reference_add(alphabet, b"500", b"2")
    assert len(steps) == 3
    assert bytes(s.sum_digit for s in steps) == b"700"
    assert [s.flag for s in steps] == [
        SpecialToken.FLAG_CONTINUE,
        SpecialToken.FLAG_CONTINUE,
        SpecialToken.FLAG_FINISHED,
    ]


@pytest.mark.parametrize("base", [2, 3, 4, 8, 10, 12, 16, 37, 240])
def test_matches_integer_addition_across_bases_and_random_alphabets(base: int):
    """Oracle sum matches exact Python integer arithmetic across bases 2..240."""
    rng = random.Random(10_000 + base)
    for _ in range(40):
        alphabet = create_alphabet(SymbolSet.ALL, sample_size=base, rng=rng)
        len_a = rng.randint(1, 25)
        len_b = rng.randint(1, 25)
        a = tuple(rng.choice(alphabet) for _ in range(len_a))
        b = tuple(rng.choice(alphabet) for _ in range(len_b))

        steps = reference_add(alphabet, a, b)
        n = max(len_a, len_b)
        assert len(steps) in (n, n + 1)

        # Every intermediate step continues; only the final step halts.
        assert all(s.flag == SpecialToken.FLAG_CONTINUE for s in steps[:-1])
        assert steps[-1].flag == SpecialToken.FLAG_FINISHED
        assert steps[-1].carry_out == alphabet[0]

        # Carry chain consistency between adjacent steps.
        assert steps[0].carry_in == alphabet[0]
        for prev_step, next_step in itertools.pairwise(steps):
            assert next_step.k == prev_step.k + 1
            assert next_step.carry_in == prev_step.carry_out

        # Exact numerical agreement with Python big-int addition.
        expected_sum = _digits_to_int(a, alphabet) + _digits_to_int(b, alphabet)
        actual_sum = _digits_to_int([s.sum_digit for s in steps], alphabet)
        assert actual_sum == expected_sum

        # Overflow column k = n occurs iff sum >= B^n.
        if expected_sum >= base**n:
            assert len(steps) == n + 1
            assert steps[-1].sum_digit == alphabet[1]
        else:
            assert len(steps) == n


def test_full_carry_ripple_max_digits():
    """(B^L - 1) + 1 ripples a carry across every column and overflows at k = L."""
    alphabet = b"ABCD"  # base 4: A=0, B=1, C=2, D=3
    a = b"DDDDD"  # 4^5 - 1 = 1023
    b = b"B"  # 1
    steps = reference_add(alphabet, a, b)
    assert len(steps) == 6
    assert bytes(s.sum_digit for s in steps) == b"AAAAAB"  # 4^5 = 1024
    assert bytes(s.carry_out for s in steps) == b"BBBBBA"


@pytest.mark.parametrize("bad_k", [-1, 4, 10])
def test_rejects_out_of_bounds_k(bad_k: int):
    alphabet = SymbolSet.DECIMAL
    with pytest.raises(IndexError, match="Step index k=.* out of range"):
        reference_step(alphabet, b"789", b"52", k=bad_k, carry=ord("0"))


def test_rejects_invalid_carry_symbol():
    alphabet = SymbolSet.DECIMAL
    # '2' is in alphabet, but not a valid carry (must be s_0='0' or s_1='1')
    with pytest.raises(ValueError, match="Incoming carry .* must be s_0 .* or s_1"):
        reference_step(alphabet, b"789", b"52", k=0, carry=ord("2"))


def test_rejects_empty_operands():
    alphabet = SymbolSet.DECIMAL
    with pytest.raises(ValueError, match="Operand a must contain at least 1 symbol"):
        reference_add(alphabet, b"", b"1")
    with pytest.raises(ValueError, match="Operand b must contain at least 1 symbol"):
        reference_add(alphabet, b"1", b"")


def test_rejects_operand_symbol_outside_alphabet():
    alphabet = b"0123"
    with pytest.raises(ValueError, match="Operand a symbol .* not found in alphabet"):
        reference_add(alphabet, b"19", b"0")
    with pytest.raises(ValueError, match="Operand b symbol .* not found in alphabet"):
        reference_add(alphabet, b"10", b"9")


def test_rejects_invalid_alphabet():
    with pytest.raises(ValueError, match="alphabet size must be between 2"):
        reference_add([ord("0")], b"0", b"0")
    with pytest.raises(ValueError, match="duplicates"):
        reference_add(b"00", b"0", b"0")
    with pytest.raises(ValueError, match="not a symbol"):
        reference_add([int(SpecialToken.BOS), ord("0")], [ord("0")], [ord("0")])
