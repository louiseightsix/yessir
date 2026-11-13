"""Unit tests for the `sir_yes_sir` ground-truth reference rule (`reference.py`)."""

import pytest

from experiments.sir_yes_sir.reference import reference_step
from yessir.special_tokens import SpecialToken


def test_example_1_canonical_base_4():
    """Example 1 from spec: alphabet = '0123', x = '130' -> o = '201'."""
    alphabet = b"0123"
    x = b"130"

    # k = 0: x[0] = '1' -> o_0 = '2', flag = FLAG_CONTINUE
    o_0, flag_0 = reference_step(alphabet, x, k=0)
    assert o_0 == ord("2")
    assert flag_0 == SpecialToken.FLAG_CONTINUE

    # k = 1: x[1] = '3' -> o_1 = '0' (wrap to start), flag = FLAG_CONTINUE
    o_1, flag_1 = reference_step(alphabet, x, k=1)
    assert o_1 == ord("0")
    assert flag_1 == SpecialToken.FLAG_CONTINUE

    # k = 2: x[2] = '0' -> o_2 = '1', flag = FLAG_FINISHED (last step)
    o_2, flag_2 = reference_step(alphabet, x, k=2)
    assert o_2 == ord("1")
    assert flag_2 == SpecialToken.FLAG_FINISHED


def test_example_2_random_base_3():
    """Example 2 from spec: alphabet = ['q', '#', '7'], x = ['#', '7', 'q', '7'] -> o = ['7', 'q', '#', 'q']."""
    alphabet = [ord("q"), ord("#"), ord("7")]
    x = [ord("#"), ord("7"), ord("q"), ord("7")]

    # k = 0: x[0] = '#' -> o_0 = '7', flag = FLAG_CONTINUE
    assert reference_step(alphabet, x, k=0) == (
        ord("7"),
        SpecialToken.FLAG_CONTINUE,
    )

    # k = 1: x[1] = '7' -> o_1 = 'q' (wrap), flag = FLAG_CONTINUE
    assert reference_step(alphabet, x, k=1) == (
        ord("q"),
        SpecialToken.FLAG_CONTINUE,
    )

    # k = 2: x[2] = 'q' -> o_2 = '#', flag = FLAG_CONTINUE
    assert reference_step(alphabet, x, k=2) == (
        ord("#"),
        SpecialToken.FLAG_CONTINUE,
    )

    # k = 3: x[3] = '7' -> o_3 = 'q' (wrap), flag = FLAG_FINISHED
    assert reference_step(alphabet, x, k=3) == (
        ord("q"),
        SpecialToken.FLAG_FINISHED,
    )


def test_single_symbol_sequence():
    """An input of length 1 should immediately emit FLAG_FINISHED on k=0."""
    alphabet = b"AB"
    x = b"A"
    o, flag = reference_step(alphabet, x, k=0)
    assert o == ord("B")
    assert flag == SpecialToken.FLAG_FINISHED


@pytest.mark.parametrize("bad_k", [-1, 3, 10])
def test_rejects_out_of_bounds_k(bad_k: int):
    with pytest.raises(IndexError, match="Step index k=.* out of range"):
        reference_step(b"0123", b"130", k=bad_k)


def test_rejects_alphabet_too_small():
    with pytest.raises(ValueError, match="Alphabet must contain at least 2 symbols"):
        reference_step([ord("0")], [ord("0")], k=0)


def test_rejects_symbol_not_in_alphabet():
    alphabet = b"0123"
    x = b"190"  # '9' is not in alphabet
    with pytest.raises(ValueError, match="Input symbol .* not found in alphabet"):
        reference_step(alphabet, x, k=1)
