"""Unit tests for the sir_yes_sir ground-truth reference rule."""

import pytest

from experiments.sir_yes_sir.reference import step
from yessir.vocab import SpecialToken


def test_example_1_canonical_base_4():
    """Example 1 from spec: axiom = '0123', x = '130'."""
    axiom = b"0123"
    x = b"130"

    # k = 0: x[0] = '1' -> o_0 = '2', w_0 = '0' (no wrap), flag = NOTIFY_CONTINUE
    o_0, w_0, flag_0 = step(axiom, x, k=0)
    assert o_0 == ord("2")
    assert w_0 == ord("0")
    assert flag_0 == SpecialToken.NOTIFY_CONTINUE

    # k = 1: x[1] = '3' -> o_1 = '0' (wrap to start), w_1 = '1' (wrap flag s_1), flag = NOTIFY_CONTINUE
    o_1, w_1, flag_1 = step(axiom, x, k=1)
    assert o_1 == ord("0")
    assert w_1 == ord("1")
    assert flag_1 == SpecialToken.NOTIFY_CONTINUE

    # k = 2: x[2] = '0' -> o_2 = '1', w_2 = '0' (no wrap), flag = NOTIFY_FINISHED (last step)
    o_2, w_2, flag_2 = step(axiom, x, k=2)
    assert o_2 == ord("1")
    assert w_2 == ord("0")
    assert flag_2 == SpecialToken.NOTIFY_FINISHED


def test_example_2_random_base_3():
    """Example 2 from spec: axiom = ['q', '#', '7'], x = ['#', '7', 'q', '7']."""
    axiom = [ord("q"), ord("#"), ord("7")]
    x = [ord("#"), ord("7"), ord("q"), ord("7")]

    # k = 0: x[0] = '#' -> o_0 = '7', w_0 = 'q' (s_0), flag = NOTIFY_CONTINUE
    assert step(axiom, x, k=0) == (ord("7"), ord("q"), SpecialToken.NOTIFY_CONTINUE)

    # k = 1: x[1] = '7' -> o_1 = 'q' (wrap), w_1 = '#' (s_1), flag = NOTIFY_CONTINUE
    assert step(axiom, x, k=1) == (ord("q"), ord("#"), SpecialToken.NOTIFY_CONTINUE)

    # k = 2: x[2] = 'q' -> o_2 = '#', w_2 = 'q' (s_0), flag = NOTIFY_CONTINUE
    assert step(axiom, x, k=2) == (ord("#"), ord("q"), SpecialToken.NOTIFY_CONTINUE)

    # k = 3: x[3] = '7' -> o_3 = 'q' (wrap), w_3 = '#' (s_1), flag = NOTIFY_FINISHED
    assert step(axiom, x, k=3) == (ord("q"), ord("#"), SpecialToken.NOTIFY_FINISHED)


def test_single_symbol_sequence():
    """An input of length 1 should immediately emit NOTIFY_FINISHED on k=0."""
    axiom = b"AB"
    x = b"A"
    o, w, flag = step(axiom, x, k=0)
    assert o == ord("B")
    assert w == ord("A")
    assert flag == SpecialToken.NOTIFY_FINISHED


@pytest.mark.parametrize("bad_k", [-1, 3, 10])
def test_rejects_out_of_bounds_k(bad_k: int):
    with pytest.raises(IndexError, match="Step index k=.* out of range"):
        step(b"0123", b"130", k=bad_k)


def test_rejects_axiom_too_small():
    with pytest.raises(ValueError, match="Axiom must contain at least 2 symbols"):
        step([ord("0")], [ord("0")], k=0)


def test_rejects_symbol_not_in_axiom():
    axiom = b"0123"
    x = b"190"  # '9' is not in axiom
    with pytest.raises(ValueError, match="Input symbol .* not found in axiom"):
        step(axiom, x, k=1)
