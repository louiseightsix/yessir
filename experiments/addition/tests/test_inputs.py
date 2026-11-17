"""Unit tests for `addition` shared input primitives and problem/step generators (`inputs.py`)."""

import argparse
import itertools

import pytest

from experiments.addition.inputs import (
    AdditionProblem,
    AdditionToken,
    EmbeddingMode,
    StepOrder,
    TokenType,
    add_common_dataset_args,
    build_addressed_prefix_example,
    build_prefix_example,
    encode_address,
    iter_problem_steps,
    iter_problems,
    make_example,
    validate_problem,
)
from experiments.addition.reference import reference_add
from yessir.alphabet import SymbolSet
from yessir.special_tokens import SpecialToken
from yessir.vocab import EXPERIMENT_SPECIAL_RANGE, TOKEN_TYPE_RANGE


def test_enums_fall_in_valid_vocab_ranges():
    for token_type in TokenType:
        assert token_type in TOKEN_TYPE_RANGE, f"{token_type.name} = {token_type}"
    for tok in AdditionToken:
        assert tok in EXPERIMENT_SPECIAL_RANGE, f"{tok.name} = 0x{tok:02X}"
    assert tuple(StepOrder) == (StepOrder.SUM_FIRST, StepOrder.CARRY_FIRST)


def test_encode_address_lsd_first_in_prompt_alphabet():
    """Verify `encode_address` matches the `addressed_ar.md` examples."""
    # Base 10: addr(0) = ('0',), addr(9) = ('9',), addr(24) = ('4', '2')
    dec = SymbolSet.DECIMAL
    assert encode_address(0, dec) == (ord("0"),)
    assert encode_address(9, dec) == (ord("9"),)
    assert encode_address(24, dec) == (ord("4"), ord("2"))
    assert encode_address(105, dec) == (ord("5"), ord("0"), ord("1"))

    # Base 2 with custom symbols ('q' = 0, '#' = 1): addr(5) = ('#', 'q', '#')
    bin_alpha = (ord("q"), ord("#"))
    assert encode_address(0, bin_alpha) == (ord("q"),)
    assert encode_address(1, bin_alpha) == (ord("#"),)
    assert encode_address(2, bin_alpha) == (ord("q"), ord("#"))
    assert encode_address(5, bin_alpha) == (ord("#"), ord("q"), ord("#"))


@pytest.mark.parametrize("bad_j", [-1, -10, True])
def test_encode_address_rejects_invalid_index(bad_j):
    with pytest.raises(ValueError, match="non-negative int"):
        encode_address(bad_j, SymbolSet.DECIMAL)


def test_encode_address_rejects_short_alphabet():
    with pytest.raises(ValueError, match="at least 2 symbols"):
        encode_address(0, b"0")


def test_build_prefix_example_layout_and_embedding_modes():
    """Verify `[BOS] alphabet [SEP] a [SEP] b [SEP]` layout for Example 2 (`task.md` §2)."""
    alphabet = [ord("q"), ord("#"), ord("7")]
    a = [ord("7"), ord("#")]
    b = [ord("#")]

    ex_pe = build_prefix_example(
        alphabet, a, b, embedding_mode=EmbeddingMode.TOKEN_TYPE_PE
    )
    assert ex_pe.input_ids == (
        SpecialToken.BOS,
        ord("q"),
        ord("#"),
        ord("7"),
        SpecialToken.SEP,
        ord("7"),
        ord("#"),
        SpecialToken.SEP,
        ord("#"),
        SpecialToken.SEP,
    )
    assert ex_pe.token_type_ids == (
        TokenType.SPECIAL,
        TokenType.ALPHABET,
        TokenType.ALPHABET,
        TokenType.ALPHABET,
        TokenType.SPECIAL,
        TokenType.OPERAND_A,
        TokenType.OPERAND_A,
        TokenType.SPECIAL,
        TokenType.OPERAND_B,
        TokenType.SPECIAL,
    )
    assert ex_pe.position_ids == (
        0,
        0,
        1,
        2,
        0,
        0,
        1,
        0,
        0,
        0,
    )

    ex_type_only = build_prefix_example(
        alphabet, a, b, embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE
    )
    assert ex_type_only.input_ids == ex_pe.input_ids
    assert ex_type_only.token_type_ids == ex_pe.token_type_ids
    assert ex_type_only.position_ids is None

    ex_nope = build_prefix_example(alphabet, a, b, embedding_mode=EmbeddingMode.NOPE)
    assert ex_nope.input_ids == ex_pe.input_ids
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_build_addressed_prefix_with_and_without_alphabet():
    """Verify addressed prefix layout with multi-digit addresses and zero-preamble mode."""
    # Base 2 ('q' = 0, '#' = 1). L_a = 3 > B = 2, so column 2 has the 2-digit address ('q', '#').
    alphabet = (ord("q"), ord("#"))
    a = (ord("#"), ord("q"), ord("#"))  # L_a = 3: j=0 ('q'), j=1 ('#'), j=2 ('q', '#')
    b = (ord("#"), ord("#"))  # L_b = 2: j=0 ('q'), j=1 ('#')

    ex_with = build_addressed_prefix_example(alphabet, a, b, include_alphabet=True)
    IDX = int(AdditionToken.IDX)
    assert ex_with.input_ids == (
        SpecialToken.BOS,
        ord("q"),
        ord("#"),
        SpecialToken.SEP,
        # a_0 at j=0 (addr = 'q'), a_1 at j=1 (addr = '#'), a_2 at j=2 (addr = 'q', '#')
        IDX,
        ord("q"),
        ord("#"),
        IDX,
        ord("#"),
        ord("q"),
        IDX,
        ord("q"),
        ord("#"),
        ord("#"),
        SpecialToken.SEP,
        # b_0 at j=0, b_1 at j=1
        IDX,
        ord("q"),
        ord("#"),
        IDX,
        ord("#"),
        ord("#"),
        SpecialToken.SEP,
    )
    # Multi-digit address tokens at j=2 all share position_id = 2 and TokenType.ADDRESS.
    assert ex_with.token_type_ids == (
        TokenType.SPECIAL,
        TokenType.ALPHABET,
        TokenType.ALPHABET,
        TokenType.SPECIAL,
        TokenType.SPECIAL,
        TokenType.ADDRESS,
        TokenType.OPERAND_A,
        TokenType.SPECIAL,
        TokenType.ADDRESS,
        TokenType.OPERAND_A,
        TokenType.SPECIAL,
        TokenType.ADDRESS,
        TokenType.ADDRESS,
        TokenType.OPERAND_A,
        TokenType.SPECIAL,
        TokenType.SPECIAL,
        TokenType.ADDRESS,
        TokenType.OPERAND_B,
        TokenType.SPECIAL,
        TokenType.ADDRESS,
        TokenType.OPERAND_B,
        TokenType.SPECIAL,
    )
    assert ex_with.position_ids == (
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
    )

    # Zero-preamble mode (`include_alphabet=False`) omits `q # [SEP]` right after `[BOS]`.
    ex_zero = build_addressed_prefix_example(alphabet, a, b, include_alphabet=False)
    assert ex_zero.input_ids == (SpecialToken.BOS, *ex_with.input_ids[4:])
    assert ex_zero.token_type_ids == (TokenType.SPECIAL, *ex_with.token_type_ids[4:])
    assert ex_zero.position_ids == (0, *ex_with.position_ids[4:])


def test_validate_problem_enforces_zero_preamble_min_length():
    alphabet = b"012"  # B = 3
    with pytest.raises(ValueError, match="Zero-preamble mode .* lengths >= base"):
        validate_problem(alphabet, b"01", b"012", include_alphabet=False)
    with pytest.raises(ValueError, match="Zero-preamble mode .* lengths >= base"):
        validate_problem(alphabet, b"012", b"01", include_alphabet=False)


def test_make_example_catches_length_mismatch_even_in_nope():
    with pytest.raises(ValueError, match="Mismatched sequence lengths"):
        make_example([0, 1], [TokenType.SPECIAL], [0, 1], EmbeddingMode.NOPE)
    with pytest.raises(ValueError, match="Unknown embedding_mode"):
        make_example([0], [TokenType.SPECIAL], [0], "bad")  # ty: ignore[invalid-argument-type]


def test_iter_problems_determinism_bases_lengths_and_zero_preamble():
    stream_a = list(
        itertools.islice(
            iter_problems(
                bases=(2, 4, 8),
                min_length=2,
                max_length=10,
                include_alphabet=False,
                seed=123,
            ),
            50,
        )
    )
    stream_b = list(
        itertools.islice(
            iter_problems(
                bases=(2, 4, 8),
                min_length=2,
                max_length=10,
                include_alphabet=False,
                seed=123,
            ),
            50,
        )
    )
    assert stream_a == stream_b
    for prob in stream_a:
        base = len(prob.alphabet)
        assert base in (2, 4, 8)
        # Zero-preamble forces L_a, L_b >= base
        assert max(2, base) <= len(prob.a) <= 10
        assert max(2, base) <= len(prob.b) <= 10


def test_prior_mixture_and_permuted_canonical_alphabets():
    # p_prior = 1.0 always emits canonical ascending HEX[:B]
    canonical_probs = list(
        itertools.islice(iter_problems(bases=(10, 12), p_prior=1.0, seed=7), 20)
    )
    for prob in canonical_probs:
        assert prob.alphabet == tuple(SymbolSet.HEX[: len(prob.alphabet)])

    # permuted_canonical = True uses the exact same canonical symbol set, shuffled
    perm_probs = list(
        itertools.islice(
            iter_problems(bases=(10,), permuted_canonical=True, seed=7), 20
        )
    )
    canonical_10 = tuple(SymbolSet.DECIMAL)
    assert all(set(p.alphabet) == set(canonical_10) for p in perm_probs)
    assert all(p.alphabet != canonical_10 for p in perm_probs)

    # A shuffle of 2 symbols returns the ascending order half the time; the
    # permuted alphabet must never be the canonical one, so base 2 is always '10'.
    perm_base_2 = list(
        itertools.islice(iter_problems(bases=(2,), permuted_canonical=True, seed=7), 50)
    )
    assert all(p.alphabet == (ord("1"), ord("0")) for p in perm_base_2)


def test_carry_chain_injection_produces_multi_column_carries():
    """With `p_chain=1.0` and `min_length >= 4` in base 10, every problem has a >=2-step carry chain."""
    probs = list(
        itertools.islice(
            iter_problems(
                bases=(10,), min_length=4, max_length=6, p_chain=1.0, seed=99
            ),
            30,
        )
    )
    for prob in probs:
        steps = reference_add(prob.alphabet, prob.a, prob.b)
        s_1 = prob.alphabet[1]
        # At least 2 consecutive columns (start i and propagated i + 1) must emit carry_out == s_1.
        has_consecutive_carries = any(
            s_prev.carry_out == s_1 and s_curr.carry_out == s_1
            for s_prev, s_curr in itertools.pairwise(steps)
        )
        assert has_consecutive_carries


def test_iter_problem_steps_decouples_problem_stream_and_oversamples_terminal():
    """Changing `p_terminal` or `all_steps` preserves the exact underlying problem sequence."""
    base_problems = list(
        itertools.islice(
            iter_problems(bases=(4,), min_length=5, max_length=5, seed=55), 40
        )
    )
    steps_p0 = list(
        itertools.islice(
            iter_problem_steps(
                bases=(4,), min_length=5, max_length=5, p_terminal=0.0, seed=55
            ),
            40,
        )
    )
    steps_p1 = list(
        itertools.islice(
            iter_problem_steps(
                bases=(4,), min_length=5, max_length=5, p_terminal=1.0, seed=55
            ),
            40,
        )
    )
    assert [p for p, _, _ in steps_p0] == base_problems
    assert [p for p, _, _ in steps_p1] == base_problems
    # p_terminal = 1.0 always selects k = K (the final FLAG_FINISHED step)
    assert all(k == len(steps) - 1 for _, steps, k in steps_p1)
    assert any(k < len(steps) - 1 for _, steps, k in steps_p0)

    # all_steps = True yields 0 .. K sequentially for each problem
    expected_first = reference_add(
        base_problems[0].alphabet, base_problems[0].a, base_problems[0].b
    )
    seq_items = list(
        itertools.islice(
            iter_problem_steps(
                bases=(4,), min_length=5, max_length=5, all_steps=True, seed=55
            ),
            len(expected_first),
        )
    )
    assert [k for _, _, k in seq_items] == list(range(len(expected_first)))
    assert all(
        isinstance(p, AdditionProblem) and p == base_problems[0]
        for p, _, _ in seq_items
    )


def test_iter_problems_eagerly_rejects_invalid_sampling_params():
    with pytest.raises(ValueError, match="at least 1 base"):
        iter_problems(bases=())
    with pytest.raises(ValueError, match="duplicates"):
        iter_problems(bases=(2, 2))
    with pytest.raises(ValueError, match="p_prior must be in 0.0..1.0"):
        iter_problems(p_prior=1.5)
    with pytest.raises(ValueError, match="cannot be combined with p_prior"):
        iter_problems(permuted_canonical=True, p_prior=0.5)
    with pytest.raises(
        ValueError, match="Canonical numerals .* support bases up to 16"
    ):
        iter_problems(bases=(20,), p_prior=0.1)
    with pytest.raises(ValueError, match="include_alphabet=False"):
        iter_problems(bases=(10,), max_length=8, include_alphabet=False)
    with pytest.raises(ValueError, match="p_terminal must be in 0.0..1.0"):
        iter_problem_steps(p_terminal=-0.1)
    with pytest.raises(ValueError, match="min_length must be >= 1"):
        iter_problems(min_length=0)
    with pytest.raises(ValueError, match="must be >= min_length"):
        iter_problems(min_length=5, max_length=4)
    with pytest.raises(ValueError, match="p_chain must be in 0.0..1.0"):
        iter_problems(p_chain=1.5)
    with pytest.raises(ValueError, match="exceeds symbol_pool size"):
        iter_problems(bases=(5,), symbols=b"0123")


@pytest.mark.parametrize("bad_length", [1.5, "3", True])
def test_iter_problems_eagerly_rejects_non_int_lengths(bad_length):
    with pytest.raises(TypeError, match="min_length must be an int"):
        iter_problems(min_length=bad_length)
    with pytest.raises(TypeError, match="max_length must be an int"):
        iter_problems(max_length=bad_length)


def test_add_common_dataset_args_defaults_and_parsing():
    parser = argparse.ArgumentParser()
    add_common_dataset_args(parser)

    defaults = parser.parse_args([])
    assert defaults.bases == [2, 4, 8, 10]
    assert defaults.min_length == 1
    assert defaults.max_length == 20
    assert defaults.p_prior == 0.0
    assert defaults.permuted_canonical is False
    assert defaults.p_chain == 0.25
    assert defaults.embedding_mode is EmbeddingMode.TOKEN_TYPE_PE
    assert defaults.seed == 42

    args = parser.parse_args(
        [
            "--bases", "3", "12",
            "--min-length", "21",
            "--max-length", "40",
            "--p-prior", "0.5",
            "--permuted-canonical",
            "--p-chain", "0.0",
            "--embedding-mode", "nope",
            "--seed", "7",
        ]
    )  # fmt: skip
    assert args.bases == [3, 12]
    assert args.min_length == 21
    assert args.max_length == 40
    assert args.p_prior == 0.5
    assert args.permuted_canonical is True
    assert args.p_chain == 0.0
    assert args.embedding_mode is EmbeddingMode.NOPE
    assert args.seed == 7

    with pytest.raises(SystemExit):
        parser.parse_args(["--embedding-mode", "pe"])
