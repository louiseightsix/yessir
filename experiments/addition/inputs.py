"""Shared model input primitives, special tokens, address encoding, and problem/step generators for `addition`."""

import argparse
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum, unique

from experiments.addition.reference import (
    ColumnStep,
    _validate_inputs,
    reference_add,
)
from yessir.alphabet import SymbolSet, create_alphabet
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence, TokenTypeId


@unique
class TokenType(IntEnum):
    """Section identifiers (`token_type_ids` in `0..15`) across all `addition` variants (`spec/task.md` §2)."""

    SPECIAL = (
        0  # Delimiters ([BOS], [SEP], [EOS], [IDX]) and query tokens ([S], [C], [F])
    )
    ALPHABET = 1  # Alphabet symbols s_0 .. s_{B-1}
    OPERAND_A = 2  # Operand a digits a_0 .. a_{L_a-1}
    OPERAND_B = 3  # Operand b digits b_0 .. b_{L_b-1}
    CARRY = 4  # Incoming carry c_k (at k) and outgoing carry c_{k+1} (at k + 1)
    OUTPUT = 5  # Emitted sum digits o_j
    ADDRESS = 6  # Base-B column address digits addr(j) in addressed variants


@unique
class AdditionToken(IntEnum):
    """Experiment-specific special tokens in `vocab.EXPERIMENT_SPECIAL_RANGE` (`0xF8..0xFF`)."""

    QUERY_SUM = 0xF8  # [S]: queries or marks the sum digit o_k
    QUERY_CARRY = 0xF9  # [C]: queries or marks the outgoing carry c_{k+1}
    QUERY_FLAG = 0xFA  # [F]: queries the halting flag flag_k
    IDX = 0xFB  # [IDX]: opens a base-B column-addressed entry [IDX] addr(j) ...


@unique
class EmbeddingMode(StrEnum):
    """Which of `token_type_ids` and `position_ids` are attached to an `Example` (`spec/task.md` §2)."""

    NOPE = "nope"  # input_ids only (token_type_ids=None, position_ids=None)
    TOKEN_TYPE_NOPE = (
        "token_type_nope"  # input_ids + token_type_ids (position_ids=None)
    )
    TOKEN_TYPE_PE = "token_type_pe"  # input_ids + token_type_ids + position_ids


@unique
class StepOrder(StrEnum):
    """Within-column emission order for autoregressive variants (`step_ar`, `ar`, `addressed_*`)."""

    SUM_FIRST = "sum_first"  # Emits o_k before c_{k+1}
    CARRY_FIRST = "carry_first"  # Emits c_{k+1} before o_k


@dataclass(frozen=True)
class AdditionProblem:
    """One validated or sampled addition problem instance `(alphabet, a, b)`."""

    alphabet: tuple[Token, ...]
    a: tuple[Token, ...]
    b: tuple[Token, ...]


def encode_address(j: int, alphabet: TokenSequence) -> tuple[Token, ...]:
    """Encode non-negative column index `j` as an `LSD -> MSD` base-`B` numeral in `alphabet`.

    Follows `spec/variants/addressed_ar.md`:
    - `encode_address(0, alphabet) == (s_0,)`
    - For `j > 0`, returns the non-empty tuple `(s_{d_0}, s_{d_1}, ...)` where
      `j = sum(d_p * B**p)` (`0 <= j < B` gives `(s_j,)`; `B <= j < B**2` gives
      `(s_{j % B}, s_{j // B})`).
    """
    if isinstance(j, bool) or not isinstance(j, int) or j < 0:
        raise ValueError(f"Column index j must be a non-negative int, got {j!r}")
    base = len(alphabet)
    if base < 2:
        raise ValueError(f"Alphabet must contain at least 2 symbols, got {base}")

    if j == 0:
        return (alphabet[0],)

    digits: list[Token] = []
    remaining = j
    while remaining > 0:
        digits.append(alphabet[remaining % base])
        remaining //= base
    return tuple(digits)


def validate_problem(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    include_alphabet: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate `(alphabet, a, b)` and return all three as tuples of token ints.

    When `include_alphabet=False` (zero-preamble mode in `addressed_step_ar` and
    `addressed_ar`), both operands must have length `>= len(alphabet)` so all `B`
    symbols appear in ascending order as single-digit addresses `addr(0)..addr(B-1)`.
    """
    a_tokens = tuple(a)
    b_tokens = tuple(b)
    # The alphabet and operand checks are the oracle's, so a problem that builds
    # a prefix is always one `reference_add` accepts.
    alphabet_tokens, _ = _validate_inputs(alphabet, a_tokens, b_tokens)

    if not include_alphabet:
        base = len(alphabet_tokens)
        if len(a_tokens) < base or len(b_tokens) < base:
            raise ValueError(
                f"Zero-preamble mode (include_alphabet=False) requires operand "
                f"lengths >= base ({base}), got ({len(a_tokens)}, {len(b_tokens)})"
            )

    return alphabet_tokens, a_tokens, b_tokens


def make_example(
    input_ids: Sequence[Token],
    token_type_ids: Sequence[TokenTypeId],
    position_ids: Sequence[int],
    embedding_mode: EmbeddingMode,
) -> Example:
    """Assemble an `Example` from aligned `(input_ids, token_type_ids, position_ids)` for `embedding_mode`.

    Length alignment across all three sequences is checked unconditionally before
    dropping slices for `TOKEN_TYPE_NOPE` or `NOPE`, so coordinate-construction
    bugs are caught regardless of the active embedding mode.
    """
    n_tokens = len(input_ids)
    if len(token_type_ids) != n_tokens or len(position_ids) != n_tokens:
        raise ValueError(
            f"Mismatched sequence lengths: input_ids={n_tokens}, "
            f"token_type_ids={len(token_type_ids)}, position_ids={len(position_ids)}"
        )

    if embedding_mode == EmbeddingMode.NOPE:
        return Example(input_ids=tuple(input_ids))
    if embedding_mode == EmbeddingMode.TOKEN_TYPE_NOPE:
        return Example(
            input_ids=tuple(input_ids),
            token_type_ids=tuple(token_type_ids),
        )
    if embedding_mode == EmbeddingMode.TOKEN_TYPE_PE:
        return Example(
            input_ids=tuple(input_ids),
            token_type_ids=tuple(token_type_ids),
            position_ids=tuple(position_ids),
        )
    raise ValueError(f"Unknown embedding_mode: {embedding_mode!r}")


def build_prefix_tokens(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
) -> tuple[list[int], list[TokenTypeId], list[int]]:
    """Build mutable `(input_ids, token_type_ids, position_ids)` lists for `[BOS] alphabet [SEP] a [SEP] b [SEP]`.

    Shared by `cls`, `step_fork`, `step_ar`, and `ar` (`spec/task.md` §2).
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(alphabet, a, b)

    input_ids: list[int] = [
        int(SpecialToken.BOS),
        *alphabet_tokens,
        int(SpecialToken.SEP),
        *a_tokens,
        int(SpecialToken.SEP),
        *b_tokens,
        int(SpecialToken.SEP),
    ]
    token_type_ids: list[TokenTypeId] = [
        TokenType.SPECIAL,
        *([TokenType.ALPHABET] * len(alphabet_tokens)),
        TokenType.SPECIAL,
        *([TokenType.OPERAND_A] * len(a_tokens)),
        TokenType.SPECIAL,
        *([TokenType.OPERAND_B] * len(b_tokens)),
        TokenType.SPECIAL,
    ]
    position_ids: list[int] = [
        0,
        *range(len(alphabet_tokens)),
        0,
        *range(len(a_tokens)),
        0,
        *range(len(b_tokens)),
        0,
    ]
    return input_ids, token_type_ids, position_ids


def build_prefix_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build the shared unaddressed prefix `Example` (`[BOS] alphabet [SEP] a [SEP] b [SEP]`)."""
    input_ids, token_type_ids, position_ids = build_prefix_tokens(alphabet, a, b)
    return make_example(input_ids, token_type_ids, position_ids, embedding_mode)


def append_index_header(
    input_ids: list[int],
    token_type_ids: list[TokenTypeId],
    position_ids: list[int],
    j: int,
    alphabet: TokenSequence,
) -> None:
    """Append `[IDX] addr(j, alphabet)` at `position_id = j` to the three parallel lists."""
    addr = encode_address(j, alphabet)
    input_ids.append(int(AdditionToken.IDX))
    token_type_ids.append(TokenType.SPECIAL)
    position_ids.append(j)

    input_ids.extend(addr)
    token_type_ids.extend([TokenType.ADDRESS] * len(addr))
    position_ids.extend([j] * len(addr))


def build_addressed_prefix_tokens(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    include_alphabet: bool = True,
) -> tuple[list[int], list[TokenTypeId], list[int]]:
    """Build mutable `(input_ids, token_type_ids, position_ids)` lists for the addressed problem prefix.

    Shared by `addressed_step_ar` and `addressed_ar`:
    `[BOS] (alphabet [SEP])? [IDX] addr(0) a_0 ... [SEP] [IDX] addr(0) b_0 ... [SEP]`.
    """
    alphabet_tokens, a_tokens, b_tokens = validate_problem(
        alphabet, a, b, include_alphabet=include_alphabet
    )

    input_ids: list[int] = [int(SpecialToken.BOS)]
    token_type_ids: list[TokenTypeId] = [TokenType.SPECIAL]
    position_ids: list[int] = [0]

    if include_alphabet:
        input_ids.extend(alphabet_tokens)
        token_type_ids.extend([TokenType.ALPHABET] * len(alphabet_tokens))
        position_ids.extend(range(len(alphabet_tokens)))

        input_ids.append(int(SpecialToken.SEP))
        token_type_ids.append(TokenType.SPECIAL)
        position_ids.append(0)

    for j, sym in enumerate(a_tokens):
        append_index_header(input_ids, token_type_ids, position_ids, j, alphabet_tokens)
        input_ids.append(sym)
        token_type_ids.append(TokenType.OPERAND_A)
        position_ids.append(j)

    input_ids.append(int(SpecialToken.SEP))
    token_type_ids.append(TokenType.SPECIAL)
    position_ids.append(0)

    for j, sym in enumerate(b_tokens):
        append_index_header(input_ids, token_type_ids, position_ids, j, alphabet_tokens)
        input_ids.append(sym)
        token_type_ids.append(TokenType.OPERAND_B)
        position_ids.append(j)

    input_ids.append(int(SpecialToken.SEP))
    token_type_ids.append(TokenType.SPECIAL)
    position_ids.append(0)

    return input_ids, token_type_ids, position_ids


def build_addressed_prefix_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build the shared addressed prefix `Example` for `addressed_step_ar` and `addressed_ar`."""
    input_ids, token_type_ids, position_ids = build_addressed_prefix_tokens(
        alphabet, a, b, include_alphabet=include_alphabet
    )
    return make_example(input_ids, token_type_ids, position_ids, embedding_mode)


def validate_sampling_params(
    *,
    bases: Sequence[int],
    min_length: int,
    max_length: int,
    symbols: TokenSequence = SymbolSet.ALL,
    p_prior: float = 0.0,
    permuted_canonical: bool = False,
    p_chain: float = 0.25,
    include_alphabet: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate problem-sampling parameters and return `(bases_tuple, symbol_pool)`."""
    bases_tuple = tuple(bases)
    if not bases_tuple:
        raise ValueError("bases must contain at least 1 base")
    if len(set(bases_tuple)) != len(bases_tuple):
        raise ValueError(f"bases must not contain duplicates, got {bases_tuple}")
    for base in bases_tuple:
        if isinstance(base, bool) or not isinstance(base, int) or base < 2:
            raise ValueError(f"Each base must be an int >= 2, got {base!r}")

    for name, length in (("min_length", min_length), ("max_length", max_length)):
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"{name} must be an int, got {length!r}")
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")
    if max_length < min_length:
        raise ValueError(
            f"max_length ({max_length}) must be >= min_length ({min_length})"
        )

    if not (0.0 <= p_prior <= 1.0):
        raise ValueError(f"p_prior must be in 0.0..1.0, got {p_prior}")
    if permuted_canonical and p_prior != 0.0:
        raise ValueError("permuted_canonical=True cannot be combined with p_prior > 0")
    if not (0.0 <= p_chain <= 1.0):
        raise ValueError(f"p_chain must be in 0.0..1.0, got {p_chain}")

    symbol_pool = create_alphabet(symbols)
    max_base = max(bases_tuple)

    if (permuted_canonical or p_prior > 0.0) and max_base > len(SymbolSet.HEX):
        raise ValueError(
            f"Canonical numerals (SymbolSet.HEX) support bases up to "
            f"{len(SymbolSet.HEX)}, got base {max_base}"
        )
    if (not permuted_canonical and p_prior < 1.0) and max_base > len(symbol_pool):
        raise ValueError(
            f"max(bases) ({max_base}) exceeds symbol_pool size ({len(symbol_pool)})"
        )
    if not include_alphabet and max_length < max_base:
        raise ValueError(
            f"max_length ({max_length}) must be >= max(bases) ({max_base}) "
            f"when include_alphabet=False"
        )

    return bases_tuple, symbol_pool


def sample_problem(
    rng: random.Random,
    *,
    bases: tuple[int, ...],
    min_length: int,
    max_length: int,
    symbol_pool: tuple[int, ...],
    p_prior: float = 0.0,
    permuted_canonical: bool = False,
    p_chain: float = 0.25,
    include_alphabet: bool = True,
) -> AdditionProblem:
    """Sample one `AdditionProblem` from already-validated sampling parameters (`spec/task.md` §1)."""
    base = rng.choice(bases)

    if permuted_canonical:
        # A shuffle returns the ascending order 1 time in B! (half the time in
        # base 2), which would not be a permuted alphabet at all, so redraw
        # until the order differs from the canonical one.
        canonical = create_alphabet(SymbolSet.HEX[:base])
        alphabet = canonical
        while alphabet == canonical:
            alphabet = create_alphabet(SymbolSet.HEX[:base], rng=rng)
    elif p_prior > 0.0 and rng.random() < p_prior:
        alphabet = create_alphabet(SymbolSet.HEX[:base])
    else:
        alphabet = create_alphabet(symbol_pool, sample_size=base, rng=rng)

    eff_min = max(min_length, base) if not include_alphabet else min_length
    len_a = rng.randint(eff_min, max_length)
    len_b = rng.randint(eff_min, max_length)

    a_vals = [rng.randrange(base) for _ in range(len_a)]
    b_vals = [rng.randrange(base) for _ in range(len_b)]

    # Carry-chain injection (`spec/task.md` §1): with probability `p_chain`,
    # set starting column `i` to generate a carry (`v_a + v_b >= B`) and the
    # next `m` columns to propagate it (`v_a + v_b == B - 1`).
    n_shared = min(len_a, len_b)
    if p_chain > 0.0 and n_shared >= 2 and rng.random() < p_chain:
        m = rng.randint(1, n_shared - 1)
        i = rng.randint(0, n_shared - 1 - m)
        a_vals[i] = rng.randint(1, base - 1)
        b_vals[i] = rng.randint(base - a_vals[i], base - 1)
        for j in range(i + 1, i + 1 + m):
            a_vals[j] = rng.randrange(base)
            b_vals[j] = base - 1 - a_vals[j]

    a = tuple(alphabet[v] for v in a_vals)
    b = tuple(alphabet[v] for v in b_vals)
    return AdditionProblem(alphabet=alphabet, a=a, b=b)


def iter_problems(
    *,
    bases: Sequence[int] = (2, 4, 8, 10),
    min_length: int = 1,
    max_length: int = 20,
    symbols: TokenSequence = SymbolSet.ALL,
    p_prior: float = 0.0,
    permuted_canonical: bool = False,
    p_chain: float = 0.25,
    include_alphabet: bool = True,
    seed: int = 42,
) -> Iterator[AdditionProblem]:
    """Validate parameters eagerly and return an infinite deterministic stream of `AdditionProblem`s."""
    bases_tuple, symbol_pool = validate_sampling_params(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        p_prior=p_prior,
        permuted_canonical=permuted_canonical,
        p_chain=p_chain,
        include_alphabet=include_alphabet,
    )

    def _stream() -> Iterator[AdditionProblem]:
        rng = random.Random(seed)
        while True:
            yield sample_problem(
                rng,
                bases=bases_tuple,
                min_length=min_length,
                max_length=max_length,
                symbol_pool=symbol_pool,
                p_prior=p_prior,
                permuted_canonical=permuted_canonical,
                p_chain=p_chain,
                include_alphabet=include_alphabet,
            )

    return _stream()


def iter_problem_steps(
    *,
    bases: Sequence[int] = (2, 4, 8, 10),
    min_length: int = 1,
    max_length: int = 20,
    symbols: TokenSequence = SymbolSet.ALL,
    p_prior: float = 0.0,
    permuted_canonical: bool = False,
    p_chain: float = 0.25,
    include_alphabet: bool = True,
    p_terminal: float = 0.2,
    all_steps: bool = False,
    seed: int = 42,
) -> Iterator[tuple[AdditionProblem, tuple[ColumnStep, ...], int]]:
    """Yield `(problem, steps, k)` items for step-by-step harnessed variants.

    Step-index selection uses a separate PRNG stream (`f"{seed}:step"`) decoupled
    from `iter_problems(seed=seed)`, so changing `p_terminal` or `all_steps`
    keeps the underlying sequence of generated `AdditionProblem`s identical.
    """
    if not (0.0 <= p_terminal <= 1.0):
        raise ValueError(f"p_terminal must be in 0.0..1.0, got {p_terminal}")

    problems = iter_problems(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        p_prior=p_prior,
        permuted_canonical=permuted_canonical,
        p_chain=p_chain,
        include_alphabet=include_alphabet,
        seed=seed,
    )

    def _stream() -> Iterator[tuple[AdditionProblem, tuple[ColumnStep, ...], int]]:
        step_rng = random.Random(f"{seed}:step")
        for problem in problems:
            steps = reference_add(problem.alphabet, problem.a, problem.b)
            if all_steps:
                for k in range(len(steps)):
                    yield problem, steps, k
            else:
                last_k = len(steps) - 1
                k = (
                    last_k
                    if step_rng.random() < p_terminal
                    else step_rng.randint(0, last_k)
                )
                yield problem, steps, k

    return _stream()


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared `addition` dataset inspection CLI flags to `parser`."""
    parser.add_argument(
        "--bases",
        type=int,
        nargs="+",
        default=[2, 4, 8, 10],
        help="Bases B to sample from (default: 2 4 8 10)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum operand length (default: 1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=20,
        help="Maximum operand length (default: 20)",
    )
    parser.add_argument(
        "--p-prior",
        type=float,
        default=0.0,
        help="Fraction of problems using ascending canonical numerals (default: 0.0)",
    )
    parser.add_argument(
        "--permuted-canonical",
        action="store_true",
        default=False,
        help="Use shuffled canonical numerals per problem (default: False)",
    )
    parser.add_argument(
        "--p-chain",
        type=float,
        default=0.25,
        help="Probability of injecting a multi-column carry chain (default: 0.25)",
    )
    parser.add_argument(
        "--embedding-mode",
        type=EmbeddingMode,
        choices=tuple(EmbeddingMode),
        default=EmbeddingMode.TOKEN_TYPE_PE,
        help="Embedding mode for token_type_ids and position_ids (default: token_type_pe)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
