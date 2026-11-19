"""Ground-truth reference implementation for `addition`.

Task: In-context column addition (`experiments/addition/spec/task.md` §1).
Given an ordered alphabet `[s_0, ..., s_{B-1}]` defining base `B = len(alphabet)`
and digit values `val(s_i) = i`, together with two unpadded operands `a` and `b`
written least-significant digit first (`LSD -> MSD`), each column step `k`
(`0 <= k <= K`) combines `a_k` (`0` if `k >= len(a)`), `b_k` (`0` if `k >= len(b)`),
and incoming carry `c_k` (`s_0` or `s_1`) to emit:
- `sum_digit` (`o_k`): `alphabet[total % B]`
- `carry_out` (`c_{k+1}`): `alphabet[total // B]` (`s_0` or `s_1`)
- `flag` (`flag_k`): `FLAG_FINISHED` if `k >= max(len(a), len(b)) - 1` and
  `c_{k+1} == s_0`, else `FLAG_CONTINUE`.
"""

from dataclasses import dataclass

from yessir.alphabet import create_alphabet
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


@dataclass(frozen=True)
class ColumnStep:
    """Ground-truth state and outputs for column `k` of an addition problem.

    `carry_in` is `c_k`, `sum_digit` is `o_k`, `carry_out` is `c_{k+1}`, and
    `flag` is `flag_k`. Named fields avoid positional mix-ups between
    `carry_in`, `sum_digit`, and `carry_out`, which are all `Token` ints.
    """

    k: int
    carry_in: Token
    sum_digit: Token
    carry_out: Token
    flag: SpecialToken


def _validate_inputs(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Validate `(alphabet, a, b)` and return `(alphabet_tokens, rank)`.

    `rank` maps each alphabet symbol `s_i` to its digit value `i`.
    """
    alphabet_tokens = create_alphabet(alphabet)
    rank = {symbol: idx for idx, symbol in enumerate(alphabet_tokens)}

    if len(a) == 0:
        raise ValueError("Operand a must contain at least 1 symbol")
    if len(b) == 0:
        raise ValueError("Operand b must contain at least 1 symbol")

    for idx, symbol in enumerate(a):
        if symbol not in rank:
            raise ValueError(
                f"Operand a symbol 0x{symbol:02X} at index {idx} not found in alphabet"
            )
    for idx, symbol in enumerate(b):
        if symbol not in rank:
            raise ValueError(
                f"Operand b symbol 0x{symbol:02X} at index {idx} not found in alphabet"
            )

    return alphabet_tokens, rank


def _compute_step(
    alphabet_tokens: tuple[int, ...],
    rank: dict[int, int],
    a: TokenSequence,
    b: TokenSequence,
    k: int,
    carry: Token,
) -> ColumnStep:
    """Compute the `ColumnStep` at column `k` on already-validated `(alphabet, a, b)`."""
    n = max(len(a), len(b))
    if not (0 <= k <= n):
        raise IndexError(
            f"Step index k={k} out of range 0..{n} for operand lengths "
            f"({len(a)}, {len(b)})"
        )

    s_0, s_1 = alphabet_tokens[0], alphabet_tokens[1]
    if carry not in (s_0, s_1):
        raise ValueError(
            f"Incoming carry 0x{carry:02X} at step k={k} must be "
            f"s_0 (0x{s_0:02X}) or s_1 (0x{s_1:02X})"
        )
    # Two (k, carry) states can never occur in a real problem, so reject them
    # rather than answer: column 0 always starts with no carry, and the
    # overflow column k == n only runs when column n - 1 carried out.
    if k == 0 and carry != s_0:
        raise ValueError(
            f"Incoming carry at step k=0 must be s_0 (0x{s_0:02X}), got 0x{carry:02X}"
        )
    if k == n and carry != s_1:
        raise ValueError(
            f"Incoming carry at overflow step k={k} must be s_1 (0x{s_1:02X}), "
            f"got 0x{carry:02X}"
        )

    base = len(alphabet_tokens)
    v_a = rank[a[k]] if k < len(a) else 0
    v_b = rank[b[k]] if k < len(b) else 0
    v_c = 0 if carry == s_0 else 1

    # Because v_a <= B - 1, v_b <= B - 1, and v_c <= 1, total <= 2 * B - 1 < 2 * B,
    # so total // B is always 0 (s_0) or 1 (s_1). At k == n (final carry overflow),
    # v_a == v_b == 0 and v_c == 1, and B >= 2 gives total == 1 < B, so
    # carry_out == s_0 and halting is guaranteed by column n at the latest.
    total = v_a + v_b + v_c
    sum_digit = alphabet_tokens[total % base]
    carry_out = alphabet_tokens[total // base]
    flag = (
        SpecialToken.FLAG_FINISHED
        if k >= n - 1 and carry_out == s_0
        else SpecialToken.FLAG_CONTINUE
    )
    return ColumnStep(
        k=k,
        carry_in=carry,
        sum_digit=sum_digit,
        carry_out=carry_out,
        flag=flag,
    )


def reference_step(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    k: int,
    carry: Token,
) -> ColumnStep:
    """Compute the ground-truth transition at column `k` as a `ColumnStep`.

    Args:
        alphabet: Ordered alphabet of `B >= 2` distinct symbols `[s_0, ..., s_{B-1}]`.
        a: Non-empty operand `a` in `LSD -> MSD` order, drawn from `alphabet`.
        b: Non-empty operand `b` in `LSD -> MSD` order, drawn from `alphabet`.
        k: 0-indexed column step (`0 <= k <= max(len(a), len(b))`).
        carry: Incoming carry symbol `c_k` (`alphabet[0]` for `0` or `alphabet[1]` for `1`).
            Must be `alphabet[0]` at `k = 0` and `alphabet[1]` at the overflow
            column `k = max(len(a), len(b))`.

    Returns:
        A `ColumnStep` echoing `k` and `carry` (as `carry_in`), with:
        - `sum_digit` (`o_k`): `alphabet[total % B]`.
        - `carry_out` (`c_{k+1}`): `alphabet[total // B]` (`s_0` or `s_1`).
        - `flag` (`flag_k`): `FLAG_FINISHED` if `k >= max(len(a), len(b)) - 1` and
          `c_{k+1} == s_0`, else `FLAG_CONTINUE`.

    Raises:
        IndexError: If `k` is outside `0 .. max(len(a), len(b))`.
        ValueError: If the alphabet or operands are invalid, or if `carry` is
            not a carry symbol that can reach column `k`.
    """
    alphabet_tokens, rank = _validate_inputs(alphabet, a, b)
    return _compute_step(alphabet_tokens, rank, a, b, k, carry)


def _reference_add_trusted(
    alphabet_tokens: tuple[int, ...],
    a: TokenSequence,
    b: TokenSequence,
) -> tuple[ColumnStep, ...]:
    """Run all `K + 1` ground-truth column steps on already-validated `(alphabet_tokens, a, b)`."""
    rank = {symbol: idx for idx, symbol in enumerate(alphabet_tokens)}
    steps: list[ColumnStep] = []
    carry: Token = alphabet_tokens[0]
    k = 0
    while True:
        step = _compute_step(alphabet_tokens, rank, a, b, k, carry)
        steps.append(step)
        if step.flag == SpecialToken.FLAG_FINISHED:
            return tuple(steps)
        carry = step.carry_out
        k += 1


def reference_add(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
) -> tuple[ColumnStep, ...]:
    """Run all `K + 1` ground-truth column steps (`k = 0 .. K`) for `a + b`.

    Starts at `k = 0` with `carry_in = alphabet[0]` (`s_0`) and advances until
    `flag == FLAG_FINISHED`, returning `max(len(a), len(b))` steps when there is
    no final carry overflow (`c_N == s_0`) or `max(len(a), len(b)) + 1` steps
    when column `N - 1` overflows (`c_N == s_1`).
    """
    alphabet_tokens, _ = _validate_inputs(alphabet, a, b)
    return _reference_add_trusted(alphabet_tokens, a, b)
