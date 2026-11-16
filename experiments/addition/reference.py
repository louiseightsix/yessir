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

    Named fields avoid positional mix-ups between `carry_in`, `sum_digit`, and
    `carry_out` (all `Token` ints, and swapped in sequence order between
    `sum_first` and `carry_first` variants), while keeping `carry_in` (`c_k`)
    alongside `carry_out` (`c_{k+1}`) for step-prompt construction and
    `carry_accuracy` scoring.
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
    """Validate `(alphabet, a, b)` and return `(alphabet_tokens, symbol_to_value)`."""
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
) -> tuple[Token, Token, SpecialToken]:
    """Compute `(o_k, c_{k+1}, flag_k)` on already-validated `(alphabet, a, b)`."""
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

    base = len(alphabet_tokens)
    v_a = rank[a[k]] if k < len(a) else 0
    v_b = rank[b[k]] if k < len(b) else 0
    v_c = 0 if carry == s_0 else 1

    # Because v_a <= B - 1, v_b <= B - 1, and v_c <= 1, total <= 2 * B - 1 < 2 * B,
    # so total // B is always 0 (s_0) or 1 (s_1). At k == n (final carry overflow),
    # v_a == v_b == 0 and B >= 2 give total == v_c < B, so carry_out == s_0 and
    # halting is guaranteed by column n at the latest.
    total = v_a + v_b + v_c
    sum_digit = alphabet_tokens[total % base]
    carry_out = alphabet_tokens[total // base]
    flag = (
        SpecialToken.FLAG_FINISHED
        if k >= n - 1 and carry_out == s_0
        else SpecialToken.FLAG_CONTINUE
    )
    return sum_digit, carry_out, flag


def reference_step(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    k: int,
    carry: Token,
) -> tuple[Token, Token, SpecialToken]:
    """Compute the ground-truth `(o_k, c_{k+1}, flag_k)` transition at column `k`.

    Args:
        alphabet: Ordered alphabet of `B >= 2` distinct symbols `[s_0, ..., s_{B-1}]`.
        a: Non-empty operand `a` in `LSD -> MSD` order, drawn from `alphabet`.
        b: Non-empty operand `b` in `LSD -> MSD` order, drawn from `alphabet`.
        k: 0-indexed column step (`0 <= k <= max(len(a), len(b))`).
        carry: Incoming carry symbol `c_k` (`alphabet[0]` for `0` or `alphabet[1]` for `1`).

    Returns:
        `(o_k, c_{k+1}, flag_k)`:
        - `o_k`: Sum digit `alphabet[total % B]`.
        - `c_{k+1}`: Outgoing carry `alphabet[total // B]` (`s_0` or `s_1`).
        - `flag_k`: `FLAG_FINISHED` if `k >= max(len(a), len(b)) - 1` and
          `c_{k+1} == s_0`, else `FLAG_CONTINUE`.
    """
    alphabet_tokens, rank = _validate_inputs(alphabet, a, b)
    return _compute_step(alphabet_tokens, rank, a, b, k, carry)


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
    alphabet_tokens, rank = _validate_inputs(alphabet, a, b)
    steps: list[ColumnStep] = []
    carry: Token = alphabet_tokens[0]
    k = 0
    while True:
        sum_digit, carry_out, flag = _compute_step(
            alphabet_tokens, rank, a, b, k, carry
        )
        steps.append(
            ColumnStep(
                k=k,
                carry_in=carry,
                sum_digit=sum_digit,
                carry_out=carry_out,
                flag=flag,
            )
        )
        if flag == SpecialToken.FLAG_FINISHED:
            return tuple(steps)
        carry = carry_out
        k += 1
