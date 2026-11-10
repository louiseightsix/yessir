"""Ground-truth reference implementation for sir_yes_sir.

Task: In-context successor (next symbol mod B, no carry).
Given an ordered alphabet [s_0, ..., s_{B-1}] and an input sequence x,
at step k where x_k = s_i:
- o_k: next symbol in the alphabet, s_{(i+1) mod B}
- w_k: s_1 if wrapped around (i == B - 1) else s_0
- flag_k: FLAG_FINISHED if k == len(x) - 1 else FLAG_CONTINUE
"""

from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


def reference_step(
    alphabet: TokenSequence,
    x: TokenSequence,
    k: int,
) -> tuple[Token, Token, SpecialToken]:
    """Compute the ground-truth transition at step k.

    Args:
        alphabet: Ordered alphabet of B >= 2 distinct symbols [s_0, ..., s_{B-1}].
        x: Input sequence of symbols drawn from the alphabet.
        k: 0-indexed step index (0 <= k < len(x)).

    Returns:
        (o_k, w_k, flag_k):
        - o_k: Successor symbol s_{(i+1) mod B}.
        - w_k: Wrap indicator (s_1 if i == B - 1 else s_0).
        - flag_k: Halting flag (FLAG_FINISHED if k == len(x) - 1 else FLAG_CONTINUE).
    """
    if not (0 <= k < len(x)):
        raise IndexError(
            f"Step index k={k} out of range for sequence of length {len(x)}"
        )

    base = len(alphabet)
    if base < 2:
        raise ValueError(f"Alphabet must contain at least 2 symbols, got {base}")

    current_symbol = x[k]
    try:
        i = alphabet.index(current_symbol)
    except ValueError:
        raise ValueError(
            f"Input symbol 0x{current_symbol:02X} at step k={k} not found in alphabet"
        ) from None

    # Primary output: next symbol mod B
    o_k = alphabet[(i + 1) % base]

    # Wrap output: s_1 if wrapped around from last symbol to first, else s_0
    w_k = alphabet[1] if i == base - 1 else alphabet[0]

    # Halting flag: finished if this is the final input element
    flag_k = (
        SpecialToken.FLAG_FINISHED if k == len(x) - 1 else SpecialToken.FLAG_CONTINUE
    )

    return o_k, w_k, flag_k
