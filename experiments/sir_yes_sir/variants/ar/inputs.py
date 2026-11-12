"""Autoregressive input construction (`build_ar_example`) for the `ar` variant of `sir_yes_sir`."""

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    TokenType,
    sections_to_example,
    validate_problem,
)
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import TokenSequence


def build_ar_example(
    alphabet: TokenSequence,
    x: TokenSequence,
    trace: TokenSequence = (),
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build the autoregressive (`ar`) input `Example` for `(alphabet, x)` and `trace`.

    The harness is passive: the problem prefix ends at the second `[SEP]`, and any
    `trace` tokens emitted so far follow in a single `TokenType.OUTPUT` section with
    sequential `position_ids = 0, 1, ..., len(trace) - 1`.

    For `alphabet = "0123"`, `x = "130"`, and a complete `OUT_FIRST` trace:

        input_ids:       [BOS] '0' '1' '2' '3' [SEP] '1' '3' '0' [SEP] '2' '0' [FLAG_CONTINUE] '0' '1' [FLAG_CONTINUE] '1' '0' [FLAG_FINISHED]
        token_type_ids:    0    1   1   1   1    0    2   2   2    0    3   3         3         3   3         3         3   3         3
        position_ids:      0    0   1   2   3    0    0   1   2    0    0   1         2         3   4         5         6   7         8

    At the start of generation (`trace = ()`), the `Example` ends at the second `[SEP]`.
    """
    # TODO(ar): When building the AR harness, revisit two design choices:
    # 1. `position_ids` for `trace`: counting `0, 1, 2, ...` puts step k's tokens at
    #    `3k .. 3k+2` instead of aligning them at `position_id = k` with `x_k`. Decide
    #    whether to keep sequential `0, 1, 2, ...` or assign step-aligned `position_id = k`.
    # 2. `StepOrder` (`OUT_FIRST`: `(o_k, w_k, flag_k)` vs. `WRAP_FIRST`: `(w_k, o_k, flag_k)`):
    #    make within-step emission order configurable in the AR trace builder/parser.
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    trace_tokens = tuple(trace)

    sections: list[tuple[tuple[int, ...], TokenType]] = [
        ((SpecialToken.BOS,), TokenType.SPECIAL),
        (alphabet_tokens, TokenType.ALPHABET),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        (x_tokens, TokenType.INPUT),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        (trace_tokens, TokenType.OUTPUT),
    ]
    return sections_to_example(sections, embedding_mode)
