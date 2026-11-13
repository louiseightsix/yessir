"""Parallel classification input construction (`build_cls_example`) for the `cls` variant of `sir_yes_sir`."""

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    TokenType,
    sections_to_example,
    validate_problem,
)
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import TokenSequence


def build_cls_example(
    alphabet: TokenSequence,
    x: TokenSequence,
    output_history: TokenSequence = (),
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build the state-machine input `Example` for step `k = len(output_history)`.

    The three pieces of task state (`alphabet`, `x`, `output_history`) are glued
    together end-to-end with `[BOS]`, `[SEP]`, and `[STEP]` into a single sequence.
    For example, with `alphabet = "0123"`, `x = "130"`, `output_history = "2"` (`k = 1`):

        input_ids:       [BOS]  '0' '1' '2' '3'  [SEP]  '1' '3' '0'  [SEP]  '2'  [STEP]
        token_type_ids:    0     1   1   1   1     0     2   2   2     0     3     3
        position_ids:      0     0   1   2   3     0     0   1   2     0     0     1

    Notice that in the `OUTPUT` section (`token_type_ids == 3`), `[STEP]` sits at
    `position_id = k = 1`, aligning it with `x_1 = '3'` in the `INPUT` section.
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    alphabet_set = set(alphabet_tokens)

    out_tokens = tuple(output_history)
    k = len(out_tokens)
    if k >= len(x_tokens):
        raise ValueError(
            f"step index k={k} (from output_history length) out of range "
            f"for input sequence of length {len(x_tokens)}"
        )

    for idx, sym in enumerate(out_tokens):
        if sym not in alphabet_set:
            raise ValueError(
                f"output symbol {sym!r} at position {idx} not found in alphabet"
            )

    sections: list[tuple[tuple[int, ...], TokenType]] = [
        ((SpecialToken.BOS,), TokenType.SPECIAL),
        (alphabet_tokens, TokenType.ALPHABET),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        (x_tokens, TokenType.INPUT),
        ((SpecialToken.SEP,), TokenType.SPECIAL),
        ((*out_tokens, SpecialToken.STEP), TokenType.OUTPUT),
    ]
    return sections_to_example(sections, embedding_mode)
