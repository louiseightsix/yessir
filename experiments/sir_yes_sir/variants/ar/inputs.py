"""Autoregressive input and trace construction (`StepOrder`, `build_ar_trace`, `build_ar_example`, `build_ar_labelled_example`) for the `ar` variant of `sir_yes_sir`."""

from enum import StrEnum, unique

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    TokenType,
    sections_to_example,
    validate_problem,
)
from experiments.sir_yes_sir.reference import reference_step
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import TokenSequence


@unique
class StepOrder(StrEnum):
    """Within-step token emission order in an autoregressive (`ar`) trace.

    Why this matters in `ar` (unlike `sm_mh`):
    In `sm_mh`, all three heads read the `[STEP]` hidden state in parallel. In `ar`,
    tokens within step `k` are emitted sequentially and condition on one another:
    - `OUT_FIRST`: emits `(o_k, w_k, flag_k)` at each step `k`.
    - `WRAP_FIRST`: emits `(w_k, o_k, flag_k)` at each step `k`.
    """

    OUT_FIRST = "out_first"
    WRAP_FIRST = "wrap_first"


def build_ar_trace(
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.OUT_FIRST,
) -> tuple[int, ...]:
    """Build the flat ground-truth `3 * len(x)` token trace for `(alphabet, x)`."""
    step_order = StepOrder(step_order)
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)

    trace: list[int] = []
    for k in range(len(x_tokens)):
        o_k, w_k, flag_k = reference_step(alphabet_tokens, x_tokens, k)
        if step_order == StepOrder.OUT_FIRST:
            trace.extend((o_k, w_k, int(flag_k)))
        else:
            trace.extend((w_k, o_k, int(flag_k)))
    return tuple(trace)


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


def build_ar_labelled_example(
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.OUT_FIRST,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> LabelledExample:
    """Build a teacher-forced next-token `LabelledExample` for `(alphabet, x)`.

    If the prompt prefix `[BOS] alphabet [SEP] x [SEP]` has length `P` and the
    ground-truth trace has length `3L` (`trace[0 .. 3L - 1]`, ending with
    `FLAG_FINISHED`):
    - `example` has `input_ids = prefix + trace[:-1]` (length `P + 3L - 1`).
    - `labels` has `(IGNORE_INDEX,) * (P - 1) + trace` (length `P + 3L - 1`), so
      loss is computed starting at index `P - 1` (the second `[SEP]`, predicting
      `trace[0]`) through `trace[-2]` (predicting `trace[-1] == FLAG_FINISHED`).
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    trace = build_ar_trace(alphabet_tokens, x_tokens, step_order=step_order)
    example = build_ar_example(
        alphabet_tokens,
        x_tokens,
        trace=trace[:-1],
        embedding_mode=embedding_mode,
    )
    labels = (IGNORE_INDEX,) * (len(example) - len(trace)) + trace
    return LabelledExample(example=example, labels=labels)
