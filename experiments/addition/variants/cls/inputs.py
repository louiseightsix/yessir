"""Input sequence and step-label construction (`build_cls_example`, `build_cls_step_labelled_example`) for the `cls` variant of `addition`."""

from collections.abc import Sequence

from experiments.addition.inputs import (
    EmbeddingMode,
    TokenType,
    build_prefix_tokens,
    make_example,
)
from experiments.addition.reference import ColumnStep, reference_add
from yessir.dataset import LabelledExample
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


def build_cls_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    carry: Token,
    output: TokenSequence = (),
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
    validate: bool = True,
) -> Example:
    """Build the step-`k` (`k = len(output)`) `Example` for `cls` (`spec/variants/cls.md` §1).

    Sequence layout at column `k`:
        input_ids:       [BOS] alphabet [SEP] a [SEP] b [SEP]  c_k  [SEP]  o_0 ... o_{k-1}
        token_type_ids:    0      1       0   2   0   3   0     4     0     5  ...    5
        position_ids:      0    0..B-1    0 0..La 0 0..Lb 0     k     0     1  ...    k

    Why `output` `position_ids` start at `1` (`o_0` at `1`, ..., `o_{k-1}` at `k`):
    At `k == 0` (`output = ()`), the predicting final token is the fourth `[SEP]`
    at `position_id = 0`; at `k >= 1`, the predicting final token is `o_{k-1}` at
    `position_id = k`. In both cases, the predicting position, `c_k` (`CARRY`),
    `a_k` (`OPERAND_A`), and `b_k` (`OPERAND_B`) all share `position_id = k`.
    """
    input_ids, token_type_ids, position_ids = build_prefix_tokens(
        alphabet, a, b, validate=validate
    )
    out_tokens = tuple(output)
    k = len(out_tokens)

    # Incoming carry `c_k` at `(TokenType.CARRY, k)` and fourth `[SEP]` at `(TokenType.SPECIAL, 0)`.
    input_ids.extend((int(carry), int(SpecialToken.SEP), *out_tokens))
    token_type_ids.extend(
        (
            TokenType.CARRY,
            TokenType.SPECIAL,
            *([TokenType.OUTPUT] * k),
        )
    )
    position_ids.extend((k, 0, *range(1, k + 1)))

    return make_example(input_ids, token_type_ids, position_ids, embedding_mode)


def build_cls_step_labelled_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    k: int,
    *,
    steps: Sequence[ColumnStep] | None = None,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
    validate: bool = True,
) -> LabelledExample:
    """Build a teacher-forced single-step `LabelledExample` for column `k` (`0 <= k <= K`).

    `labels` is the 1D 3-tuple `(o_k, c_{k+1}, int(flag_k))` supervising
    `ParallelClassificationHead.heads[0..2]` at the final token (`lengths - 1`).
    """
    oracle_steps = tuple(steps) if steps is not None else reference_add(alphabet, a, b)
    if not (0 <= k < len(oracle_steps)):
        raise IndexError(f"Step index k={k} out of range 0..{len(oracle_steps) - 1}")

    step = oracle_steps[k]
    output_prefix = tuple(s.sum_digit for s in oracle_steps[:k])
    example = build_cls_example(
        alphabet,
        a,
        b,
        step.carry_in,
        output_prefix,
        embedding_mode=embedding_mode,
        validate=validate,
    )
    return LabelledExample(
        example=example,
        labels=(step.sum_digit, step.carry_out, int(step.flag)),
    )
