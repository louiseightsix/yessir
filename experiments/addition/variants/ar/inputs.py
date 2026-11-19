"""Pure-LM (`input_ids`-only) input sequence and full-trace label construction (`build_ar_trace_tokens`, `build_ar_example`, `build_ar_labelled_example`) for the `ar` variant of `addition`."""

from collections.abc import Sequence

from experiments.addition.inputs import (
    StepOrder,
    build_prefix_tokens,
    pack_step_order,
    step_query_tokens,
)
from experiments.addition.reference import ColumnStep, reference_add
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


def build_ar_trace_tokens(
    steps: Sequence[ColumnStep],
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
) -> tuple[Token, ...]:
    """Build the ground-truth column trace tokens across `k = 0 .. K` (excluding trailing `[EOS]`)."""
    order = StepOrder(step_order)
    q_0, q_1, _ = step_query_tokens(order)
    trace: list[Token] = []
    for step in steps:
        t_0, t_1 = pack_step_order(step.sum_digit, step.carry_out, order)
        if use_query_tokens:
            trace.extend((int(q_0), t_0, int(q_1), t_1))
        else:
            trace.extend((t_0, t_1))
    return tuple(trace)


def build_ar_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    trace: TokenSequence = (),
    *,
    validate: bool = True,
) -> Example:
    """Build a pure-LM (`input_ids`-only) `ar` `Example` appending `trace` tokens to `[BOS] alphabet [SEP] a [SEP] b [SEP]`."""
    input_ids, _, _ = build_prefix_tokens(alphabet, a, b, validate=validate)
    input_ids.extend(int(tok) for tok in trace)
    return Example(input_ids=tuple(input_ids))


def build_ar_labelled_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    steps: Sequence[ColumnStep] | None = None,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    validate: bool = True,
) -> LabelledExample:
    """Build the full-trace teacher-forced `LabelledExample` for `(alphabet, a, b)` (`spec/variants/ar.md` §3).

    Appends `trace` to the shared prefix of length `P = B + L_a + L_b + 4` and
    assigns pre-shifted `labels = (IGNORE_INDEX,) * (P - 1) + (*trace, EOS)` with
    no side-channel `token_type_ids` or `position_ids`.
    """
    oracle_steps = tuple(steps) if steps is not None else reference_add(alphabet, a, b)
    order = StepOrder(step_order)
    trace = build_ar_trace_tokens(
        oracle_steps, step_order=order, use_query_tokens=use_query_tokens
    )
    example = build_ar_example(alphabet, a, b, trace, validate=validate)
    prefix_len = len(example) - len(trace)
    labels = (IGNORE_INDEX,) * (prefix_len - 1) + (*trace, int(SpecialToken.EOS))
    return LabelledExample(example=example, labels=labels)
