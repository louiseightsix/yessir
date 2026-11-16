"""Input sequence and full-trace label construction (`build_ar_trace_tokens`, `build_ar_example`, `build_ar_labelled_example`) for the `ar` variant of `addition`."""

from collections.abc import Sequence

from experiments.addition.inputs import (
    EmbeddingMode,
    StepOrder,
    TokenType,
    build_prefix_tokens,
    make_example,
    pack_step_order,
    step_query_tokens,
)
from experiments.addition.reference import ColumnStep, reference_add
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


def _trace_slot_metadata(
    slot_idx: int,
    *,
    step_order: StepOrder,
    use_query_tokens: bool,
) -> tuple[TokenType, int]:
    """Return deterministic `(token_type_id, position_id)` for trace slot `slot_idx` (`spec/variants/ar.md` §1, §3).

    For column `k`, `o_k` sits at `(TokenType.OUTPUT, k)`, `c_{k+1}` sits at
    `(TokenType.CARRY, k + 1)` (matching column `k + 1`'s operands `a_{k+1}` and
    `b_{k+1}`), and any `[S]` / `[C]` query tokens sit at `(TokenType.SPECIAL, k)`.
    """
    order = StepOrder(step_order)
    if not use_query_tokens:
        k, rem = divmod(slot_idx, 2)
        is_sum = (rem == 0) if order == StepOrder.SUM_FIRST else (rem == 1)
        return (TokenType.OUTPUT, k) if is_sum else (TokenType.CARRY, k + 1)

    k, rem = divmod(slot_idx, 4)
    if rem in (0, 2):
        return (TokenType.SPECIAL, k)
    is_sum = (rem == 1) if order == StepOrder.SUM_FIRST else (rem == 3)
    return (TokenType.OUTPUT, k) if is_sum else (TokenType.CARRY, k + 1)


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
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build an `ar` `Example` appending partial or complete `trace` tokens to the shared problem prefix."""
    order = StepOrder(step_order)
    input_ids, token_type_ids, position_ids = build_prefix_tokens(alphabet, a, b)

    for slot_idx, tok in enumerate(trace):
        tt, pos = _trace_slot_metadata(
            slot_idx, step_order=order, use_query_tokens=use_query_tokens
        )
        input_ids.append(int(tok))
        token_type_ids.append(tt)
        position_ids.append(pos)

    return make_example(input_ids, token_type_ids, position_ids, embedding_mode)


def build_ar_labelled_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    steps: Sequence[ColumnStep] | None = None,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> LabelledExample:
    """Build the full-trace teacher-forced `LabelledExample` for `(alphabet, a, b)` (`spec/variants/ar.md` §3).

    Appends `trace` to the shared prefix of length `P = B + L_a + L_b + 4` and
    assigns pre-shifted `labels = (IGNORE_INDEX,) * (P - 1) + (*trace, EOS)`.
    """
    oracle_steps = tuple(steps) if steps is not None else reference_add(alphabet, a, b)
    order = StepOrder(step_order)
    trace = build_ar_trace_tokens(
        oracle_steps, step_order=order, use_query_tokens=use_query_tokens
    )
    example = build_ar_example(
        alphabet,
        a,
        b,
        trace,
        step_order=order,
        use_query_tokens=use_query_tokens,
        embedding_mode=embedding_mode,
    )
    prefix_len = len(example) - len(trace)
    labels = (IGNORE_INDEX,) * (prefix_len - 1) + (*trace, int(SpecialToken.EOS))
    return LabelledExample(example=example, labels=labels)
