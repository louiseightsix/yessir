"""Input sequence and full-trace label construction (`build_addressed_ar_trace_tokens`, `build_addressed_ar_example`, `build_addressed_ar_labelled_example`) for the `addressed_ar` variant of `addition`."""

from collections.abc import Sequence

from experiments.addition.inputs import (
    AdditionToken,
    EmbeddingMode,
    StepOrder,
    TokenType,
    build_addressed_prefix_tokens,
    encode_address,
    make_example,
    pack_step_order,
    step_query_tokens,
)
from experiments.addition.reference import ColumnStep, reference_add
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence


def _column_body_slot_metadata(
    k: int,
    body_idx: int,
    *,
    step_order: StepOrder,
) -> tuple[TokenType, int]:
    """Return `(token_type_id, position_id)` for unqueried body offset `body_idx` (`0` or `1`) within column `k`."""
    is_sum = (body_idx == 0) if step_order == StepOrder.SUM_FIRST else (body_idx == 1)
    return (TokenType.OUTPUT, k) if is_sum else (TokenType.CARRY, k + 1)


def build_addressed_ar_trace_tokens(
    steps: Sequence[ColumnStep],
    alphabet: TokenSequence,
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
) -> tuple[Token, ...]:
    """Build the ground-truth addressed trace tokens across `k = 0 .. K` (excluding trailing `[EOS]`)."""
    alphabet_tokens = tuple(alphabet)
    order = StepOrder(step_order)
    q_0, q_1, _ = step_query_tokens(order)
    idx_tok = int(AdditionToken.IDX)
    trace: list[Token] = []

    for k, step in enumerate(steps):
        addr_k = encode_address(k, alphabet_tokens)
        t_0, t_1 = pack_step_order(step.sum_digit, step.carry_out, order)
        trace.append(idx_tok)
        trace.extend(addr_k)
        if use_query_tokens:
            trace.extend((int(q_0), t_0, int(q_1), t_1))
        else:
            trace.extend((t_0, t_1))

    return tuple(trace)


def build_addressed_ar_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    trace: TokenSequence = (),
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build an `addressed_ar` `Example` appending partial or complete `trace` tokens to the addressed prefix."""
    order = StepOrder(step_order)
    input_ids, token_type_ids, position_ids = build_addressed_prefix_tokens(
        alphabet, a, b, include_alphabet=include_alphabet
    )
    alphabet_tokens = tuple(alphabet)
    idx_tok = int(AdditionToken.IDX)
    q_sum = int(AdditionToken.QUERY_SUM)
    q_carry = int(AdditionToken.QUERY_CARRY)

    k = 0
    in_segment = False
    slot = 0
    addr_len = len(encode_address(0, alphabet_tokens))
    pending_query: int | None = None

    for raw_tok in trace:
        tok = int(raw_tok)
        if tok == idx_tok:
            if in_segment:
                k += 1
                addr_len = len(encode_address(k, alphabet_tokens))
            in_segment = True
            slot = 1
            pending_query = None
            tt, pos = TokenType.SPECIAL, k
        else:
            if not in_segment:
                in_segment = True
                slot = 1
            if use_query_tokens:
                if tok in (q_sum, q_carry):
                    pending_query = tok
                    tt, pos = TokenType.SPECIAL, k
                elif pending_query == q_sum:
                    pending_query = None
                    tt, pos = TokenType.OUTPUT, k
                elif pending_query == q_carry:
                    pending_query = None
                    tt, pos = TokenType.CARRY, k + 1
                else:
                    tt, pos = TokenType.ADDRESS, k
            else:
                if slot <= addr_len:
                    tt, pos = TokenType.ADDRESS, k
                elif slot <= addr_len + 2:
                    tt, pos = _column_body_slot_metadata(
                        k,
                        slot - 1 - addr_len,
                        step_order=order,
                    )
                else:
                    # Hold column `k` on overlong malformed segments until `[IDX]` arrives.
                    tt, pos = TokenType.SPECIAL, k
            slot += 1

        input_ids.append(tok)
        token_type_ids.append(tt)
        position_ids.append(pos)

    return make_example(input_ids, token_type_ids, position_ids, embedding_mode)


def build_addressed_ar_labelled_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    *,
    steps: Sequence[ColumnStep] | None = None,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> LabelledExample:
    """Build the full-trace teacher-forced `LabelledExample` for `(alphabet, a, b)` (`spec/variants/addressed_ar.md` §3).

    Appends `trace` to the addressed problem prefix of length `P` and assigns
    pre-shifted `labels = (IGNORE_INDEX,) * (P - 1) + (*trace, EOS)`.
    """
    oracle_steps = tuple(steps) if steps is not None else reference_add(alphabet, a, b)
    order = StepOrder(step_order)
    trace = build_addressed_ar_trace_tokens(
        oracle_steps,
        alphabet,
        step_order=order,
        use_query_tokens=use_query_tokens,
    )
    example = build_addressed_ar_example(
        alphabet,
        a,
        b,
        trace,
        step_order=order,
        use_query_tokens=use_query_tokens,
        include_alphabet=include_alphabet,
        embedding_mode=embedding_mode,
    )
    prefix_len = len(example) - len(trace)
    labels = (IGNORE_INDEX,) * (prefix_len - 1) + (*trace, int(SpecialToken.EOS))
    return LabelledExample(example=example, labels=labels)
