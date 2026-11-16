"""Input sequence and step-label construction (`build_addressed_step_ar_example`, `build_addressed_step_ar_step_labelled_example`) for the `addressed_step_ar` variant of `addition`."""

from collections.abc import Sequence

from experiments.addition.inputs import (
    EmbeddingMode,
    StepOrder,
    TokenType,
    append_index_header,
    build_addressed_prefix_tokens,
    make_example,
    pack_step_order,
    step_query_tokens,
    unpack_step_order,
)
from experiments.addition.reference import ColumnStep, reference_add
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence

__all__ = [
    "build_addressed_step_ar_example",
    "build_addressed_step_ar_step_labelled_example",
    "pack_step_order",
    "step_query_tokens",
    "unpack_step_order",
]


def _step_slot_metadata(
    k: int, step_order: StepOrder
) -> tuple[tuple[TokenType, int], tuple[TokenType, int]]:
    """Return `((tt_0, pos_0), (tt_1, pos_1))` for `(t_0, t_1)` at column `k` (`spec/variants/addressed_step_ar.md` §1)."""
    sum_meta = (TokenType.OUTPUT, k)
    carry_meta = (TokenType.CARRY, k + 1)
    if StepOrder(step_order) == StepOrder.SUM_FIRST:
        return sum_meta, carry_meta
    return carry_meta, sum_meta


def build_addressed_step_ar_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    carry: Token,
    output: TokenSequence = (),
    step_tokens: TokenSequence = (),
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build a step-`k` (`k = len(output)`) `Example` for `addressed_step_ar` (`spec/variants/addressed_step_ar.md` §1, §3).

    `step_tokens` holds `0`, `1` (`(t_0,)`), or `2` (`(t_0, t_1)`) tokens already
    emitted within column `k` after the step-`k` index hint `[IDX] addr(k)`:
    - `0` tokens: step-`k` prompt ending at `[IDX] addr(k)` (`use_query_tokens=False`)
      or `[IDX] addr(k) Q_0` (`use_query_tokens=True`).
    - `1` token: appends `t_0` (plus `Q_1` when `use_query_tokens=True`), ready to predict `t_1`.
    - `2` tokens: appends `t_0` and `t_1` (interleaved with `Q_1` and `Q_2` when
      `use_query_tokens=True`), ready to predict `flag_k` (and matching the full
      teacher-forced sequence for step `k`).
    """
    emitted = tuple(int(t) for t in step_tokens)
    if len(emitted) > 2:
        raise ValueError(
            f"step_tokens may contain at most 2 tokens (t_0, t_1), got {len(emitted)}"
        )

    input_ids, token_type_ids, position_ids = build_addressed_prefix_tokens(
        alphabet, a, b, include_alphabet=include_alphabet
    )
    alphabet_tokens = tuple(alphabet)
    out_tokens = tuple(int(t) for t in output)
    k = len(out_tokens)

    # Incoming carry `[IDX] addr(k) c_k [SEP]`.
    append_index_header(input_ids, token_type_ids, position_ids, k, alphabet_tokens)
    input_ids.extend((int(carry), int(SpecialToken.SEP)))
    token_type_ids.extend((TokenType.CARRY, TokenType.SPECIAL))
    position_ids.extend((k, 0))

    # Historical outputs `[IDX] addr(0) o_0 ... [IDX] addr(k-1) o_{k-1} [SEP]`.
    for j, o_j in enumerate(out_tokens):
        append_index_header(input_ids, token_type_ids, position_ids, j, alphabet_tokens)
        input_ids.append(o_j)
        token_type_ids.append(TokenType.OUTPUT)
        position_ids.append(j)
    input_ids.append(int(SpecialToken.SEP))
    token_type_ids.append(TokenType.SPECIAL)
    position_ids.append(0)

    # Step-`k` index hint `[IDX] addr(k)` (length `P_k`).
    append_index_header(input_ids, token_type_ids, position_ids, k, alphabet_tokens)

    slot_metas = _step_slot_metadata(k, step_order)
    if not use_query_tokens:
        for idx, tok in enumerate(emitted):
            tt, pos = slot_metas[idx]
            input_ids.append(tok)
            token_type_ids.append(tt)
            position_ids.append(pos)
    else:
        q_tokens = step_query_tokens(step_order)
        input_ids.append(int(q_tokens[0]))
        token_type_ids.append(TokenType.SPECIAL)
        position_ids.append(k)
        for idx, tok in enumerate(emitted):
            tt, pos = slot_metas[idx]
            input_ids.extend((tok, int(q_tokens[idx + 1])))
            token_type_ids.extend((tt, TokenType.SPECIAL))
            position_ids.extend((pos, k))

    return make_example(input_ids, token_type_ids, position_ids, embedding_mode)


def build_addressed_step_ar_step_labelled_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    k: int,
    *,
    steps: Sequence[ColumnStep] | None = None,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> LabelledExample:
    """Build the teacher-forced `LabelledExample` for column `k` (`0 <= k <= K`; `spec/variants/addressed_step_ar.md` §3).

    Supervises the 3 active-step targets `(t_0, t_1, flag_k)` in a single forward pass:
    - `use_query_tokens=False`: `labels` ends with `(t_0, t_1, flag_k)` at the last
      3 positions (`addr(k)[-1] -> t_0`, `t_0 -> t_1`, `t_1 -> flag_k`).
    - `use_query_tokens=True`: `labels` ends with `(t_0, -100, t_1, -100, flag_k)` at
      the last 5 positions (`Q_0 -> t_0`, `Q_1 -> t_1`, `Q_2 -> flag_k`).
    """
    oracle_steps = tuple(steps) if steps is not None else reference_add(alphabet, a, b)
    if not (0 <= k < len(oracle_steps)):
        raise IndexError(f"Step index k={k} out of range 0..{len(oracle_steps) - 1}")

    step = oracle_steps[k]
    output_prefix = tuple(s.sum_digit for s in oracle_steps[:k])
    t_0, t_1 = pack_step_order(step.sum_digit, step.carry_out, step_order)
    flag_tok = int(step.flag)

    example = build_addressed_step_ar_example(
        alphabet,
        a,
        b,
        step.carry_in,
        output_prefix,
        (t_0, t_1),
        step_order=step_order,
        use_query_tokens=use_query_tokens,
        include_alphabet=include_alphabet,
        embedding_mode=embedding_mode,
    )
    if not use_query_tokens:
        masked_len = len(example) - 3
        labels = (IGNORE_INDEX,) * masked_len + (t_0, t_1, flag_tok)
    else:
        masked_len = len(example) - 5
        labels = (IGNORE_INDEX,) * masked_len + (
            t_0,
            IGNORE_INDEX,
            t_1,
            IGNORE_INDEX,
            flag_tok,
        )
    return LabelledExample(example=example, labels=labels)
