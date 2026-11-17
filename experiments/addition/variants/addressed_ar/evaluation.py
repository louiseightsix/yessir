"""Teacher-forced evaluation (`evaluate_addressed_ar`) for the `addressed_ar` variant of `addition`."""

import itertools
from collections.abc import Iterable

import torch

from experiments.addition.evaluation import (
    EvalMetrics,
    StepPrediction,
    score_predictions,
)
from experiments.addition.inputs import (
    AdditionProblem,
    AdditionToken,
    EmbeddingMode,
    StepOrder,
    encode_address,
    step_query_tokens,
    unpack_step_order,
)
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.addressed_ar.inputs import (
    build_addressed_ar_example,
    build_addressed_ar_trace_tokens,
)
from yessir.batch import collate
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken


@torch.no_grad()
def evaluate_addressed_ar(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> EvalMetrics:
    """Evaluate `model` under teacher forcing across `num_problems` full-trace `addressed_ar` problems.

    Each problem is evaluated in a single forward pass across all `len(targets)`
    supervised trace positions (`P - 1 .. P + len(trace) - 1`), parsed column by
    column across `[IDX] addr(k) ...` entries into `StepPrediction`s
    (`spec/variants/addressed_ar.md` §3–§4), and scored via `score_predictions`.
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = StepOrder(step_order)
    q_0, q_1, _ = step_query_tokens(order)
    idx_tok = int(AdditionToken.IDX)
    eos = int(SpecialToken.EOS)
    body_len = 4 if use_query_tokens else 2

    problem_list = list(itertools.islice(problems, num_problems))
    if len(problem_list) < num_problems:
        raise ValueError(
            f"problems stream exhausted early: needed {num_problems}, got {len(problem_list)}"
        )

    oracle_per_problem: list[tuple[tuple[int, ...], tuple[ColumnStep, ...], int]] = []
    examples: list[Example] = []

    for prob in problem_list:
        steps = reference_add(prob.alphabet, prob.a, prob.b)
        trace = build_addressed_ar_trace_tokens(
            steps,
            prob.alphabet,
            step_order=order,
            use_query_tokens=use_query_tokens,
        )
        ex = build_addressed_ar_example(
            prob.alphabet,
            prob.a,
            prob.b,
            trace,
            step_order=order,
            use_query_tokens=use_query_tokens,
            include_alphabet=include_alphabet,
            embedding_mode=embedding_mode,
        )
        num_targets = len(trace) + 1
        oracle_per_problem.append((prob.alphabet, steps, num_targets))
        examples.append(ex)

    model.eval()
    device = model.device
    preds_per_problem: list[list[int]] = []

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        meta_chunk = oracle_per_problem[start : start + batch_size]
        batch = collate(chunk).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(
                "evaluate_addressed_ar requires a Transformer with SubspaceLMHead"
            )
        argmax_ids = logits.argmax(dim=-1)
        for b_idx, (_, _, num_targets) in enumerate(meta_chunk):
            n = int(batch.lengths[b_idx].item())
            # Supervised positions span `n - num_targets .. n - 1` (`P - 1 .. P + len(trace) - 1`).
            preds_per_problem.append(argmax_ids[b_idx, n - num_targets : n].tolist())

    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = []
    for (alphabet, steps, _), preds in zip(
        oracle_per_problem, preds_per_problem, strict=True
    ):
        last_k = len(steps) - 1
        step_preds: list[StepPrediction] = []
        col_offset = 0

        for k in range(len(steps)):
            addr_k = encode_address(k, alphabet)
            addr_len = len(addr_k)
            entry_len = 1 + addr_len + body_len
            col_slice = preds[col_offset : col_offset + entry_len]
            after_col_pred = preds[col_offset + entry_len]
            col_offset += entry_len

            header_ok = (col_slice[0] == idx_tok) and (
                tuple(col_slice[1 : 1 + addr_len]) == addr_k
            )
            body = col_slice[1 + addr_len :]
            if use_query_tokens:
                pred_q0, pred_t0, pred_q1, pred_t1 = body
                syntax_ok = (
                    header_ok and (pred_q0 == int(q_0)) and (pred_q1 == int(q_1))
                )
            else:
                pred_t0, pred_t1 = body
                syntax_ok = header_ok

            pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, order)
            no_premature_eos = all(tok != eos for tok in col_slice)
            expected_after = eos if k == last_k else idx_tok
            halt_ok = no_premature_eos and (after_col_pred == expected_after)

            step_preds.append(
                StepPrediction(
                    sum_digit=pred_o,
                    carry_out=pred_c,
                    syntax_ok=syntax_ok,
                    halt_ok=halt_ok,
                )
            )

        assert col_offset == len(preds) - 1
        evaluated.append((alphabet, steps, tuple(step_preds)))

    return score_predictions(evaluated, step_order=order)
