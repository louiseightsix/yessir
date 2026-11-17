"""Teacher-forced evaluation (`evaluate_ar`) for the `ar` variant of `addition`."""

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
    EmbeddingMode,
    StepOrder,
    step_query_tokens,
    unpack_step_order,
)
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.ar.inputs import (
    build_ar_example,
    build_ar_trace_tokens,
)
from yessir.batch import collate
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken


@torch.no_grad()
def evaluate_ar(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> EvalMetrics:
    """Evaluate `model` under teacher forcing across `num_problems` full-trace `ar` problems.

    Each problem is evaluated in a single forward pass across all `len(targets)`
    supervised positions (`P - 1 .. P + len(trace) - 1`), unpacked column by column
    into `StepPrediction`s (`spec/variants/ar.md` §3–§4), and scored via `score_predictions`.
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = StepOrder(step_order)
    q_0, q_1, _ = step_query_tokens(order)
    eos = int(SpecialToken.EOS)

    problem_list = list(itertools.islice(problems, num_problems))
    if len(problem_list) < num_problems:
        raise ValueError(
            f"problems stream exhausted early: needed {num_problems}, got {len(problem_list)}"
        )

    oracle_per_problem: list[tuple[tuple[int, ...], tuple[ColumnStep, ...], int]] = []
    examples: list[Example] = []

    for prob in problem_list:
        steps = reference_add(prob.alphabet, prob.a, prob.b)
        trace = build_ar_trace_tokens(
            steps, step_order=order, use_query_tokens=use_query_tokens
        )
        ex = build_ar_example(
            prob.alphabet,
            prob.a,
            prob.b,
            trace,
            step_order=order,
            use_query_tokens=use_query_tokens,
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
            raise ValueError("evaluate_ar requires a Transformer with SubspaceLMHead")
        argmax_ids = logits.argmax(dim=-1)
        for b_idx, (_, _, num_targets) in enumerate(meta_chunk):
            n = int(batch.lengths[b_idx].item())
            # Supervised positions span `n - num_targets .. n - 1` (`P - 1 .. P + len(trace) - 1`).
            preds_per_problem.append(argmax_ids[b_idx, n - num_targets : n].tolist())

    stride = 4 if use_query_tokens else 2
    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = []
    for (alphabet, steps, _), preds in zip(
        oracle_per_problem, preds_per_problem, strict=True
    ):
        last_k = len(steps) - 1
        final_pred = preds[-1]
        step_preds: list[StepPrediction] = []
        for k in range(len(steps)):
            col_slice = preds[stride * k : stride * (k + 1)]
            if use_query_tokens:
                pred_q0, pred_t0, pred_q1, pred_t1 = col_slice
                syntax_ok = (pred_q0 == int(q_0)) and (pred_q1 == int(q_1))
            else:
                pred_t0, pred_t1 = col_slice
                # Without query tokens, `col_slice[0]` (`preds[2 * k]`) is the column
                # boundary after `[SEP]` (`k = 0`) or column `k - 1` (`k >= 1`), so a
                # premature `[EOS]` there is a halting/syntax failure rather than an
                # out-of-alphabet digit error (`spec/task.md` §3, `spec/variants/ar.md` §4).
                syntax_ok = pred_t0 != eos
            pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, order)
            no_premature_eos = all(tok != eos for tok in col_slice)
            halt_ok = (
                no_premature_eos and (final_pred == eos)
                if k == last_k
                else no_premature_eos
            )
            step_preds.append(
                StepPrediction(
                    sum_digit=pred_o,
                    carry_out=pred_c,
                    syntax_ok=syntax_ok,
                    halt_ok=halt_ok,
                )
            )
        evaluated.append((alphabet, steps, tuple(step_preds)))

    return score_predictions(evaluated, step_order=order)
