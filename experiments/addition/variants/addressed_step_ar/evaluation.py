"""Teacher-forced evaluation (`evaluate_addressed_step_ar`) for the `addressed_step_ar` variant of `addition`."""

import itertools
from collections.abc import Iterable

import torch

from experiments.addition.evaluation import (
    EvalMetrics,
    StepPrediction,
    score_predictions,
)
from experiments.addition.inputs import AdditionProblem, EmbeddingMode, StepOrder
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.addressed_step_ar.inputs import (
    build_addressed_step_ar_example,
    pack_step_order,
    unpack_step_order,
)
from yessir.batch import collate
from yessir.example import Example
from yessir.model import Transformer


@torch.no_grad()
def evaluate_addressed_step_ar(
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
    """Evaluate `model` under teacher forcing across all `K + 1` steps of `num_problems` problems.

    Each column `k` (`0 <= k <= K`) is evaluated in a single teacher-forced forward
    pass; greedy predictions are read from the 3 supervised positions relative to
    each row's unpadded length `n = batch.lengths[b]` (`(n - 3, n - 2, n - 1)` when
    `use_query_tokens=False`, or `(n - 5, n - 3, n - 1)` when `use_query_tokens=True`),
    unpacked according to `step_order`, and scored (`spec/variants/addressed_step_ar.md` §3–§4).
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = StepOrder(step_order)

    problem_list = list(itertools.islice(problems, num_problems))
    if len(problem_list) < num_problems:
        raise ValueError(
            f"problems stream exhausted early: needed {num_problems}, got {len(problem_list)}"
        )

    oracle_per_problem: list[tuple[tuple[int, ...], tuple[ColumnStep, ...]]] = []
    flat_examples: list[Example] = []

    for prob in problem_list:
        steps = reference_add(prob.alphabet, prob.a, prob.b)
        oracle_per_problem.append((prob.alphabet, steps))
        sum_digits = tuple(s.sum_digit for s in steps)
        for k, step in enumerate(steps):
            t_0, t_1 = pack_step_order(step.sum_digit, step.carry_out, order)
            flat_examples.append(
                build_addressed_step_ar_example(
                    prob.alphabet,
                    prob.a,
                    prob.b,
                    step.carry_in,
                    sum_digits[:k],
                    (t_0, t_1),
                    step_order=order,
                    use_query_tokens=use_query_tokens,
                    include_alphabet=include_alphabet,
                    embedding_mode=embedding_mode,
                )
            )

    model.eval()
    device = model.device
    flat_preds: list[tuple[int, int, int]] = []
    offsets = (-5, -3, -1) if use_query_tokens else (-3, -2, -1)

    for start in range(0, len(flat_examples), batch_size):
        chunk = flat_examples[start : start + batch_size]
        batch = collate(chunk).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(
                "evaluate_addressed_step_ar requires a Transformer with SubspaceLMHead"
            )
        row_indices = torch.arange(len(chunk), device=device)
        preds_t0 = (
            logits[row_indices, batch.lengths + offsets[0]].argmax(dim=-1).tolist()
        )
        preds_t1 = (
            logits[row_indices, batch.lengths + offsets[1]].argmax(dim=-1).tolist()
        )
        preds_flag = (
            logits[row_indices, batch.lengths + offsets[2]].argmax(dim=-1).tolist()
        )
        flat_preds.extend(zip(preds_t0, preds_t1, preds_flag, strict=True))

    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = []
    cursor = 0
    for alphabet, steps in oracle_per_problem:
        n_steps = len(steps)
        raw_slice = flat_preds[cursor : cursor + n_steps]
        cursor += n_steps
        step_preds: list[StepPrediction] = []
        for step, (pred_t0, pred_t1, pred_flag) in zip(steps, raw_slice, strict=True):
            pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, order)
            step_preds.append(
                StepPrediction(
                    sum_digit=pred_o,
                    carry_out=pred_c,
                    halt_ok=(pred_flag == int(step.flag)),
                )
            )
        evaluated.append((alphabet, steps, tuple(step_preds)))

    return score_predictions(evaluated, step_order=order)
