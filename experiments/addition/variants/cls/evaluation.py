"""Teacher-forced evaluation (`evaluate_cls`) for the `cls` variant of `addition`."""

import itertools
from collections.abc import Iterable

import torch

from experiments.addition.evaluation import (
    EvalMetrics,
    StepPrediction,
    score_predictions,
)
from experiments.addition.inputs import AdditionProblem, EmbeddingMode
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.cls.inputs import build_cls_example
from yessir.batch import collate
from yessir.example import Example
from yessir.model import Transformer


@torch.no_grad()
def evaluate_cls(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> EvalMetrics:
    """Evaluate `model` under teacher forcing across all `K + 1` steps of `num_problems` problems.

    Because the incoming carry slot `c_k` is updated in place from step to step,
    each column `k` (`0 <= k <= K`) has its own step `Example`. All step examples
    across the evaluated problems are flattened into `batch_size` chunks for
    batched GPU/CPU forward passes and then re-grouped by problem for scoring
    (`spec/variants/cls.md` §3–§4).
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

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
            flat_examples.append(
                build_cls_example(
                    prob.alphabet,
                    prob.a,
                    prob.b,
                    step.carry_in,
                    sum_digits[:k],
                    embedding_mode=embedding_mode,
                )
            )

    model.eval()
    device = model.device
    flat_preds: list[tuple[int, int, int]] = []

    for start in range(0, len(flat_examples), batch_size):
        chunk = flat_examples[start : start + batch_size]
        batch = collate(chunk).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, tuple) or len(logits) != 3:
            raise ValueError(
                "evaluate_cls requires a Transformer with a 3-head ParallelClassificationHead"
            )
        logits_o, logits_c, logits_flag = logits
        preds_o = logits_o.argmax(dim=-1).tolist()
        preds_c = logits_c.argmax(dim=-1).tolist()
        preds_flag = logits_flag.argmax(dim=-1).tolist()
        flat_preds.extend(zip(preds_o, preds_c, preds_flag, strict=True))

    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = []
    cursor = 0
    for alphabet, steps in oracle_per_problem:
        n_steps = len(steps)
        raw_slice = flat_preds[cursor : cursor + n_steps]
        cursor += n_steps
        step_preds = tuple(
            StepPrediction(
                sum_digit=pred_o,
                carry_out=pred_c,
                halt_ok=(pred_flag == int(step.flag)),
            )
            for step, (pred_o, pred_c, pred_flag) in zip(steps, raw_slice, strict=True)
        )
        evaluated.append((alphabet, steps, step_preds))

    return score_predictions(evaluated)
