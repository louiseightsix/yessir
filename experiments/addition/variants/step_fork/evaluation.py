"""Teacher-forced evaluation (`evaluate_step_fork`) for the `step_fork` variant of `addition`."""

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
from experiments.addition.variants.step_fork.inputs import build_step_fork_examples
from yessir.batch import collate
from yessir.example import Example
from yessir.model import Transformer


@torch.no_grad()
def evaluate_step_fork(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> EvalMetrics:
    """Evaluate `model` under teacher forcing across all `K + 1` steps of `num_problems` problems.

    Each column `k` (`0 <= k <= K`) expands into 3 branched query `Example`s
    (`[S]`, `[C]`, `[F]`). All branched examples across the evaluated problems are
    batched in chunks of `3 * batch_size` rows (`batch_size` column transitions per
    forward pass), decoded at each row's unpadded final query position
    (`batch.lengths - 1`), and re-grouped per step and problem (`spec/variants/step_fork.md` §3–§4).
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
            flat_examples.extend(
                build_step_fork_examples(
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
    flat_tokens: list[int] = []
    row_chunk_size = 3 * batch_size

    for start in range(0, len(flat_examples), row_chunk_size):
        chunk = flat_examples[start : start + row_chunk_size]
        batch = collate(chunk).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(
                "evaluate_step_fork requires a Transformer with SubspaceLMHead"
            )
        row_indices = torch.arange(len(chunk), device=device)
        final_logits = logits[row_indices, batch.lengths - 1]
        flat_tokens.extend(final_logits.argmax(dim=-1).tolist())

    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = []
    cursor = 0
    for alphabet, steps in oracle_per_problem:
        step_preds: list[StepPrediction] = []
        for step in steps:
            pred_o, pred_c, pred_flag = flat_tokens[cursor : cursor + 3]
            cursor += 3
            step_preds.append(
                StepPrediction(
                    sum_digit=pred_o,
                    carry_out=pred_c,
                    halt_ok=(pred_flag == int(step.flag)),
                )
            )
        evaluated.append((alphabet, steps, tuple(step_preds)))

    return score_predictions(evaluated)
