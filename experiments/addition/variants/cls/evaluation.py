"""Free-running rollout evaluation (`evaluate_cls`) for the `cls` variant of `addition`."""

import itertools
from collections.abc import Iterable, Sequence

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
from yessir.special_tokens import SpecialToken


def _predict_cls_chunks(
    model: Transformer,
    examples: Sequence[Example],
    *,
    batch_size: int,
    device: torch.device,
) -> list[tuple[int, int, int]]:
    """Run `examples` in `batch_size` chunks and return `(pred_o, pred_c, pred_flag)` per row."""
    preds: list[tuple[int, int, int]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
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
        preds.extend(zip(preds_o, preds_c, preds_flag, strict=True))
    return preds


@torch.no_grad()
def evaluate_cls(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> EvalMetrics:
    """Evaluate `model` via batched free-running rollout across `num_problems` problems.

    Runs unassisted step-by-step execution starting from `carry = s_0` and
    `output = ()` (stopping per problem on `FLAG_FINISHED` or `max(L_a, L_b) + 1`),
    and computes `oracle_step_accuracy` across any reference columns not already
    visited on the ground-truth prefix during rollout (`spec/variants/cls.md` §3–§4).
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

    oracle_steps: list[tuple[ColumnStep, ...]] = []
    oracle_outputs: list[tuple[int, ...]] = []
    for prob in problem_list:
        steps = reference_add(prob.alphabet, prob.a, prob.b)
        oracle_steps.append(steps)
        oracle_outputs.append(tuple(s.sum_digit for s in steps))

    model.eval()
    device = model.device
    fin_tok = int(SpecialToken.FLAG_FINISHED)

    carries: list[int] = [prob.alphabet[0] for prob in problem_list]
    outputs: list[list[int]] = [[] for _ in problem_list]
    max_cols: list[int] = [max(len(prob.a), len(prob.b)) + 1 for prob in problem_list]
    emitted: list[list[tuple[int, int, int]]] = [[] for _ in problem_list]
    oracle_ok: list[list[bool | None]] = [[None] * len(steps) for steps in oracle_steps]

    active = list(range(num_problems))
    col_idx = 0
    while active:
        step_examples = [
            build_cls_example(
                problem_list[i].alphabet,
                problem_list[i].a,
                problem_list[i].b,
                carries[i],
                outputs[i],
                embedding_mode=embedding_mode,
            )
            for i in active
        ]
        step_preds = _predict_cls_chunks(
            model, step_examples, batch_size=batch_size, device=device
        )

        next_active: list[int] = []
        for i, (pred_o, pred_c, pred_flag) in zip(active, step_preds, strict=True):
            steps = oracle_steps[i]
            if (
                col_idx < len(steps)
                and carries[i] == steps[col_idx].carry_in
                and tuple(outputs[i]) == oracle_outputs[i][:col_idx]
            ):
                oracle_ok[i][col_idx] = (pred_o == steps[col_idx].sum_digit) and (
                    pred_c == steps[col_idx].carry_out
                )

            emitted[i].append((pred_o, pred_c, pred_flag))
            if pred_flag != fin_tok and (col_idx + 1) < max_cols[i]:
                carries[i] = pred_c
                outputs[i].append(pred_o)
                next_active.append(i)

        active = next_active
        col_idx += 1

    fallback_coords: list[tuple[int, int]] = []
    fallback_examples: list[Example] = []
    for i, (prob, steps) in enumerate(zip(problem_list, oracle_steps, strict=True)):
        for k, step in enumerate(steps):
            if oracle_ok[i][k] is None:
                fallback_coords.append((i, k))
                fallback_examples.append(
                    build_cls_example(
                        prob.alphabet,
                        prob.a,
                        prob.b,
                        step.carry_in,
                        oracle_outputs[i][:k],
                        embedding_mode=embedding_mode,
                    )
                )

    if fallback_examples:
        fallback_preds = _predict_cls_chunks(
            model, fallback_examples, batch_size=batch_size, device=device
        )
        for (i, k), (pred_o, pred_c, _) in zip(
            fallback_coords, fallback_preds, strict=True
        ):
            step = oracle_steps[i][k]
            oracle_ok[i][k] = (pred_o == step.sum_digit) and (pred_c == step.carry_out)

    total_oracle_steps = sum(len(steps) for steps in oracle_steps)
    oracle_ok_count = sum(1 for row in oracle_ok for ok in row if ok is True)

    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = []
    for prob, steps, raw_emitted in zip(
        problem_list, oracle_steps, emitted, strict=True
    ):
        step_preds_tuple = tuple(
            StepPrediction(
                sum_digit=pred_o,
                carry_out=pred_c,
                halt_ok=(k < len(steps) and pred_flag == int(steps[k].flag)),
            )
            for k, (pred_o, pred_c, pred_flag) in enumerate(raw_emitted)
        )
        evaluated.append((prob.alphabet, steps, step_preds_tuple))

    return score_predictions(
        evaluated,
        oracle_step_accuracy=oracle_ok_count / total_oracle_steps,
    )
