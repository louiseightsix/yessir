"""Free-running rollout evaluation (`evaluate_addressed_step_ar`) for the `addressed_step_ar` variant of `addition`."""

import itertools
from collections.abc import Iterable, Sequence

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
from yessir.special_tokens import SpecialToken


def _predict_last_token_chunks(
    model: Transformer,
    examples: Sequence[Example],
    *,
    batch_size: int,
    device: torch.device,
) -> list[int]:
    """Run `examples` in `batch_size` chunks and return greedy `argmax` at `lengths - 1`."""
    preds: list[int] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
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
        preds.extend(logits[row_indices, batch.lengths - 1].argmax(dim=-1).tolist())
    return preds


def _predict_oracle_pairs_chunks(
    model: Transformer,
    examples: Sequence[Example],
    *,
    use_query_tokens: bool,
    batch_size: int,
    device: torch.device,
) -> list[tuple[int, int]]:
    """Read teacher-forced `(pred_t0, pred_t1)` in one pass per chunk for `oracle_step_accuracy`."""
    offsets = (-5, -3) if use_query_tokens else (-3, -2)
    pairs: list[tuple[int, int]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
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
        pairs.extend(zip(preds_t0, preds_t1, strict=True))
    return pairs


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
    per_base_metrics: dict[int, EvalMetrics] | None = None,
) -> EvalMetrics:
    """Evaluate `model` via batched free-running rollout across `num_problems` problems.

    At each active column `k`, decodes `(t_0, t_1, flag_k)` autoregressively in 3
    batched passes (eliminating intra-step oracle conditioning), feeds the model's
    own `(o_k, c_{k+1})` into column `k + 1` until `FLAG_FINISHED` or
    `max(L_a, L_b) + 1`, and computes `oracle_step_accuracy` across any reference
    columns not already visited on the ground-truth prefix (`spec/variants/addressed_step_ar.md` §3–§4).
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
        partial: list[list[int]] = [[] for _ in active]
        for _ in range(3):
            pass_examples = [
                build_addressed_step_ar_example(
                    problem_list[i].alphabet,
                    problem_list[i].a,
                    problem_list[i].b,
                    carries[i],
                    outputs[i],
                    partial[pos],
                    step_order=order,
                    use_query_tokens=use_query_tokens,
                    include_alphabet=include_alphabet,
                    embedding_mode=embedding_mode,
                )
                for pos, i in enumerate(active)
            ]
            pass_preds = _predict_last_token_chunks(
                model, pass_examples, batch_size=batch_size, device=device
            )
            for pos, tok in enumerate(pass_preds):
                partial[pos].append(tok)

        next_active: list[int] = []
        for i, (t_0, t_1, pred_flag) in zip(active, partial, strict=True):
            pred_o, pred_c = unpack_step_order(t_0, t_1, order)
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
                t_0, t_1 = pack_step_order(step.sum_digit, step.carry_out, order)
                fallback_examples.append(
                    build_addressed_step_ar_example(
                        prob.alphabet,
                        prob.a,
                        prob.b,
                        step.carry_in,
                        oracle_outputs[i][:k],
                        (t_0, t_1),
                        step_order=order,
                        use_query_tokens=use_query_tokens,
                        include_alphabet=include_alphabet,
                        embedding_mode=embedding_mode,
                    )
                )

    if fallback_examples:
        fallback_pairs = _predict_oracle_pairs_chunks(
            model,
            fallback_examples,
            use_query_tokens=use_query_tokens,
            batch_size=batch_size,
            device=device,
        )
        for (i, k), (pred_t0, pred_t1) in zip(
            fallback_coords, fallback_pairs, strict=True
        ):
            pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, order)
            step = oracle_steps[i][k]
            oracle_ok[i][k] = (pred_o == step.sum_digit) and (pred_c == step.carry_out)

    oracle_ok_counts = [sum(1 for ok in row if ok is True) for row in oracle_ok]

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
        step_order=order,
        oracle_ok_per_problem=oracle_ok_counts,
        per_base_metrics=per_base_metrics,
    )
