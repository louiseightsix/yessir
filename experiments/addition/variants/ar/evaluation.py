"""Free-running rollout evaluation (`evaluate_ar`) for the `ar` variant of `addition`."""

import itertools
from collections.abc import Iterable, Sequence

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
    _trace_slot_metadata,
    build_ar_example,
    build_ar_trace_tokens,
)
from yessir.batch import collate, collate_left_padded
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken


@torch.no_grad()
def _rollout_ar_chunk(
    model: Transformer,
    problems_chunk: Sequence[AdditionProblem],
    max_tokens_chunk: Sequence[int],
    *,
    step_order: StepOrder,
    use_query_tokens: bool,
    embedding_mode: EmbeddingMode,
    device: torch.device,
) -> list[list[int]]:
    """Run batched greedy rollout on `problems_chunk` using `KVCache` for `Transformer`."""
    eos = int(SpecialToken.EOS)
    pad = int(SpecialToken.PAD)
    chunk_size = len(problems_chunk)
    emitted: list[list[int]] = [[] for _ in range(chunk_size)]
    done: list[bool] = [False] * chunk_size

    if isinstance(model, Transformer):
        prefix_examples = [
            build_ar_example(
                prob.alphabet,
                prob.a,
                prob.b,
                (),
                step_order=step_order,
                use_query_tokens=use_query_tokens,
                embedding_mode=embedding_mode,
            )
            for prob in problems_chunk
        ]
        prefill = collate_left_padded(prefix_examples).to(device)
        kv_cache = model.init_kv_cache(
            int(prefill.input_ids.size(1)) + max(max_tokens_chunk)
        )
        logits = model(
            prefill.input_ids,
            token_type_ids=prefill.token_type_ids,
            position_ids=prefill.position_ids,
            key_padding_mask=prefill.key_padding_mask,
            kv_cache=kv_cache,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("evaluate_ar requires a Transformer with SubspaceLMHead")

        next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
        for b_idx, tok in enumerate(next_tokens):
            emitted[b_idx].append(tok)
            if tok == eos or len(emitted[b_idx]) >= max_tokens_chunk[b_idx]:
                done[b_idx] = True

        mode = EmbeddingMode(embedding_mode)
        use_tt = mode in (EmbeddingMode.TOKEN_TYPE_PE, EmbeddingMode.TOKEN_TYPE_NOPE)
        use_pe = mode == EmbeddingMode.TOKEN_TYPE_PE
        slot_idx = 0
        while not all(done):
            tt, pos = _trace_slot_metadata(
                slot_idx,
                step_order=step_order,
                use_query_tokens=use_query_tokens,
            )
            step_ids = torch.tensor(
                [
                    [pad if done[b_idx] else emitted[b_idx][-1]]
                    for b_idx in range(chunk_size)
                ],
                dtype=torch.long,
                device=device,
            )
            step_tt = (
                torch.full((chunk_size, 1), int(tt), dtype=torch.long, device=device)
                if use_tt
                else None
            )
            step_pos = (
                torch.full((chunk_size, 1), pos, dtype=torch.long, device=device)
                if use_pe
                else None
            )
            logits = model(
                step_ids,
                token_type_ids=step_tt,
                position_ids=step_pos,
                kv_cache=kv_cache,
            )
            assert isinstance(logits, torch.Tensor)
            next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
            for b_idx, tok in enumerate(next_tokens):
                if not done[b_idx]:
                    emitted[b_idx].append(tok)
                    if tok == eos or len(emitted[b_idx]) >= max_tokens_chunk[b_idx]:
                        done[b_idx] = True
            slot_idx += 1
        return emitted

    while not all(done):
        active = [b_idx for b_idx, is_done in enumerate(done) if not is_done]
        step_examples = [
            build_ar_example(
                problems_chunk[b_idx].alphabet,
                problems_chunk[b_idx].a,
                problems_chunk[b_idx].b,
                emitted[b_idx],
                step_order=step_order,
                use_query_tokens=use_query_tokens,
                embedding_mode=embedding_mode,
            )
            for b_idx in active
        ]
        batch = collate(step_examples).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("evaluate_ar requires a Transformer with SubspaceLMHead")
        row_indices = torch.arange(len(active), device=device)
        next_tokens = logits[row_indices, batch.lengths - 1].argmax(dim=-1).tolist()
        for b_idx, tok in zip(active, next_tokens, strict=True):
            emitted[b_idx].append(tok)
            if tok == eos or len(emitted[b_idx]) >= max_tokens_chunk[b_idx]:
                done[b_idx] = True
    return emitted


def _parse_ar_rollout(
    emitted: Sequence[int],
    *,
    num_steps: int,
    step_order: StepOrder,
    use_query_tokens: bool,
) -> tuple[StepPrediction, ...]:
    """Unpack emitted free-running `ar` tokens into up to `num_steps` `StepPrediction`s."""
    eos = int(SpecialToken.EOS)
    if not emitted or emitted[0] == eos:
        return ()

    q_0, q_1, _ = step_query_tokens(step_order)
    stride = 4 if use_query_tokens else 2
    last_k = num_steps - 1
    step_preds: list[StepPrediction] = []

    for k in range(num_steps):
        start = stride * k
        if start >= len(emitted) or emitted[start] == eos:
            break
        col_slice = emitted[start : start + stride]

        if len(col_slice) == stride:
            if use_query_tokens:
                pred_q0, pred_t0, pred_q1, pred_t1 = col_slice
                syntax_ok = (pred_q0 == int(q_0)) and (pred_q1 == int(q_1))
            else:
                pred_t0, pred_t1 = col_slice
                syntax_ok = True
            pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, step_order)
            after_idx = start + stride
            if k == last_k:
                halt_ok = (
                    (pred_t1 != eos)
                    and (len(emitted) == after_idx + 1)
                    and (emitted[after_idx] == eos)
                )
            else:
                halt_ok = (
                    (pred_t1 != eos)
                    and (after_idx < len(emitted))
                    and (emitted[after_idx] != eos)
                )
            step_preds.append(
                StepPrediction(
                    sum_digit=pred_o,
                    carry_out=pred_c,
                    syntax_ok=syntax_ok,
                    halt_ok=halt_ok,
                )
            )
        else:
            # Mid-column premature `[EOS]` with `use_query_tokens=True` (`len(col_slice)` is 2 or 3).
            if len(col_slice) == 2:
                syntax_ok = col_slice[0] == int(q_0)
                pred_o, pred_c = unpack_step_order(eos, eos, step_order)
            else:
                syntax_ok = False
                pred_o, pred_c = unpack_step_order(col_slice[1], eos, step_order)
            step_preds.append(
                StepPrediction(
                    sum_digit=pred_o,
                    carry_out=pred_c,
                    syntax_ok=syntax_ok,
                    halt_ok=False,
                )
            )
            break

    return tuple(step_preds)


def _compute_ar_oracle_step_accuracy(
    model: Transformer,
    problems: Sequence[AdditionProblem],
    oracle_steps: Sequence[tuple[ColumnStep, ...]],
    *,
    batch_size: int,
    step_order: StepOrder,
    use_query_tokens: bool,
    embedding_mode: EmbeddingMode,
    device: torch.device,
) -> float:
    """Compute teacher-forced `oracle_step_accuracy` across all `K + 1` columns in one pass per chunk."""
    eos = int(SpecialToken.EOS)
    q_0, q_1, _ = step_query_tokens(step_order)
    stride = 4 if use_query_tokens else 2

    examples: list[Example] = []
    num_targets_list: list[int] = []
    for prob, steps in zip(problems, oracle_steps, strict=True):
        trace = build_ar_trace_tokens(
            steps, step_order=step_order, use_query_tokens=use_query_tokens
        )
        examples.append(
            build_ar_example(
                prob.alphabet,
                prob.a,
                prob.b,
                trace,
                step_order=step_order,
                use_query_tokens=use_query_tokens,
                embedding_mode=embedding_mode,
            )
        )
        num_targets_list.append(len(trace) + 1)

    total_steps = 0
    oracle_ok_count = 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        steps_chunk = oracle_steps[start : start + batch_size]
        targets_chunk = num_targets_list[start : start + batch_size]
        batch = collate(chunk).to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(logits, torch.Tensor)
        argmax_ids = logits.argmax(dim=-1)
        for b_idx, (steps, num_targets) in enumerate(
            zip(steps_chunk, targets_chunk, strict=True)
        ):
            n = int(batch.lengths[b_idx].item())
            preds = argmax_ids[b_idx, n - num_targets : n].tolist()
            for k, step in enumerate(steps):
                total_steps += 1
                col_slice = preds[stride * k : stride * (k + 1)]
                if use_query_tokens:
                    pred_q0, pred_t0, pred_q1, pred_t1 = col_slice
                    syntax_ok = (pred_q0 == int(q_0)) and (pred_q1 == int(q_1))
                else:
                    pred_t0, pred_t1 = col_slice
                    syntax_ok = pred_t0 != eos
                pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, step_order)
                if (
                    syntax_ok
                    and (pred_o == step.sum_digit)
                    and (pred_c == step.carry_out)
                ):
                    oracle_ok_count += 1

    return oracle_ok_count / total_steps


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
    """Evaluate `model` via KV-cached free-running rollout across `num_problems` full-trace `ar` problems.

    Decodes each problem autoregressively from its `[BOS] ... [SEP]` prompt until
    `SpecialToken.EOS` or `(4 if use_query_tokens else 2) * (max(L_a, L_b) + 1) + 1`
    tokens (`spec/variants/ar.md` §3–§4), unpacks emitted columns into
    `StepPrediction`s, and records single-pass teacher-forced `oracle_step_accuracy`
    alongside rollout metrics via `score_predictions`.
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = StepOrder(step_order)
    tokens_per_col = 4 if use_query_tokens else 2

    problem_list = list(itertools.islice(problems, num_problems))
    if len(problem_list) < num_problems:
        raise ValueError(
            f"problems stream exhausted early: needed {num_problems}, got {len(problem_list)}"
        )

    oracle_steps = [
        reference_add(prob.alphabet, prob.a, prob.b) for prob in problem_list
    ]
    max_tokens_all = [
        tokens_per_col * (max(len(prob.a), len(prob.b)) + 1) + 1
        for prob in problem_list
    ]

    model.eval()
    device = model.device
    emitted_per_problem: list[list[int]] = []

    for start in range(0, num_problems, batch_size):
        emitted_per_problem.extend(
            _rollout_ar_chunk(
                model,
                problem_list[start : start + batch_size],
                max_tokens_all[start : start + batch_size],
                step_order=order,
                use_query_tokens=use_query_tokens,
                embedding_mode=embedding_mode,
                device=device,
            )
        )

    oracle_step_acc = _compute_ar_oracle_step_accuracy(
        model,
        problem_list,
        oracle_steps,
        batch_size=batch_size,
        step_order=order,
        use_query_tokens=use_query_tokens,
        embedding_mode=embedding_mode,
        device=device,
    )

    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = [
        (
            prob.alphabet,
            steps,
            _parse_ar_rollout(
                emitted,
                num_steps=len(steps),
                step_order=order,
                use_query_tokens=use_query_tokens,
            ),
        )
        for prob, steps, emitted in zip(
            problem_list, oracle_steps, emitted_per_problem, strict=True
        )
    ]

    return score_predictions(
        evaluated,
        step_order=order,
        oracle_step_accuracy=oracle_step_acc,
    )
