"""Free-running rollout evaluation (`evaluate_cls`) for the `cls` variant of `addition`."""

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
    TokenType,
    build_prefix_tokens,
    make_example,
)
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.cls.inputs import build_cls_example
from yessir.batch import collate, collate_left_padded
from yessir.blocks import KVCache
from yessir.example import Example
from yessir.heads import ParallelClassificationHead
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import autocast_context


def _predict_cls_chunks(
    model: Transformer,
    examples: Sequence[Example],
    *,
    batch_size: int,
    device: torch.device,
) -> list[tuple[int, int, int]]:
    """Uncached fallback helper running `examples` in `batch_size` chunks."""
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


def _run_cls_tail_step(
    model: Transformer,
    carries: Sequence[int],
    outputs: Sequence[Sequence[int]],
    *,
    col_idx: int,
    use_tt: bool,
    use_pe: bool,
    device: torch.device,
    kv_cache: KVCache,
) -> list[tuple[int, int, int]]:
    """Run `c_k [SEP] o_0 ... o_{k-1}` (uniform length `col_idx + 2`) against `kv_cache`."""
    sep = int(SpecialToken.SEP)
    n_rows = len(carries)
    tail_len = col_idx + 2
    tail_ids = torch.tensor(
        [[carries[r], sep, *outputs[r]] for r in range(n_rows)],
        dtype=torch.long,
        device=device,
    )
    tail_lengths = torch.full((n_rows,), tail_len, dtype=torch.long, device=device)
    tail_tt = (
        torch.tensor(
            [int(TokenType.CARRY), int(TokenType.SPECIAL)]
            + [int(TokenType.OUTPUT)] * col_idx,
            dtype=torch.long,
            device=device,
        )
        .unsqueeze(0)
        .expand(n_rows, -1)
        if use_tt
        else None
    )
    tail_pos = (
        torch.tensor(
            [col_idx, 0, *range(1, col_idx + 1)],
            dtype=torch.long,
            device=device,
        )
        .unsqueeze(0)
        .expand(n_rows, -1)
        if use_pe
        else None
    )
    logits = model(
        tail_ids,
        tail_lengths,
        token_type_ids=tail_tt,
        position_ids=tail_pos,
        kv_cache=kv_cache,
    )
    assert isinstance(logits, tuple) and len(logits) == 3
    logits_o, logits_c, logits_flag = logits
    return list(
        zip(
            logits_o.argmax(dim=-1).tolist(),
            logits_c.argmax(dim=-1).tolist(),
            logits_flag.argmax(dim=-1).tolist(),
            strict=True,
        )
    )


def _evaluate_cls_cached_chunk(
    model: Transformer,
    probs_chunk: Sequence[AdditionProblem],
    oracle_steps_chunk: Sequence[tuple[ColumnStep, ...]],
    oracle_outputs_chunk: Sequence[tuple[int, ...]],
    *,
    embedding_mode: EmbeddingMode,
    device: torch.device,
) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
    """Evaluate one `batch_size` problem chunk by caching `[BOS] alphabet [SEP] a [SEP] b [SEP]`."""
    if (
        not isinstance(model.head, ParallelClassificationHead)
        or len(model.head.heads) != 3
    ):
        raise ValueError(
            "evaluate_cls requires a Transformer with a 3-head ParallelClassificationHead"
        )

    chunk_size = len(probs_chunk)
    fin_tok = int(SpecialToken.FLAG_FINISHED)
    headers = [
        make_example(
            *build_prefix_tokens(prob.alphabet, prob.a, prob.b), embedding_mode
        )
        for prob in probs_chunk
    ]
    prefill = collate_left_padded(headers).to(device)
    header_len = int(prefill.input_ids.size(1))
    use_tt = prefill.token_type_ids is not None
    use_pe = prefill.position_ids is not None

    header_cache = model.init_kv_cache(header_len)
    dummy_lengths = torch.ones((chunk_size,), dtype=torch.long, device=device)
    model(
        prefill.input_ids,
        dummy_lengths,
        token_type_ids=prefill.token_type_ids,
        position_ids=prefill.position_ids,
        key_padding_mask=prefill.key_padding_mask,
        kv_cache=header_cache,
    )

    max_cols = [max(len(p.a), len(p.b)) + 1 for p in probs_chunk]
    max_seq_len = header_len + max(max_cols) + 1
    work_cache = header_cache.select_rows(range(chunk_size), max_seq_len=max_seq_len)

    carries: list[int] = [p.alphabet[0] for p in probs_chunk]
    outputs: list[list[int]] = [[] for _ in probs_chunk]
    emitted: list[list[tuple[int, int, int]]] = [[] for _ in probs_chunk]
    oracle_ok: list[list[bool | None]] = [
        [None] * len(steps) for steps in oracle_steps_chunk
    ]

    active = list(range(chunk_size))
    col_idx = 0
    while active:
        step_preds = _run_cls_tail_step(
            model,
            [carries[i] for i in active],
            [outputs[i] for i in active],
            col_idx=col_idx,
            use_tt=use_tt,
            use_pe=use_pe,
            device=device,
            kv_cache=work_cache,
        )

        next_active: list[int] = []
        for i, (pred_o, pred_c, pred_flag) in zip(active, step_preds, strict=True):
            steps = oracle_steps_chunk[i]
            if (
                col_idx < len(steps)
                and carries[i] == steps[col_idx].carry_in
                and tuple(outputs[i]) == oracle_outputs_chunk[i][:col_idx]
            ):
                oracle_ok[i][col_idx] = (pred_o == steps[col_idx].sum_digit) and (
                    pred_c == steps[col_idx].carry_out
                )

            emitted[i].append((pred_o, pred_c, pred_flag))
            if pred_flag != fin_tok and (col_idx + 1) < max_cols[i]:
                carries[i] = pred_c
                outputs[i].append(pred_o)
                next_active.append(i)

        if next_active:
            if len(next_active) == len(active):
                work_cache.truncate(header_len)
            else:
                work_cache = header_cache.select_rows(
                    next_active, max_seq_len=max_seq_len
                )
        active = next_active
        col_idx += 1

    max_oracle_cols = max(len(steps) for steps in oracle_steps_chunk)
    for k in range(max_oracle_cols):
        need_rows = [
            i
            for i in range(chunk_size)
            if k < len(oracle_steps_chunk[i]) and oracle_ok[i][k] is None
        ]
        if not need_rows:
            continue
        fb_cache = header_cache.select_rows(need_rows, max_seq_len=header_len + k + 2)
        fb_preds = _run_cls_tail_step(
            model,
            [oracle_steps_chunk[i][k].carry_in for i in need_rows],
            [oracle_outputs_chunk[i][:k] for i in need_rows],
            col_idx=k,
            use_tt=use_tt,
            use_pe=use_pe,
            device=device,
            kv_cache=fb_cache,
        )
        for i, (pred_o, pred_c, _) in zip(need_rows, fb_preds, strict=True):
            step = oracle_steps_chunk[i][k]
            oracle_ok[i][k] = (pred_o == step.sum_digit) and (pred_c == step.carry_out)

    oracle_ok_counts = [sum(1 for ok in row if ok is True) for row in oracle_ok]
    return emitted, oracle_ok_counts


@torch.no_grad()
def evaluate_cls(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
    per_base_metrics: dict[int, EvalMetrics] | None = None,
    dtype: str | torch.dtype | None = None,
) -> EvalMetrics:
    """Evaluate `model` via batched `KVCache`-accelerated free-running rollout across `num_problems` problems.

    Prefills each problem's static `[BOS] alphabet [SEP] a [SEP] b [SEP]` header once
    into `KVCache`, runs unassisted step-by-step execution starting from `carry = s_0`
    and `output = ()` (stopping per problem on `FLAG_FINISHED` or `max(L_a, L_b) + 1`),
    and computes `oracle_step_accuracy` across any reference columns not already
    visited on the ground-truth prefix during rollout (`spec/variants/cls.md` §3–§4).
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    device = model.device
    amp_ctx = autocast_context(device, dtype)

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

    if isinstance(model, Transformer):
        emitted: list[list[tuple[int, int, int]]] = []
        oracle_ok_counts: list[int] = []
        with amp_ctx:
            for start in range(0, num_problems, batch_size):
                end = min(num_problems, start + batch_size)
                chunk_emitted, chunk_oracle_counts = _evaluate_cls_cached_chunk(
                    model,
                    problem_list[start:end],
                    oracle_steps[start:end],
                    oracle_outputs[start:end],
                    embedding_mode=embedding_mode,
                    device=device,
                )
                emitted.extend(chunk_emitted)
                oracle_ok_counts.extend(chunk_oracle_counts)
    else:
        fin_tok = int(SpecialToken.FLAG_FINISHED)
        carries: list[int] = [prob.alphabet[0] for prob in problem_list]
        outputs: list[list[int]] = [[] for _ in problem_list]
        max_cols: list[int] = [
            max(len(prob.a), len(prob.b)) + 1 for prob in problem_list
        ]
        emitted = [[] for _ in problem_list]
        oracle_ok: list[list[bool | None]] = [
            [None] * len(steps) for steps in oracle_steps
        ]

        with amp_ctx:
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
                for i, (pred_o, pred_c, pred_flag) in zip(
                    active, step_preds, strict=True
                ):
                    steps = oracle_steps[i]
                    if (
                        col_idx < len(steps)
                        and carries[i] == steps[col_idx].carry_in
                        and tuple(outputs[i]) == oracle_outputs[i][:col_idx]
                    ):
                        oracle_ok[i][col_idx] = (
                            pred_o == steps[col_idx].sum_digit
                        ) and (pred_c == steps[col_idx].carry_out)

                    emitted[i].append((pred_o, pred_c, pred_flag))
                    if pred_flag != fin_tok and (col_idx + 1) < max_cols[i]:
                        carries[i] = pred_c
                        outputs[i].append(pred_o)
                        next_active.append(i)

                active = next_active
                col_idx += 1

            fallback_coords: list[tuple[int, int]] = []
            fallback_examples: list[Example] = []
            for i, (prob, steps) in enumerate(
                zip(problem_list, oracle_steps, strict=True)
            ):
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
                    oracle_ok[i][k] = (pred_o == step.sum_digit) and (
                        pred_c == step.carry_out
                    )

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
        oracle_ok_per_problem=oracle_ok_counts,
        per_base_metrics=per_base_metrics,
    )
