"""Free-running rollout evaluation (`evaluate_step_fork`) for the `step_fork` variant of `addition`."""

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
    AdditionToken,
    EmbeddingMode,
    TokenType,
    build_prefix_tokens,
    make_example,
)
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.step_fork.inputs import (
    STEP_FORK_QUERY_TOKENS,
    build_step_fork_examples,
)
from yessir.batch import collate, collate_left_padded
from yessir.blocks import KVCache
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import autocast_context


def _predict_fork_chunks(
    model: Transformer,
    branch_examples: Sequence[Example],
    *,
    batch_size: int,
    device: torch.device,
) -> list[tuple[int, int, int]]:
    """Uncached fallback helper running `[S], [C], [F]` triples in `3 * batch_size` row chunks."""
    flat_tokens: list[int] = []
    row_chunk_size = 3 * batch_size
    for start in range(0, len(branch_examples), row_chunk_size):
        chunk = branch_examples[start : start + row_chunk_size]
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

    return [
        (flat_tokens[idx], flat_tokens[idx + 1], flat_tokens[idx + 2])
        for idx in range(0, len(flat_tokens), 3)
    ]


def _run_step_fork_queries(
    model: Transformer,
    carries: Sequence[int],
    outputs: Sequence[Sequence[int]],
    query_tokens: Sequence[AdditionToken],
    *,
    col_idx: int,
    use_tt: bool,
    use_pe: bool,
    device: torch.device,
    kv_cache: KVCache,
) -> list[tuple[int, ...]]:
    """Prefill `c_k [SEP] o_0 ... o_{k-1}` once into `kv_cache`, then query each `q_tok` via `truncate`."""
    sep = int(SpecialToken.SEP)
    n_rows = len(carries)
    tail_ids = torch.tensor(
        [[carries[r], sep, *outputs[r]] for r in range(n_rows)],
        dtype=torch.long,
        device=device,
    )
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
            [col_idx, 0, *range(col_idx)],
            dtype=torch.long,
            device=device,
        )
        .unsqueeze(0)
        .expand(n_rows, -1)
        if use_pe
        else None
    )
    model(
        tail_ids,
        token_type_ids=tail_tt,
        position_ids=tail_pos,
        kv_cache=kv_cache,
    )
    base_len = kv_cache.seq_len

    q_tt = (
        torch.full((n_rows, 1), int(TokenType.SPECIAL), dtype=torch.long, device=device)
        if use_tt
        else None
    )
    q_pos = (
        torch.full((n_rows, 1), col_idx, dtype=torch.long, device=device)
        if use_pe
        else None
    )
    per_query_preds: list[list[int]] = []
    for q_idx, q_tok in enumerate(query_tokens):
        if q_idx > 0:
            kv_cache.truncate(base_len)
        q_ids = torch.full((n_rows, 1), int(q_tok), dtype=torch.long, device=device)
        logits = model(
            q_ids,
            token_type_ids=q_tt,
            position_ids=q_pos,
            kv_cache=kv_cache,
        )
        assert isinstance(logits, torch.Tensor)
        per_query_preds.append(logits[:, 0, :].argmax(dim=-1).tolist())

    return [tuple(col) for col in zip(*per_query_preds, strict=True)]


def _evaluate_step_fork_cached_chunk(
    model: Transformer,
    probs_chunk: Sequence[AdditionProblem],
    oracle_steps_chunk: Sequence[tuple[ColumnStep, ...]],
    oracle_outputs_chunk: Sequence[tuple[int, ...]],
    *,
    embedding_mode: EmbeddingMode,
    device: torch.device,
) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
    """Evaluate one `batch_size` problem chunk by caching `[BOS] ... b [SEP]` and intra-step base prefixes."""
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
    prefill_out = model(
        prefill.input_ids,
        token_type_ids=prefill.token_type_ids,
        position_ids=prefill.position_ids,
        key_padding_mask=prefill.key_padding_mask,
        kv_cache=header_cache,
    )
    if not isinstance(prefill_out, torch.Tensor) or prefill_out.ndim != 3:
        raise ValueError(
            "evaluate_step_fork requires a Transformer with SubspaceLMHead"
        )

    max_cols = [max(len(p.a), len(p.b)) + 1 for p in probs_chunk]
    max_seq_len = header_len + max(max_cols) + 2
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
        step_preds = _run_step_fork_queries(
            model,
            [carries[i] for i in active],
            [outputs[i] for i in active],
            STEP_FORK_QUERY_TOKENS,
            col_idx=col_idx,
            use_tt=use_tt,
            use_pe=use_pe,
            device=device,
            kv_cache=work_cache,
        )

        next_active: list[int] = []
        for i, pred_tup in zip(active, step_preds, strict=True):
            pred_o, pred_c, pred_flag = pred_tup[0], pred_tup[1], pred_tup[2]
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
    oracle_queries = (AdditionToken.QUERY_SUM, AdditionToken.QUERY_CARRY)
    for k in range(max_oracle_cols):
        need_rows = [
            i
            for i in range(chunk_size)
            if k < len(oracle_steps_chunk[i]) and oracle_ok[i][k] is None
        ]
        if not need_rows:
            continue
        fb_cache = header_cache.select_rows(need_rows, max_seq_len=header_len + k + 3)
        fb_preds = _run_step_fork_queries(
            model,
            [oracle_steps_chunk[i][k].carry_in for i in need_rows],
            [oracle_outputs_chunk[i][:k] for i in need_rows],
            oracle_queries,
            col_idx=k,
            use_tt=use_tt,
            use_pe=use_pe,
            device=device,
            kv_cache=fb_cache,
        )
        for i, pred_pair in zip(need_rows, fb_preds, strict=True):
            step = oracle_steps_chunk[i][k]
            oracle_ok[i][k] = (pred_pair[0] == step.sum_digit) and (
                pred_pair[1] == step.carry_out
            )

    oracle_ok_counts = [sum(1 for ok in row if ok is True) for row in oracle_ok]
    return emitted, oracle_ok_counts


@torch.no_grad()
def evaluate_step_fork(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
    per_base_metrics: dict[int, EvalMetrics] | None = None,
    dtype: str | torch.dtype | None = None,
) -> EvalMetrics:
    """Evaluate `model` via batched `KVCache`-accelerated rollout across `num_problems` problems.

    Prefills each problem's static `[BOS] alphabet [SEP] a [SEP] b [SEP]` header once
    into `KVCache`, encodes column `k`'s `c_k [SEP] o_0 ... o_{k-1}` tail once per
    step before querying `[S]`, `[C]`, and `[F]` via 1-token `KVCache.truncate`
    rewinds, and computes `oracle_step_accuracy` across any reference columns not
    already visited on the ground-truth prefix (`spec/variants/step_fork.md` §3–§4).
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
                chunk_emitted, chunk_oracle_counts = _evaluate_step_fork_cached_chunk(
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
                branch_examples = [
                    ex
                    for i in active
                    for ex in build_step_fork_examples(
                        problem_list[i].alphabet,
                        problem_list[i].a,
                        problem_list[i].b,
                        carries[i],
                        outputs[i],
                        embedding_mode=embedding_mode,
                    )
                ]
                step_preds = _predict_fork_chunks(
                    model, branch_examples, batch_size=batch_size, device=device
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
            fallback_branches: list[Example] = []
            for i, (prob, steps) in enumerate(
                zip(problem_list, oracle_steps, strict=True)
            ):
                for k, step in enumerate(steps):
                    if oracle_ok[i][k] is None:
                        fallback_coords.append((i, k))
                        fallback_branches.extend(
                            build_step_fork_examples(
                                prob.alphabet,
                                prob.a,
                                prob.b,
                                step.carry_in,
                                oracle_outputs[i][:k],
                                embedding_mode=embedding_mode,
                            )
                        )

            if fallback_branches:
                fallback_preds = _predict_fork_chunks(
                    model, fallback_branches, batch_size=batch_size, device=device
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
