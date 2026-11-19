"""Free-running rollout evaluation (`evaluate_addressed_step_ar`) for the `addressed_step_ar` variant of `addition`."""

import itertools
from collections.abc import Iterable, Iterator, Sequence

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
    TokenType,
    append_index_header,
    build_addressed_prefix_tokens,
    encode_address,
    make_example,
    step_query_tokens,
)
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.addressed_step_ar.inputs import (
    _step_slot_metadata,
    build_addressed_step_ar_example,
    pack_step_order,
    unpack_step_order,
)
from yessir.batch import collate, collate_left_padded
from yessir.blocks import KVCache
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import autocast_context

# Caps per-chunk KV-cache / padded token allocation (`chunk_len * max_seq_len <= 65_536`)
# so default `eval_batch_size = 64` on `B = 2, L = 40` (`802` tokens -> `51,328`)
# runs unsplit while large `eval_batch_size` values (`256`–`512`) sub-chunk safely.
MAX_EVAL_CHUNK_TOKENS = 65_536


def _iter_token_bounded_chunks(
    examples: Sequence[Example],
    *,
    batch_size: int,
    max_chunk_tokens: int | None = None,
) -> Iterator[Sequence[Example]]:
    """Yield slices of `examples` of at most `batch_size` items and `max_chunk_tokens` padded tokens."""
    budget = MAX_EVAL_CHUNK_TOKENS if max_chunk_tokens is None else max_chunk_tokens
    for start in range(0, len(examples), batch_size):
        outer = examples[start : start + batch_size]
        max_len = max(len(ex) for ex in outer)
        sub_size = max(1, min(len(outer), budget // max(1, max_len)))
        for sub in range(0, len(outer), sub_size):
            yield outer[sub : sub + sub_size]


def _predict_last_token_chunks(
    model: Transformer,
    examples: Sequence[Example],
    *,
    batch_size: int,
    device: torch.device,
    max_chunk_tokens: int | None = None,
) -> list[int]:
    """Uncached fallback helper running `examples` in token-bounded `batch_size` chunks."""
    preds: list[int] = []
    for chunk in _iter_token_bounded_chunks(
        examples, batch_size=batch_size, max_chunk_tokens=max_chunk_tokens
    ):
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
    max_chunk_tokens: int | None = None,
) -> list[tuple[int, int]]:
    """Uncached fallback helper reading teacher-forced `(pred_t0, pred_t1)` per chunk."""
    offsets = (-5, -3) if use_query_tokens else (-3, -2)
    pairs: list[tuple[int, int]] = []
    for chunk in _iter_token_bounded_chunks(
        examples, batch_size=batch_size, max_chunk_tokens=max_chunk_tokens
    ):
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


def _build_addressed_tail_example(
    alphabet: tuple[int, ...],
    carry: int,
    output: Sequence[int],
    step_tokens: Sequence[int] = (),
    *,
    order: StepOrder,
    use_query_tokens: bool,
    embedding_mode: EmbeddingMode,
) -> Example:
    """Build the post-header tail `[IDX] addr(k) c_k [SEP] ... [IDX] addr(k) ...` at `k = len(output)`."""
    k = len(output)
    input_ids: list[int] = []
    token_type_ids: list[int] = []
    position_ids: list[int] = []

    append_index_header(input_ids, token_type_ids, position_ids, k, alphabet)
    input_ids.extend((int(carry), int(SpecialToken.SEP)))
    token_type_ids.extend((TokenType.CARRY, TokenType.SPECIAL))
    position_ids.extend((k, 0))

    for j, o_j in enumerate(output):
        append_index_header(input_ids, token_type_ids, position_ids, j, alphabet)
        input_ids.append(int(o_j))
        token_type_ids.append(TokenType.OUTPUT)
        position_ids.append(j)
    input_ids.append(int(SpecialToken.SEP))
    token_type_ids.append(TokenType.SPECIAL)
    position_ids.append(0)

    append_index_header(input_ids, token_type_ids, position_ids, k, alphabet)

    slot_metas = _step_slot_metadata(k, order)
    if not use_query_tokens:
        for idx, tok in enumerate(step_tokens):
            tt, pos = slot_metas[idx]
            input_ids.append(int(tok))
            token_type_ids.append(tt)
            position_ids.append(pos)
    else:
        q_tokens = step_query_tokens(order)
        input_ids.append(int(q_tokens[0]))
        token_type_ids.append(TokenType.SPECIAL)
        position_ids.append(k)
        for idx, tok in enumerate(step_tokens):
            tt, pos = slot_metas[idx]
            input_ids.extend((int(tok), int(q_tokens[idx + 1])))
            token_type_ids.extend((tt, TokenType.SPECIAL))
            position_ids.extend((pos, k))

    return make_example(input_ids, token_type_ids, position_ids, embedding_mode)


def _max_addressed_tail_len(
    alphabet: tuple[int, ...],
    max_cols: int,
    *,
    use_query_tokens: bool,
) -> int:
    """Exact maximum tail length at `k = max_cols - 1` (including emitted `(t_0, t_1)` and query tokens)."""
    last_k = max_cols - 1
    addr_k_len = len(encode_address(last_k, alphabet))
    hist_len = sum(1 + len(encode_address(j, alphabet)) + 1 for j in range(last_k))
    step_suffix_len = 5 if use_query_tokens else 2
    # `[IDX] addr(k) c_k [SEP]` + historical outputs + `[SEP]` + `[IDX] addr(k)` + suffix
    return (1 + addr_k_len + 2) + hist_len + 1 + (1 + addr_k_len) + step_suffix_len


def _append_intra_step_token(
    model: Transformer,
    emitted_tok: torch.Tensor,
    *,
    slot_tt: TokenType,
    slot_pos: int,
    col_idx: int,
    next_query_tok: int | None,
    use_tt: bool,
    use_pe: bool,
    device: torch.device,
    kv_cache: KVCache,
) -> torch.Tensor:
    """Append `emitted_tok` (plus `next_query_tok` if `use_query_tokens`) to `kv_cache` and return greedy next token."""
    n_rows = int(emitted_tok.size(0))
    if next_query_tok is None:
        step_ids = emitted_tok.unsqueeze(1)
        step_tt = (
            torch.full((n_rows, 1), int(slot_tt), dtype=torch.long, device=device)
            if use_tt
            else None
        )
        step_pos = (
            torch.full((n_rows, 1), slot_pos, dtype=torch.long, device=device)
            if use_pe
            else None
        )
    else:
        q_col = torch.full((n_rows,), next_query_tok, dtype=torch.long, device=device)
        step_ids = torch.stack((emitted_tok, q_col), dim=1)
        step_tt = (
            torch.tensor(
                [[int(slot_tt), int(TokenType.SPECIAL)]],
                dtype=torch.long,
                device=device,
            ).expand(n_rows, -1)
            if use_tt
            else None
        )
        step_pos = (
            torch.tensor([[slot_pos, col_idx]], dtype=torch.long, device=device).expand(
                n_rows, -1
            )
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
    return logits[:, -1, :].argmax(dim=-1)


def _evaluate_addressed_step_ar_cached_chunk(
    model: Transformer,
    headers_chunk: Sequence[Example],
    probs_chunk: Sequence[AdditionProblem],
    oracle_steps_chunk: Sequence[tuple[ColumnStep, ...]],
    oracle_outputs_chunk: Sequence[tuple[int, ...]],
    max_tail_lens_chunk: Sequence[int],
    *,
    order: StepOrder,
    use_query_tokens: bool,
    embedding_mode: EmbeddingMode,
    device: torch.device,
) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
    """Evaluate one token-bounded problem chunk by caching addressed headers and intra-column decoding."""
    chunk_size = len(probs_chunk)
    fin_tok = int(SpecialToken.FLAG_FINISHED)
    q_tokens = step_query_tokens(order)

    prefill = collate_left_padded(headers_chunk).to(device)
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
            "evaluate_addressed_step_ar requires a Transformer with SubspaceLMHead"
        )

    max_cols = [max(len(p.a), len(p.b)) + 1 for p in probs_chunk]
    max_seq_len = header_len + max(max_tail_lens_chunk)
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
        slot_metas = _step_slot_metadata(col_idx, order)
        tail_examples = [
            _build_addressed_tail_example(
                probs_chunk[i].alphabet,
                carries[i],
                outputs[i],
                (),
                order=order,
                use_query_tokens=use_query_tokens,
                embedding_mode=embedding_mode,
            )
            for i in active
        ]
        lp_tail = collate_left_padded(tail_examples).to(device)
        tail_mask = (
            None if bool(lp_tail.key_padding_mask.all()) else lp_tail.key_padding_mask
        )
        logits_0 = model(
            lp_tail.input_ids,
            token_type_ids=lp_tail.token_type_ids,
            position_ids=lp_tail.position_ids,
            key_padding_mask=tail_mask,
            kv_cache=work_cache,
        )
        assert isinstance(logits_0, torch.Tensor)
        preds_t0 = logits_0[:, -1, :].argmax(dim=-1)

        preds_t1 = _append_intra_step_token(
            model,
            preds_t0,
            slot_tt=slot_metas[0][0],
            slot_pos=slot_metas[0][1],
            col_idx=col_idx,
            next_query_tok=int(q_tokens[1]) if use_query_tokens else None,
            use_tt=use_tt,
            use_pe=use_pe,
            device=device,
            kv_cache=work_cache,
        )
        preds_flag = _append_intra_step_token(
            model,
            preds_t1,
            slot_tt=slot_metas[1][0],
            slot_pos=slot_metas[1][1],
            col_idx=col_idx,
            next_query_tok=int(q_tokens[2]) if use_query_tokens else None,
            use_tt=use_tt,
            use_pe=use_pe,
            device=device,
            kv_cache=work_cache,
        )

        next_active: list[int] = []
        for i, t_0, t_1, pred_flag in zip(
            active,
            preds_t0.tolist(),
            preds_t1.tolist(),
            preds_flag.tolist(),
            strict=True,
        ):
            pred_o, pred_c = unpack_step_order(t_0, t_1, order)
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
    t0_offset = -3 if use_query_tokens else -2
    for k in range(max_oracle_cols):
        need_rows = [
            i
            for i in range(chunk_size)
            if k < len(oracle_steps_chunk[i]) and oracle_ok[i][k] is None
        ]
        if not need_rows:
            continue
        fb_tails: list[Example] = []
        for i in need_rows:
            st = oracle_steps_chunk[i][k]
            t_0, _ = pack_step_order(st.sum_digit, st.carry_out, order)
            fb_tails.append(
                _build_addressed_tail_example(
                    probs_chunk[i].alphabet,
                    st.carry_in,
                    oracle_outputs_chunk[i][:k],
                    (t_0,),
                    order=order,
                    use_query_tokens=use_query_tokens,
                    embedding_mode=embedding_mode,
                )
            )
        lp_fb = collate_left_padded(fb_tails).to(device)
        fb_len = int(lp_fb.input_ids.size(1))
        fb_mask = None if bool(lp_fb.key_padding_mask.all()) else lp_fb.key_padding_mask
        fb_cache = header_cache.select_rows(need_rows, max_seq_len=header_len + fb_len)
        fb_logits = model(
            lp_fb.input_ids,
            token_type_ids=lp_fb.token_type_ids,
            position_ids=lp_fb.position_ids,
            key_padding_mask=fb_mask,
            kv_cache=fb_cache,
        )
        assert isinstance(fb_logits, torch.Tensor)
        fb_t0 = fb_logits[:, t0_offset, :].argmax(dim=-1).tolist()
        fb_t1 = fb_logits[:, -1, :].argmax(dim=-1).tolist()
        for i, pred_t0, pred_t1 in zip(need_rows, fb_t0, fb_t1, strict=True):
            pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, order)
            st = oracle_steps_chunk[i][k]
            oracle_ok[i][k] = (pred_o == st.sum_digit) and (pred_c == st.carry_out)

    oracle_ok_counts = [sum(1 for ok in row if ok is True) for row in oracle_ok]
    return emitted, oracle_ok_counts


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
    dtype: str | torch.dtype | None = None,
) -> EvalMetrics:
    """Evaluate `model` via batched `KVCache`-accelerated rollout across `num_problems` problems.

    Prefills each problem's static addressed header once into `KVCache`, decodes
    `(t_0, t_1, flag_k)` incrementally within each active column `k`, feeds the
    model's own `(o_k, c_{k+1})` into column `k + 1` until `FLAG_FINISHED` or
    `max(L_a, L_b) + 1`, and computes `oracle_step_accuracy` across any reference
    columns not already visited on the ground-truth prefix
    (`spec/variants/addressed_step_ar.md` §3–§4).
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = StepOrder(step_order)

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
        headers_all = [
            make_example(
                *build_addressed_prefix_tokens(
                    prob.alphabet,
                    prob.a,
                    prob.b,
                    include_alphabet=include_alphabet,
                ),
                embedding_mode,
            )
            for prob in problem_list
        ]
        max_tail_lens_all = [
            _max_addressed_tail_len(
                prob.alphabet,
                max(len(prob.a), len(prob.b)) + 1,
                use_query_tokens=use_query_tokens,
            )
            for prob in problem_list
        ]

        emitted: list[list[tuple[int, int, int]]] = []
        oracle_ok_counts: list[int] = []
        with amp_ctx:
            for start in range(0, num_problems, batch_size):
                outer_end = min(num_problems, start + batch_size)
                max_total_len = max(
                    len(headers_all[i]) + max_tail_lens_all[i]
                    for i in range(start, outer_end)
                )
                sub_size = max(
                    1,
                    min(
                        outer_end - start,
                        MAX_EVAL_CHUNK_TOKENS // max(1, max_total_len),
                    ),
                )
                for sub in range(start, outer_end, sub_size):
                    sub_end = min(outer_end, sub + sub_size)
                    chunk_emitted, chunk_oracle_counts = (
                        _evaluate_addressed_step_ar_cached_chunk(
                            model,
                            headers_all[sub:sub_end],
                            problem_list[sub:sub_end],
                            oracle_steps[sub:sub_end],
                            oracle_outputs[sub:sub_end],
                            max_tail_lens_all[sub:sub_end],
                            order=order,
                            use_query_tokens=use_query_tokens,
                            embedding_mode=embedding_mode,
                            device=device,
                        )
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
                        t_0, t_1 = pack_step_order(
                            step.sum_digit, step.carry_out, order
                        )
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
        step_order=order,
        oracle_ok_per_problem=oracle_ok_counts,
        per_base_metrics=per_base_metrics,
    )
