"""Free-running rollout evaluation (`evaluate_addressed_ar`) for the `addressed_ar` variant of `addition`."""

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
    StepOrder,
    encode_address,
    step_query_tokens,
    unpack_step_order,
)
from experiments.addition.reference import ColumnStep, reference_add
from experiments.addition.variants.addressed_ar.inputs import (
    build_addressed_ar_example,
    build_addressed_ar_trace_tokens,
)
from yessir.batch import collate, collate_left_padded
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken


@torch.no_grad()
def _rollout_addressed_ar_chunk(
    model: Transformer,
    problems_chunk: Sequence[AdditionProblem],
    max_tokens_chunk: Sequence[int],
    *,
    include_alphabet: bool,
    device: torch.device,
) -> list[list[int]]:
    """Run batched pure-LM (`input_ids`-only) greedy rollout on `problems_chunk` using `KVCache`."""
    eos = int(SpecialToken.EOS)
    pad = int(SpecialToken.PAD)
    chunk_size = len(problems_chunk)
    emitted: list[list[int]] = [[] for _ in range(chunk_size)]
    done: list[bool] = [False] * chunk_size

    if isinstance(model, Transformer):
        prefix_examples = [
            build_addressed_ar_example(
                prob.alphabet,
                prob.a,
                prob.b,
                (),
                include_alphabet=include_alphabet,
            )
            for prob in problems_chunk
        ]
        prefill = collate_left_padded(prefix_examples).to(device)
        kv_cache = model.init_kv_cache(
            int(prefill.input_ids.size(1)) + max(max_tokens_chunk)
        )
        logits = model(
            prefill.input_ids,
            key_padding_mask=prefill.key_padding_mask,
            kv_cache=kv_cache,
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(
                "evaluate_addressed_ar requires a Transformer with SubspaceLMHead"
            )

        next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
        for b_idx, tok in enumerate(next_tokens):
            emitted[b_idx].append(tok)
            if tok == eos or len(emitted[b_idx]) >= max_tokens_chunk[b_idx]:
                done[b_idx] = True

        while not all(done):
            step_ids = torch.tensor(
                [
                    [pad if done[b_idx] else emitted[b_idx][-1]]
                    for b_idx in range(chunk_size)
                ],
                dtype=torch.long,
                device=device,
            )
            logits = model(step_ids, kv_cache=kv_cache)
            assert isinstance(logits, torch.Tensor)
            next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
            for b_idx, tok in enumerate(next_tokens):
                if not done[b_idx]:
                    emitted[b_idx].append(tok)
                    if tok == eos or len(emitted[b_idx]) >= max_tokens_chunk[b_idx]:
                        done[b_idx] = True
        return emitted

    while not all(done):
        active = [b_idx for b_idx, is_done in enumerate(done) if not is_done]
        step_examples = [
            build_addressed_ar_example(
                problems_chunk[b_idx].alphabet,
                problems_chunk[b_idx].a,
                problems_chunk[b_idx].b,
                emitted[b_idx],
                include_alphabet=include_alphabet,
            )
            for b_idx in active
        ]
        batch = collate(step_examples).to(device)
        logits = model(batch.input_ids, batch.lengths)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(
                "evaluate_addressed_ar requires a Transformer with SubspaceLMHead"
            )
        row_indices = torch.arange(len(active), device=device)
        next_tokens = logits[row_indices, batch.lengths - 1].argmax(dim=-1).tolist()
        for b_idx, tok in zip(active, next_tokens, strict=True):
            emitted[b_idx].append(tok)
            if tok == eos or len(emitted[b_idx]) >= max_tokens_chunk[b_idx]:
                done[b_idx] = True
    return emitted


def _parse_addressed_ar_rollout(
    emitted: Sequence[int],
    alphabet: Sequence[int],
    *,
    num_steps: int,
    step_order: StepOrder,
    use_query_tokens: bool,
) -> tuple[StepPrediction, ...]:
    """Segment emitted `addressed_ar` tokens on `[IDX]` into `StepPrediction`s."""
    idx_tok = int(AdditionToken.IDX)
    eos = int(SpecialToken.EOS)
    body_len = 4 if use_query_tokens else 2
    q_0, q_1, _ = step_query_tokens(step_order)

    ended_with_eos = bool(emitted and emitted[-1] == eos)
    stream = list(emitted[:-1]) if ended_with_eos else list(emitted)
    if not stream:
        return ()

    segments: list[tuple[bool, list[int]]] = []
    for tok in stream:
        if tok == idx_tok:
            segments.append((True, []))
        elif not segments:
            segments.append((False, [tok]))
        else:
            segments[-1][1].append(tok)

    num_segments = len(segments)
    last_k = num_steps - 1
    step_preds: list[StepPrediction] = []

    for k, (starts_with_idx, raw_seg) in enumerate(segments):
        is_last_seg = k == num_segments - 1
        expected_addr = encode_address(k, alphabet) if k < num_steps else None
        seg_tokens = raw_seg
        if (
            is_last_seg
            and not ended_with_eos
            and starts_with_idx
            and expected_addr is not None
            and len(seg_tokens) > len(expected_addr) + body_len
            and tuple(seg_tokens[: len(expected_addr)]) == expected_addr
        ):
            # Rollout hit `max_tokens` without emitting `[EOS]` after completing
            # column `k`; trim trailing overrun tokens (`halt_ok` is already `False`).
            seg_tokens = seg_tokens[: len(expected_addr) + body_len]

        has_full_entry = len(seg_tokens) >= 1 + body_len
        if k < last_k:
            halt_ok = has_full_entry and (k + 1 < num_segments) and segments[k + 1][0]
        elif k == last_k:
            halt_ok = has_full_entry and (num_segments == num_steps) and ended_with_eos
        else:
            halt_ok = False

        if has_full_entry:
            pred_addr = tuple(seg_tokens[:-body_len])
            body = seg_tokens[-body_len:]
            address_ok = bool(
                starts_with_idx
                and expected_addr is not None
                and pred_addr == expected_addr
            )
            if use_query_tokens:
                pred_q0, pred_t0, pred_q1, pred_t1 = body
                queries_ok = (pred_q0 == int(q_0)) and (pred_q1 == int(q_1))
            else:
                pred_t0, pred_t1 = body
                queries_ok = True
            syntax_ok = address_ok and queries_ok
            pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, step_order)
        elif (
            is_last_seg
            and ended_with_eos
            and starts_with_idx
            and expected_addr is not None
            and len(seg_tokens) >= len(expected_addr)
            and tuple(seg_tokens[: len(expected_addr)]) == expected_addr
        ):
            # Valid `[IDX] + addr(k)` header followed by premature `[EOS]` inside body.
            address_ok = True
            rem = seg_tokens[len(expected_addr) :]
            if not use_query_tokens:
                pred_t0 = rem[0] if len(rem) >= 1 else eos
                syntax_ok = True
                pred_o, pred_c = unpack_step_order(pred_t0, eos, step_order)
            elif len(rem) == 1 and rem[0] == int(q_0):
                syntax_ok = True
                pred_o, pred_c = unpack_step_order(eos, eos, step_order)
            elif len(rem) == 3:
                syntax_ok = (rem[0] == int(q_0)) and (rem[2] == int(q_1))
                pred_o, pred_c = unpack_step_order(rem[1], eos, step_order)
            else:
                syntax_ok = False
                pred_t0 = rem[1] if len(rem) >= 2 else eos
                pred_o, pred_c = unpack_step_order(pred_t0, eos, step_order)
        else:
            address_ok = False
            syntax_ok = False
            pred_o, pred_c = eos, eos

        step_preds.append(
            StepPrediction(
                sum_digit=pred_o,
                carry_out=pred_c,
                syntax_ok=syntax_ok,
                halt_ok=halt_ok,
                address_ok=address_ok,
            )
        )

    return tuple(step_preds)


def _compute_addressed_ar_oracle_counts(
    model: Transformer,
    problems: Sequence[AdditionProblem],
    oracle_steps: Sequence[tuple[ColumnStep, ...]],
    *,
    batch_size: int,
    step_order: StepOrder,
    use_query_tokens: bool,
    include_alphabet: bool,
    device: torch.device,
) -> list[int]:
    """Return per-problem teacher-forced column match counts across all `K + 1` columns."""
    idx_tok = int(AdditionToken.IDX)
    q_0, q_1, _ = step_query_tokens(step_order)
    body_len = 4 if use_query_tokens else 2

    examples: list[Example] = []
    num_targets_list: list[int] = []
    for prob, steps in zip(problems, oracle_steps, strict=True):
        trace = build_addressed_ar_trace_tokens(
            steps,
            prob.alphabet,
            step_order=step_order,
            use_query_tokens=use_query_tokens,
        )
        examples.append(
            build_addressed_ar_example(
                prob.alphabet,
                prob.a,
                prob.b,
                trace,
                include_alphabet=include_alphabet,
            )
        )
        num_targets_list.append(len(trace) + 1)

    oracle_ok_per_problem: list[int] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        probs_chunk = problems[start : start + batch_size]
        steps_chunk = oracle_steps[start : start + batch_size]
        targets_chunk = num_targets_list[start : start + batch_size]
        batch = collate(chunk).to(device)
        logits = model(batch.input_ids, batch.lengths)
        assert isinstance(logits, torch.Tensor)
        argmax_ids = logits.argmax(dim=-1)
        for b_idx, (prob, steps, num_targets) in enumerate(
            zip(probs_chunk, steps_chunk, targets_chunk, strict=True)
        ):
            n = int(batch.lengths[b_idx].item())
            preds = argmax_ids[b_idx, n - num_targets : n].tolist()
            col_offset = 0
            prob_ok = 0
            for k, step in enumerate(steps):
                addr_k = encode_address(k, prob.alphabet)
                addr_len = len(addr_k)
                entry_len = 1 + addr_len + body_len
                col_slice = preds[col_offset : col_offset + entry_len]
                col_offset += entry_len

                header_ok = (col_slice[0] == idx_tok) and (
                    tuple(col_slice[1 : 1 + addr_len]) == addr_k
                )
                body = col_slice[1 + addr_len :]
                if use_query_tokens:
                    pred_q0, pred_t0, pred_q1, pred_t1 = body
                    syntax_ok = (
                        header_ok and (pred_q0 == int(q_0)) and (pred_q1 == int(q_1))
                    )
                else:
                    pred_t0, pred_t1 = body
                    syntax_ok = header_ok
                pred_o, pred_c = unpack_step_order(pred_t0, pred_t1, step_order)
                if (
                    syntax_ok
                    and (pred_o == step.sum_digit)
                    and (pred_c == step.carry_out)
                ):
                    prob_ok += 1
            oracle_ok_per_problem.append(prob_ok)

    return oracle_ok_per_problem


@torch.no_grad()
def evaluate_addressed_ar(
    model: Transformer,
    problems: Iterable[AdditionProblem],
    *,
    num_problems: int,
    batch_size: int = 64,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    use_query_tokens: bool = False,
    include_alphabet: bool = True,
    per_base_metrics: dict[int, EvalMetrics] | None = None,
) -> EvalMetrics:
    """Evaluate `model` via KV-cached free-running rollout across `num_problems` `addressed_ar` problems.

    Decodes each problem autoregressively (`input_ids` only) from its addressed
    prompt until `SpecialToken.EOS` or the maximum well-formed trace length for
    `max(L_a, L_b) + 1` columns (`spec/variants/addressed_ar.md` §3–§4), segments
    `[IDX] addr(k) ...` entries into `StepPrediction`s (recording per-step
    `address_accuracy` alongside `syntax_ok`-gated `step_accuracy`), and records
    single-pass teacher-forced `oracle_step_accuracy`.
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = StepOrder(step_order)
    body_len = 4 if use_query_tokens else 2

    problem_list = list(itertools.islice(problems, num_problems))
    if len(problem_list) < num_problems:
        raise ValueError(
            f"problems stream exhausted early: needed {num_problems}, got {len(problem_list)}"
        )

    oracle_steps = [
        reference_add(prob.alphabet, prob.a, prob.b) for prob in problem_list
    ]
    max_tokens_all = [
        sum(
            1 + len(encode_address(k, prob.alphabet)) + body_len
            for k in range(max(len(prob.a), len(prob.b)) + 1)
        )
        + 1
        for prob in problem_list
    ]

    model.eval()
    device = model.device
    emitted_per_problem: list[list[int]] = []

    for start in range(0, num_problems, batch_size):
        emitted_per_problem.extend(
            _rollout_addressed_ar_chunk(
                model,
                problem_list[start : start + batch_size],
                max_tokens_all[start : start + batch_size],
                include_alphabet=include_alphabet,
                device=device,
            )
        )

    oracle_ok_counts = _compute_addressed_ar_oracle_counts(
        model,
        problem_list,
        oracle_steps,
        batch_size=batch_size,
        step_order=order,
        use_query_tokens=use_query_tokens,
        include_alphabet=include_alphabet,
        device=device,
    )

    evaluated: list[
        tuple[tuple[int, ...], tuple[ColumnStep, ...], tuple[StepPrediction, ...]]
    ] = [
        (
            prob.alphabet,
            steps,
            _parse_addressed_ar_rollout(
                emitted,
                prob.alphabet,
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
        oracle_ok_per_problem=oracle_ok_counts,
        track_addresses=True,
        per_base_metrics=per_base_metrics,
    )
