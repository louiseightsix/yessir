"""Free-running rollout evaluation (`evaluate_cls`) for the `cls` variant of `sir_yes_sir`."""

import itertools
from collections.abc import Iterable, Sequence

import torch

from experiments.sir_yes_sir.evaluation import EvalMetrics
from experiments.sir_yes_sir.inputs import TokenType
from yessir.batch import collate, collate_left_padded
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken


def _validate_cls_item(item: LabelledExample) -> list[tuple[int, int]]:
    """Validate a single-pass `cls` `LabelledExample` and return its `L` `(o_k, flag_k)` targets."""
    if len(item.labels) != len(item.example):
        raise ValueError(
            f"expected per-position labels ({len(item.example)}), got {len(item.labels)}"
        )
    if not isinstance(item.labels[0], tuple) or len(item.labels[0]) != 2:
        raise ValueError(
            "expected per-position 2-tuple labels (o_k, flag_k) for cls evaluation"
        )
    target_steps: list[tuple[int, int]] = [
        (step[0], step[1])
        for step in item.labels
        if isinstance(step, tuple) and step[0] != IGNORE_INDEX
    ]
    if not target_steps:
        raise ValueError("expected at least 1 non-ignored step in cls LabelledExample")
    if target_steps[-1][1] != SpecialToken.FLAG_FINISHED:
        raise ValueError(
            f"expected final cls step flag to be FLAG_FINISHED ({int(SpecialToken.FLAG_FINISHED)}), "
            f"got {target_steps[-1][1]!r}"
        )
    for k, (_, flag) in enumerate(target_steps[:-1]):
        if flag != SpecialToken.FLAG_CONTINUE:
            raise ValueError(
                f"expected non-final cls step {k} flag to be FLAG_CONTINUE ({int(SpecialToken.FLAG_CONTINUE)}), "
                f"got {flag!r}"
            )
    return target_steps


def _slice_prefix_example(example: Example, prefix_len: int) -> Example:
    """Slice `example` down to its `[BOS] alphabet [SEP] x [SEP]` prompt prefix."""
    return Example(
        input_ids=example.input_ids[:prefix_len],
        token_type_ids=(
            example.token_type_ids[:prefix_len]
            if example.token_type_ids is not None
            else None
        ),
        position_ids=(
            example.position_ids[:prefix_len]
            if example.position_ids is not None
            else None
        ),
    )


def _extend_prefix_example(prefix: Example, emitted_symbols: Sequence[int]) -> Example:
    """Append `emitted_symbols` (`TokenType.OUTPUT`, `position_ids = 1..k`) to `prefix`."""
    if not emitted_symbols:
        return prefix
    k = len(emitted_symbols)
    return Example(
        input_ids=prefix.input_ids + tuple(emitted_symbols),
        token_type_ids=(
            prefix.token_type_ids + (int(TokenType.OUTPUT),) * k
            if prefix.token_type_ids is not None
            else None
        ),
        position_ids=(
            prefix.position_ids + tuple(range(1, k + 1))
            if prefix.position_ids is not None
            else None
        ),
    )


@torch.no_grad()
def _rollout_cls_chunk(
    model: Transformer,
    prefixes_chunk: Sequence[Example],
    max_steps_chunk: Sequence[int],
    *,
    device: torch.device,
) -> list[list[tuple[int, int]]]:
    """Run batched free-running `cls` rollout on `prefixes_chunk` using `KVCache` for `Transformer`."""
    fin_tok = int(SpecialToken.FLAG_FINISHED)
    pad = int(SpecialToken.PAD)
    chunk_size = len(prefixes_chunk)
    emitted: list[list[tuple[int, int]]] = [[] for _ in range(chunk_size)]
    done: list[bool] = [False] * chunk_size

    if isinstance(model, Transformer):
        prefill = collate_left_padded(prefixes_chunk).to(device)
        prefill_len = int(prefill.input_ids.size(1))
        kv_cache = model.init_kv_cache(prefill_len + max(max_steps_chunk))
        prefill_lengths = torch.full(
            (chunk_size,), prefill_len, dtype=torch.long, device=device
        )
        out = model(
            prefill.input_ids,
            prefill_lengths,
            token_type_ids=prefill.token_type_ids,
            position_ids=prefill.position_ids,
            key_padding_mask=prefill.key_padding_mask,
            kv_cache=kv_cache,
        )
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError(
                "evaluate_cls requires a Transformer with a 2-head ParallelClassificationHead"
            )
        logits_o, logits_flag = out
        preds_o = logits_o.argmax(dim=-1).tolist()
        preds_flag = logits_flag.argmax(dim=-1).tolist()
        for b_idx, (o_k, flag_k) in enumerate(zip(preds_o, preds_flag, strict=True)):
            emitted[b_idx].append((o_k, flag_k))
            if flag_k == fin_tok or len(emitted[b_idx]) >= max_steps_chunk[b_idx]:
                done[b_idx] = True

        use_tt = prefill.token_type_ids is not None
        use_pe = prefill.position_ids is not None
        step_lengths = torch.ones((chunk_size,), dtype=torch.long, device=device)
        slot_idx = 0

        while not all(done):
            step_ids = torch.tensor(
                [
                    [pad if done[b_idx] else emitted[b_idx][-1][0]]
                    for b_idx in range(chunk_size)
                ],
                dtype=torch.long,
                device=device,
            )
            step_tt = (
                torch.full(
                    (chunk_size, 1),
                    int(TokenType.OUTPUT),
                    dtype=torch.long,
                    device=device,
                )
                if use_tt
                else None
            )
            step_pos = (
                torch.full(
                    (chunk_size, 1), slot_idx + 1, dtype=torch.long, device=device
                )
                if use_pe
                else None
            )
            out = model(
                step_ids,
                step_lengths,
                token_type_ids=step_tt,
                position_ids=step_pos,
                kv_cache=kv_cache,
            )
            assert isinstance(out, tuple) and len(out) == 2
            logits_o, logits_flag = out
            preds_o = logits_o.argmax(dim=-1).tolist()
            preds_flag = logits_flag.argmax(dim=-1).tolist()
            for b_idx, (o_k, flag_k) in enumerate(
                zip(preds_o, preds_flag, strict=True)
            ):
                if not done[b_idx]:
                    emitted[b_idx].append((o_k, flag_k))
                    if (
                        flag_k == fin_tok
                        or len(emitted[b_idx]) >= max_steps_chunk[b_idx]
                    ):
                        done[b_idx] = True
            slot_idx += 1
        return emitted

    while not all(done):
        active = [b_idx for b_idx, is_done in enumerate(done) if not is_done]
        step_examples = [
            _extend_prefix_example(
                prefixes_chunk[b_idx], [step[0] for step in emitted[b_idx]]
            )
            for b_idx in active
        ]
        batch = collate(step_examples).to(device)
        out = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError(
                "evaluate_cls requires a Transformer with a 2-head ParallelClassificationHead"
            )
        logits_o, logits_flag = out
        preds_o = logits_o.argmax(dim=-1).tolist()
        preds_flag = logits_flag.argmax(dim=-1).tolist()
        for b_idx, o_k, flag_k in zip(active, preds_o, preds_flag, strict=True):
            emitted[b_idx].append((o_k, flag_k))
            if flag_k == fin_tok or len(emitted[b_idx]) >= max_steps_chunk[b_idx]:
                done[b_idx] = True
    return emitted


@torch.no_grad()
def evaluate_cls(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    num_problems: int,
    batch_size: int = 64,
) -> EvalMetrics:
    """Evaluate `model` via batched free-running rollout on `num_problems` problems from `dataset`.

    Starting from each problem's `[BOS] alphabet [SEP] x [SEP]` prompt, feeds the
    model's own predicted `o_k` tokens forward step by step (`KVCache`-accelerated)
    and stops per problem as soon as `pred_flag == FLAG_FINISHED` or `L` steps have
    been emitted (`experiments/sir_yes_sir/spec/variants/cls.md` §2–§3), matching
    `greedy_generate_cls` and preventing unreached post-halt steps from scoring in
    `step_accuracy`.
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    items = list(itertools.islice(dataset, num_problems))
    if len(items) < num_problems:
        raise ValueError(
            f"dataset exhausted early: needed {num_problems} problems, got {len(items)}"
        )

    targets_per_problem: list[list[tuple[int, int]]] = []
    prefixes: list[Example] = []
    max_steps_all: list[int] = []
    for item in items:
        target_steps = _validate_cls_item(item)
        n_steps = len(target_steps)
        prefix_len = len(item.example) - n_steps + 1
        targets_per_problem.append(target_steps)
        prefixes.append(_slice_prefix_example(item.example, prefix_len))
        max_steps_all.append(n_steps)

    model.eval()
    device = model.device
    em_count = 0
    total_steps = 0
    step_ok_count = 0
    halt_ok_count = 0

    for start in range(0, num_problems, batch_size):
        emitted_chunk = _rollout_cls_chunk(
            model,
            prefixes[start : start + batch_size],
            max_steps_all[start : start + batch_size],
            device=device,
        )
        targets_chunk = targets_per_problem[start : start + batch_size]

        for emitted_steps, target_steps in zip(
            emitted_chunk, targets_chunk, strict=True
        ):
            n_steps = len(target_steps)
            total_steps += n_steps
            for k in range(min(n_steps, len(emitted_steps))):
                if emitted_steps[k][0] == target_steps[k][0]:
                    step_ok_count += 1

            halt_ok = (len(emitted_steps) == n_steps) and all(
                emitted_steps[k][1] == target_steps[k][1] for k in range(n_steps)
            )
            if halt_ok:
                halt_ok_count += 1
            if emitted_steps == target_steps:
                em_count += 1

    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=step_ok_count / total_steps,
        halt_accuracy=halt_ok_count / num_problems,
    )
