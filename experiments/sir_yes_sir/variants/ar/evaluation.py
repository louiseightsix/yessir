"""Free-running rollout evaluation (`evaluate_ar`) for the `ar` variant of `sir_yes_sir`."""

import itertools
from collections.abc import Iterable, Sequence

import torch

from experiments.sir_yes_sir.evaluation import EvalMetrics, parse_prompt_prefix
from yessir.batch import collate, collate_left_padded
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import autocast_context


def _validate_ar_item(item: LabelledExample) -> list[int]:
    """Validate an `ar` per-position `LabelledExample` and return its `L + 1` target tokens."""
    if len(item.labels) != len(item.example):
        raise ValueError(
            f"expected per-position labels ({len(item.example)}), got {len(item.labels)}"
        )
    target_tokens = [t for t in item.labels if isinstance(t, int) and t != IGNORE_INDEX]
    if len(target_tokens) < 2:
        raise ValueError(
            f"expected non-ignored AR target length >= 2 (L >= 1 symbols + EOS), got {len(target_tokens)}"
        )
    if target_tokens[-1] != SpecialToken.EOS:
        raise ValueError(
            f"expected final AR target token to be EOS ({int(SpecialToken.EOS)}), "
            f"got {target_tokens[-1]!r}"
        )
    for k, tok in enumerate(target_tokens[:-1]):
        if tok == SpecialToken.EOS:
            raise ValueError(f"unexpected EOS at non-final target step k={k}")
    return target_tokens


def _slice_prefix_example(example: Example, prefix_len: int) -> Example:
    """Slice `example` down to its `[BOS] alphabet [SEP] x [SEP]` pure `input_ids` prompt prefix."""
    return Example(input_ids=example.input_ids[:prefix_len])


def _extend_prefix_example(prefix: Example, emitted_symbols: Sequence[int]) -> Example:
    """Append `emitted_symbols` directly to `prefix.input_ids`."""
    if not emitted_symbols:
        return prefix
    return Example(input_ids=prefix.input_ids + tuple(emitted_symbols))


@torch.no_grad()
def _rollout_ar_chunk(
    model: Transformer,
    prefixes_chunk: Sequence[Example],
    max_tokens_chunk: Sequence[int],
    *,
    device: torch.device,
) -> list[list[int]]:
    """Run batched pure-LM (`input_ids`-only) `ar` rollout on `prefixes_chunk` using `KVCache`."""
    eos = int(SpecialToken.EOS)
    pad = int(SpecialToken.PAD)
    chunk_size = len(prefixes_chunk)
    emitted: list[list[int]] = [[] for _ in range(chunk_size)]
    done: list[bool] = [False] * chunk_size

    if isinstance(model, Transformer):
        prefill = collate_left_padded(prefixes_chunk).to(device)
        kv_cache = model.init_kv_cache(
            int(prefill.input_ids.size(1)) + max(max_tokens_chunk)
        )
        logits = model(
            prefill.input_ids,
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
            _extend_prefix_example(prefixes_chunk[b_idx], emitted[b_idx])
            for b_idx in active
        ]
        batch = collate(step_examples).to(device)
        logits = model(batch.input_ids, batch.lengths)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("evaluate_ar requires a Transformer with SubspaceLMHead")
        row_indices = torch.arange(len(active), device=device)
        next_tokens = logits[row_indices, batch.lengths - 1].argmax(dim=-1).tolist()
        for b_idx, tok in zip(active, next_tokens, strict=True):
            emitted[b_idx].append(tok)
            if tok == eos or len(emitted[b_idx]) >= max_tokens_chunk[b_idx]:
                done[b_idx] = True
    return emitted


@torch.no_grad()
def evaluate_ar(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    num_problems: int,
    batch_size: int = 64,
    dtype: str | torch.dtype | None = None,
) -> EvalMetrics:
    """Evaluate `model` via batched KV-cached free-running rollout on `num_problems` `ar` problems.

    Starting from each problem's `[BOS] alphabet [SEP] x [SEP]` prompt, generates up
    to `L + 1` tokens autoregressively (`KVCache`-accelerated) and stops per problem
    as soon as `SpecialToken.EOS` is emitted (`experiments/sir_yes_sir/spec/variants/ar.md` §2–§3),
    matching `greedy_generate_ar`.
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    device = model.device
    amp_ctx = autocast_context(device, dtype)

    items = list(itertools.islice(dataset, num_problems))
    if len(items) < num_problems:
        raise ValueError(
            f"dataset exhausted early: needed {num_problems} problems, got {len(items)}"
        )

    eos = int(SpecialToken.EOS)
    targets_per_problem: list[list[int]] = []
    prefixes: list[Example] = []
    problems_ax: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    max_tokens_all: list[int] = []
    for item in items:
        targets = _validate_ar_item(item)
        n_steps = len(targets) - 1
        prefix_len = len(item.example) - n_steps
        prefix = _slice_prefix_example(item.example, prefix_len)
        targets_per_problem.append(targets)
        prefixes.append(prefix)
        problems_ax.append(parse_prompt_prefix(prefix.input_ids))
        max_tokens_all.append(len(targets))

    model.eval()
    em_count = 0
    total_steps = 0
    step_ok_count = 0
    halt_ok_count = 0
    wrap_steps = 0
    wrap_ok_count = 0

    with amp_ctx:
        for start in range(0, num_problems, batch_size):
            emitted_chunk = _rollout_ar_chunk(
                model,
                prefixes[start : start + batch_size],
                max_tokens_all[start : start + batch_size],
                device=device,
            )
            targets_chunk = targets_per_problem[start : start + batch_size]
            ax_chunk = problems_ax[start : start + batch_size]

            for emitted, targets, (alphabet, x) in zip(
                emitted_chunk, targets_chunk, ax_chunk, strict=True
            ):
                n_steps = len(targets) - 1
                total_steps += n_steps
                last_sym = alphabet[-1]
                symbol_preds = (
                    emitted[:-1]
                    if (emitted and emitted[-1] == eos)
                    else emitted[:n_steps]
                )
                for k in range(n_steps):
                    is_wrap = x[k] == last_sym
                    if is_wrap:
                        wrap_steps += 1
                    if k < len(symbol_preds) and symbol_preds[k] == targets[k]:
                        step_ok_count += 1
                        if is_wrap:
                            wrap_ok_count += 1

                halt_ok = (len(emitted) == n_steps + 1) and (emitted[-1] == eos)
                if halt_ok:
                    halt_ok_count += 1
                if emitted == targets:
                    em_count += 1

    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=step_ok_count / total_steps,
        halt_accuracy=halt_ok_count / num_problems,
        wrap_around_accuracy=(wrap_ok_count / wrap_steps if wrap_steps > 0 else None),
    )
