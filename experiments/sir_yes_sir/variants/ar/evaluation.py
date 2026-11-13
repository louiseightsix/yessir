"""Teacher-forced evaluation (`evaluate_ar`) for the `ar` variant of `sir_yes_sir`."""

import itertools
from collections.abc import Iterable

import torch

from experiments.sir_yes_sir.evaluation import EvalMetrics
from yessir.batch import collate_labelled
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken


def _validate_ar_item(item: LabelledExample) -> int:
    """Validate an `ar` per-position `LabelledExample` and return its step count `L`."""
    if len(item.labels) != len(item.example):
        raise ValueError(
            f"expected per-position labels ({len(item.example)}), got {len(item.labels)}"
        )
    target_tokens = [t for t in item.labels if t != IGNORE_INDEX]
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
    return len(target_tokens) - 1


@torch.no_grad()
def evaluate_ar(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    num_problems: int,
    batch_size: int = 64,
) -> EvalMetrics:
    """Evaluate `model` on `num_problems` problems from an `ar` `dataset`.

    Why teacher-forced next-token evaluation gives exact match (`exact_match`) for `ar`:
    Under causal masking and right-padding invariance, free-running greedy generation
    (`greedy_generate_ar`) feeds the exact same prefix into `model` as teacher forcing up to the
    first token mismatch. Therefore `greedy_generate_ar` emits `(o_0, ..., o_{L-1}, EOS)` if and
    only if every non-ignored position's greedy `argmax` matches `labels`.

    Metrics (`experiments/sir_yes_sir/spec/variants/ar.md` §3):
    - `exact_match` (per problem): every prediction in `o_0, ..., o_{L-1}, EOS` matches.
    - `step_accuracy` (per input step `k` in `0 .. L - 1`): fraction of the `L` symbol
      steps where `preds[k] == o_k`.
    - `halt_accuracy` (per problem): fraction of problems that halt at the exact right
      step (`preds[k] != EOS` for all `k < L` and `preds[L] == EOS`).

    Args:
        model: `Transformer` configured with `SubspaceLMHead`.
        dataset: Stream of per-position `LabelledExample`s (one per problem).
        num_problems: Number of problems to evaluate (`>= 1`).
        batch_size: Maximum number of problems per forward pass (`>= 1`).

    Returns:
        `EvalMetrics` with `exact_match`, `step_accuracy`, and `halt_accuracy`.
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
    for item in items:
        _validate_ar_item(item)

    model.eval()
    device = model.device
    em_count = 0
    total_steps = 0
    step_ok_count = 0
    halt_ok_count = 0

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch, labels = collate_labelled(chunk)
        batch = batch.to(device)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(logits, torch.Tensor)
        batch_preds = logits.argmax(dim=-1).cpu()

        for b in range(len(chunk)):
            valid_mask = labels[b] != IGNORE_INDEX
            preds = batch_preds[b][valid_mask].tolist()
            targets = labels[b][valid_mask].tolist()

            n_steps = len(targets) - 1
            total_steps += n_steps
            for k in range(n_steps):
                if preds[k] == targets[k]:
                    step_ok_count += 1

            halt_ok = (
                all(preds[k] != SpecialToken.EOS for k in range(n_steps))
                and preds[n_steps] == SpecialToken.EOS
            )
            if halt_ok:
                halt_ok_count += 1
            if preds == targets:
                em_count += 1

    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=step_ok_count / total_steps,
        halt_accuracy=halt_ok_count / num_problems,
    )
