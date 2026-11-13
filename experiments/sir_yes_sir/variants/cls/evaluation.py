"""Teacher-forced evaluation (`evaluate_cls`) for the `cls` variant of `sir_yes_sir`."""

import itertools
from collections.abc import Iterable

import torch

from experiments.sir_yes_sir.evaluation import EvalMetrics
from yessir.batch import collate_labelled
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken


def _validate_cls_item(item: LabelledExample) -> int:
    """Validate a single-pass `cls` `LabelledExample` and return its step count `L`."""
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
    return len(target_steps)


@torch.no_grad()
def evaluate_cls(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    num_problems: int,
    batch_size: int = 64,
) -> EvalMetrics:
    """Evaluate `model` with single-pass teacher forcing on `num_problems` problems from `dataset`.

    Why single-pass teacher-forced evaluation gives exact match (`exact_match`) for `cls`:
    Under causal masking and right-padding invariance, the hidden state at query index
    `P - 1 + k` sees the exact same prefix `prefix + (o_0, ..., o_{k-1})` as step `k` of
    `greedy_generate_cls` up to the first prediction mismatch.

    Metrics (`experiments/sir_yes_sir/spec/variants/cls.md` §3):
    - `exact_match` (per problem): every step `k` in `0 .. L - 1` predicts both
      `pred_o[k] == o_k` and `pred_flag[k] == flag_k`.
    - `step_accuracy` (per input step `k` in `0 .. L - 1`): fraction of the `L` steps
      across all evaluated problems where `pred_o[k] == o_k`.
    - `halt_accuracy` (per problem): fraction of problems that halt at the exact right
      step (`pred_flag[k] == FLAG_CONTINUE` for `k < L - 1` and `pred_flag[L - 1] == FLAG_FINISHED`).

    Args:
        model: `Transformer` configured with a 2-head `ParallelClassificationHead`.
        dataset: Stream of single-pass `cls` `LabelledExample`s (one per problem).
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
        _validate_cls_item(item)

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

        logits_o, logits_flag = model(
            batch.input_ids,
            lengths=None,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        batch_pred_o = logits_o.argmax(dim=-1).cpu()
        batch_pred_flag = logits_flag.argmax(dim=-1).cpu()

        for b in range(len(chunk)):
            valid_mask = labels[b, :, 0] != IGNORE_INDEX
            pred_o = batch_pred_o[b][valid_mask].tolist()
            pred_flag = batch_pred_flag[b][valid_mask].tolist()
            target_o = labels[b, :, 0][valid_mask].tolist()
            target_flag = labels[b, :, 1][valid_mask].tolist()

            n_steps = len(target_o)
            total_steps += n_steps
            for k in range(n_steps):
                if pred_o[k] == target_o[k]:
                    step_ok_count += 1

            halt_ok = pred_flag == target_flag
            if halt_ok:
                halt_ok_count += 1
            if pred_o == target_o and halt_ok:
                em_count += 1

    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=step_ok_count / total_steps,
        halt_accuracy=halt_ok_count / num_problems,
    )
