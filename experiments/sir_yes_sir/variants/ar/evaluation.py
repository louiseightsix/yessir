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
    """Validate an `ar` sequence-labelled `LabelledExample` and return its step count `L`."""
    if len(item.labels) != len(item.example):
        raise ValueError(
            f"expected sequence-length labels ({len(item.example)}), got {len(item.labels)}"
        )
    target_tokens = [t for t in item.labels if t != IGNORE_INDEX]
    if not target_tokens or len(target_tokens) % 3 != 0:
        raise ValueError(
            f"expected non-ignored AR target length to be a positive multiple of 3, got {len(target_tokens)}"
        )
    if target_tokens[-1] != SpecialToken.FLAG_FINISHED:
        raise ValueError(
            f"expected final AR target token to be FLAG_FINISHED ({int(SpecialToken.FLAG_FINISHED)}), "
            f"got {target_tokens[-1]!r}"
        )
    for k in range(len(target_tokens) // 3 - 1):
        flag = target_tokens[3 * k + 2]
        if flag != SpecialToken.FLAG_CONTINUE:
            raise ValueError(
                f"expected intermediate step {k} flag to be FLAG_CONTINUE ({int(SpecialToken.FLAG_CONTINUE)}), "
                f"got {flag!r}"
            )
    return len(target_tokens) // 3


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
    Under causal masking and right-padding invariance, free-running greedy rollout
    (`rollout_ar`) feeds the exact same prefix into `model` as teacher forcing up to the
    first token mismatch. Therefore `rollout_ar` emits the ground-truth `(o_k, w_k, flag_k)`
    across all `k = 0 .. L - 1` if and only if every non-ignored position's greedy
    argmax matches `labels`.

    Args:
        model: `Transformer` configured with `TiedIdentityHead`.
        dataset: Stream of sequence-labelled `LabelledExample`s (one per problem).
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
    em_count = 0
    total_steps = 0
    step_ok_count = 0
    halt_ok_count = 0

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch, labels = collate_labelled(chunk)
        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        assert isinstance(logits, torch.Tensor)
        preds = logits.argmax(dim=-1)

        for b in range(len(chunk)):
            valid_mask = labels[b] != IGNORE_INDEX
            prob_preds = preds[b][valid_mask].tolist()
            prob_targets = labels[b][valid_mask].tolist()

            n_steps = len(prob_targets) // 3
            prob_all_ok = True
            for k in range(n_steps):
                pred_step = prob_preds[3 * k : 3 * k + 3]
                targ_step = prob_targets[3 * k : 3 * k + 3]
                step_ok = pred_step == targ_step
                halt_ok = pred_step[2] == targ_step[2]

                total_steps += 1
                if step_ok:
                    step_ok_count += 1
                else:
                    prob_all_ok = False
                if halt_ok:
                    halt_ok_count += 1

            if prob_all_ok:
                em_count += 1

    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=step_ok_count / total_steps,
        halt_accuracy=halt_ok_count / total_steps,
    )
