"""Teacher-forced evaluation (`EvalMetrics`, `evaluate_sm_mh`) for `sir_yes_sir`."""

# TODO(addition): Once we build `experiments/addition/`, re-evaluate which parts
# of evaluation (e.g. batched greedy prediction over `LabelledExample`s or common
# metric containers) are genuinely shared across experiments and belong in `yessir/`.

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from yessir.batch import collate
from yessir.dataset import LabelledExample
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import pack_multitask_targets


@dataclass(frozen=True)
class EvalMetrics:
    """Teacher-forced evaluation metrics over `sir_yes_sir` problems.

    Attributes:
        exact_match: Fraction of problems in `0.0..1.0` where every step `k`
            in `0 .. L - 1` predicted all three outputs `(o_k, w_k, flag_k)`
            correctly (`experiments/sir_yes_sir/spec/README.md` §3).
        step_accuracy: Fraction of individual steps in `0.0..1.0` where all
            three heads `(o_k, w_k, flag_k)` matched `labels`.
        halt_accuracy: Fraction of individual steps in `0.0..1.0` where the
            halting head (`labels[2]`) matched `flag_k` (`FLAG_CONTINUE` vs.
            `FLAG_FINISHED`).
    """

    exact_match: float
    step_accuracy: float
    halt_accuracy: float


def _collect_problems(
    dataset: Iterable[LabelledExample],
    num_problems: int,
) -> list[list[LabelledExample]]:
    """Draw `num_problems` complete problems from `dataset`, grouped by `FLAG_FINISHED`."""
    problems: list[list[LabelledExample]] = []
    current: list[LabelledExample] = []

    for item in dataset:
        if len(item.labels) != 3:
            raise ValueError(
                f"expected 3 labels (o_k, w_k, flag_k), got {len(item.labels)}"
            )
        if not current:
            # At step k = 0 of a state-machine problem, output_history is empty,
            # so the second [SEP] sits immediately before [STEP] at input_ids[-2].
            input_ids = item.example.input_ids
            if (
                len(input_ids) < 2
                or input_ids[-1] != SpecialToken.STEP
                or input_ids[-2] != SpecialToken.SEP
            ):
                raise ValueError(
                    "stream must start at step k=0 of a problem "
                    "(expected [SEP] immediately before [STEP])"
                )
        flag = item.labels[2]
        if flag not in (SpecialToken.FLAG_CONTINUE, SpecialToken.FLAG_FINISHED):
            raise ValueError(
                f"expected flag label to be FLAG_CONTINUE ({int(SpecialToken.FLAG_CONTINUE)}) "
                f"or FLAG_FINISHED ({int(SpecialToken.FLAG_FINISHED)}), got {flag!r}"
            )
        current.append(item)
        if flag == SpecialToken.FLAG_FINISHED:
            problems.append(current)
            current = []
            if len(problems) == num_problems:
                return problems

    raise ValueError(
        f"dataset exhausted early: needed {num_problems} complete problems, "
        f"got {len(problems)} (with {len(current)} trailing unfinished steps)"
    )


@torch.no_grad()
def evaluate_sm_mh(
    model: Transformer,
    dataset: Iterable[LabelledExample],
    *,
    num_problems: int,
    batch_size: int = 64,
) -> EvalMetrics:
    """Evaluate `model` with teacher forcing on `num_problems` problems from `dataset`.

    `dataset` must start at a problem boundary (`k = 0`, e.g. a fresh `SirYesSirDataset`
    with an evaluation seed disjoint from training); starting mid-problem raises a
    `ValueError`.

    Why teacher-forced step evaluation also gives exact match (`exact_match`) for `sir_yes_sir`:
    Per `experiments/sir_yes_sir/spec/README.md` §3, an instance is an exact match if
    and only if the model emits the ground-truth `(o_k, w_k, flag_k)` at every step
    `k = 0 .. L - 1`. Until the first error on a problem, `run_sm_mh`'s `output_history`
    is identical to the teacher-forced `output_history` in `dataset`, and right-padding
    invariance produces the same logits up to floating-point rounding as step-by-step
    execution.
    """
    if num_problems < 1:
        raise ValueError(f"num_problems must be >= 1, got {num_problems}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    model.eval()
    problems = _collect_problems(dataset, num_problems)
    flat_items = [item for problem in problems for item in problem]

    step_ok_list: list[bool] = []
    halt_ok_list: list[bool] = []

    for start in range(0, len(flat_items), batch_size):
        chunk = flat_items[start : start + batch_size]
        batch = collate([item.example for item in chunk])
        targets = pack_multitask_targets(chunk)

        logits = model(
            batch.input_ids,
            batch.lengths,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
        )
        preds = torch.stack([head.argmax(dim=-1) for head in logits], dim=1)
        matches = preds == targets

        step_ok_list.extend(matches.all(dim=1).tolist())
        halt_ok_list.extend(matches[:, 2].tolist())

    em_count = 0
    cursor = 0
    for problem in problems:
        n_steps = len(problem)
        if all(step_ok_list[cursor : cursor + n_steps]):
            em_count += 1
        cursor += n_steps

    total_steps = len(flat_items)
    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=sum(step_ok_list) / total_steps,
        halt_accuracy=sum(halt_ok_list) / total_steps,
    )
