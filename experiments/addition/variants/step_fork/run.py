"""Experiment runner (`run_addition_step_fork`) and `STEP_FORK_VARIANT` descriptor for `addition`."""

from functools import partial
from pathlib import Path
from typing import Any

from experiments.addition.config import STEP_FORK_DESCRIPTION, AdditionStepForkConfig
from experiments.addition.train_and_evaluate import train_and_evaluate
from experiments.addition.variants.step_fork.dataset import AdditionStepForkDataset
from experiments.addition.variants.step_fork.evaluation import evaluate_step_fork
from yessir.experiment import ExperimentVariant
from yessir.heads import HeadType


def run_addition_step_fork(
    config: AdditionStepForkConfig,
    out_dir: Path | None = None,
    *,
    device: str = "auto",
    dtype: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the `step_fork` variant of `addition` according to `config`.

    Passes `train_batch_size = 3 * config.batch_size` so each optimisation step
    collates all 3 branched queries (`[S]`, `[C]`, `[F]`) for `config.batch_size`
    column transitions (`spec/variants/step_fork.md` §3).
    """
    train_dataset = AdditionStepForkDataset(
        bases=config.parsed_bases,
        min_length=config.min_length,
        max_length=config.max_length,
        symbols=config.symbol_pool,
        p_prior=config.p_prior,
        p_chain=config.p_chain,
        p_terminal=config.p_terminal,
        all_steps=config.all_steps,
        embedding_mode=config.embedding_mode,
        seed=config.seed,
    )
    evaluate_fn = partial(evaluate_step_fork, embedding_mode=config.embedding_mode)
    return train_and_evaluate(
        variant_name="step_fork",
        config=config,
        out_dir=out_dir,
        head_type=HeadType.SUBSPACE_LM,
        train_dataset=train_dataset,
        evaluate_fn=evaluate_fn,
        train_batch_size=AdditionStepForkDataset.BRANCHES_PER_STEP * config.batch_size,
        device=device,
        dtype=dtype,
    )


STEP_FORK_VARIANT = ExperimentVariant(
    name="step_fork",
    description=STEP_FORK_DESCRIPTION,
    config_cls=AdditionStepForkConfig,
    run=run_addition_step_fork,
)
