"""Experiment runner (`run_addition_addressed_step_ar`) and `ADDRESSED_STEP_AR_VARIANT` descriptor for `addition`."""

from functools import partial
from pathlib import Path
from typing import Any

from experiments.addition.config import (
    ADDRESSED_STEP_AR_DESCRIPTION,
    AdditionAddressedStepArConfig,
)
from experiments.addition.train_and_evaluate import train_and_evaluate
from experiments.addition.variants.addressed_step_ar.dataset import (
    AdditionAddressedStepArDataset,
)
from experiments.addition.variants.addressed_step_ar.evaluation import (
    evaluate_addressed_step_ar,
)
from yessir.experiment import ExperimentVariant
from yessir.heads import HeadType


def run_addition_addressed_step_ar(
    config: AdditionAddressedStepArConfig,
    out_dir: Path | None = None,
    *,
    device: str = "auto",
    dtype: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the `addressed_step_ar` variant of `addition` according to `config`."""
    train_dataset = AdditionAddressedStepArDataset(
        bases=config.parsed_bases,
        min_length=config.min_length,
        max_length=config.max_length,
        symbols=config.symbol_pool,
        p_prior=config.p_prior,
        p_chain=config.p_chain,
        p_terminal=config.p_terminal,
        all_steps=config.all_steps,
        step_order=config.step_order,
        use_query_tokens=config.use_query_tokens,
        include_alphabet=config.include_alphabet,
        embedding_mode=config.embedding_mode,
        seed=config.seed,
    )
    evaluate_fn = partial(
        evaluate_addressed_step_ar,
        step_order=config.step_order,
        use_query_tokens=config.use_query_tokens,
        include_alphabet=config.include_alphabet,
        embedding_mode=config.embedding_mode,
    )
    return train_and_evaluate(
        variant_name="addressed_step_ar",
        config=config,
        out_dir=out_dir,
        head_type=HeadType.SUBSPACE_LM,
        train_dataset=train_dataset,
        evaluate_fn=evaluate_fn,
        include_alphabet=config.include_alphabet,
        device=device,
        dtype=dtype,
    )


ADDRESSED_STEP_AR_VARIANT = ExperimentVariant(
    name="addressed_step_ar",
    description=ADDRESSED_STEP_AR_DESCRIPTION,
    config_cls=AdditionAddressedStepArConfig,
    run=run_addition_addressed_step_ar,
)
