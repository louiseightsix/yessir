"""Experiment runner (`run_addition_addressed_ar`) and `ADDRESSED_AR_VARIANT` descriptor for `addition`."""

from functools import partial
from pathlib import Path
from typing import Any

from experiments.addition.config import (
    ADDRESSED_AR_DESCRIPTION,
    AdditionAddressedArConfig,
)
from experiments.addition.train_and_evaluate import train_and_evaluate
from experiments.addition.variants.addressed_ar.dataset import (
    AdditionAddressedArDataset,
)
from experiments.addition.variants.addressed_ar.evaluation import evaluate_addressed_ar
from yessir.experiment import ExperimentVariant
from yessir.heads import HeadType


def run_addition_addressed_ar(
    config: AdditionAddressedArConfig,
    out_dir: Path | None = None,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the full-trace `addressed_ar` variant of `addition` according to `config`."""
    train_dataset = AdditionAddressedArDataset(
        bases=config.parsed_bases,
        min_length=config.min_length,
        max_length=config.max_length,
        symbols=config.symbol_pool,
        p_prior=config.p_prior,
        p_chain=config.p_chain,
        step_order=config.step_order,
        use_query_tokens=config.use_query_tokens,
        include_alphabet=config.include_alphabet,
        seed=config.seed,
    )
    evaluate_fn = partial(
        evaluate_addressed_ar,
        step_order=config.step_order,
        use_query_tokens=config.use_query_tokens,
        include_alphabet=config.include_alphabet,
    )
    return train_and_evaluate(
        variant_name="addressed_ar",
        config=config,
        out_dir=out_dir,
        head_type=HeadType.SUBSPACE_LM,
        train_dataset=train_dataset,
        evaluate_fn=evaluate_fn,
        include_alphabet=config.include_alphabet,
        device=device,
    )


ADDRESSED_AR_VARIANT = ExperimentVariant(
    name="addressed_ar",
    description=ADDRESSED_AR_DESCRIPTION,
    config_cls=AdditionAddressedArConfig,
    run=run_addition_addressed_ar,
)
