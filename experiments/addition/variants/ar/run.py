"""Experiment runner (`run_addition_ar`) and `AR_VARIANT` descriptor for `addition`."""

from functools import partial
from pathlib import Path
from typing import Any

from experiments.addition.config import AR_DESCRIPTION, AdditionArConfig
from experiments.addition.train_and_evaluate import train_and_evaluate
from experiments.addition.variants.ar.dataset import AdditionArDataset
from experiments.addition.variants.ar.evaluation import evaluate_ar
from yessir.experiment import ExperimentVariant
from yessir.heads import HeadType


def run_addition_ar(
    config: AdditionArConfig,
    out_dir: Path | None = None,
    *,
    device: str = "auto",
    dtype: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the full-trace `ar` variant of `addition` according to `config`."""
    train_dataset = AdditionArDataset(
        bases=config.parsed_bases,
        min_length=config.min_length,
        max_length=config.max_length,
        symbols=config.symbol_pool,
        p_prior=config.p_prior,
        p_chain=config.p_chain,
        step_order=config.step_order,
        use_query_tokens=config.use_query_tokens,
        seed=config.seed,
    )
    evaluate_fn = partial(
        evaluate_ar,
        step_order=config.step_order,
        use_query_tokens=config.use_query_tokens,
    )
    return train_and_evaluate(
        variant_name="ar",
        config=config,
        out_dir=out_dir,
        head_type=HeadType.SUBSPACE_LM,
        train_dataset=train_dataset,
        evaluate_fn=evaluate_fn,
        device=device,
        dtype=dtype,
    )


AR_VARIANT = ExperimentVariant(
    name="ar",
    description=AR_DESCRIPTION,
    config_cls=AdditionArConfig,
    run=run_addition_ar,
)
