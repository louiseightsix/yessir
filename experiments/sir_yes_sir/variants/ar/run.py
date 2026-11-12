"""Experiment runner (`run_sir_yes_sir_ar`) and `AR_VARIANT` descriptor for `sir_yes_sir`."""

from pathlib import Path
from typing import Any

from experiments.sir_yes_sir.runner import run_sir_yes_sir_variant
from experiments.sir_yes_sir.variants.ar.config import SirYesSirArConfig
from experiments.sir_yes_sir.variants.ar.dataset import SirYesSirArDataset
from experiments.sir_yes_sir.variants.ar.evaluation import evaluate_ar
from yessir.experiment import ExperimentVariant


def run_sir_yes_sir_ar(
    config: SirYesSirArConfig,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate the `ar` variant of `sir_yes_sir` according to `config`."""
    train_dataset = SirYesSirArDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbol_pool,
        step_order=config.step_order,
        embedding_mode=config.embedding_mode,
        seed=config.seed,
    )
    eval_dataset = SirYesSirArDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbol_pool,
        step_order=config.step_order,
        embedding_mode=config.embedding_mode,
        seed=config.eval_seed,
    )
    return run_sir_yes_sir_variant(
        variant_name="ar",
        config=config,
        out_dir=out_dir,
        num_classes=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluate_fn=evaluate_ar,
    )


AR_VARIANT = ExperimentVariant(
    name="ar",
    description="Autoregressive trace harness read through TiedIdentityHead.",
    config_cls=SirYesSirArConfig,
    run=run_sir_yes_sir_ar,
)
