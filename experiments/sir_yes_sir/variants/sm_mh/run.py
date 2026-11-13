"""Experiment runner (`run_sir_yes_sir_sm_mh`) and `SM_MH_VARIANT` descriptor for `sir_yes_sir`."""

from pathlib import Path
from typing import Any

from experiments.sir_yes_sir.config import SM_MH_DESCRIPTION, SirYesSirConfig
from experiments.sir_yes_sir.train_and_evaluate import train_and_evaluate
from experiments.sir_yes_sir.variants.sm_mh.dataset import SirYesSirSmMhDataset
from experiments.sir_yes_sir.variants.sm_mh.evaluation import evaluate_sm_mh
from yessir.experiment import ExperimentVariant
from yessir.heads import HeadType
from yessir.vocab import VOCAB_SIZE


def run_sir_yes_sir_sm_mh(
    config: SirYesSirConfig,
    out_dir: Path | None = None,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate the `sm_mh` variant of `sir_yes_sir` according to `config`."""
    train_dataset = SirYesSirSmMhDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbol_pool,
        embedding_mode=config.embedding_mode,
        seed=config.seed,
    )
    eval_dataset = SirYesSirSmMhDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbol_pool,
        embedding_mode=config.embedding_mode,
        seed=config.eval_seed,
    )
    return train_and_evaluate(
        variant_name="sm_mh",
        config=config,
        out_dir=out_dir,
        head_type=HeadType.MULTITASK,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluate_fn=evaluate_sm_mh,
        device=device,
    )


SM_MH_VARIANT = ExperimentVariant(
    name="sm_mh",
    description=SM_MH_DESCRIPTION,
    config_cls=SirYesSirConfig,
    run=run_sir_yes_sir_sm_mh,
)
