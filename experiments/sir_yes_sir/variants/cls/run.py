"""Experiment runner (`run_sir_yes_sir_cls`) and `CLS_VARIANT` descriptor for `sir_yes_sir`."""

from pathlib import Path
from typing import Any

from experiments.sir_yes_sir.config import CLS_DESCRIPTION, SirYesSirConfig
from experiments.sir_yes_sir.train_and_evaluate import train_and_evaluate
from experiments.sir_yes_sir.variants.cls.dataset import SirYesSirClsDataset
from experiments.sir_yes_sir.variants.cls.evaluation import evaluate_cls
from yessir.experiment import ExperimentVariant
from yessir.heads import HeadType
from yessir.vocab import VOCAB_SIZE


def run_sir_yes_sir_cls(
    config: SirYesSirConfig,
    out_dir: Path | None = None,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate the `cls` variant of `sir_yes_sir` according to `config`."""
    train_dataset = SirYesSirClsDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbol_pool,
        embedding_mode=config.embedding_mode,
        seed=config.seed,
    )
    eval_dataset = SirYesSirClsDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbol_pool,
        embedding_mode=config.embedding_mode,
        seed=config.eval_seed,
    )
    return train_and_evaluate(
        variant_name="cls",
        config=config,
        out_dir=out_dir,
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluate_fn=evaluate_cls,
        device=device,
    )


CLS_VARIANT = ExperimentVariant(
    name="cls",
    description=CLS_DESCRIPTION,
    config_cls=SirYesSirConfig,
    run=run_sir_yes_sir_cls,
)
