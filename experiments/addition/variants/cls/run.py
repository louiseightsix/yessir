"""Experiment runner (`run_addition_cls`) and `CLS_VARIANT` descriptor for `addition`."""

from functools import partial
from pathlib import Path
from typing import Any

from experiments.addition.config import CLS_DESCRIPTION, AdditionClsConfig
from experiments.addition.train_and_evaluate import train_and_evaluate
from experiments.addition.variants.cls.dataset import AdditionClsDataset
from experiments.addition.variants.cls.evaluation import evaluate_cls
from yessir.experiment import ExperimentVariant
from yessir.heads import HeadType
from yessir.vocab import VOCAB_SIZE


def run_addition_cls(
    config: AdditionClsConfig,
    out_dir: Path | None = None,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate the `cls` variant of `addition` according to `config`."""
    train_dataset = AdditionClsDataset(
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
    evaluate_fn = partial(evaluate_cls, embedding_mode=config.embedding_mode)
    return train_and_evaluate(
        variant_name="cls",
        config=config,
        out_dir=out_dir,
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        train_dataset=train_dataset,
        evaluate_fn=evaluate_fn,
        device=device,
    )


CLS_VARIANT = ExperimentVariant(
    name="cls",
    description=CLS_DESCRIPTION,
    config_cls=AdditionClsConfig,
    run=run_addition_cls,
)
