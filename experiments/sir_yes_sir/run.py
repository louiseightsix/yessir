"""Experiment runner and `Experiment` descriptor for `sir_yes_sir`."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.dataset import SirYesSirDataset
from experiments.sir_yes_sir.evaluation import evaluate_sm_mh
from yessir.artefacts import write_config_json, write_eval_json
from yessir.experiment import Experiment, run_experiment_cli
from yessir.metrics import MetricsLogger
from yessir.model import Transformer
from yessir.training import seed_everything, train
from yessir.vocab import VOCAB_SIZE


def run_sir_yes_sir(
    config: SirYesSirConfig,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate a `sir_yes_sir` model according to `config`."""
    logger: MetricsLogger | None = None
    if out_dir is not None:
        write_config_json(
            out_dir,
            experiment_name="sir_yes_sir",
            config=config,
        )
        logger = MetricsLogger(out_dir / "train.jsonl")

    seed_everything(config.seed)

    model = Transformer(
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        use_token_type=config.use_token_type,
        d_position=config.d_position,
        d_scratch=config.d_scratch,
    )

    train_dataset = SirYesSirDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbols,
        embedding_mode=config.embedding_mode,
        seed=config.seed,
    )
    eval_dataset = SirYesSirDataset(
        alphabet_size=config.alphabet_size,
        x_length=config.x_length,
        symbols=config.symbols,
        embedding_mode=config.embedding_mode,
        seed=config.eval_seed,
    )

    losses = train(
        model,
        train_dataset,
        steps=config.steps,
        batch_size=config.batch_size,
        lr=config.lr,
    )
    if logger is not None:
        for step_idx, step_loss in enumerate(losses, start=1):
            logger.log(step_idx, {"loss": step_loss})

    metrics = evaluate_sm_mh(
        model,
        eval_dataset,
        num_problems=config.eval_problems,
        batch_size=config.eval_batch_size,
    )

    eval_results: dict[str, Any] = {
        "exact_match": metrics.exact_match,
        "step_accuracy": metrics.step_accuracy,
        "halt_accuracy": metrics.halt_accuracy,
    }
    if out_dir is not None:
        write_eval_json(out_dir, eval_results)

    print(
        f"[sir_yes_sir] step={config.steps} "
        f"loss={losses[-1]:.4f} "
        f"exact_match={metrics.exact_match:.2%} "
        f"step_acc={metrics.step_accuracy:.2%} "
        f"halt_acc={metrics.halt_accuracy:.2%}"
    )
    if out_dir is not None:
        print(f"[sir_yes_sir] artefacts written to {out_dir}")
    return eval_results


SIR_YES_SIR_EXPERIMENT = Experiment(
    name="sir_yes_sir",
    description="In-context successor (next symbol mod B, no carry) smoke-test experiment.",
    config_cls=SirYesSirConfig,
    run=run_sir_yes_sir,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run `sir_yes_sir` from the command line."""
    run_experiment_cli(SIR_YES_SIR_EXPERIMENT, argv)


if __name__ == "__main__":
    main()
