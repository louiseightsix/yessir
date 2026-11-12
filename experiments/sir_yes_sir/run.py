"""Experiment runner and `Experiment` descriptor for `sir_yes_sir`."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.dataset import SirYesSirDataset
from experiments.sir_yes_sir.evaluation import evaluate_sm_mh
from yessir.experiment import Experiment, run_experiment_cli
from yessir.model import Transformer
from yessir.training import seed_everything, train
from yessir.vocab import VOCAB_SIZE


def run_sir_yes_sir(
    config: SirYesSirConfig,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate a `sir_yes_sir` model according to `config`."""
    _ = out_dir  # Used once Stage A step 9 artefact recording lands.
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
    metrics = evaluate_sm_mh(
        model,
        eval_dataset,
        num_problems=config.eval_problems,
        batch_size=config.eval_batch_size,
    )

    summary: dict[str, Any] = {
        "final_loss": losses[-1],
        "exact_match": metrics.exact_match,
        "step_accuracy": metrics.step_accuracy,
        "halt_accuracy": metrics.halt_accuracy,
    }
    print(
        f"[sir_yes_sir] step={config.steps} "
        f"loss={summary['final_loss']:.4f} "
        f"exact_match={metrics.exact_match:.2%} "
        f"step_acc={metrics.step_accuracy:.2%} "
        f"halt_acc={metrics.halt_accuracy:.2%}"
    )
    return summary


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
