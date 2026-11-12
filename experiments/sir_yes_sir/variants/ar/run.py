"""Experiment runner (`run_sir_yes_sir_ar`) and `AR_VARIANT` descriptor for `sir_yes_sir`."""

from pathlib import Path
from typing import Any

from experiments.sir_yes_sir.variants.ar.config import SirYesSirArConfig
from experiments.sir_yes_sir.variants.ar.dataset import SirYesSirArDataset
from experiments.sir_yes_sir.variants.ar.evaluation import evaluate_ar
from yessir.artefacts import write_config_json, write_eval_json
from yessir.experiment import ExperimentVariant
from yessir.metrics import MetricsLogger
from yessir.model import Transformer
from yessir.training import seed_everything, train


def run_sir_yes_sir_ar(
    config: SirYesSirArConfig,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate the `ar` variant of `sir_yes_sir` according to `config`."""
    dest = str(out_dir) if out_dir is not None else "no-save"
    print(
        f"[sir_yes_sir:ar] Starting run: B={config.alphabet_size}, L={config.x_length}, "
        f"step_order={config.step_order.value}, mode={config.embedding_mode.value}, "
        f"layers={config.n_layers}, heads={config.n_heads}, d_model={config.d_model}, "
        f"steps={config.steps}, batch_size={config.batch_size}, lr={config.lr}, "
        f"seed={config.seed} -> {dest}",
        flush=True,
    )

    logger: MetricsLogger | None = None
    if out_dir is not None:
        write_config_json(
            out_dir,
            experiment_name="sir_yes_sir",
            variant_name="ar",
            config=config,
        )
        logger = MetricsLogger(out_dir / "train.jsonl")

    seed_everything(config.seed)

    model = Transformer(
        num_classes=None,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        use_token_type=config.use_token_type,
        d_position=config.d_position,
        d_scratch=config.d_scratch,
    )

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

    def _on_step(step_idx: int, step_loss: float) -> None:
        if logger is not None:
            logger.log(step_idx, {"loss": step_loss})
        print(
            f"[sir_yes_sir:ar] step={step_idx}/{config.steps} loss={step_loss:.4f}",
            flush=True,
        )

    train(
        model,
        train_dataset,
        steps=config.steps,
        batch_size=config.batch_size,
        lr=config.lr,
        on_step=_on_step,
    )

    metrics = evaluate_ar(
        model,
        eval_dataset,
        num_problems=config.num_eval_problems,
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
        f"[sir_yes_sir:ar] eval ({config.num_eval_problems} problems): "
        f"exact_match={metrics.exact_match:.2%} "
        f"step_acc={metrics.step_accuracy:.2%} "
        f"halt_acc={metrics.halt_accuracy:.2%}",
        flush=True,
    )
    if out_dir is not None:
        print(f"[sir_yes_sir:ar] artefacts written to {out_dir}", flush=True)
    return eval_results


AR_VARIANT = ExperimentVariant(
    name="ar",
    description="Autoregressive trace harness read through TiedIdentityHead.",
    config_cls=SirYesSirArConfig,
    run=run_sir_yes_sir_ar,
)
