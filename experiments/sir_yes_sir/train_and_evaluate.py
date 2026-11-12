"""Shared training, evaluation, and artefact-recording loop (`train_and_evaluate`) for `sir_yes_sir` variants."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.evaluation import EvalMetrics
from yessir.artefacts import write_config_json, write_eval_json
from yessir.dataset import LabelledExample
from yessir.heads import HeadType
from yessir.metrics import MetricsLogger
from yessir.model import Transformer
from yessir.training import seed_everything, train


def train_and_evaluate(
    *,
    variant_name: str,
    config: SirYesSirConfig,
    out_dir: Path | None,
    head_type: HeadType,
    num_classes: Sequence[int] | None = None,
    train_dataset: Iterable[LabelledExample],
    eval_dataset: Iterable[LabelledExample],
    evaluate_fn: Callable[..., EvalMetrics],
) -> dict[str, Any]:
    """Run training, evaluation, and artefact logging for one `sir_yes_sir` variant."""
    tag = f"[sir_yes_sir:{variant_name}]"
    dest = str(out_dir) if out_dir is not None else "no-save"
    print(
        f"{tag} Starting run: B={config.alphabet_size}, L={config.x_length}, "
        f"mode={config.embedding_mode.value}, layers={config.n_layers}, heads={config.n_heads}, "
        f"d_model={config.d_model}, steps={config.steps}, batch_size={config.batch_size}, "
        f"lr={config.lr}, seed={config.seed} -> {dest}",
        flush=True,
    )

    logger: MetricsLogger | None = None
    if out_dir is not None:
        write_config_json(
            out_dir,
            experiment_name="sir_yes_sir",
            variant_name=variant_name,
            config=config,
        )
        logger = MetricsLogger(out_dir / "train.jsonl")

    seed_everything(config.seed)

    model = Transformer(
        head_type=head_type,
        num_classes=num_classes,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        use_token_type=config.use_token_type,
        d_position=config.d_position,
        d_scratch=config.d_scratch,
    )

    def _on_step(step_idx: int, step_loss: float) -> None:
        if logger is not None:
            logger.log(step_idx, {"loss": step_loss})
        print(
            f"{tag} step={step_idx}/{config.steps} loss={step_loss:.4f}",
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

    metrics = evaluate_fn(
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
        f"{tag} eval ({config.num_eval_problems} problems): "
        f"exact_match={metrics.exact_match:.2%} "
        f"step_acc={metrics.step_accuracy:.2%} "
        f"halt_acc={metrics.halt_accuracy:.2%}",
        flush=True,
    )
    if out_dir is not None:
        print(f"{tag} artefacts written to {out_dir}", flush=True)
    return eval_results
