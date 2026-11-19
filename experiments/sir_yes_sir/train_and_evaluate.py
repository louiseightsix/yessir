"""Shared training, periodic/final evaluation, and artefact-recording loop (`train_and_evaluate`) for `sir_yes_sir` variants."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.evaluation import EvalMetrics
from yessir.artefacts import write_config_json, write_eval_json
from yessir.dataset import LabelledExample
from yessir.heads import HeadType
from yessir.metrics import MetricsLogger
from yessir.model import Transformer
from yessir.training import (
    build_lr_scheduler,
    build_optimizer,
    dtype_name,
    resolve_device,
    resolve_dtype,
    seed_everything,
    train,
)


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
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> dict[str, Any]:
    """Run training, periodic/final evaluation, and artefact logging for one `sir_yes_sir` variant."""
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)
    resolved_dtype_name = dtype_name(resolved_dtype)
    tag = f"[sir_yes_sir:{variant_name}]"
    dest = str(out_dir) if out_dir is not None else "no-save"
    interval = config.effective_eval_every
    eval_every_str = str(interval) if interval is not None else "final"
    print(
        f"{tag} Starting run: B={config.alphabet_size}, L={config.x_length}, "
        f"mode={config.embedding_mode_label}, layers={config.n_layers}, heads={config.n_heads}, "
        f"d_model={config.d_model}, max_steps={config.max_steps}, eval_every={eval_every_str}, "
        f"batch_size={config.batch_size}, accum={config.grad_accum_steps}, lr={config.lr}, "
        f"wd={config.weight_decay}, warmup={config.warmup_steps}, seed={config.seed}, "
        f"device={resolved_device}, dtype={resolved_dtype_name} -> {dest}",
        flush=True,
    )

    train_logger: MetricsLogger | None = None
    eval_logger: MetricsLogger | None = None
    if out_dir is not None:
        write_config_json(
            out_dir,
            experiment_name="sir_yes_sir",
            variant_name=variant_name,
            config=config,
            device=str(resolved_device),
            dtype=resolved_dtype_name,
        )
        train_logger = MetricsLogger(out_dir / "train.jsonl")
        eval_logger = MetricsLogger(out_dir / "eval.jsonl")

    seed_everything(config.seed)

    model = Transformer(
        head_type=head_type,
        num_classes=num_classes,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        use_token_type=config.use_token_type,
        d_position=config.effective_d_position,
        d_scratch=config.effective_d_scratch,
    ).to(resolved_device)

    optimizer = build_optimizer(model, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
    )
    train_stream = iter(train_dataset)

    tokens_seen = 0

    def _make_on_step(step_offset: int) -> Callable[[int, float, int], None]:
        def _on_step(local_step: int, step_loss: float, step_tokens: int) -> None:
            nonlocal tokens_seen
            tokens_seen += step_tokens
            global_step = step_offset + local_step
            if (
                global_step == 1
                or global_step % config.log_every == 0
                or global_step == config.max_steps
            ):
                current_lr = float(optimizer.param_groups[0]["lr"])
                if train_logger is not None:
                    train_logger.log(
                        global_step,
                        {
                            "loss": step_loss,
                            "lr": current_lr,
                            "tokens_seen": tokens_seen,
                        },
                    )
                print(
                    f"{tag} step={global_step}/{config.max_steps} "
                    f"loss={step_loss:.4f} lr={current_lr:.2e} tokens={tokens_seen}",
                    flush=True,
                )

        return _on_step

    completed_steps = 0
    eval_results: dict[str, Any] = {}
    while completed_steps < config.max_steps:
        remaining = config.max_steps - completed_steps
        chunk_steps = remaining if interval is None else min(interval, remaining)
        train(
            model,
            train_stream,
            max_steps=chunk_steps,
            batch_size=config.batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accum_steps=config.grad_accum_steps,
            max_grad_norm=config.max_grad_norm,
            dtype=resolved_dtype,
            on_step_tokens=_make_on_step(completed_steps),
        )
        completed_steps += chunk_steps

        metrics = evaluate_fn(
            model,
            eval_dataset,
            num_problems=config.num_eval_problems,
            batch_size=config.eval_batch_size,
            dtype=resolved_dtype,
        )
        eval_results = {
            "exact_match": metrics.exact_match,
            "step_accuracy": metrics.step_accuracy,
            "halt_accuracy": metrics.halt_accuracy,
        }
        split_metrics: dict[str, float | None] = {
            **eval_results,
            "wrap_around_accuracy": metrics.wrap_around_accuracy,
        }
        if eval_logger is not None:
            eval_logger.log(
                completed_steps,
                {**eval_results, "splits": {"id": split_metrics}},
            )
        print(
            f"{tag} eval ({config.num_eval_problems} problems): "
            f"step={completed_steps}/{config.max_steps} "
            f"exact_match={metrics.exact_match:.2%} "
            f"step_acc={metrics.step_accuracy:.2%} "
            f"halt_acc={metrics.halt_accuracy:.2%}",
            flush=True,
        )

    if out_dir is not None:
        write_eval_json(out_dir, eval_results)
        print(f"{tag} artefacts written to {out_dir}", flush=True)
    return eval_results
