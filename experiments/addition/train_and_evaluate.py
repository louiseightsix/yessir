"""Shared training, periodic/final multi-split evaluation, and artefact-recording loop (`train_and_evaluate`) for `addition` variants."""

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from experiments.addition.config import AdditionConfig
from experiments.addition.evaluation import EvalMetrics
from experiments.addition.inputs import AdditionProblem, iter_problems
from yessir.artefacts import write_config_json, write_eval_json
from yessir.dataset import LabelledExample
from yessir.heads import HeadType
from yessir.metrics import MetricsLogger, json_safe
from yessir.model import Transformer
from yessir.training import build_optimizer, resolve_device, seed_everything, train


class EvaluateProblemsFn(Protocol):
    """Callable contract for a variant's rollout evaluator over an `AdditionProblem` stream."""

    def __call__(
        self,
        model: Transformer,
        problems: Iterable[AdditionProblem],
        *,
        num_problems: int,
        batch_size: int = 64,
        per_base_metrics: dict[int, EvalMetrics] | None = None,
    ) -> EvalMetrics: ...


def write_eval_splits_json(
    out_dir: Path,
    eval_splits: dict[str, dict[str, float | None]],
) -> Path:
    """Write the full `EvalMetrics` diagnostic breakdown per split to `out_dir / 'eval_splits.json'`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval_splits.json"
    path.write_text(
        json.dumps(json_safe(eval_splits), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def train_and_evaluate(
    *,
    variant_name: str,
    config: AdditionConfig,
    out_dir: Path | None,
    head_type: HeadType,
    num_classes: Sequence[int] | None = None,
    train_dataset: Iterable[LabelledExample],
    evaluate_fn: EvaluateProblemsFn | Callable[..., EvalMetrics],
    train_batch_size: int | None = None,
    include_alphabet: bool = True,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train one `Transformer` and evaluate it periodically and at `max_steps` across `config.eval_splits()`.

    Training runs in chunks of `config.effective_eval_every` steps sharing a single
    `AdamW` optimiser and a single persistent `iter(train_dataset)` stream, while
    re-seeding each evaluation split from its fixed `split.seed` at every checkpoint
    so accuracy curves are directly comparable across steps without perturbing the
    training trajectory.

    Writes three evaluation artefacts when `out_dir` is given:
    - `eval.jsonl`: one JSON record per evaluated checkpoint (including `max_steps`)
      with `step`, `elapsed_sec`, top-level headline pairs matching `eval.json`, and
      a nested `"splits"` dict holding all 11 `EvalMetrics` fields per split (plus
      `{split}_base{b}` per-base slices for multi-base splits).
    - `eval.json`: final-step compact headline pairs (`exact_match`, `step_accuracy`)
      per split for `web/` tables and charts.
    - `eval_splits.json`: final-step full 11-field diagnostic `EvalMetrics` keyed by
      split name (plus `{split}_base{b}` per-base slices for multi-base splits).
    """
    resolved_device = resolve_device(device)
    tag = f"[addition:{variant_name}]"
    dest = str(out_dir) if out_dir is not None else "no-save"
    interval = config.effective_eval_every
    eval_every_str = str(interval) if interval is not None else "final"
    print(
        f"{tag} Starting run: bases={config.bases}, L={config.min_length}..{config.max_length}, "
        f"mode={config.embedding_mode.value}, layers={config.n_layers}, heads={config.n_heads}, "
        f"d_model={config.d_model}, max_steps={config.max_steps}, eval_every={eval_every_str}, "
        f"batch_size={config.batch_size}, lr={config.lr}, seed={config.seed}, "
        f"device={resolved_device} -> {dest}",
        flush=True,
    )

    train_logger: MetricsLogger | None = None
    eval_logger: MetricsLogger | None = None
    if out_dir is not None:
        write_config_json(
            out_dir,
            experiment_name="addition",
            variant_name=variant_name,
            config=config,
            device=str(resolved_device),
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

    optimizer = build_optimizer(model, lr=config.lr)
    train_stream = iter(train_dataset)
    splits = config.eval_splits()

    def _make_on_step(step_offset: int) -> Callable[[int, float], None]:
        def _on_step(local_step: int, step_loss: float) -> None:
            global_step = step_offset + local_step
            if (
                global_step == 1
                or global_step % config.log_every == 0
                or global_step == config.max_steps
            ):
                if train_logger is not None:
                    train_logger.log(global_step, {"loss": step_loss})
                print(
                    f"{tag} step={global_step}/{config.max_steps} loss={step_loss:.4f}",
                    flush=True,
                )

        return _on_step

    effective_batch_size = (
        config.batch_size if train_batch_size is None else train_batch_size
    )

    completed_steps = 0
    eval_results: dict[str, Any] = {}
    eval_splits_results: dict[str, dict[str, float | None]] = {}

    while completed_steps < config.max_steps:
        remaining = config.max_steps - completed_steps
        chunk_steps = remaining if interval is None else min(interval, remaining)
        train(
            model,
            train_stream,
            max_steps=chunk_steps,
            batch_size=effective_batch_size,
            optimizer=optimizer,
            on_step=_make_on_step(completed_steps),
        )
        completed_steps += chunk_steps

        eval_results = {}
        eval_splits_results = {}
        for split in splits:
            problems = iter_problems(
                bases=split.bases,
                min_length=split.min_length,
                max_length=split.max_length,
                symbols=config.symbol_pool,
                p_prior=split.p_prior,
                permuted_canonical=split.permuted_canonical,
                p_chain=split.p_chain,
                include_alphabet=include_alphabet,
                seed=split.seed,
            )
            per_base: dict[int, EvalMetrics] | None = (
                {} if len(split.bases) > 1 else None
            )
            if per_base is not None:
                metrics = evaluate_fn(
                    model,
                    problems,
                    num_problems=config.num_eval_problems,
                    batch_size=config.eval_batch_size,
                    per_base_metrics=per_base,
                )
            else:
                metrics = evaluate_fn(
                    model,
                    problems,
                    num_problems=config.num_eval_problems,
                    batch_size=config.eval_batch_size,
                )
            eval_results[f"{split.prefix}exact_match"] = metrics.exact_match
            eval_results[f"{split.prefix}step_accuracy"] = metrics.step_accuracy
            eval_splits_results[split.name] = metrics.to_dict()
            if per_base is not None:
                for b in split.bases:
                    if b in per_base:
                        eval_splits_results[f"{split.name}_base{b}"] = per_base[
                            b
                        ].to_dict()

            carry_str = (
                f"{metrics.carry_accuracy:.2%}"
                if metrics.carry_accuracy is not None
                else "n/a"
            )
            print(
                f"{tag} eval[{split.name}] ({config.num_eval_problems} problems): "
                f"step={completed_steps}/{config.max_steps} "
                f"exact_match={metrics.exact_match:.2%} "
                f"step_acc={metrics.step_accuracy:.2%} "
                f"digit_acc={metrics.digit_accuracy:.2%} "
                f"carry_acc={carry_str} "
                f"halt_acc={metrics.halt_accuracy:.2%}",
                flush=True,
            )

        if eval_logger is not None:
            eval_logger.log(
                completed_steps,
                {**eval_results, "splits": eval_splits_results},
            )

    if out_dir is not None:
        write_eval_json(out_dir, eval_results)
        write_eval_splits_json(out_dir, eval_splits_results)
        print(f"{tag} artefacts written to {out_dir}", flush=True)

    return eval_results
