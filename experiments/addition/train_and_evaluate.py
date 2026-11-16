"""Shared training, multi-split evaluation, and artefact-recording loop (`train_and_evaluate`) for `addition` variants."""

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
from yessir.training import resolve_device, seed_everything, train


class EvaluateProblemsFn(Protocol):
    """Callable contract for a variant's teacher-forced evaluator over an `AdditionProblem` stream."""

    def __call__(
        self,
        model: Transformer,
        problems: Iterable[AdditionProblem],
        *,
        num_problems: int,
        batch_size: int = 64,
    ) -> EvalMetrics: ...


def write_eval_splits_json(
    out_dir: Path,
    eval_splits: dict[str, dict[str, float | None]],
) -> Path:
    """Write the full 9-metric + 4-way error diagnostic breakdown per split to `out_dir / 'eval_splits.json'`."""
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
    """Train one `Transformer` and evaluate it once across `config.eval_splits()`.

    Writes two evaluation artefacts from the same in-memory `EvalMetrics` objects:
    - `eval.json`: compact headline pairs (`exact_match`, `step_accuracy`) per split
      for `web/` tables and charts.
    - `eval_splits.json`: full 9-field diagnostic `EvalMetrics` (including
      `digit_accuracy`, `carry_accuracy`, `halt_accuracy`, and the 4 first-error
      rates) keyed by split name.
    """
    resolved_device = resolve_device(device)
    tag = f"[addition:{variant_name}]"
    dest = str(out_dir) if out_dir is not None else "no-save"
    print(
        f"{tag} Starting run: bases={config.bases}, L={config.min_length}..{config.max_length}, "
        f"mode={config.embedding_mode.value}, layers={config.n_layers}, heads={config.n_heads}, "
        f"d_model={config.d_model}, max_steps={config.max_steps}, batch_size={config.batch_size}, "
        f"lr={config.lr}, seed={config.seed}, device={resolved_device} -> {dest}",
        flush=True,
    )

    logger: MetricsLogger | None = None
    if out_dir is not None:
        write_config_json(
            out_dir,
            experiment_name="addition",
            variant_name=variant_name,
            config=config,
            device=str(resolved_device),
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
        d_position=config.effective_d_position,
        d_scratch=config.effective_d_scratch,
    ).to(resolved_device)

    def _on_step(step_idx: int, step_loss: float) -> None:
        if (
            step_idx == 1
            or step_idx % config.log_every == 0
            or step_idx == config.max_steps
        ):
            if logger is not None:
                logger.log(step_idx, {"loss": step_loss})
            print(
                f"{tag} step={step_idx}/{config.max_steps} loss={step_loss:.4f}",
                flush=True,
            )

    effective_batch_size = (
        config.batch_size if train_batch_size is None else train_batch_size
    )
    train(
        model,
        train_dataset,
        max_steps=config.max_steps,
        batch_size=effective_batch_size,
        lr=config.lr,
        on_step=_on_step,
    )

    eval_results: dict[str, Any] = {}
    eval_splits_results: dict[str, dict[str, float | None]] = {}

    for split in config.eval_splits():
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
        metrics = evaluate_fn(
            model,
            problems,
            num_problems=config.num_eval_problems,
            batch_size=config.eval_batch_size,
        )
        eval_results[f"{split.prefix}exact_match"] = metrics.exact_match
        eval_results[f"{split.prefix}step_accuracy"] = metrics.step_accuracy
        eval_splits_results[split.name] = metrics.to_dict()

        carry_str = (
            f"{metrics.carry_accuracy:.2%}"
            if metrics.carry_accuracy is not None
            else "n/a"
        )
        print(
            f"{tag} eval[{split.name}] ({config.num_eval_problems} problems): "
            f"exact_match={metrics.exact_match:.2%} "
            f"step_acc={metrics.step_accuracy:.2%} "
            f"digit_acc={metrics.digit_accuracy:.2%} "
            f"carry_acc={carry_str} "
            f"halt_acc={metrics.halt_accuracy:.2%}",
            flush=True,
        )

    if out_dir is not None:
        write_eval_json(out_dir, eval_results)
        write_eval_splits_json(out_dir, eval_splits_results)
        print(f"{tag} artefacts written to {out_dir}", flush=True)

    return eval_results
