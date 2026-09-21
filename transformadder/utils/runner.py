"""Shared experiment runner: CLI, training loop, and evaluation cadence common to all variants.
Each script in experiments/ declares its Variant and delegates here, so that hyperparameters
(spec/06_implementation.md Section 7.3) and evaluation are identical across the model matrix.
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from transformadder.core.constants import IGNORE_LABEL, VOCAB_SIZE
from transformadder.core.data import ProceduralArithmeticGenerator
from transformadder.core.model import TransformAdder
from transformadder.core.training import build_optimizer, build_scheduler, seed_everything
from transformadder.utils.evaluation import evaluate
from transformadder.utils.logging import MetricsLogger, json_safe
from transformadder.utils.stepper import InteractiveStepper

REGIMES = {
    "bimodal": [2, 10],
    "multimodal": [2, 4, 8, 10],
    "base10": [10],
}


@dataclass
class Variant:
    name: str         # e.g. "SM-MH-T1-L6-SwiGLU-Causal"
    paradigm: str     # "sm_mh" | "sm_seq" | "sm_seq_rev" | "ar"
    tier: int         # 1 | 2 | 3 | 4 | 5
    log_dir: str      # e.g. "experiments/SM-MH-T1-L6-SwiGLU-Causal-seed42"
    n_layers: int = 6
    ffn_type: str = "swiglu"   # "swiglu" | "gelu"
    attn_type: str = "causal"  # "causal" | "prefix_lm"
    seed: int = 42
    train_steps: int = 100_000
    eval_interval: int = 5000
    batch_size: int = 256
    eval_problems: int = 1000
    final_eval_problems: int = 10_000
    train_axiom: str = "random"   # "random" | "canonical" (NC-1)
    eval_axioms: str = "both"     # "both" | "random" | "canonical"
    regime: str = "multimodal"    # key of REGIMES


def parse_args(variant: Variant):
    parser = argparse.ArgumentParser(description=f"Train and evaluate {variant.name}")
    parser.add_argument("--paradigm", choices=["sm_mh", "sm_seq", "sm_seq_rev", "ar"], default=variant.paradigm, help="Runtime paradigm")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5], default=variant.tier, help="Inductive bias tier")
    parser.add_argument("--n-layers", type=int, default=variant.n_layers, help="Transformer layer depth")
    parser.add_argument("--ffn-type", choices=["swiglu", "gelu"], default=variant.ffn_type, help="FFN non-linearity")
    parser.add_argument("--attn-type", choices=["causal", "prefix_lm"], default=variant.attn_type, help="Attention visibility")
    parser.add_argument("--log-dir", type=str, default=variant.log_dir, help="Directory or GCS path for saving metrics and artifacts")

    parser.add_argument("--interactive", action="store_true", help="Enable interactive step-by-step educational mode")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 training step and a tiny evaluation to verify the pipeline")
    parser.add_argument("--regime", choices=sorted(REGIMES), default=variant.regime, help="Training base regime")
    parser.add_argument("--seed", type=int, default=variant.seed, help="Seed for initialisation and data generation")
    parser.add_argument("--train-steps", type=int, default=variant.train_steps, help="Total optimisation steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear warm-up steps")
    parser.add_argument("--batch-size", type=int, default=variant.batch_size, help="Samples per batch (SM: step transitions; AR: full traces)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--eval-interval", type=int, default=variant.eval_interval, help="Optimisation steps between evaluations")
    parser.add_argument("--eval-problems", type=int, default=variant.eval_problems, help="Problems per evaluation cell during training")
    final_default = getattr(variant, "final_eval_problems", 250 if variant.train_steps >= 50_000 else variant.eval_problems)
    parser.add_argument("--final-eval-problems", type=int, default=final_default, help="Problems per evaluation cell at the end of training")
    parser.add_argument("--train-axiom", choices=["random", "canonical"], default=variant.train_axiom, help="Training axiom alphabet: random symbols (default) or fixed canonical ASCII numerals (NC-1 negative control)")
    parser.add_argument("--eval-axioms", choices=["both", "random", "canonical"], default=variant.eval_axioms, help="Evaluation axiom sets: both (default), random only, or canonical only")
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--log-path", type=str, default=f"{variant.log_dir}/metrics.jsonl", help="Path for JSONL metric logs")
    parsed, _ = parser.parse_known_args()
    return parsed


def compute_loss(model, variant: Variant, batch: dict) -> tuple[torch.Tensor, dict]:
    """Loss objectives of spec/06_implementation.md Section 7.1."""
    if variant.paradigm == "sm_mh":
        logits_sum, logits_carry, logits_flag = model.forward_sm_mh(
            batch["token_ids"], batch["role_ids"], batch["sig_ids"], batch["lengths"]
        )
        loss_sum = F.cross_entropy(logits_sum.float(), batch["target_sum"])
        loss_carry = F.cross_entropy(logits_carry.float(), batch["target_carry"])
        loss_flag = F.cross_entropy(logits_flag.float(), batch["target_flag"])
        parts = {"sum": loss_sum.item(), "carry": loss_carry.item(), "flag": loss_flag.item()}
        return loss_sum + loss_carry + 0.5 * loss_flag, parts

    # Labels are already aligned to input positions (model-emitted tokens only)
    logits = model(batch["token_ids"], batch["role_ids"], batch["sig_ids"])
    loss = F.cross_entropy(
        logits.float().reshape(-1, VOCAB_SIZE), batch["labels"].reshape(-1), ignore_index=IGNORE_LABEL
    )
    return loss, {}


def run_experiment(variant: Variant, args: argparse.Namespace = None):
    if args is None:
        args = parse_args(variant)
    if not hasattr(args, "log_path") or not args.log_path:
        args.log_path = f"{variant.log_dir}/metrics.jsonl"
    if not hasattr(args, "device") or not args.device:
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if not hasattr(args, "regime") or not args.regime:
        args.regime = variant.regime
    if not hasattr(args, "seed") or args.seed is None:
        args.seed = variant.seed
    if not hasattr(args, "train_steps") or args.train_steps is None:
        args.train_steps = variant.train_steps
    if not hasattr(args, "warmup_steps") or args.warmup_steps is None:
        args.warmup_steps = 1000
    if not hasattr(args, "batch_size") or args.batch_size is None:
        args.batch_size = variant.batch_size
    if not hasattr(args, "lr") or args.lr is None:
        args.lr = 3e-4
    if not hasattr(args, "eval_interval") or args.eval_interval is None:
        args.eval_interval = variant.eval_interval
    if not hasattr(args, "eval_problems") or args.eval_problems is None:
        args.eval_problems = variant.eval_problems
    if not hasattr(args, "final_eval_problems") or args.final_eval_problems is None:
        args.final_eval_problems = getattr(variant, "final_eval_problems", 250 if args.train_steps >= 50_000 else args.eval_problems)
    if not hasattr(args, "train_axiom") or not args.train_axiom:
        args.train_axiom = variant.train_axiom
    if not hasattr(args, "eval_axioms") or not args.eval_axioms:
        args.eval_axioms = variant.eval_axioms
    if not hasattr(args, "interactive"):
        args.interactive = False
    if not hasattr(args, "dry_run"):
        args.dry_run = False
    # The architecture is fixed by the Variant; say so if a flag asked for something else
    for field in ("paradigm", "tier", "n_layers", "ffn_type", "attn_type"):
        requested = getattr(args, field, None)
        if requested is not None and requested != getattr(variant, field):
            print(f"[!] Ignoring --{field.replace('_', '-')}={requested}: variant {variant.name} "
                  f"fixes {field}={getattr(variant, field)}.", flush=True)
    if variant.attn_type != "causal":
        # prefix_lm_mask exists in core/blocks.py but is not threaded through training, the
        # harnesses or evaluation; refuse rather than train causally under a PrefixLM label.
        raise NotImplementedError(f"attn_type={variant.attn_type!r} is not wired into the runner yet.")
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_bases = REGIMES[args.regime]

    print("=" * 70)
    print(f"TRANSFORMADDER EXPERIMENT: {variant.name}")
    print(f"Regime: {args.regime} {train_bases} | Seed: {args.seed} | Device: {device}")
    print(f"Axioms: Train={args.train_axiom} | Eval={args.eval_axioms}")
    print("=" * 70)

    model = TransformAdder(
        tier=variant.tier,
        n_layers=variant.n_layers,
        head_type="multi_head" if variant.paradigm == "sm_mh" else "single_head",
        ffn_type=variant.ffn_type,
    ).to(device)
    print(f"Model initialised. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.dry_run:
        print("[Dry-run mode activated: 1 training step and a tiny evaluation]")
        args.train_steps, args.batch_size, args.warmup_steps = 1, 4, 1
        args.eval_problems = args.final_eval_problems = 2

    optimizer = build_optimizer(model, lr=args.lr)
    warmup = min(args.warmup_steps, max(1, args.train_steps // 10))
    scheduler = build_scheduler(optimizer, total_steps=args.train_steps, warmup_steps=warmup)
    logger = MetricsLogger(args.log_path)
    stepper = InteractiveStepper(enabled=args.interactive)
    is_canonical_train = args.train_axiom == "canonical"
    data_gen = ProceduralArithmeticGenerator(bases=train_bases, canonical=is_canonical_train, seed=args.seed)
    make_batch = getattr(data_gen, f"{variant.paradigm}_batch")
    use_amp = device.type == "cuda"

    final_results = None
    for step in range(1, args.train_steps + 1):
        model.train()
        batch = {k: v.to(device) if v is not None else None
                 for k, v in make_batch(args.batch_size, variant.tier).items()}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, parts = compute_loss(model, variant, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 50 == 0 or args.dry_run:
            detail = " (" + ", ".join(f"{k}: {v:.3f}" for k, v in parts.items()) + ")" if parts else ""
            print(f"Step {step:6d} | Loss: {loss.item():.4f}{detail} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        final = step == args.train_steps
        do_eval = final or (step % args.eval_interval == 0) or (step == 1 and args.interactive)
        if do_eval:
            final_n = getattr(args, "final_eval_problems", getattr(args, "eval_problems", 1000))
            n = final_n if final else (2 if (step == 1 and args.interactive) else args.eval_problems)
            results = evaluate(
                model,
                variant.paradigm,
                variant.tier,
                train_bases,
                num_problems=n,
                eval_axioms=args.eval_axioms,
                stepper=stepper,
            )
            logger.log(step=step, metrics={"loss": loss.item(), **results})
            logger.print_eval_summary(step=step, results=results, train_bases=train_bases)
            if final:
                final_results = results

    # Save summary.json to experiment directory
    if final_results is not None:
        import json
        from pathlib import Path
        summary_path = Path(args.log_path).parent / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_data = {
            "variant": variant.name,
            "paradigm": variant.paradigm,
            "tier": variant.tier,
            "seed": args.seed,
            "regime": args.regime,
            "n_layers": variant.n_layers,
            "ffn_type": variant.ffn_type,
            "attn_type": variant.attn_type,
            "final_step": args.train_steps,
            "results": json_safe(final_results),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Evaluation summary saved to: {summary_path}")

    if args.dry_run:
        print("Dry run completed successfully.")
