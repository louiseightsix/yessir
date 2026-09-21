"""TransformAdder CLI & Vertex AI Custom Job Entrypoint.

This module can be executed directly as:
    python -m transformadder.cli [options]

It supports both local execution and execution inside Google Cloud Vertex AI
pre-built training containers. When running inside Vertex AI, it automatically
detects environment variables (AIP_MODEL_DIR, AIP_CHECKPOINT_DIR) and maps
Cloud Storage URIs (gs://bucket/path) to Cloud Storage FUSE mounts (/gcs/bucket/path).
"""

import argparse
import os
from pathlib import Path

from transformadder.utils.runner import REGIMES, Variant, run_experiment


def gcs_to_fuse(path: str) -> str:
    """Converts a Cloud Storage URI (gs://bucket/...) to a Vertex AI FUSE mount (/gcs/bucket/...)."""
    if path.startswith("gs://"):
        return "/gcs/" + path[5:]
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="TransformAdder: Zero-Shot Algorithmic Generalisation CLI & Cloud Runner"
    )
    # Architectural parameters
    parser.add_argument("--paradigm", choices=["sm_mh", "sm_seq", "ar"], default="sm_mh", help="Runtime paradigm")
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Inductive bias tier (1: Split PE, 2: Role NoPE, 3: Generic NoPE, 4: Generic NoPE+Role, 5: Dual Delimiter)",
    )
    parser.add_argument("--n-layers", type=int, default=6, help="Transformer layer depth")
    parser.add_argument("--ffn-type", choices=["swiglu", "gelu"], default="swiglu", help="FFN non-linearity")
    parser.add_argument("--attn-type", choices=["causal", "prefix_lm"], default="causal", help="Attention visibility")
    parser.add_argument("--regime", choices=sorted(REGIMES), default="multimodal", help="Training base regime")
    parser.add_argument("--seed", type=int, default=42, help="Seed for model initialisation and procedural data generation")

    # Optimisation parameters
    parser.add_argument("--train-steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear warm-up steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Steps between evaluations")
    parser.add_argument("--eval-problems", type=int, default=1000, help="Problems per evaluation cell during training")
    parser.add_argument("--final-eval-problems", type=int, default=None, help="Problems per evaluation cell at final evaluation")

    # Axiom symbol regimes
    parser.add_argument("--train-axiom", choices=["random", "canonical"], default="random", help="Training axiom symbols")
    parser.add_argument("--eval-axioms", choices=["both", "random", "canonical"], default="both", help="Evaluation axiom sets")

    # Storage and execution flags
    parser.add_argument("--log-dir", type=str, default=None, help="Directory or GCS path for saving metrics and artifacts")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda, mps, cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step smoke test")
    parser.add_argument("--interactive", action="store_true", help="Enable educational step-by-step mode")

    return parser.parse_args()


def build_variant_from_args(args) -> Variant:
    p_map = {"sm_mh": "SM-MH", "sm_seq": "SM-Seq", "ar": "AR"}
    p_str = p_map.get(args.paradigm, args.paradigm.upper())
    ffn_str = "SwiGLU" if args.ffn_type == "swiglu" else "GELU"
    attn_str = "PrefixLM" if args.attn_type == "prefix_lm" else "Causal"
    suffix = "-NC1Canon" if args.train_axiom == "canonical" else ""
    exp_name = f"{p_str}-T{args.tier}-L{args.n_layers}-{ffn_str}-{attn_str}-{args.regime}{suffix}-seed{args.seed}"

    # Determine log_dir
    log_dir = args.log_dir
    if not log_dir:
        # Check Vertex AI standard environment variables
        aip_dir = os.environ.get("AIP_MODEL_DIR") or os.environ.get("AIP_CHECKPOINT_DIR")
        if aip_dir:
            log_dir = gcs_to_fuse(aip_dir)
        else:
            log_dir = str(Path("experiments") / exp_name)
    else:
        log_dir = gcs_to_fuse(log_dir)

    final_eval = args.final_eval_problems
    if final_eval is None:
        final_eval = 10_000 if args.train_steps >= 50_000 else args.eval_problems

    return Variant(
        name=exp_name,
        paradigm=args.paradigm,
        tier=args.tier,
        log_dir=log_dir,
        n_layers=args.n_layers,
        ffn_type=args.ffn_type,
        attn_type=args.attn_type,
        seed=args.seed,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        eval_problems=args.eval_problems,
        final_eval_problems=final_eval,
        train_axiom=args.train_axiom,
        eval_axioms=args.eval_axioms,
        regime=args.regime,
    )


def main():
    args = parse_args()
    variant = build_variant_from_args(args)
    run_experiment(variant, args=args)


if __name__ == "__main__":
    main()
