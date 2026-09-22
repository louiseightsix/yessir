#!/usr/bin/env python3
"""Interactive Experiment Configurator & Runner for TransformAdder.

Interactively configures an architectural variant, generates a dedicated
reproducible experiment directory in experiments/, creates self-contained
run.sh and run.py scripts with an explicit random seed, and optionally launches execution.

Usage:
    python scripts/configure_experiment.py                 # Interactive configuration wizard
    python scripts/configure_experiment.py --help          # CLI flag options
    python scripts/configure_experiment.py --dry-run ...   # Non-interactive / scripted execution
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformadder.utils.runner import Variant, run_experiment

PARADIGM_CHOICES = [
    ("sm_mh", "State Machine: Multi-Head Parallel Projections (W_sum, W_carry, W_flag)"),
    ("sm_seq", "State Machine: Sequential Single-Head Tied Copy Circuit (o -> c -> FLAG)"),
    ("sm_seq_rev", "State Machine: Sequential Carry-First Micro-AR (c -> o -> FLAG)"),
    ("sm_fork", "State Machine: Branching Parallel Queries ([S] -> sum, [C] -> carry, [F] -> flag)"),
    ("ar", "Autoregressive: Autonomous Trace Rollout with In-Context Scratchpad"),
]

TIER_CHOICES = [
    ("1", "Tier 1: 2D Split PE (Learned Role 64d + Sinusoidal Significance 192d)"),
    ("2", "Tier 2: Role-Anchored Semantic Delimiters + NoPE (Zero Coordinates)"),
    ("3", "Tier 3: Generic Delimiters + NoPE (Zero Coordinates, Single Delimiter)"),
    ("4", "Tier 4: Generic Delimiters + Role Embeddings (Hybrid: Scope Subspace + Single Anchor)"),
    ("5", "Tier 5: Dual Delimiters [ROLE] [ORIGIN] + NoPE (Decoupled Scope and Zero Origin)"),
    ("6", "Tier 6: Axiom Key-Value Addressing (Pure NoPE + Explicit Little-Endian Numeral Addresses)"),
    ("7", "Tier 7: Axiom-Free KV Addressing (Zero-Preamble NoPE + Option A Full-Span Addresses)"),
]

FFN_CHOICES = [
    ("swiglu", "SwiGLU: Bilinear Gated Feed-Forward (d_ffn=1376)"),
    ("gelu", "GELU: Standard 2-Layer MLP (d_ffn=2048)"),
]

ATTN_CHOICES = [
    ("causal", "Strict Causal Attention (Lower-Triangular Mask)"),
    ("prefix_lm", "Prefix-LM (Bidirectional on Prompt, Causal on Trace)"),
]

REGIME_CHOICES = [
    ("multimodal", "Diverse Multi-Base: {2, 4, 8, 10} (Standard Curriculum)"),
    ("bimodal", "Bimodal Base: {2, 10}"),
    ("base10", "Single-Base Control: {10} Only (Memorisation Baseline)"),
]

PRESET_CHOICES = [
    ("full", "Full Benchmark: 100,000 steps, eval every 5,000, batch 256"),
    ("standard", "Standard Run:   10,000 steps, eval every 1,000, batch 256"),
    ("quick", "Fast Prototype:  1,000 steps, eval every 200, batch 128"),
    ("custom", "Custom: Manually specify steps, eval interval, and batch size"),
]

TRAIN_AXIOM_CHOICES = [
    ("random", "Random Symbols: Dynamic per-problem bytes from 240-byte pool (Forces in-context pointer circuit)"),
    ("canonical", "Fixed Canonical: Standard ASCII '0'..'9' (NC-1 Negative Control: tests static memorisation)"),
]

EVAL_AXIOM_CHOICES = [
    ("both", "Both: Evaluate Random and Canonical side-by-side (Tracks grokking)"),
    ("random", "Random Only: Pure in-context pointer evaluation (2x faster checkpointing)"),
    ("canonical", "Canonical Only: Evaluate standard numerals only"),
]

TARGET_CHOICES = [
    ("local", "Local Machine: Run directly on Apple Silicon (MPS), CUDA, or CPU"),
    ("vertex", "Google Cloud: Submit as a managed Vertex AI Custom Training Job"),
]

ACCELERATOR_CHOICES = [
    ("NVIDIA_L4", "NVIDIA L4 (24GB Ada Lovelace): Fast, modern, cost-effective (Recommended)"),
    ("NVIDIA_TESLA_T4", "NVIDIA Tesla T4 (16GB Turing): Budget cloud GPU"),
    ("NVIDIA_A100_80GB", "NVIDIA A100 (80GB Ampere): Maximum throughput for large runs"),
    ("none", "CPU Only: 8 vCPUs (n1-standard-8)"),
]

ACCELERATOR_MACHINE_MAP = {
    "NVIDIA_L4": "g2-standard-4",
    "NVIDIA_TESLA_T4": "n1-standard-4",
    "NVIDIA_A100_80GB": "a2-ultragpu-1g",
    "none": "n1-standard-8",
}


def get_default_gcp_project() -> str:
    """Attempt to detect the active GCP project using gcloud CLI."""
    import shutil
    import subprocess
    if shutil.which("gcloud"):
        try:
            res = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=False,
            )
            val = res.stdout.strip()
            if val and val != "(unset)":
                return val
        except Exception:
            pass
    return ""


def prompt_menu(title: str, options: list[tuple[str, str]], default_idx: int = 0) -> str:
    """Prompt user with a clean numbered menu and return selected key."""
    print(f"\n[?] {title}:")
    for i, (key, desc) in enumerate(options):
        tag = " (default)" if i == default_idx else ""
        print(f"    [{i + 1}] {key:<10} - {desc}{tag}")

    while True:
        choice = input(f"    Select [1-{len(options)}] (default {default_idx + 1}): ").strip()
        if not choice:
            return options[default_idx][0]
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        # Allow typing the key directly
        for key, _ in options:
            if choice.lower() == key.lower():
                return key
        print(f"    Invalid choice '{choice}'. Please enter a number between 1 and {len(options)}.")


def prompt_text(prompt: str, default: str) -> str:
    """Prompt user for a text/numerical string with default."""
    val = input(f"\n[?] {prompt} [default: {default}]: ").strip()
    return val if val else default


def prompt_confirm(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no confirmation."""
    def_str = "Y/n" if default else "y/N"
    val = input(f"\n[?] {prompt} [{def_str}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "1", "true")


def build_experiment_name(
    paradigm: str, tier: int, n_layers: int, ffn: str, attn: str, regime: str, seed: int, train_axiom: str = "random"
) -> str:
    p_map = {
        "sm_mh": "SM-MH",
        "sm_seq": "SM-Seq",
        "sm_seq_rev": "SM-SeqRev",
        "sm_fork": "SM-Fork",
        "ar": "AR",
    }
    p_str = p_map.get(paradigm, paradigm.upper())
    ffn_str = "SwiGLU" if ffn == "swiglu" else "GELU"
    attn_str = "PrefixLM" if attn == "prefix_lm" else "Causal"
    suffix = "-NC1Canon" if train_axiom == "canonical" else ""
    return f"{p_str}-T{tier}-L{n_layers}-{ffn_str}-{attn_str}-{regime}{suffix}-seed{seed}"


def create_experiment_bundle(
    paradigm: str,
    tier: int,
    n_layers: int,
    ffn_type: str,
    attn_type: str,
    regime: str,
    seed: int,
    train_steps: int = 100_000,
    eval_interval: int = 5000,
    batch_size: int = 256,
    eval_problems: int = 100,
    train_axiom: str = "random",
    eval_axioms: str = "both",
    final_eval_problems: int = None,
    target: str = "local",
    gcp_project: str = "",
    gcp_region: str = "us-central1",
    gcp_bucket: str = "",
    gcp_accelerator: str = "NVIDIA_L4",
    gcp_machine_type: str = None,
) -> tuple[Path, str]:
    """Create dedicated subdirectory inside experiments/ with config.json, run.sh, submit_vertex.sh, and run.py."""
    exp_name = build_experiment_name(paradigm, tier, n_layers, ffn_type, attn_type, regime, seed, train_axiom=train_axiom)
    exp_dir = REPO_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if final_eval_problems is None:
        final_eval_problems = 250 if train_steps >= 50_000 else eval_problems

    machine_type = gcp_machine_type or ACCELERATOR_MACHINE_MAP.get(gcp_accelerator, "g2-standard-4")

    # 1. Save config.json
    config_data = {
        "experiment_name": exp_name,
        "paradigm": paradigm,
        "tier": tier,
        "n_layers": n_layers,
        "ffn_type": ffn_type,
        "attn_type": attn_type,
        "regime": regime,
        "seed": seed,
        "train_steps": train_steps,
        "eval_interval": eval_interval,
        "batch_size": batch_size,
        "eval_problems": eval_problems,
        "final_eval_problems": final_eval_problems,
        "train_axiom": train_axiom,
        "eval_axioms": eval_axioms,
        "target": target,
        "gcp_project": gcp_project,
        "gcp_region": gcp_region,
        "gcp_bucket": gcp_bucket,
        "gcp_accelerator": gcp_accelerator,
        "gcp_machine_type": machine_type,
        "created_at": datetime.now().isoformat(),
        "reproducible_command": f"./experiments/{exp_name}/run.sh",
        "vertex_command": f"./experiments/{exp_name}/submit_vertex.sh",
    }
    config_path = exp_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # 2. Save standalone run.py
    run_script_content = f'''"""Standalone Experiment Runner for: {exp_name}
Deterministic execution seed: {seed}
Created at: {config_data["created_at"]}
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformadder.utils.runner import Variant, run_experiment

EXP_DIR = Path(__file__).resolve().parent

VARIANT = Variant(
    name="{exp_name}",
    paradigm="{paradigm}",
    tier={tier},
    log_dir=str(EXP_DIR),
    n_layers={n_layers},
    ffn_type="{ffn_type}",
    attn_type="{attn_type}",
    seed={seed},
    train_steps={train_steps},
    eval_interval={eval_interval},
    batch_size={batch_size},
    eval_problems={eval_problems},
    final_eval_problems={final_eval_problems},
    train_axiom="{train_axiom}",
    eval_axioms="{eval_axioms}",
    regime="{regime}",
)

if __name__ == "__main__":
    run_experiment(VARIANT)
'''
    run_script_path = exp_dir / "run.py"
    with open(run_script_path, "w", encoding="utf-8") as f:
        f.write(run_script_content)
    run_script_path.chmod(0o755)

    # 3. Save standalone run.sh (Local execution)
    sh_script_content = f'''#!/usr/bin/env bash
# Standalone execution shell script for: {exp_name}
# Random seed: {seed}
set -e

EXP_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../.." && pwd)"

# Resolve active or platform-appropriate virtual environment
VENV_DIR=""
OS_TYPE="$(uname -s | tr '[:upper:]' '[:lower:]')"
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ] && "$VIRTUAL_ENV/bin/python" -c "import sys" >/dev/null 2>&1; then
    VENV_DIR="$VIRTUAL_ENV"
elif [ -x "$REPO_DIR/.venv/bin/python" ] && "$REPO_DIR/.venv/bin/python" -c "import sys" >/dev/null 2>&1; then
    VENV_DIR="$REPO_DIR/.venv"
elif [ -x "$REPO_DIR/.venv-$OS_TYPE/bin/python" ] && "$REPO_DIR/.venv-$OS_TYPE/bin/python" -c "import sys" >/dev/null 2>&1; then
    VENV_DIR="$REPO_DIR/.venv-$OS_TYPE"
fi

if [ -n "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

exec python "$EXP_DIR/run.py" "$@"
'''
    sh_script_path = exp_dir / "run.sh"
    with open(sh_script_path, "w", encoding="utf-8") as f:
        f.write(sh_script_content)
    sh_script_path.chmod(0o755)

    # 4. Save standalone submit_vertex.sh (Google Cloud Vertex AI Custom Job)
    vertex_script_content = f'''#!/usr/bin/env bash
# Standalone Google Cloud Vertex AI Custom Job submission for: {exp_name}
# Random seed: {seed}
# Created at: {config_data["created_at"]}
set -e

PROJECT_ID="${{GCP_PROJECT:-{gcp_project}}}"
REGION="${{GCP_REGION:-{gcp_region}}}"
BUCKET="${{GCP_BUCKET:-{gcp_bucket}}}"
ACCELERATOR="${{GCP_ACCELERATOR:-{gcp_accelerator}}}"
MACHINE_TYPE="${{GCP_MACHINE_TYPE:-{machine_type}}}"

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: GCP project ID is required." >&2
    echo "Provide it via environment variable: export GCP_PROJECT=your-project-id" >&2
    exit 1
fi

if [ -z "$BUCKET" ]; then
    echo "ERROR: Cloud Storage staging bucket is required (e.g. gs://my-bucket)." >&2
    echo "Provide it via environment variable: export GCP_BUCKET=gs://my-bucket" >&2
    exit 1
fi

# Clean up bucket URI prefix if trailing slash exists
BUCKET="${{BUCKET%/}}"

EXP_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../.." && pwd)"

CLEAN_EXP_NAME="$(echo "{exp_name}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr '.' '-')"
JOB_NAME="ta-${{CLEAN_EXP_NAME:0:45}}-$(date +%s)"

echo "Building Python package from $REPO_DIR..."
cd "$REPO_DIR"
PYTHON_BIN=""
OS_TYPE="$(uname -s | tr '[:upper:]' '[:lower:]')"
for candidate in "$REPO_DIR/.venv/bin/python" "$REPO_DIR/.venv-$OS_TYPE/bin/python"; do
    if [ -x "$candidate" ] && "$candidate" -c "import sys" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
"$PYTHON_BIN" "$REPO_DIR/setup.py" sdist --formats=gztar --dist-dir="$REPO_DIR/dist" > /dev/null

PACKAGE_FILE="$(ls -t "$REPO_DIR"/dist/transformadder-*.tar.gz | head -n 1)"
TIMESTAMP="$(date +%s)"
PACKAGE_GCS_URI="$BUCKET/packages/transformadder-${{TIMESTAMP}}.tar.gz"

echo "Verifying staging bucket $BUCKET..."
if command -v gcloud >/dev/null 2>&1; then
    if ! gcloud storage buckets describe "$BUCKET" >/dev/null 2>&1; then
        echo "[*] Bucket $BUCKET does not exist. Creating bucket in $REGION ($PROJECT_ID)..."
        gcloud storage buckets create "$BUCKET" --project="$PROJECT_ID" --location="$REGION"
    fi
elif command -v gsutil >/dev/null 2>&1; then
    if ! gsutil ls -b "$BUCKET" >/dev/null 2>&1; then
        echo "[*] Bucket $BUCKET does not exist. Creating bucket in $REGION ($PROJECT_ID)..."
        gsutil mb -p "$PROJECT_ID" -l "$REGION" "$BUCKET"
    fi
fi

echo "Uploading package to $PACKAGE_GCS_URI..."
if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "$PACKAGE_FILE" "$PACKAGE_GCS_URI"
elif command -v gsutil >/dev/null 2>&1; then
    gsutil cp "$PACKAGE_FILE" "$PACKAGE_GCS_URI"
else
    echo "ERROR: Neither 'gcloud storage' nor 'gsutil' found." >&2
    exit 1
fi

if [ "$ACCELERATOR" = "none" ] || [ -z "$ACCELERATOR" ]; then
    IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-4.py310:latest"
    WORKER_POOL_SPEC="machine-type=$MACHINE_TYPE,executor-image-uri=$IMAGE_URI,python-module=transformadder.cli"
else
    IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
    WORKER_POOL_SPEC="machine-type=$MACHINE_TYPE,accelerator-type=$ACCELERATOR,accelerator-count=1,executor-image-uri=$IMAGE_URI,python-module=transformadder.cli"
fi

echo "========================================================================"
echo "SUBMITTING VERTEX AI CUSTOM JOB"
echo "========================================================================"
echo "  Job Name:     $JOB_NAME"
echo "  Project:      $PROJECT_ID"
echo "  Region:       $REGION"
echo "  Package:      $PACKAGE_GCS_URI"
echo "  Accelerator:  $ACCELERATOR (Machine: $MACHINE_TYPE)"
echo "  Image:        $IMAGE_URI"
echo "========================================================================"

JOB_RESOURCE_NAME=$(gcloud ai custom-jobs create \\
    --region="$REGION" \\
    --project="$PROJECT_ID" \\
    --display-name="$JOB_NAME" \\
    --python-package-uris="$PACKAGE_GCS_URI" \\
    --worker-pool-spec="$WORKER_POOL_SPEC" \\
    --args="--paradigm={paradigm},--tier={tier},--n-layers={n_layers},--ffn-type={ffn_type},--attn-type={attn_type},--regime={regime},--seed={seed},--train-steps={train_steps},--eval-interval={eval_interval},--batch-size={batch_size},--eval-problems={eval_problems},--final-eval-problems={final_eval_problems},--train-axiom={train_axiom},--eval-axioms={eval_axioms},--log-dir=$BUCKET/experiments/{exp_name}" \\
    --format="value(name)")

echo ""
echo "[✓] Custom Job submitted successfully!"
echo "  Resource: $JOB_RESOURCE_NAME"
echo ""
echo "Cloud Console URL:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "To stream training logs live in your terminal, run:"
echo "  gcloud ai custom-jobs stream-logs $JOB_RESOURCE_NAME --region=$REGION --project=$PROJECT_ID"
'''
    vertex_script_path = exp_dir / "submit_vertex.sh"
    with open(vertex_script_path, "w", encoding="utf-8") as f:
        f.write(vertex_script_content)
    vertex_script_path.chmod(0o755)

    return exp_dir, exp_name


def main():
    parser = argparse.ArgumentParser(description="Configure and run TransformAdder experiments.")
    parser.add_argument("--paradigm", choices=["sm_mh", "sm_seq", "sm_seq_rev", "sm_fork", "ar"], help="Mechanic / runtime paradigm")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="Inductive bias tier")
    parser.add_argument("--n-layers", type=int, default=6, help="Transformer layer depth")
    parser.add_argument("--ffn-type", choices=["swiglu", "gelu"], default="swiglu", help="FFN non-linearity")
    parser.add_argument("--attn-type", choices=["causal", "prefix_lm"], default="causal", help="Attention visibility")
    parser.add_argument("--regime", choices=["multimodal", "bimodal", "base10"], default="multimodal", help="Curriculum")
    parser.add_argument("--seed", type=int, default=None, help="Explicit random seed for 100% repeatability")
    parser.add_argument("--train-steps", type=int, default=None, help="Total training steps")
    parser.add_argument("--eval-interval", type=int, default=None, help="Steps between evaluations")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--eval-problems", type=int, default=None, help="Evaluation problems per cell")
    parser.add_argument("--train-axiom", choices=["random", "canonical"], default=None, help="Training axiom alphabet: random symbols (default) or fixed canonical ASCII numerals (NC-1)")
    parser.add_argument("--eval-axioms", choices=["both", "random", "canonical"], default=None, help="Evaluation axiom sets: both (default), random only, or canonical only")
    parser.add_argument("--target", choices=["local", "vertex"], default="local", help="Execution target: local machine or Google Cloud Vertex AI")
    parser.add_argument("--vertex", action="store_true", help="Shortcut for --target vertex")
    parser.add_argument("--gcp-project", type=str, default=None, help="Google Cloud Project ID for Vertex AI Custom Job")
    parser.add_argument("--gcp-region", type=str, default="us-central1", help="Google Cloud region for Vertex AI Custom Job")
    parser.add_argument("--gcp-bucket", type=str, default=None, help="GCS staging bucket (e.g. gs://my-bucket) for Vertex AI")
    parser.add_argument("--gcp-accelerator", choices=["NVIDIA_L4", "NVIDIA_TESLA_T4", "NVIDIA_A100_80GB", "none"], default="NVIDIA_L4", help="Accelerator type for Vertex AI Custom Job")
    parser.add_argument("--gcp-machine-type", type=str, default=None, help="Compute machine type for Vertex AI (auto-selected by default)")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 step verification immediately")
    parser.add_argument("--interactive", action="store_true", help="Enable educational step-by-step mode")
    parser.add_argument("--no-run", action="store_true", help="Generate experiment directory only; do not execute")
    args = parser.parse_args()

    # If key parameters are omitted and stdin is a TTY (or no CLI arguments were given), run interactive wizard
    is_interactive_mode = (args.paradigm is None or args.tier is None or args.seed is None) and sys.stdin.isatty()

    if is_interactive_mode:
        print("=" * 72)
        print("   TRANSFORMADDER: INTERACTIVE EXPERIMENT CONFIGURATOR")
        print("=" * 72)
        print("Configure an experiment across the architectural design space.")
        print("Each run receives an explicit random seed and a dedicated directory.")

        paradigm = prompt_menu("Select Runtime Paradigm / Mechanic", PARADIGM_CHOICES, default_idx=0)
        tier = int(prompt_menu("Select Inductive Bias Tier", TIER_CHOICES, default_idx=0))
        n_layers = int(prompt_text("Enter Number of Transformer Layers (N_layers)", default="6"))
        ffn_type = prompt_menu("Select FFN Non-Linearity Engine", FFN_CHOICES, default_idx=0)
        attn_type = prompt_menu("Select Attention Visibility Regime", ATTN_CHOICES, default_idx=0)
        regime = prompt_menu("Select Training Base Curriculum", REGIME_CHOICES, default_idx=0)

        # Training symbol alphabet and evaluation axiom sets
        if args.train_axiom is not None:
            train_axiom = args.train_axiom
        else:
            train_axiom = prompt_menu("Select Training Symbol Alphabet Regime", TRAIN_AXIOM_CHOICES, default_idx=0)

        if args.eval_axioms is not None:
            eval_axioms = args.eval_axioms
        else:
            eval_axioms = prompt_menu("Select Evaluation Axiom Sets", EVAL_AXIOM_CHOICES, default_idx=0)

        # Training duration and evaluation preset
        if args.train_steps is not None or args.eval_interval is not None:
            train_steps = args.train_steps if args.train_steps is not None else 100_000
            eval_interval = args.eval_interval if args.eval_interval is not None else (5_000 if train_steps >= 50_000 else max(50, train_steps // 10))
            batch_size = args.batch_size if args.batch_size is not None else 256
            eval_problems = args.eval_problems if args.eval_problems is not None else 100
        else:
            preset = prompt_menu("Select Training Duration & Optimisation Preset", PRESET_CHOICES, default_idx=0)
            if preset == "full":
                train_steps = 100_000
                eval_interval = 5_000
                batch_size = 256
                eval_problems = 100
            elif preset == "standard":
                train_steps = 10_000
                eval_interval = 1_000
                batch_size = 256
                eval_problems = 100
            elif preset == "quick":
                train_steps = 1_000
                eval_interval = 200
                batch_size = 128
                eval_problems = 100
            else:  # custom
                train_steps = int(prompt_text("Enter Total Training Steps", default="100000"))
                eval_interval = int(prompt_text("Enter Evaluation Interval (steps between evals)", default=str(max(50, train_steps // 20))))
                batch_size = int(prompt_text("Enter Batch Size", default="256"))
                eval_problems = int(prompt_text("Enter Number of Evaluation Problems per cell", default="100"))

        print("\n" + "-" * 72)
        print("RANDOM SEED CONFIGURATION:")
        print("The random seed is a primary parameter defining this unique run.")
        print("The same seed gives bitwise-identical data, and run-to-run repeatable training on a")
        print("fixed hardware/software stack (spec/06_implementation.md Section 1).")
        print("-" * 72)
        seed = int(prompt_text("Enter Experiment Random Seed", default="42"))

        # Execution Target
        if args.vertex:
            target = "vertex"
        elif args.target != "local":
            target = args.target
        else:
            target = prompt_menu("Select Execution Target", TARGET_CHOICES, default_idx=0)

        if target == "vertex":
            default_proj = args.gcp_project or get_default_gcp_project() or "my-gcp-project"
            gcp_project = prompt_text("Enter Google Cloud Project ID", default=default_proj)
            gcp_region = prompt_text("Enter Google Cloud Region", default=args.gcp_region or "us-central1")
            default_bucket = args.gcp_bucket or f"gs://{gcp_project}-vertex-staging"
            gcp_bucket = prompt_text("Enter Cloud Storage Bucket for Staging & Artifacts (gs://...)", default=default_bucket)
            gcp_accelerator = prompt_menu("Select Cloud Compute Accelerator", ACCELERATOR_CHOICES, default_idx=0)
            gcp_machine_type = args.gcp_machine_type or ACCELERATOR_MACHINE_MAP.get(gcp_accelerator, "g2-standard-4")
        else:
            gcp_project = args.gcp_project or ""
            gcp_region = args.gcp_region or "us-central1"
            gcp_bucket = args.gcp_bucket or ""
            gcp_accelerator = args.gcp_accelerator or "NVIDIA_L4"
            gcp_machine_type = args.gcp_machine_type or ACCELERATOR_MACHINE_MAP.get(gcp_accelerator, "g2-standard-4")

        dry_run = prompt_confirm("Execute as a quick smoke-test / dry-run (1 step)?", default=False)
        interactive_mode = prompt_confirm("Enable educational step-by-step stepper mode?", default=False) if target == "local" else False
        launch_now = prompt_confirm("Launch this experiment immediately after creation?", default=True)

    else:
        # Scripted or CLI argument mode
        paradigm = args.paradigm if args.paradigm else "sm_mh"
        tier = args.tier if args.tier is not None else 1
        n_layers = args.n_layers
        ffn_type = args.ffn_type
        attn_type = args.attn_type
        regime = args.regime
        seed = args.seed if args.seed is not None else 42
        train_axiom = args.train_axiom if args.train_axiom is not None else "random"
        eval_axioms = args.eval_axioms if args.eval_axioms is not None else "both"
        train_steps = args.train_steps if args.train_steps is not None else 100_000
        eval_interval = args.eval_interval if args.eval_interval is not None else (5_000 if train_steps >= 50_000 else max(50, train_steps // 10))
        batch_size = args.batch_size if args.batch_size is not None else 256
        eval_problems = args.eval_problems if args.eval_problems is not None else 100
        target = "vertex" if args.vertex else args.target
        gcp_project = args.gcp_project or get_default_gcp_project() or ""
        gcp_region = args.gcp_region or "us-central1"
        gcp_bucket = args.gcp_bucket or ""
        gcp_accelerator = args.gcp_accelerator or "NVIDIA_L4"
        gcp_machine_type = args.gcp_machine_type or ACCELERATOR_MACHINE_MAP.get(gcp_accelerator, "g2-standard-4")
        dry_run = args.dry_run
        interactive_mode = args.interactive if target == "local" else False
        launch_now = not args.no_run

    final_eval_problems = 250 if train_steps >= 50_000 else eval_problems

    # Create experiment bundle inside experiments/
    exp_dir, exp_name = create_experiment_bundle(
        paradigm=paradigm,
        tier=tier,
        n_layers=n_layers,
        ffn_type=ffn_type,
        attn_type=attn_type,
        regime=regime,
        seed=seed,
        train_steps=train_steps,
        eval_interval=eval_interval,
        batch_size=batch_size,
        eval_problems=eval_problems,
        final_eval_problems=final_eval_problems,
        train_axiom=train_axiom,
        eval_axioms=eval_axioms,
        target=target,
        gcp_project=gcp_project,
        gcp_region=gcp_region,
        gcp_bucket=gcp_bucket,
        gcp_accelerator=gcp_accelerator,
        gcp_machine_type=gcp_machine_type,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT BUNDLE CREATED:")
    print(f"  Directory:     {exp_dir}")
    print(f"  Shell (Local): ./experiments/{exp_name}/run.sh")
    print(f"  Shell (Cloud): ./experiments/{exp_name}/submit_vertex.sh")
    print(f"  Python:        python experiments/{exp_name}/run.py")
    print(f"  Config:        {exp_dir / 'config.json'}")
    print(f"  Target:        {target.upper()}")
    if target == "vertex":
        print(f"  GCP Project:   {gcp_project or '(unset)'}")
        print(f"  GCP Region:    {gcp_region}")
        print(f"  GCP Bucket:    {gcp_bucket or '(unset)'}")
        print(f"  Accelerator:   {gcp_accelerator} ({gcp_machine_type})")
    print(f"  Seed:          {seed}")
    print(f"  Train Axiom:   {train_axiom}")
    print(f"  Eval Axioms:   {eval_axioms}")
    print(f"  Train Steps:   {train_steps:,} (eval every {eval_interval:,} steps)")
    print(f"  Batch Size:    {batch_size}")
    print("=" * 72)

    if launch_now:
        if target == "vertex":
            print(f"\n[*] Submitting Vertex AI Custom Job: {exp_name}...\n")
            sh_vertex = exp_dir / "submit_vertex.sh"
            import shutil
            import subprocess
            if not shutil.which("gcloud"):
                print("[!] Notice: 'gcloud' CLI was not found in current PATH.")
                print(f"[✓] Standalone Vertex AI submission script generated at:\n    ./experiments/{exp_name}/submit_vertex.sh")
                print("\nTo submit the job from your host machine terminal:")
                print(f"    ./experiments/{exp_name}/submit_vertex.sh")
            elif not gcp_project or not gcp_bucket:
                print("[!] Notice: GCP project or GCS bucket not configured.")
                print(f"[✓] Standalone Vertex AI submission script generated at:\n    ./experiments/{exp_name}/submit_vertex.sh")
                print("\nRun with environment variables:")
                print(f"    GCP_PROJECT=my-project GCP_BUCKET=gs://my-bucket ./experiments/{exp_name}/submit_vertex.sh")
            else:
                try:
                    subprocess.run(["bash", str(sh_vertex)], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"\n[!] Vertex AI submission failed with exit code {e.returncode}")
        else:
            print(f"\n[*] Launching experiment: {exp_name}...\n")
            # Construct variant and invoke run_experiment directly
            variant = Variant(
                name=exp_name,
                paradigm=paradigm,
                tier=tier,
                log_dir=str(exp_dir),
                n_layers=n_layers,
                ffn_type=ffn_type,
                attn_type=attn_type,
                seed=seed,
                train_steps=train_steps,
                eval_interval=eval_interval,
                batch_size=batch_size,
                eval_problems=eval_problems,
                final_eval_problems=final_eval_problems,
                train_axiom=train_axiom,
                eval_axioms=eval_axioms,
                regime=regime,
            )

            # Set sys.argv for parse_args within run_experiment
            runner_args = [
                f"experiments/{exp_name}/run.py",
                "--regime", regime,
                "--seed", str(seed),
                "--train-steps", str(train_steps),
                "--eval-interval", str(eval_interval),
                "--batch-size", str(batch_size),
                "--eval-problems", str(eval_problems),
                "--final-eval-problems", str(final_eval_problems),
                "--train-axiom", train_axiom,
                "--eval-axioms", eval_axioms,
            ]
            if dry_run:
                runner_args.append("--dry-run")
            if interactive_mode:
                runner_args.append("--interactive")

            sys.argv = runner_args
            run_experiment(variant)
    else:
        print(f"\nExperiment ready. Run at any time with:")
        print(f"  Local: ./experiments/{exp_name}/run.sh")
        print(f"  Cloud: ./experiments/{exp_name}/submit_vertex.sh")


if __name__ == "__main__":
    main()
