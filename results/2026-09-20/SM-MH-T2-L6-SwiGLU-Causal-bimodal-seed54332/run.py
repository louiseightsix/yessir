"""Standalone Experiment Runner for: SM-MH-T2-L6-SwiGLU-Causal-bimodal-seed54332
Deterministic execution seed: 54332
Created at: 2026-09-18T19:50:43.664191
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformadder.utils.runner import Variant, run_experiment

EXP_DIR = Path(__file__).resolve().parent

VARIANT = Variant(
    name="SM-MH-T2-L6-SwiGLU-Causal-bimodal-seed54332",
    paradigm="sm_mh",
    tier=2,
    log_dir=str(EXP_DIR),
    n_layers=6,
    ffn_type="swiglu",
    attn_type="causal",
    seed=54332,
    train_steps=100000,
    eval_interval=5000,
    batch_size=256,
    eval_problems=100,
    final_eval_problems=10000,
    train_axiom="random",
    eval_axioms="both",
    regime="bimodal",
)

if __name__ == "__main__":
    run_experiment(VARIANT)
