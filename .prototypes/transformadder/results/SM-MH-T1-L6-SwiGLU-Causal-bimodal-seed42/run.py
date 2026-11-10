"""Standalone Experiment Runner for: SM-MH-T1-L6-SwiGLU-Causal-bimodal-seed42
Deterministic execution seed: 42
Created at: 2026-09-18T15:35:12.909729
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformadder.utils.runner import Variant, run_experiment

EXP_DIR = Path(__file__).resolve().parent

VARIANT = Variant(
    name="SM-MH-T1-L6-SwiGLU-Causal-bimodal-seed42",
    paradigm="sm_mh",
    tier=1,
    log_dir=str(EXP_DIR),
    n_layers=6,
    ffn_type="swiglu",
    attn_type="causal",
    seed=42,
    train_steps=1000,
    eval_interval=200,
    batch_size=128,
    eval_problems=100,
    final_eval_problems=100,
    train_axiom="random",
    eval_axioms="both",
    regime="bimodal",
)

if __name__ == "__main__":
    run_experiment(VARIANT)
