"""Standalone Experiment Runner for: SM-Fork-T6-L6-SwiGLU-Causal-multimodal-seed3256
Deterministic execution seed: 3256
Created at: 2026-09-21T17:33:53.517962
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformadder.utils.runner import Variant, run_experiment

EXP_DIR = Path(__file__).resolve().parent

VARIANT = Variant(
    name="SM-Fork-T6-L6-SwiGLU-Causal-multimodal-seed3256",
    paradigm="sm_fork",
    tier=6,
    log_dir=str(EXP_DIR),
    n_layers=6,
    ffn_type="swiglu",
    attn_type="causal",
    seed=3256,
    train_steps=100000,
    eval_interval=5000,
    batch_size=256,
    eval_problems=100,
    final_eval_problems=250,
    train_axiom="random",
    eval_axioms="random",
    regime="multimodal",
)

if __name__ == "__main__":
    run_experiment(VARIANT)
