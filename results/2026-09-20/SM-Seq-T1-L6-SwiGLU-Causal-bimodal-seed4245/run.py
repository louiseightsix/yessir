"""Standalone Experiment Runner for: SM-Seq-T1-L6-SwiGLU-Causal-bimodal-seed4245
Deterministic execution seed: 4245
Created at: 2026-09-18T19:53:46.192233
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformadder.utils.runner import Variant, run_experiment

EXP_DIR = Path(__file__).resolve().parent

VARIANT = Variant(
    name="SM-Seq-T1-L6-SwiGLU-Causal-bimodal-seed4245",
    paradigm="sm_seq",
    tier=1,
    log_dir=str(EXP_DIR),
    n_layers=6,
    ffn_type="swiglu",
    attn_type="causal",
    seed=4245,
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
