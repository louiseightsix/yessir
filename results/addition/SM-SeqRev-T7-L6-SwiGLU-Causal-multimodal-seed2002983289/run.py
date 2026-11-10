"""Standalone Experiment Runner for: SM-SeqRev-T7-L6-SwiGLU-Causal-multimodal-seed2002983289
Deterministic execution seed: 2002983289
Created at: 2026-09-21T18:06:17.035590
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformadder.utils.runner import Variant, run_experiment

EXP_DIR = Path(__file__).resolve().parent

VARIANT = Variant(
    name="SM-SeqRev-T7-L6-SwiGLU-Causal-multimodal-seed2002983289",
    paradigm="sm_seq_rev",
    tier=7,
    log_dir=str(EXP_DIR),
    n_layers=6,
    ffn_type="swiglu",
    attn_type="causal",
    seed=2002983289,
    train_steps=100000,
    eval_interval=5000,
    batch_size=256,
    grad_accum_steps=2,
    eval_problems=100,
    final_eval_problems=250,
    train_axiom="random",
    eval_axioms="random",
    regime="multimodal",
)

if __name__ == "__main__":
    run_experiment(VARIANT)
