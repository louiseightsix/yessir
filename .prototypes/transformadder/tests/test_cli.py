"""Tests for CLI entrypoint and Vertex AI submission generation."""

import json
from pathlib import Path
from unittest.mock import patch

from transformadder.scripts.configure_experiment import create_experiment_bundle
from transformadder.cli import build_variant_from_args, gcs_to_fuse


def test_gcs_to_fuse_translation():
    assert gcs_to_fuse("gs://my-bucket/models/exp1") == "/gcs/my-bucket/models/exp1"
    assert gcs_to_fuse("/local/path/exp1") == "/local/path/exp1"
    assert gcs_to_fuse("experiments/exp1") == "experiments/exp1"


def test_cli_tier_options(monkeypatch):
    import sys
    from transformadder.cli import parse_args
    monkeypatch.setattr(sys, "argv", ["transformadder", "--tier", "4"])
    args = parse_args()
    assert args.tier == 4

    monkeypatch.setattr(sys, "argv", ["transformadder", "--tier", "5"])
    args = parse_args()
    assert args.tier == 5

    monkeypatch.setattr(sys, "argv", ["transformadder", "--paradigm", "sm_seq_rev", "--tier", "4"])
    args = parse_args()
    assert args.paradigm == "sm_seq_rev"
    assert args.tier == 4

    monkeypatch.setattr(sys, "argv", ["transformadder", "--paradigm", "sm_fork", "--tier", "5"])
    args = parse_args()
    assert args.paradigm == "sm_fork"
    assert args.tier == 5


def test_cli_variant_builder(monkeypatch):
    class MockArgs:
        paradigm = "sm_mh"
        tier = 1
        n_layers = 6
        ffn_type = "swiglu"
        attn_type = "causal"
        regime = "multimodal"
        seed = 42
        train_steps = 1000
        warmup_steps = 100
        batch_size = 128
        lr = 3e-4
        eval_interval = 200
        eval_problems = 100
        final_eval_problems = None
        train_axiom = "random"
        eval_axioms = "both"
        log_dir = "gs://my-bucket/experiments/test-run"
        device = None
        dry_run = False
        interactive = False

    variant = build_variant_from_args(MockArgs())
    assert variant.name == "SM-MH-T1-L6-SwiGLU-Causal-multimodal-seed42"
    assert variant.log_dir == "/gcs/my-bucket/experiments/test-run"
    assert variant.regime == "multimodal"
    assert variant.final_eval_problems == 100  # Inherited from eval_problems when steps < 50k

    MockArgs.paradigm = "sm_fork"
    variant_fork = build_variant_from_args(MockArgs())
    assert variant_fork.name == "SM-Fork-T1-L6-SwiGLU-Causal-multimodal-seed42"
    assert variant_fork.grad_accum_steps == 1

    MockArgs.tier = 6
    variant_t6_fork = build_variant_from_args(MockArgs())
    assert variant_t6_fork.grad_accum_steps == 8

    MockArgs.paradigm = "sm_seq"
    variant_t6_seq = build_variant_from_args(MockArgs())
    assert variant_t6_seq.grad_accum_steps == 2

    MockArgs.grad_accum_steps = 4
    variant_t6_custom = build_variant_from_args(MockArgs())
    assert variant_t6_custom.grad_accum_steps == 4


def test_create_experiment_bundle_vertex(tmp_path, monkeypatch):
    # Route generated run bundles to tmp_path instead of .prototypes/transformadder/results/
    monkeypatch.setattr("transformadder.scripts.configure_experiment.RUNS_DIR", tmp_path)

    exp_dir, exp_name = create_experiment_bundle(
        paradigm="sm_mh",
        tier=1,
        n_layers=6,
        ffn_type="swiglu",
        attn_type="causal",
        regime="bimodal",
        seed=99,
        train_steps=10000,
        eval_interval=1000,
        batch_size=256,
        eval_problems=500,
        train_axiom="canonical",
        eval_axioms="canonical",
        target="vertex",
        gcp_project="my-project",
        gcp_region="us-central1",
        gcp_bucket="gs://my-bucket",
        gcp_accelerator="NVIDIA_L4",
    )

    assert exp_dir.exists()
    assert (exp_dir / "config.json").exists()
    assert (exp_dir / "run.py").exists()
    assert (exp_dir / "run.sh").exists()
    assert (exp_dir / "submit_vertex.sh").exists()

    with open(exp_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    assert config["target"] == "vertex"
    assert config["gcp_project"] == "my-project"
    assert config["gcp_bucket"] == "gs://my-bucket"
    assert config["gcp_accelerator"] == "NVIDIA_L4"
    assert config["train_axiom"] == "canonical"

    # The standalone runner must train the regime its name and config.json declare
    assert 'regime="bimodal"' in (exp_dir / "run.py").read_text(encoding="utf-8")

    with open(exp_dir / "submit_vertex.sh", encoding="utf-8") as f:
        sh_content = f.read()
    assert "gcloud ai custom-jobs create" in sh_content
    assert "uv build --sdist" in sh_content
    assert "buckets create" in sh_content
    assert "--python-package-uris=\"$PACKAGE_GCS_URI\"" in sh_content
    assert "--worker-pool-spec=\"$WORKER_POOL_SPEC\"" in sh_content
    assert "transformadder.cli" in sh_content
    assert "--train-axiom=canonical" in sh_content
    assert "stream-logs $JOB_RESOURCE_NAME" in sh_content
