"""Tests for Instance Telemetry, 4-way Error Taxonomy, and Prior Suppression Metrics.
Conforms to spec/05_evaluation_and_metrics.md.
"""

import json
from pathlib import Path
import pytest
import math

from prototypes.transformadder.core.constants import CANONICAL_AXIOM
from prototypes.transformadder.core.rule import get_canonical_alphabet, sample_alphabet
from prototypes.transformadder.utils.telemetry import (
    InstanceTelemetryLogger,
    calculate_distance_regressors,
    classify_first_error,
    compute_chance_adjusted_prior_intrusion,
)


def test_classify_first_error_perfect_match():
    inst = {
        "axiom": [48, 49, 50],
        "base": 3,
        "steps": [{"o": 48, "total": 0}, {"o": 49, "total": 1}],
        "expected_out": [48, 49],
    }
    res = classify_first_error(inst, out=[48, 49], status="finished")
    assert res["first_error_type"] is None
    assert res["first_error_step_k"] is None


def test_classify_first_error_prior_intrusion_under_permuted():
    # Canonical order for base 10 is '0'..'9' (ASCII 48..57)
    # Permuted axiom swaps '0' (48) and '1' (49)
    perm_axiom = [49, 48, 50, 51, 52, 53, 54, 55, 56, 57]
    inst = {
        "axiom": perm_axiom,
        "base": 10,
        "alphabet_condition": "canonical_permuted",
        # Step has total = 0.
        # Under active axiom, 0 is represented by perm_axiom[0] = 49 ('1').
        # But under canonical order, 0 is represented by CANONICAL_AXIOM[0] = 48 ('0').
        "steps": [{"o": 49, "total": 0}],
        "expected_out": [49],
    }

    # Model emits 48 ('0') -> Prior intrusion!
    res = classify_first_error(inst, out=[48], status="finished")
    assert res["first_error_type"] == "prior_intrusion"
    assert res["first_error_step_k"] == 0
    assert res["first_predicted_token"] == 48
    assert res["first_expected_token"] == 49
    assert res["canonical_expected_token"] == 48


def test_classify_first_error_out_of_alphabet():
    inst = {
        "axiom": [48, 49],
        "base": 2,
        "alphabet_condition": "canonical_order",
        "steps": [{"o": 48, "total": 0}],
        "expected_out": [48],
    }
    # Model emits 99 (not in [48, 49])
    res = classify_first_error(inst, out=[99], status="finished")
    assert res["first_error_type"] == "out_of_alphabet"
    assert res["first_error_step_k"] == 0
    assert res["first_predicted_token"] == 99


def test_classify_first_error_procedural_error():
    inst = {
        "axiom": [48, 49, 50],
        "base": 3,
        "alphabet_condition": "symbols_random",
        "steps": [{"o": 48, "total": 0}],
        "expected_out": [48],
    }
    # Model emits 50 (in alphabet, but wrong value)
    res = classify_first_error(inst, out=[50], status="finished")
    assert res["first_error_type"] == "procedural_error"
    assert res["first_error_step_k"] == 0


def test_classify_first_error_syntax_malformation():
    inst = {
        "axiom": [48, 49],
        "base": 2,
        "steps": [{"o": 48, "total": 0}, {"o": 49, "total": 1}],
        "expected_out": [48, 49],
    }
    # Early termination: model emits only 1 token
    res = classify_first_error(inst, out=[48], status="finished")
    assert res["first_error_type"] == "syntax_malformation"
    assert res["first_error_step_k"] == 1


def test_chance_adjusted_prior_intrusion():
    # Base 10: chance = 1 / (10 - 1) = 1/9
    # Exactly at chance: R_adj should be 0.0
    r_adj_chance = compute_chance_adjusted_prior_intrusion(prior_intrusion_count=1, total_errors=9, base=10)
    assert abs(r_adj_chance - 0.0) < 1e-6

    # 100% prior intrusion: R_adj should be 1.0
    r_adj_full = compute_chance_adjusted_prior_intrusion(prior_intrusion_count=9, total_errors=9, base=10)
    assert abs(r_adj_full - 1.0) < 1e-6

    # 50% prior intrusion: (0.5 - 1/9) / (8/9) = 7/16 = 0.4375
    r_adj_half = compute_chance_adjusted_prior_intrusion(prior_intrusion_count=9, total_errors=18, base=10)
    assert abs(r_adj_half - 0.4375) < 1e-6


def test_calculate_distance_regressors():
    # Tier 1: rule_to_query_distance = prompt_len - 1 - (1 + axiom_len)
    rule_dist, entries = calculate_distance_regressors(prompt_len=100, axiom_len=10, tier=1, len_a=15, len_b=20)
    assert entries == 35
    assert rule_dist == 99 - 11 == 88

    # Tier 7: rule_to_query_distance is None
    rule_dist_t7, entries_t7 = calculate_distance_regressors(prompt_len=100, axiom_len=10, tier=7, len_a=15, len_b=20)
    assert entries_t7 == 35
    assert rule_dist_t7 is None


def test_telemetry_logger(tmp_path: Path):
    logger = InstanceTelemetryLogger(tmp_path)
    record = {
        "instance_id": "test_001",
        "outcome_em": 0,
        "per_step_accuracy": float("nan"),
        "probe_accuracy": 1.0,
    }
    logger.log_instance(record)

    lines = (tmp_path / "instances.jsonl").read_text().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["instance_id"] == "test_001"
    assert loaded["per_step_accuracy"] is None  # NaN sanitized to null
    assert loaded["probe_accuracy"] == 1.0


def test_evaluate_with_telemetry_writes_instances_jsonl(tmp_path: Path):
    """The runner always passes log_dir, so evaluate() must not crash when telemetry is on."""
    from prototypes.transformadder.core.model import TransformAdder
    from prototypes.transformadder.utils.evaluation import evaluate

    for paradigm, head in (("sm_seq", "single_head"), ("sm_mh", "multi_head"), ("ar", "single_head")):
        model = TransformAdder(tier=2, head_type=head, n_layers=1)
        log_dir = tmp_path / paradigm
        results = evaluate(model, paradigm, 2, [2, 10], num_problems=2, eval_axioms="all", log_dir=log_dir)
        assert "permuted_id_r_prior_adj" in results
        records = [json.loads(l) for l in (log_dir / "instances.jsonl").read_text().splitlines()]
        assert len(records) == 2 * 6 * 3  # problems x cells x conditions
        assert all(r["prompt_token_length"] > 0 for r in records)
