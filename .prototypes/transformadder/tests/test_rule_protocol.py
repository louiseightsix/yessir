"""Tests for Rule Protocol, AdditionRule, Prior Dialing (p_prior), and Binding Probe (p_probe).
Conforms to spec/01_research_goal.md, spec/05_evaluation_and_metrics.md, and spec/06_implementation.md.
"""

import random
import pytest
import torch

from transformadder.core.constants import (
    CANONICAL_AXIOM,
    IGNORE_LABEL,
    SIGMA_POOL,
    TOKEN_PROBE,
)
from transformadder.core.data import (
    ProceduralArithmeticGenerator,
    encode_probe,
)
from transformadder.core.rule import (
    AdditionRule,
    Rule,
    get_canonical_alphabet,
    sample_alphabet,
)
from transformadder.core.sequences import build_probe_prompt


def test_addition_rule_implements_protocol():
    rule = AdditionRule()
    assert isinstance(rule, Rule)
    assert rule.rule_family == "addition"

    rng = random.Random(42)
    inst = rule.sample_instance(rng, base=10, condition="symbols_random", min_len=2, max_len=5)
    assert "instance_id" in inst
    assert inst["rule_family"] == "addition"
    assert inst["rule_params"]["base"] == 10
    assert len(inst["rule_params"]["axiom"]) == 10
    assert inst["rule_params"]["alphabet_condition"] == "symbols_random"
    assert "a" in inst["inputs"] and "b" in inst["inputs"]
    assert "steps" in inst
    assert "expected_out" in inst


def test_sample_alphabet_conditions():
    rng = random.Random(123)
    # Canonical order
    canon = sample_alphabet(10, "canonical_order", rng)
    assert bytes(canon) == b"0123456789"

    canon12 = sample_alphabet(12, "canonical_order", rng)
    assert bytes(canon12) == b"0123456789AB"

    # Canonical permuted: same set of glyphs, but different order
    perm = sample_alphabet(10, "canonical_permuted", rng)
    assert sorted(perm) == sorted(canon)
    assert perm != canon

    # Symbols random: drawn uniformly from SIGMA_POOL
    rnd = sample_alphabet(10, "symbols_random", rng)
    assert len(rnd) == 10
    assert all(s in SIGMA_POOL for s in rnd)


def test_prior_dialing_rates():
    # p_prior = 0.0: all instances should be symbols_random
    gen_0 = ProceduralArithmeticGenerator(bases=[10], p_prior=0.0, seed=1)
    for _ in range(50):
        inst = gen_0.sample_instance()
        assert inst["alphabet_condition"] == "symbols_random"

    # p_prior = 1.0: all instances should be canonical_order
    gen_1 = ProceduralArithmeticGenerator(bases=[10], p_prior=1.0, seed=2)
    for _ in range(50):
        inst = gen_1.sample_instance()
        assert inst["alphabet_condition"] == "canonical_order"
        assert bytes(inst["axiom"]) == b"0123456789"

    # p_prior = 0.5: mixture of canonical and random
    gen_half = ProceduralArithmeticGenerator(bases=[10], p_prior=0.5, seed=3)
    canon_count = sum(1 for _ in range(200) if gen_half.sample_instance()["alphabet_condition"] == "canonical_order")
    # Expect ~100 out of 200, within 3 sigma (sigma = sqrt(200 * 0.25) ~ 7)
    assert 70 <= canon_count <= 130


def test_probe_prompt_and_encoding_across_tiers():
    gen = ProceduralArithmeticGenerator(bases=[10], seed=42)
    inst = gen.sample_instance()
    axiom = inst["axiom"]
    s_i = axiom[3]
    s_next = axiom[4]

    for tier in range(1, 8):
        prompt = build_probe_prompt(axiom, inst["a"], inst["b"], s_i, tier)
        assert prompt.tokens[-2] == TOKEN_PROBE
        assert prompt.tokens[-1] == s_i

        encoded_seq, labels = encode_probe(inst, tier, s_i)
        assert len(encoded_seq) == len(labels)
        # Exactly the last label should be s_next, all prior should be IGNORE_LABEL
        assert labels[-1] == s_next
        assert all(l == IGNORE_LABEL for l in labels[:-1])


def test_probe_batch_generation():
    gen = ProceduralArithmeticGenerator(bases=[10], seed=42)
    for tier in range(1, 8):
        batch = gen.probe_batch(batch_size=8, tier=tier)
        assert "token_ids" in batch
        assert "labels" in batch
        assert batch["token_ids"].shape[0] == 8
        assert batch["labels"].shape[0] == 8
        # Each row in batch["labels"] must have at least one valid target
        for row in batch["labels"]:
            valid_targets = (row != IGNORE_LABEL).sum().item()
            assert valid_targets == 1


def test_probe_injection_in_sequence_batches():
    # With p_probe = 0.5, probe queries should be injected into sequence batches
    gen = ProceduralArithmeticGenerator(bases=[10], p_probe=0.5, seed=10)
    batch = gen.ar_batch(batch_size=20, tier=2)
    token_ids = batch["token_ids"]
    # Check if TOKEN_PROBE appears in some sequences
    has_probe = (token_ids == TOKEN_PROBE).any().item()
    assert has_probe, "With p_probe=0.5, some sequences in ar_batch should contain TOKEN_PROBE"
