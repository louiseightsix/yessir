"""Unit tests for Tier 7: Axiom-Free Recursive Key-Value Addressing (Zero-Preamble Induction)."""

import pytest
import torch

from transformadder.core.constants import (
    BOS,
    DELIM_A,
    DELIM_AXIOM,
    DELIM_B,
    DELIM_CARRY,
    DELIM_GENERIC,
    DELIM_ORIGIN,
    DELIM_OUT,
    TOKEN_C,
    TOKEN_FLAG,
    TOKEN_S,
)
from transformadder.core.data import (
    ProceduralArithmeticGenerator,
    collate_labelled,
    encode_sm_fork,
    encode_sm_seq_rev,
)
from transformadder.core.embedding import SubspaceEmbedding
from transformadder.core.model import TransformAdder
from transformadder.core.sequences import build_prefix, build_sm_prompt, int_to_axiom_address
from transformadder.harnesses.state_machine import (
    run_state_machine_fork,
    run_state_machine_seq_rev,
)
from transformadder.tests.test_harnesses import decode
from transformadder.utils.runner import Variant, compute_loss


def test_tier7_prefix_and_sm_prompt_no_axiom_header():
    axiom = list(b"0123456789")
    a = [axiom[3], axiom[7]]
    b = [axiom[5], axiom[2]]

    # 1. Prefix: Must have NO DELIM_AXIOM and NO axiom block
    prefix = build_prefix(axiom, a, b, tier=7)
    assert prefix.roles is None
    assert prefix.sigs is None
    assert DELIM_AXIOM not in prefix.tokens

    expected_prefix = (
        [BOS, DELIM_A]
        + [DELIM_GENERIC, axiom[0], DELIM_ORIGIN, axiom[3]]  # col 0: 3
        + [DELIM_GENERIC, axiom[1], DELIM_ORIGIN, axiom[7]]  # col 1: 7
        + [DELIM_B]
        + [DELIM_GENERIC, axiom[0], DELIM_ORIGIN, axiom[5]]  # col 0: 5
        + [DELIM_GENERIC, axiom[1], DELIM_ORIGIN, axiom[2]]  # col 1: 2
    )
    assert prefix.tokens == expected_prefix

    # 2. SM Prompt: builds on prefix with KV carry, out, and query entries
    carry = axiom[1]
    out_digits = [axiom[8]]
    prompt = build_sm_prompt(axiom, a, b, carry, out_digits, tier=7, query_token=TOKEN_S)
    assert DELIM_AXIOM not in prompt.tokens

    addr_1 = [axiom[1]]
    addr_0 = [axiom[0]]
    expected_tail = (
        [DELIM_CARRY, DELIM_GENERIC]
        + addr_1
        + [DELIM_ORIGIN, carry]
        + [DELIM_OUT, DELIM_GENERIC]
        + addr_0
        + [DELIM_ORIGIN, axiom[8]]
        + [DELIM_GENERIC]
        + addr_1
        + [DELIM_ORIGIN, TOKEN_S]
    )
    assert prompt.tokens[len(expected_prefix) :] == expected_tail


def test_tier7_generator_option_a_full_span():
    # Enforces min_l = max(min_len, base) so all digits 0..B-1 appear as address keys
    gen = ProceduralArithmeticGenerator(bases=[2, 4, 8, 10], min_len=1, max_len=20, tier=7, seed=42)
    for _ in range(50):
        inst = gen.sample_instance()
        base = inst["base"]
        assert len(inst["a"]) >= base
        assert len(inst["b"]) >= base


def test_tier7_oracle_decodes_axiom_from_address_stream():
    axiom = list(b"0123456789")
    a = [axiom[i % len(axiom)] for i in range(12)]  # 12 digits, reaches rollover at col 10
    b = [axiom[(i + 3) % len(axiom)] for i in range(12)]

    prompt = build_sm_prompt(axiom, a, b, carry=axiom[0], out_digits=[axiom[3]], tier=7, query_token=TOKEN_S)
    recovered_axiom, rec_a, rec_b, rest = decode(prompt)

    assert recovered_axiom == axiom
    assert rec_a == a
    assert rec_b == b


def test_tier7_subspace_embedding():
    emb = SubspaceEmbedding(tier=7)
    assert emb.role_embed is None
    assert not hasattr(emb, "inv_freq")

    token_ids = torch.randint(0, 256, (2, 32))
    out = emb(token_ids)
    assert out.shape == (2, 32, 512)
    assert torch.equal(out[:, :, 256:], torch.zeros_like(out[:, :, 256:]))


def test_tier7_data_and_forward_fork():
    gen = ProceduralArithmeticGenerator(bases=[2, 10], min_len=2, max_len=12, tier=7, seed=123)
    inst = gen.sample_instance()

    sample = encode_sm_fork(inst, k=0, tier=7, query_token=TOKEN_S)
    batch = collate_labelled([sample])

    model = TransformAdder(tier=7, n_layers=2, head_type="single_head")
    variant = Variant(
        name="test_sm_fork_t7",
        paradigm="sm_fork",
        tier=7,
        log_dir="/tmp/test_t7",
        n_layers=2,
    )

    loss, parts = compute_loss(model, variant, batch)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_tier7_harness_rollout():
    gen = ProceduralArithmeticGenerator(bases=[2, 4], min_len=2, max_len=6, tier=7, seed=99)
    inst = gen.sample_instance()

    model = TransformAdder(tier=7, n_layers=2, head_type="single_head")

    out_digits, status = run_state_machine_fork(model, inst["axiom"], inst["a"], inst["b"], tier_format=7)
    assert status in ("finished", "malformed", "max_steps_exceeded")

    out_digits_rev, status_rev = run_state_machine_seq_rev(model, inst["axiom"], inst["a"], inst["b"], tier_format=7)
    assert status_rev in ("finished", "malformed", "max_steps_exceeded")
