"""Unit tests for Tier 6: Recursive Axiom Key-Value Addressing (Gödel Numeral Indexing)."""

import pytest
import torch

from prototypes.transformadder.core.constants import (
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
from prototypes.transformadder.core.data import (
    ProceduralArithmeticGenerator,
    collate_labelled,
    collate_sm_mh,
    encode_sm_fork,
    encode_sm_mh,
    encode_sm_seq,
    encode_sm_seq_rev,
)
from prototypes.transformadder.core.embedding import SubspaceEmbedding
from prototypes.transformadder.core.model import TransformAdder
from prototypes.transformadder.core.sequences import build_prefix, build_sm_prompt, int_to_axiom_address
from prototypes.transformadder.harnesses.state_machine import (
    run_state_machine_fork,
    run_state_machine_mh,
    run_state_machine_seq,
    run_state_machine_seq_rev,
)
from prototypes.transformadder.utils.runner import Variant, compute_loss


def test_int_to_axiom_address_base10():
    axiom = list(b"0123456789")  # ASCII 48..57
    s0, s1, s2, s5, s9 = axiom[0], axiom[1], axiom[2], axiom[5], axiom[9]

    assert int_to_axiom_address(0, axiom) == [s0]
    assert int_to_axiom_address(1, axiom) == [s1]
    assert int_to_axiom_address(9, axiom) == [s9]
    # In little-endian, 10 is [digit_0, digit_1] = [s0, s1]
    assert int_to_axiom_address(10, axiom) == [s0, s1]
    # In little-endian, 25 is [s5, s2]
    assert int_to_axiom_address(25, axiom) == [s5, s2]


def test_int_to_axiom_address_base2():
    axiom = [ord("X"), ord("Y")]  # base 2 symbols
    sx, sy = axiom[0], axiom[1]

    assert int_to_axiom_address(0, axiom) == [sx]
    assert int_to_axiom_address(1, axiom) == [sy]
    assert int_to_axiom_address(2, axiom) == [sx, sy]  # 2 in binary: 0b10 -> LSB first [sx, sy]
    assert int_to_axiom_address(3, axiom) == [sy, sy]  # 3 in binary: 0b11 -> [sy, sy]
    assert int_to_axiom_address(4, axiom) == [sx, sx, sy]  # 4 in binary: 0b100 -> [sx, sx, sy]
    assert int_to_axiom_address(7, axiom) == [sy, sy, sy]  # 7 in binary: 0b111 -> [sy, sy, sy]


def test_tier6_prefix_and_sm_prompt():
    axiom = list(b"0123456789")
    a = [axiom[3], axiom[7]]  # 2 digits: column 0 is 3, column 1 is 7
    b = [axiom[5], axiom[2]]  # 2 digits: column 0 is 5, column 1 is 2

    # 1. Prefix
    prefix = build_prefix(axiom, a, b, tier=6)
    assert prefix.roles is None
    assert prefix.sigs is None

    expected_prefix = (
        [BOS, DELIM_AXIOM]
        + axiom
        + [DELIM_A]
        + [DELIM_GENERIC, axiom[0], DELIM_ORIGIN, axiom[3]]  # col 0: 3
        + [DELIM_GENERIC, axiom[1], DELIM_ORIGIN, axiom[7]]  # col 1: 7
        + [DELIM_B]
        + [DELIM_GENERIC, axiom[0], DELIM_ORIGIN, axiom[5]]  # col 0: 5
        + [DELIM_GENERIC, axiom[1], DELIM_ORIGIN, axiom[2]]  # col 1: 2
    )
    assert prefix.tokens == expected_prefix

    # 2. SM prompt at step k=1 with out_digits=[axiom[8]] and carry=axiom[1]
    carry = axiom[1]
    out_digits = [axiom[8]]
    prompt = build_sm_prompt(axiom, a, b, carry, out_digits, tier=6, query_token=TOKEN_S)
    assert prompt.roles is None
    assert prompt.sigs is None

    # Verify tail structure
    # Carry entry at k=1: [DELIM_CARRY, DELIM_GENERIC, addr_1, DELIM_ORIGIN, carry]
    # Out block: [DELIM_OUT, DELIM_GENERIC, addr_0, DELIM_ORIGIN, out_0]
    # Query entry at k=1: [DELIM_GENERIC, addr_1, DELIM_ORIGIN, TOKEN_S]
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


def test_tier6_subspace_embedding():
    emb = SubspaceEmbedding(tier=6)
    assert emb.role_embed is None
    assert not hasattr(emb, "inv_freq")

    token_ids = torch.randint(0, 256, (2, 32))
    out = emb(token_ids)
    assert out.shape == (2, 32, 512)

    # First 256 dims are one-hot (unit norm per token)
    one_hot_norms = torch.norm(out[:, :, :256], dim=-1)
    assert torch.allclose(one_hot_norms, torch.ones_like(one_hot_norms))

    # Remaining 256 dims must be strictly zero (NoPE)
    assert torch.equal(out[:, :, 256:], torch.zeros_like(out[:, :, 256:]))


def test_tier6_data_and_forward_fork():
    gen = ProceduralArithmeticGenerator(bases=[2, 10], min_len=2, max_len=5, seed=123)
    inst = gen.sample_instance()

    sample = encode_sm_fork(inst, k=0, tier=6, query_token=TOKEN_S)
    batch = collate_labelled([sample])

    model = TransformAdder(tier=6, n_layers=2, head_type="single_head")
    variant = Variant(
        name="test_sm_fork_t6",
        paradigm="sm_fork",
        tier=6,
        log_dir="/tmp/test_t6",
        n_layers=2,
    )

    loss, parts = compute_loss(model, variant, batch)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_tier6_harness_rollout():
    axiom = list(b"01")
    a = [axiom[1], axiom[0]]
    b = [axiom[1], axiom[1]]

    model = TransformAdder(tier=6, n_layers=2, head_type="single_head")

    out_digits, status = run_state_machine_fork(model, axiom, a, b, tier_format=6)
    assert status in ("finished", "malformed", "max_steps_exceeded")

    out_digits_rev, status_rev = run_state_machine_seq_rev(model, axiom, a, b, tier_format=6)
    assert status_rev in ("finished", "malformed", "max_steps_exceeded")
