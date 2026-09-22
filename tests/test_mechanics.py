"""Tests for the reference adder, procedural data generation, prompt layouts, sample encoders,
the trace parser, and agreement between code constants and the spec vocabulary table."""

import re
from pathlib import Path

import pytest

from transformadder.core import constants as C
from transformadder.core.constants import (
    BOS,
    DELIM_A,
    DELIM_AXIOM,
    DELIM_B,
    DELIM_CARRY,
    DELIM_GENERIC,
    DELIM_OUT,
    FLAG_CLASS_CONTINUE,
    FLAG_CLASS_FINISHED,
    FLAG_CONTINUE,
    FLAG_FINISHED,
    IGNORE_LABEL,
    ROLE_AXIOM,
    ROLE_CARRY,
    ROLE_CTRL,
    ROLE_FLAG,
    ROLE_OPERAND_A,
    ROLE_OPERAND_B,
    ROLE_OUT,
    ROLE_START,
    SIGMA_POOL,
    START,
    TOKEN_C,
    TOKEN_FLAG,
    TOKEN_S,
)
from transformadder.core.data import (
    ProceduralArithmeticGenerator,
    encode_ar,
    encode_sm_fork,
    encode_sm_mh,
    encode_sm_seq,
    encode_sm_seq_rev,
)
from transformadder.core.reference import reference_add, reference_out, reference_trace
from transformadder.core.sequences import build_ar_prompt, build_sm_prompt, trace_coords
from transformadder.harnesses.autoregressive import parse_trace

SPEC_DIR = Path(__file__).resolve().parents[1] / "spec"


def _value(axiom, digits):
    val = {s: i for i, s in enumerate(axiom)}
    return sum(val[d] * len(axiom) ** i for i, d in enumerate(digits))


def test_constants_match_spec_vocabulary_table():
    text = (SPEC_DIR / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `[^`]+` \| `(\w+)` \| `(0x[0-9A-F]{2})` \|", text, flags=re.M)
    assert len(rows) == 15, "expected 15 named control tokens in spec/README.md Section 2"
    for name, byte in rows:
        assert getattr(C, name) == int(byte, 16), f"{name} disagrees with the spec"


def test_reference_base2():
    axiom = [48, 49]
    assert reference_out(reference_add(axiom, [49], [49])) == [48, 49]
    assert reference_out(reference_add(axiom, [49, 49], [49])) == [48, 48, 49]


def test_reference_base3_spec_trace():
    # a = [1, 2] is 21_3 = 7; b = [2]; 7 + 2 = 9 = 100_3 -> LSD [0, 0, 1]
    steps = reference_add([48, 49, 50], [49, 50], [50])
    assert reference_out(steps) == [48, 48, 49]
    assert reference_trace(steps) == [
        TOKEN_S, 48, TOKEN_C, 49, TOKEN_S, 48, TOKEN_C, 49, TOKEN_S, 49, TOKEN_C, 48, FLAG_FINISHED,
    ]


def test_reference_halting_invariant():
    axiom = [ord(str(i)) for i in range(10)]
    # No final carry: exactly max(|a|, |b|) steps, no dummy zero step
    steps = reference_add(axiom, [ord("1"), ord("2")], [ord("3")])
    assert [s["flag"] for s in steps] == [FLAG_CONTINUE, FLAG_FINISHED]
    # Final carry: one extra column emitting s_1
    steps = reference_add(axiom, [ord("9"), ord("9")], [ord("1")])
    assert reference_out(steps) == [ord("0"), ord("0"), ord("1")]
    assert [s["flag"] for s in steps] == [FLAG_CONTINUE, FLAG_CONTINUE, FLAG_FINISHED]
    assert steps[-1]["c_out"] == axiom[0]


def test_reference_matches_integer_arithmetic():
    gen = ProceduralArithmeticGenerator(bases=[2, 3, 10, 12], p_chain=0.5, seed=7)
    for _ in range(2000):
        inst = gen.sample_instance()
        axiom, a, b, steps = inst["axiom"], inst["a"], inst["b"], inst["steps"]
        assert _value(axiom, inst["expected_out"]) == _value(axiom, a) + _value(axiom, b)
        n = max(len(a), len(b))
        assert len(steps) in (n, n + 1)
        assert steps[-1]["flag"] == FLAG_FINISHED
        assert all(s["flag"] == FLAG_CONTINUE for s in steps[:-1])
        assert all(s["c_out"] in axiom[:2] for s in steps)
        assert len(reference_trace(steps)) <= 4 * (n + 1) + 1


def test_carry_chain_enrichment():
    def longest_chain(gen):
        best = 0
        for _ in range(300):
            run = 0
            inst = gen.sample_instance()
            for s in inst["steps"]:
                run = run + 1 if s["c_in"] != inst["axiom"][0] else 0
                best = max(best, run)
        return best

    plain = longest_chain(ProceduralArithmeticGenerator(bases=[10], p_chain=0.0, seed=1))
    enriched = longest_chain(ProceduralArithmeticGenerator(bases=[10], p_chain=1.0, seed=1))
    assert enriched > plain


def test_procedural_generator_determinism():
    gen1 = ProceduralArithmeticGenerator(bases=[2, 4, 8, 10], seed=12345)
    gen2 = ProceduralArithmeticGenerator(bases=[2, 4, 8, 10], seed=12345)
    gen3 = ProceduralArithmeticGenerator(bases=[2, 4, 8, 10], seed=99999)

    p1 = [gen1.sample_instance() for _ in range(10)]
    p2 = [gen2.sample_instance() for _ in range(10)]
    p3 = [gen3.sample_instance() for _ in range(10)]

    assert p1 == p2, "Identical seeds must produce identical instances"
    assert p1 != p3, "Different seeds must produce different instances"


def test_procedural_generator_numeral_pool_integrity():
    gen = ProceduralArithmeticGenerator(bases=[2, 4, 8, 10], seed=42)
    for _ in range(50):
        inst = gen.sample_instance()
        axiom = inst["axiom"]
        assert len(set(axiom)) == inst["base"]
        assert all(s in SIGMA_POOL for s in axiom)
        assert all(s in axiom for s in inst["a"] + inst["b"])


def test_canonical_axiom():
    inst = ProceduralArithmeticGenerator(bases=[12], canonical=True, seed=0).sample_instance()
    assert bytes(inst["axiom"]) == b"0123456789AB"


def test_sm_prompt_tier1_coordinates():
    axiom, a, b = [10, 11, 12], [11, 12], [12]
    seq = build_sm_prompt(axiom, a, b, carry=11, out_digits=[10], tier=1)  # Step k = 1
    assert seq.tokens == [BOS, 10, 11, 12, 11, 12, 12, 11, 10, TOKEN_S]
    assert seq.roles == [ROLE_CTRL] + [ROLE_AXIOM] * 3 + [ROLE_OPERAND_A] * 2 + [ROLE_OPERAND_B] + [
        ROLE_CARRY, ROLE_OUT, ROLE_OUT]
    #                  BOS  axiom     a      b   c_1 o_0 [S]
    assert seq.sigs == [0, 0, 1, 2, 0, 1, 0, 1, 0, 1]

    step0 = build_sm_prompt(axiom, a, b, carry=10, out_digits=[], tier=1)
    assert step0.tokens[-2:] == [10, TOKEN_S]
    assert (step0.roles[-2], step0.sigs[-2]) == (ROLE_CARRY, 0)
    assert (step0.roles[-1], step0.sigs[-1]) == (ROLE_OUT, 0)


def test_sm_prompt_tier2_and_tier3():
    axiom, a, b = [10, 11], [11], [10]
    t2 = build_sm_prompt(axiom, a, b, carry=10, out_digits=[11], tier=2)
    assert t2.tokens == [BOS, DELIM_AXIOM, 10, 11, DELIM_A, 11, DELIM_B, 10,
                         DELIM_CARRY, 10, DELIM_OUT, 11, TOKEN_S]
    assert t2.roles is None and t2.sigs is None

    t3_step0 = build_sm_prompt(axiom, a, b, carry=10, out_digits=[], tier=3)
    G = DELIM_GENERIC
    assert t3_step0.tokens == [BOS, 10, 11, G, 11, G, 10, G, 10, G, TOKEN_S]


def test_sm_prompt_tier4_and_tier5():
    axiom, a, b = [10, 11], [11], [10]
    G = DELIM_GENERIC
    O = C.DELIM_ORIGIN

    # Tier 4: Generic Delimiter + Role Subspace (NoPE coordinates)
    t4 = build_sm_prompt(axiom, a, b, carry=10, out_digits=[11], tier=4)
    assert t4.tokens == [BOS, 10, 11, G, 11, G, 10, G, 10, G, 11, TOKEN_S]
    assert t4.roles == [
        ROLE_CTRL, ROLE_AXIOM, ROLE_AXIOM, ROLE_CTRL, ROLE_OPERAND_A, ROLE_CTRL,
        ROLE_OPERAND_B, ROLE_CTRL, ROLE_CARRY, ROLE_CTRL, ROLE_OUT, ROLE_OUT,
    ]
    assert t4.sigs is None

    # Tier 5: Dual Delimiters [ROLE] [ORIGIN] (Pure NoPE)
    t5 = build_sm_prompt(axiom, a, b, carry=10, out_digits=[11], tier=5)
    assert t5.tokens == [
        BOS, DELIM_AXIOM, O, 10, 11, DELIM_A, O, 11, DELIM_B, O, 10,
        DELIM_CARRY, O, 10, DELIM_OUT, O, 11, TOKEN_S,
    ]
    assert t5.roles is None and t5.sigs is None


def test_ar_prompts():
    axiom, a, b = [10, 11], [11], [10]
    G = DELIM_GENERIC
    assert build_ar_prompt(axiom, a, b, tier=3).tokens == [BOS, 10, 11, G, 11, G, 10, START]
    assert build_ar_prompt(axiom, a, b, tier=2).tokens == [
        BOS, DELIM_AXIOM, 10, 11, DELIM_A, 11, DELIM_B, 10, START]
    t1 = build_ar_prompt(axiom, a, b, tier=1)
    assert (t1.tokens[-1], t1.roles[-1], t1.sigs[-1]) == (START, ROLE_START, 0)
    t4 = build_ar_prompt(axiom, a, b, tier=4)
    assert t4.tokens == [BOS, 10, 11, G, 11, G, 10, START]
    assert t4.roles[-2:] == [ROLE_OPERAND_B, ROLE_START] and t4.sigs is None
    O = C.DELIM_ORIGIN
    t5 = build_ar_prompt(axiom, a, b, tier=5)
    assert t5.tokens == [BOS, DELIM_AXIOM, O, 10, 11, DELIM_A, O, 11, DELIM_B, O, 10, START]
    assert t5.roles is None and t5.sigs is None


def test_trace_coords_follow_slot_rule():
    assert [trace_coords(i) for i in range(8)] == [
        (ROLE_OUT, 0), (ROLE_OUT, 0), (ROLE_CARRY, 1), (ROLE_CARRY, 1),
        (ROLE_OUT, 1), (ROLE_OUT, 1), (ROLE_CARRY, 2), (ROLE_CARRY, 2),
    ]


def _instance():
    axiom, a, b = [48, 49, 50], [49, 50], [50]
    steps = reference_add(axiom, a, b)
    return {"base": 3, "axiom": axiom, "a": a, "b": b, "steps": steps, "expected_out": reference_out(steps)}


def test_encode_sm_mh_targets():
    inst = _instance()
    prompt, target = encode_sm_mh(inst, k=1, tier=2)
    assert prompt.tokens[-4:] == [49, DELIM_OUT, 48, TOKEN_S]  # carry c_1 = '1', OUT_1 = ['0']
    assert target == (48, 49, FLAG_CLASS_CONTINUE)
    assert encode_sm_mh(inst, k=2, tier=2)[1] == (49, 48, FLAG_CLASS_FINISHED)


def test_encode_sm_seq_labels_only_model_emitted_positions():
    inst = _instance()
    seq, labels = encode_sm_seq(inst, k=2, tier=1)
    assert seq.tokens[-5:] == [TOKEN_S, 49, TOKEN_C, 48, TOKEN_FLAG]
    assert list(zip(seq.roles[-5:], seq.sigs[-5:])) == [
        (ROLE_OUT, 2), (ROLE_OUT, 2), (ROLE_CARRY, 3), (ROLE_CARRY, 3), (ROLE_FLAG, 2)]
    assert labels[-5:] == [49, IGNORE_LABEL, 48, IGNORE_LABEL, FLAG_FINISHED]
    assert all(l == IGNORE_LABEL for l in labels[:-5])

    # Tier 4: Roles preserved, sigs is None
    seq4, labels4 = encode_sm_seq(inst, k=2, tier=4)
    assert seq4.roles[-5:] == [ROLE_OUT, ROLE_OUT, ROLE_CARRY, ROLE_CARRY, ROLE_FLAG]
    assert seq4.sigs is None
    assert labels4[-5:] == [49, IGNORE_LABEL, 48, IGNORE_LABEL, FLAG_FINISHED]

    # Tier 5: Pure NoPE (roles is None, sigs is None)
    seq5, labels5 = encode_sm_seq(inst, k=2, tier=5)
    assert seq5.roles is None and seq5.sigs is None
    assert labels5[-5:] == [49, IGNORE_LABEL, 48, IGNORE_LABEL, FLAG_FINISHED]


def test_encode_sm_seq_rev_is_carry_first():
    inst = _instance()
    G = DELIM_GENERIC
    # Base prompt is the SM layout with the tail query [S] replaced by [C]
    for tier in (1, 2, 3, 4, 5, 6, 7):
        s = build_sm_prompt(inst["axiom"], inst["a"], inst["b"], 49, [48], tier)
        c = build_sm_prompt(inst["axiom"], inst["a"], inst["b"], 49, [48], tier, query_token=TOKEN_C)
        assert (s.tokens[:-1], s.tokens[-1], c.tokens[-1]) == (c.tokens[:-1], TOKEN_S, TOKEN_C)

    seq, labels = encode_sm_seq_rev(inst, k=2, tier=1)
    assert seq.tokens[-5:] == [TOKEN_C, 48, TOKEN_S, 49, TOKEN_FLAG]   # c_3 = '0' first, then o_2 = '1'
    assert seq.roles[-5:] == [ROLE_CARRY, ROLE_CARRY, ROLE_OUT, ROLE_OUT, ROLE_FLAG]
    assert seq.sigs[-5:] == [3, 3, 2, 2, 2]
    assert labels[-5:] == [48, IGNORE_LABEL, 49, IGNORE_LABEL, FLAG_FINISHED]
    assert all(l == IGNORE_LABEL for l in labels[:-5])

    seq4, labels4 = encode_sm_seq_rev(inst, k=2, tier=4)
    assert seq4.tokens[-9:-5] == [49, G, 48, 48] and seq4.sigs is None   # c_2 | o_0 o_1
    assert seq4.roles[-5:] == [ROLE_CARRY, ROLE_CARRY, ROLE_OUT, ROLE_OUT, ROLE_FLAG]
    assert labels4[-5:] == [48, IGNORE_LABEL, 49, IGNORE_LABEL, FLAG_FINISHED]

    seq5, labels5 = encode_sm_seq_rev(inst, k=0, tier=5)
    assert seq5.roles is None and seq5.tokens[-6:-4] == [C.DELIM_ORIGIN, TOKEN_C]   # empty OUT block
    assert labels5[-5:] == [49, IGNORE_LABEL, 48, IGNORE_LABEL, FLAG_CONTINUE]   # 1 + 2 = 10_3: c_1 = '1', o_0 = '0'


def test_encode_sm_fork_one_label_per_branch():
    inst = _instance()
    step = inst["steps"][2]                                       # o_2 = '1', c_3 = '0', FINISHED
    expected = {
        TOKEN_S: (step["o"], (ROLE_OUT, 2)),
        TOKEN_C: (step["c_out"], (ROLE_CARRY, 3)),
        TOKEN_FLAG: (FLAG_FINISHED, (ROLE_FLAG, 2)),
    }
    for tier in (1, 2, 3, 4, 5, 6, 7):
        base = build_sm_prompt(inst["axiom"], inst["a"], inst["b"], step["c_in"], inst["expected_out"][:2], tier)
        for query, (target, (role, sig)) in expected.items():
            seq, labels = encode_sm_fork(inst, k=2, tier=tier, query_token=query)
            assert seq.tokens == base.tokens[:-1] + [query]          # no transient tokens
            assert labels == [IGNORE_LABEL] * (len(seq) - 1) + [target]
            if tier in (1, 4):
                assert seq.roles[:-1] == base.roles[:-1] and seq.roles[-1] == role
            if tier == 1:
                assert seq.sigs[:-1] == base.sigs[:-1] and seq.sigs[-1] == sig

    with pytest.raises(ValueError):
        encode_sm_fork(inst, k=2, tier=1, query_token=START)


def test_encode_ar_labels_are_shifted_trace():
    inst = _instance()
    seq, labels = encode_ar(inst, tier=3)
    trace = reference_trace(inst["steps"])
    n_prompt = len(build_ar_prompt(inst["axiom"], inst["a"], inst["b"], 3))
    assert FLAG_FINISHED not in seq.tokens, "[FINISHED] is terminal and never an input"
    assert seq.tokens[n_prompt:] == trace[:-1]
    assert labels[: n_prompt - 1] == [IGNORE_LABEL] * (n_prompt - 1)
    assert labels[n_prompt - 1:] == trace
    # Next-token alignment: the label at position t is the input at position t + 1
    assert all(labels[t] == seq.tokens[t + 1] for t in range(n_prompt - 1, len(seq) - 1))


def test_parse_trace():
    good = [TOKEN_S, 48, TOKEN_C, 49, TOKEN_S, 49, TOKEN_C, 48, FLAG_FINISHED]
    assert parse_trace(good) == ([48, 49], [49, 48], "finished")
    assert parse_trace([FLAG_FINISHED])[2] == "malformed"
    assert parse_trace([TOKEN_S, 48, TOKEN_C, 49])[2] == "max_tokens_exceeded"
    assert parse_trace([TOKEN_S, 48, TOKEN_C, FLAG_FINISHED])[2] == "malformed"
    assert parse_trace([TOKEN_S, TOKEN_S, TOKEN_C, 48, FLAG_FINISHED])[2] == "malformed"
    assert parse_trace([TOKEN_C, 48, TOKEN_S, 48, FLAG_FINISHED])[2] == "malformed"
