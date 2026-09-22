"""End-to-end harness tests against a scripted oracle "model".

The oracle decodes the harness's own prompts (all six tiers) and answers with the reference adder, so a
passing test proves that prompt layout, harness control flow, halting, trace parsing, and scoring
are mutually consistent; an adversarial variant proves that failures are reported, not hidden.
"""

import torch

from transformadder.core.constants import (
    DELIM_A,
    DELIM_AXIOM,
    DELIM_B,
    DELIM_CARRY,
    DELIM_GENERIC,
    DELIM_ORIGIN,
    DELIM_OUT,
    FLAG_CLASS_CONTINUE,
    FLAG_CLASS_FINISHED,
    FLAG_CONTINUE,
    FLAG_FINISHED,
    ROLE_AXIOM,
    ROLE_CARRY,
    ROLE_FLAG,
    ROLE_OPERAND_A,
    ROLE_OPERAND_B,
    ROLE_OUT,
    START,
    TOKEN_C,
    TOKEN_FLAG,
    TOKEN_S,
)
from transformadder.core.data import ProceduralArithmeticGenerator
from transformadder.core.reference import reference_add, reference_step, reference_trace
from transformadder.core.sequences import int_to_axiom_address
from transformadder.harnesses.autoregressive import run_autoregressive_trace
from transformadder.harnesses.state_machine import (
    run_state_machine_fork,
    run_state_machine_mh,
    run_state_machine_seq,
    run_state_machine_seq_rev,
)
from transformadder.utils.evaluation import free_running_metrics


def one_hot(index: int, n: int = 256) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), n).float()


def split(tokens, *delims):
    """Splits a token list at each delimiter in order; returns the blocks after each."""
    blocks, rest = [], tokens
    for d in delims:
        i = rest.index(d)
        blocks.append(rest[:i])
        rest = rest[i + 1:]
    return blocks[1:] + [rest]


def tier6_entries(block, axiom):
    """Values of the `| addr [ORIGIN] v` entries of a Tier 6 block, checking that entry j is
    addressed by the little-endian axiom numeral of j."""
    values, i = [], 0
    while i < len(block):
        assert block[i] == DELIM_GENERIC
        j = block.index(DELIM_ORIGIN, i)
        assert block[i + 1:j] == int_to_axiom_address(len(values), axiom)
        values.append(block[j + 1])
        i = j + 2
    return values


def decode(seq):
    """Recovers (axiom, a, b, rest) from a harness-built sequence of any tier, where `rest` is
    everything after operand b with block delimiters removed: [c_k] OUT_k [S] transients (SM)
    or [START] trace (AR)."""
    tokens = seq.tokens
    if seq.roles is not None and DELIM_GENERIC not in tokens:    # Tier 1: blocks given by role
        assert len(seq.roles) == len(seq.sigs) == len(tokens)
        axiom, a, b = ([t for t, r in zip(tokens, seq.roles) if r == role]
                       for role in (ROLE_AXIOM, ROLE_OPERAND_A, ROLE_OPERAND_B))
        return axiom, a, b, tokens[1 + len(axiom) + len(a) + len(b):]
    if DELIM_ORIGIN in tokens and DELIM_GENERIC in tokens:       # Tier 6 & 7: addressed entries
        if DELIM_AXIOM in tokens:
            axiom, a_block, rest = split(tokens, DELIM_AXIOM, DELIM_A, DELIM_B)
        else:
            a_block, rest = split(tokens, DELIM_A, DELIM_B)
            axiom = []
            i = 0
            while i < len(a_block):
                assert a_block[i] == DELIM_GENERIC
                j = a_block.index(DELIM_ORIGIN, i)
                addr = a_block[i + 1:j]
                if len(addr) == 1 and addr[0] not in axiom:
                    axiom.append(addr[0])
                    i = j + 2
                else:
                    break
        a = tier6_entries(a_block, axiom)
        if START in rest:
            s = rest.index(START)
            return axiom, a, tier6_entries(rest[:s], axiom), rest[s:]
        c, o = rest.index(DELIM_CARRY), rest.index(DELIM_OUT)
        b = tier6_entries(rest[:c], axiom)
        out_section = rest[o + 1:]
        # OUT entries, then the query entry `| addr_k [ORIGIN] Q`, then any transient tokens
        q = next(i for i, t in enumerate(out_section) if t in (TOKEN_S, TOKEN_C, TOKEN_FLAG))
        g = max(i for i in range(q) if out_section[i] == DELIM_GENERIC)   # opens the query entry
        out = tier6_entries(out_section[:g], axiom)
        k = len(out)
        assert out_section[g + 1:q - 1] == int_to_axiom_address(k, axiom) and out_section[q - 1] == DELIM_ORIGIN
        assert rest[c + 1:o] == [DELIM_GENERIC] + int_to_axiom_address(k, axiom) + [DELIM_ORIGIN, rest[o - 1]]
        return axiom, a, b, [rest[o - 1]] + out + out_section[q:]
    if DELIM_ORIGIN in tokens:                                   # Tier 5: dual delimiters
        cleaned = [t for t in tokens if t != DELIM_ORIGIN]
        axiom, a, rest = split(cleaned, DELIM_AXIOM, DELIM_A, DELIM_B)
        carry_delim, out_delim = DELIM_CARRY, DELIM_OUT
    elif DELIM_AXIOM in tokens:                                  # Tier 2: role delimiters
        axiom, a, rest = split(tokens, DELIM_AXIOM, DELIM_A, DELIM_B)
        carry_delim, out_delim = DELIM_CARRY, DELIM_OUT
    else:                                                       # Tier 3 and Tier 4: generic delimiters
        axiom, a, rest = split([DELIM_GENERIC] + tokens[1:], DELIM_GENERIC, DELIM_GENERIC, DELIM_GENERIC)
        carry_delim = out_delim = DELIM_GENERIC
    if START in rest:
        return axiom, a, rest[:rest.index(START)], rest[rest.index(START):]
    n_b = rest.index(carry_delim)
    assert rest[n_b + 2] == out_delim
    return axiom, a, rest[:n_b], [rest[n_b + 1]] + rest[n_b + 3:]


class OracleModel:
    """Answers prompts of any tier with the reference adder. `never_finish` makes it adversarial."""

    def __init__(self, never_finish: bool = False):
        self.never_finish = never_finish

    def eval(self):
        return self

    def _sm_state(self, seq):
        axiom, a, b, rest = decode(seq)
        carry, tail = rest[0], rest[1:]
        is_flag_query = TOKEN_FLAG in tail and TOKEN_S not in tail and TOKEN_C not in tail
        is_carry_first = TOKEN_C in tail and (TOKEN_S not in tail or tail.index(TOKEN_C) < tail.index(TOKEN_S))
        if is_flag_query:
            f_at = tail.index(TOKEN_FLAG)
            out, transient = tail[:f_at], tail[f_at + 1:]
            if seq.roles is not None and seq.sigs is not None:
                f_idx = seq.tokens.index(TOKEN_FLAG)
                assert (seq.roles[f_idx], seq.sigs[f_idx]) == (ROLE_FLAG, len(out))
            mode = "flag"
        elif is_carry_first:
            c_at = tail.index(TOKEN_C)
            out, transient = tail[:c_at], tail[c_at + 1:]
            if seq.roles is not None and seq.sigs is not None:
                c_idx = seq.tokens.index(TOKEN_C)
                assert (seq.roles[c_idx], seq.sigs[c_idx]) == (ROLE_CARRY, len(out) + 1)
            mode = "carry_first"
        else:
            s_at = tail.index(TOKEN_S)
            out, transient = tail[:s_at], tail[s_at + 1:]
            if seq.roles is not None and seq.sigs is not None:
                s_idx = seq.tokens.index(TOKEN_S)
                assert (seq.roles[s_idx], seq.sigs[s_idx]) == (ROLE_OUT, len(out))
            mode = "sum_first"
        return reference_step(axiom, a, b, len(out), carry), transient, mode

    def forward_sm_mh_single(self, seq):
        step, transient, _ = self._sm_state(seq)
        assert transient == []
        finished = step["flag"] == FLAG_FINISHED and not self.never_finish
        flag_class = FLAG_CLASS_FINISHED if finished else FLAG_CLASS_CONTINUE
        return one_hot(step["o"]), one_hot(step["c_out"]), one_hot(flag_class, 2)

    def next_logits(self, seq):
        if START in seq.tokens:
            axiom, a, b, rest = decode(seq)
            emitted = rest[1:]
            trace = reference_trace(reference_add(axiom, a, b))
            if self.never_finish:
                trace = trace[:-1] + [TOKEN_S] * 8
            return one_hot(trace[len(emitted)])

        step, transient, mode = self._sm_state(seq)
        if mode == "flag":
            assert transient == []
            return one_hot(FLAG_CONTINUE if self.never_finish else step["flag"])
        if mode == "carry_first":
            if transient == []:
                return one_hot(step["c_out"])
            if transient == [step["c_out"], TOKEN_S]:
                return one_hot(step["o"])
            assert transient == [step["c_out"], TOKEN_S, step["o"], TOKEN_FLAG]
            return one_hot(FLAG_CONTINUE if self.never_finish else step["flag"])

        if transient == []:
            return one_hot(step["o"])
        if transient == [step["o"], TOKEN_C]:
            return one_hot(step["c_out"])
        assert transient == [step["o"], TOKEN_C, step["c_out"], TOKEN_FLAG]
        return one_hot(FLAG_CONTINUE if self.never_finish else step["flag"])

    def next_logits_batch(self, seqs):
        return torch.stack([self.next_logits(s) for s in seqs])


TIERS = (1, 2, 3, 4, 5, 6, 7)
HARNESSES = {
    "sm_mh": run_state_machine_mh,
    "sm_seq": run_state_machine_seq,
    "sm_seq_rev": run_state_machine_seq_rev,
    "sm_fork": run_state_machine_fork,
    "ar": run_autoregressive_trace,
}


def instances(n=200, tier=1):
    gen = ProceduralArithmeticGenerator(bases=[2, 3, 10, 12], max_len=12, p_chain=0.5, seed=5, tier=tier)
    return [gen.sample_instance() for _ in range(n)]


def test_oracle_rollouts_are_exact():
    for paradigm, run in HARNESSES.items():
        for tier in TIERS:
            for inst in instances(tier=tier):
                carries = []
                out, status = run(OracleModel(), inst["axiom"], inst["a"], inst["b"], tier, carries=carries)
                assert status == "finished", (paradigm, tier)
                assert out == inst["expected_out"], (paradigm, tier)
                assert carries == [s["c_out"] for s in inst["steps"]], (paradigm, tier)


def test_oracle_scores_perfectly():
    for paradigm in HARNESSES:
        for tier in TIERS:
            metrics = free_running_metrics(OracleModel(), paradigm, tier, instances(50, tier=tier))
            assert metrics["em"] == 1.0 and metrics["digit_acc"] == 1.0 and metrics["carry_acc"] == 1.0
            assert metrics["out_of_axiom"] == 0


def test_non_halting_model_is_bounded_and_scored_as_failure():
    expected_status = {
        "sm_mh": "max_steps_exceeded",
        "sm_seq": "max_steps_exceeded",
        "sm_seq_rev": "max_steps_exceeded",
        "sm_fork": "max_steps_exceeded",
    }
    for paradigm, run in HARNESSES.items():
        for tier in TIERS:
            inst = instances(1, tier=tier)[0]
            out, status = run(OracleModel(never_finish=True), inst["axiom"], inst["a"], inst["b"], tier)
            assert status != "finished"
            if paradigm in expected_status:
                assert status == expected_status[paradigm]
                assert len(out) == max(len(inst["a"]), len(inst["b"])) + 1
            metrics = free_running_metrics(OracleModel(never_finish=True), paradigm, tier, [inst])
            assert metrics["em"] == 0.0


def test_out_of_axiom_counts_carries_as_well_as_sum_digits():
    class ForeignCarryOracle(OracleModel):
        """Correct sum digits and flags, but every carry is a byte outside the axiom."""

        def forward_sm_mh_single(self, seq):
            logits_sum, _, logits_flag = super().forward_sm_mh_single(seq)
            return logits_sum, one_hot(self.foreign), logits_flag

    # One column, no carry: the rollout halts at step 0, so the foreign carry is never fed back
    axiom = list(range(100, 110))
    a, b = [axiom[1]], [axiom[2]]
    steps = reference_add(axiom, a, b)
    inst = {"base": 10, "axiom": axiom, "a": a, "b": b, "steps": steps, "expected_out": [axiom[3]]}
    model = ForeignCarryOracle()
    model.foreign = 7
    metrics = free_running_metrics(model, "sm_mh", 2, [inst])
    assert metrics["em"] == 1.0          # sum digits and halting are correct
    assert metrics["out_of_axiom"] == 1  # ... but the emitted carry lies outside the axiom


def test_metrics_log_is_standard_json(tmp_path):
    import json

    from transformadder.utils.logging import MetricsLogger

    path = tmp_path / "metrics.jsonl"
    MetricsLogger(path).log(step=1, metrics={"carry_acc": float("nan"), "em": 0.5})
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    record = json.loads(text)
    assert record["carry_acc"] is None and record["em"] == 0.5
