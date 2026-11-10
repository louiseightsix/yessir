"""Deterministic procedural arithmetic generator, sample encoders, and batch builders.
Data generation is a pure function of the seed, with zero disk I/O.

The generator is the oracle side of the experiment and is free to perform arithmetic; all ground
truth comes from core/reference.py.
"""

import random

import torch

from prototypes.transformadder.core.constants import (
    CANONICAL_AXIOM,
    FLAG_CLASS_CONTINUE,
    FLAG_CLASS_FINISHED,
    FLAG_FINISHED,
    IGNORE_LABEL,
    SIGMA_POOL,
    TOKEN_C,
    TOKEN_FLAG,
    TOKEN_PROBE,
    TOKEN_S,
)
from prototypes.transformadder.core.reference import reference_add, reference_out, reference_trace
from prototypes.transformadder.core.rule import AdditionRule, Rule, get_canonical_alphabet, sample_alphabet
from prototypes.transformadder.core.sequences import (
    Seq,
    build_ar_prompt,
    build_probe_prompt,
    build_sm_prompt,
    carry_coords,
    collate,
    flag_coords,
    sum_coords,
    trace_coords,
)


# ---------------------------------------------------------------------------
# Sample encoders (teacher-forced from the oracle state)
# ---------------------------------------------------------------------------

def encode_sm_mh(inst: dict, k: int, tier: int) -> tuple[Seq, tuple[int, int, int]]:
    """SM-MH sample for column k: base prompt and targets (o_k, c_{k+1}, flag class index)."""
    step = inst["steps"][k]
    prompt = build_sm_prompt(
        inst["axiom"], inst["a"], inst["b"], step["c_in"], inst["expected_out"][:k], tier
    )
    flag_class = FLAG_CLASS_FINISHED if step["flag"] == FLAG_FINISHED else FLAG_CLASS_CONTINUE
    return prompt, (step["o"], step["c_out"], flag_class)


def encode_sm_seq(inst: dict, k: int, tier: int) -> tuple[Seq, list[int]]:
    """SM-Seq sample for column k: input [BOS] A a b c_k OUT_k [S] o_k [C] c_{k+1} [F] and
    next-token labels aligned to the input positions.

    Exactly three positions carry loss: o_k (predicted at [S]), c_{k+1} (at [C]) and FLAG_k
    (at [F]). The triggers [C] and [F] are harness-inserted, so they are never targets.
    """
    step = inst["steps"][k]
    prompt = build_sm_prompt(
        inst["axiom"], inst["a"], inst["b"], step["c_in"], inst["expected_out"][:k], tier
    )
    seq = (
        prompt.push(step["o"], *sum_coords(k))
        .push(TOKEN_C, *carry_coords(k))
        .push(step["c_out"], *carry_coords(k))
        .push(TOKEN_FLAG, *flag_coords(k))
    )
    labels = [IGNORE_LABEL] * len(seq)
    labels[len(prompt) - 1] = step["o"]      # at [S]
    labels[len(prompt) + 1] = step["c_out"]  # at [C]
    labels[len(prompt) + 3] = step["flag"]   # at [F]
    return seq, labels


def encode_sm_seq_rev(inst: dict, k: int, tier: int) -> tuple[Seq, list[int]]:
    """SM-SeqRev sample for column k: input [BOS] A a b c_k OUT_k [C] c_{k+1} [S] o_k [F] and
    next-token labels aligned to the input positions.

    Exactly three positions carry loss: c_{k+1} (predicted at [C]), o_k (at [S]) and FLAG_k
    (at [F]). The triggers [S] and [F] are harness-inserted, so they are never targets.
    """
    step = inst["steps"][k]
    prompt = build_sm_prompt(
        inst["axiom"], inst["a"], inst["b"], step["c_in"], inst["expected_out"][:k], tier, query_token=TOKEN_C
    )
    seq = (
        prompt.push(step["c_out"], *carry_coords(k))
        .push(TOKEN_S, *sum_coords(k))
        .push(step["o"], *sum_coords(k))
        .push(TOKEN_FLAG, *flag_coords(k))
    )
    labels = [IGNORE_LABEL] * len(seq)
    labels[len(prompt) - 1] = step["c_out"]  # at [C]
    labels[len(prompt) + 1] = step["o"]      # at [S]
    labels[len(prompt) + 3] = step["flag"]   # at [F]
    return seq, labels


def encode_sm_fork(inst: dict, k: int, tier: int, query_token: int) -> tuple[Seq, list[int]]:
    """SM-Fork sample for column k: input [BOS] A a b c_k OUT_k [Q] and exactly 1 next-token
    label aligned to the query position len(prompt) - 1. Q in ([S], [C], [F]).
    """
    step = inst["steps"][k]
    prompt = build_sm_prompt(
        inst["axiom"], inst["a"], inst["b"], step["c_in"], inst["expected_out"][:k], tier, query_token=query_token
    )
    labels = [IGNORE_LABEL] * len(prompt)
    if query_token == TOKEN_S:
        target = step["o"]
    elif query_token == TOKEN_C:
        target = step["c_out"]
    elif query_token == TOKEN_FLAG:
        target = step["flag"]
    else:
        raise ValueError(f"Unknown query_token: {query_token}")
    labels[len(prompt) - 1] = target
    return prompt, labels


def encode_ar(inst: dict, tier: int, order: str = "sum_first") -> tuple[Seq, list[int]]:
    """AR sample: input [BOS] A a b [START] trace[:-1] and next-token labels aligned to the
    input positions. Every trace token carries loss, including delimiters/triggers and [FINISHED];
    [FINISHED] is terminal and is never an input.
    """
    prompt = build_ar_prompt(inst["axiom"], inst["a"], inst["b"], tier)
    trace = reference_trace(inst["steps"], order=order)
    seq = prompt
    for i, token in enumerate(trace[:-1]):
        seq = seq.push(token, *trace_coords(i, order=order))
    labels = [IGNORE_LABEL] * (len(prompt) - 1) + trace
    assert len(labels) == len(seq)
    return seq, labels


def encode_probe(inst: dict, tier: int, query_symbol: int | None = None) -> tuple[Seq, list[int]]:
    """Single-token successor binding probe sample:
    input: [BOS] A a b [PROBE] s_i
    labels: [IGNORE_LABEL] * (len(prompt) - 1) + [s_{i+1}]
    """
    axiom = inst["axiom"]
    if query_symbol is None:
        query_symbol = random.choice(axiom)
    idx = axiom.index(query_symbol)
    target = axiom[(idx + 1) % len(axiom)]
    prompt = build_probe_prompt(axiom, inst["a"], inst["b"], query_symbol, tier)
    labels = [IGNORE_LABEL] * len(prompt)
    labels[len(prompt) - 1] = target
    return prompt, labels


def collate_labels(label_lists: list[list[int]], T: int) -> torch.Tensor:
    labels = torch.full((len(label_lists), T), IGNORE_LABEL, dtype=torch.long)
    for i, row in enumerate(label_lists):
        labels[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return labels


def collate_sm_mh(samples: list[tuple[Seq, tuple[int, int, int]]]) -> dict:
    batch = collate([seq for seq, _ in samples])
    targets = torch.tensor([t for _, t in samples], dtype=torch.long)
    batch["target_sum"], batch["target_carry"], batch["target_flag"] = targets.unbind(dim=1)
    return batch


def collate_labelled(samples: list[tuple[Seq, list[int]]]) -> dict:
    batch = collate([seq for seq, _ in samples])
    batch["labels"] = collate_labels([l for _, l in samples], batch["token_ids"].size(1))
    return batch


# ---------------------------------------------------------------------------
# Procedural generator
# ---------------------------------------------------------------------------

class ProceduralArithmeticGenerator:
    """Deterministic seeded procedural generator for base-conditioned arithmetic.

    `bases` has no default, to prevent silently training the wrong regime ([2, 10] bimodal or
    [2, 4, 8, 10] multimodal).
    """

    def __init__(
        self,
        bases: list[int],
        min_len: int = 1,
        max_len: int = 20,
        p_chain: float = 0.25,
        p_terminal: float = 0.2,
        canonical: bool = False,
        p_prior: float = 0.0,
        p_probe: float = 0.0,
        alphabet_condition: str | None = None,
        seed: int = 42,
        tier: int = 1,      # Tier 7 forces operand lengths >= base (see sample_instance)
        trace_order: str = "sum_first",
        rule: Rule | None = None,
    ):
        if canonical and max(bases) > len(CANONICAL_AXIOM):
            raise ValueError(f"Canonical axiom supports bases up to {len(CANONICAL_AXIOM)}.")
        self.bases = list(bases)
        self.min_len = min_len
        self.max_len = max_len
        self.p_chain = p_chain
        self.p_terminal = p_terminal
        self.canonical = canonical
        self.p_prior = p_prior
        self.p_probe = p_probe
        self.alphabet_condition = alphabet_condition
        self.seed = seed
        self.tier = tier
        self.trace_order = trace_order
        self.rule = rule if rule is not None else AdditionRule()
        self.rng = random.Random(seed)
        self.sigma_pool = list(SIGMA_POOL)

    def sample_instance(self, alphabet_condition: str | None = None) -> dict:
        """Sample a single problem instance with its ground-truth transitions.

        Operand digits are uniform over the axiom; most-significant zeros are permitted.
        """
        base = self.rng.choice(self.bases)
        if alphabet_condition is not None:
            condition = alphabet_condition
        elif self.alphabet_condition is not None:
            condition = self.alphabet_condition
        elif self.canonical:
            condition = "canonical_order"
        else:
            if self.p_prior > 0.0 and self.rng.random() < self.p_prior:
                condition = "canonical_order"
            else:
                condition = "symbols_random"

        return self.rule.sample_instance(
            rng=self.rng,
            base=base,
            condition=condition,
            min_len=self.min_len,
            max_len=self.max_len,
            p_chain=self.p_chain,
            p_terminal=self.p_terminal,
            tier=self.tier,
        )

    def sample_sm_step(self) -> tuple[dict, int]:
        """One column transition of a fresh instance: the terminal step k = K with probability
        p_terminal (counteracting the CONTINUE / FINISHED imbalance), otherwise uniform."""
        inst = self.sample_instance()
        K = len(inst["steps"]) - 1
        k = K if self.rng.random() < self.p_terminal else self.rng.randint(0, K)
        return inst, k

    def probe_batch(self, batch_size: int, tier: int) -> dict:
        """Batch of pure auxiliary binding probe queries ([PROBE] s_i -> s_{i+1})."""
        items = []
        for _ in range(batch_size):
            inst = self.sample_instance()
            s_i = self.rng.choice(inst["axiom"])
            items.append(encode_probe(inst, tier, s_i))
        return collate_labelled(items)

    # Training batches (CPU tensors; the runner moves them to the training device)
    def sm_mh_batch(self, batch_size: int, tier: int) -> dict:
        return collate_sm_mh([encode_sm_mh(*self.sample_sm_step(), tier) for _ in range(batch_size)])

    def sm_seq_batch(self, batch_size: int, tier: int) -> dict:
        samples = []
        for _ in range(batch_size):
            if self.p_probe > 0.0 and self.rng.random() < self.p_probe:
                inst = self.sample_instance()
                s_i = self.rng.choice(inst["axiom"])
                samples.append(encode_probe(inst, tier, s_i))
            else:
                samples.append(encode_sm_seq(*self.sample_sm_step(), tier))
        return collate_labelled(samples)

    def sm_seq_rev_batch(self, batch_size: int, tier: int) -> dict:
        samples = []
        for _ in range(batch_size):
            if self.p_probe > 0.0 and self.rng.random() < self.p_probe:
                inst = self.sample_instance()
                s_i = self.rng.choice(inst["axiom"])
                samples.append(encode_probe(inst, tier, s_i))
            else:
                samples.append(encode_sm_seq_rev(*self.sample_sm_step(), tier))
        return collate_labelled(samples)

    def sm_fork_batch(self, batch_size: int, tier: int) -> dict:
        items = []
        for _ in range(batch_size):
            if self.p_probe > 0.0 and self.rng.random() < self.p_probe:
                inst = self.sample_instance()
                s_i = self.rng.choice(inst["axiom"])
                items.append(encode_probe(inst, tier, s_i))
            else:
                inst, k = self.sample_sm_step()
                items.append(encode_sm_fork(inst, k, tier, TOKEN_S))
                items.append(encode_sm_fork(inst, k, tier, TOKEN_C))
                items.append(encode_sm_fork(inst, k, tier, TOKEN_FLAG))
        return collate_labelled(items)

    def ar_batch(self, batch_size: int, tier: int, order: str | None = None) -> dict:
        trace_order = order if order is not None else self.trace_order
        samples = []
        for _ in range(batch_size):
            if self.p_probe > 0.0 and self.rng.random() < self.p_probe:
                inst = self.sample_instance()
                s_i = self.rng.choice(inst["axiom"])
                samples.append(encode_probe(inst, tier, s_i))
            else:
                samples.append(encode_ar(self.sample_instance(), tier, order=trace_order))
        return collate_labelled(samples)


ProceduralRuleGenerator = ProceduralArithmeticGenerator
