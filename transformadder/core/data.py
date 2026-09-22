"""Deterministic procedural arithmetic generator, sample encoders, and batch builders.
Data generation is a pure function of the seed, with zero disk I/O.

The generator is the oracle side of the experiment and is free to perform arithmetic; all ground
truth comes from core/reference.py.
"""

import random

import torch

from transformadder.core.constants import (
    CANONICAL_AXIOM,
    FLAG_CLASS_CONTINUE,
    FLAG_CLASS_FINISHED,
    FLAG_FINISHED,
    IGNORE_LABEL,
    SIGMA_POOL,
    TOKEN_C,
    TOKEN_FLAG,
    TOKEN_S,
)
from transformadder.core.reference import reference_add, reference_out, reference_trace
from transformadder.core.sequences import (
    Seq,
    build_ar_prompt,
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


def encode_ar(inst: dict, tier: int) -> tuple[Seq, list[int]]:
    """AR sample: input [BOS] A a b [START] trace[:-1] and next-token labels aligned to the
    input positions. Every trace token carries loss, including [S], [C] and [FINISHED];
    [FINISHED] is terminal and is never an input.
    """
    prompt = build_ar_prompt(inst["axiom"], inst["a"], inst["b"], tier)
    trace = reference_trace(inst["steps"])
    seq = prompt
    for i, token in enumerate(trace[:-1]):
        seq = seq.push(token, *trace_coords(i))
    labels = [IGNORE_LABEL] * (len(prompt) - 1) + trace
    assert len(labels) == len(seq)
    return seq, labels


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
        seed: int = 42,
        tier: int = 1,      # Tier 7 forces operand lengths >= base (see sample_instance)
    ):
        if canonical and max(bases) > len(CANONICAL_AXIOM):
            raise ValueError(f"Canonical axiom supports bases up to {len(CANONICAL_AXIOM)}.")
        self.bases = list(bases)
        self.min_len = min_len
        self.max_len = max_len
        self.p_chain = p_chain
        self.p_terminal = p_terminal
        self.canonical = canonical
        self.seed = seed
        self.tier = tier
        self.rng = random.Random(seed)
        self.sigma_pool = list(SIGMA_POOL)

    def sample_instance(self) -> dict:
        """Sample a single problem instance with its ground-truth transitions.

        Operand digits are uniform over the axiom; most-significant zeros are permitted.
        """
        base = self.rng.choice(self.bases)
        if self.canonical:
            axiom = CANONICAL_AXIOM[:base]
        else:
            axiom = self.rng.sample(self.sigma_pool, base)
        # Tier 7 has no axiom header: the model reads the axiom off the single-symbol addresses
        # of entries 0..B-1, so both operands must span at least B columns
        min_l = max(self.min_len, base) if self.tier == 7 else self.min_len
        max_l = max(self.max_len, min_l)
        len_a = self.rng.randint(min_l, max_l)
        len_b = self.rng.randint(min_l, max_l)
        a_vals = [self.rng.randrange(base) for _ in range(len_a)]
        b_vals = [self.rng.randrange(base) for _ in range(len_b)]
        if self.rng.random() < self.p_chain:
            self._inject_carry_chain(a_vals, b_vals, base)

        a = [axiom[v] for v in a_vals]
        b = [axiom[v] for v in b_vals]
        steps = reference_add(axiom, a, b)
        return {
            "base": base,
            "axiom": axiom,
            "a": a,
            "b": b,
            "steps": steps,
            "expected_out": reference_out(steps),
        }

    def _inject_carry_chain(self, a_vals: list[int], b_vals: list[int], base: int) -> None:
        """Forces one generating column (a_i + b_i >= base) followed by L propagating columns
        (a_j + b_j == base - 1). Uniform digits propagate with probability only 1/base, so long
        chains would otherwise be essentially absent in larger bases."""
        n = min(len(a_vals), len(b_vals))
        if n < 2:
            return
        L = self.rng.randint(1, n - 1)                 # number of propagating columns
        i = self.rng.randint(0, n - 1 - L)             # generating column
        a_vals[i] = self.rng.randint(1, base - 1)
        b_vals[i] = self.rng.randint(base - a_vals[i], base - 1)
        for j in range(i + 1, i + 1 + L):
            a_vals[j] = self.rng.randrange(base)
            b_vals[j] = base - 1 - a_vals[j]

    def sample_sm_step(self) -> tuple[dict, int]:
        """One column transition of a fresh instance: the terminal step k = K with probability
        p_terminal (counteracting the CONTINUE / FINISHED imbalance), otherwise uniform."""
        inst = self.sample_instance()
        K = len(inst["steps"]) - 1
        k = K if self.rng.random() < self.p_terminal else self.rng.randint(0, K)
        return inst, k

    # Training batches (CPU tensors; the runner moves them to the training device)
    def sm_mh_batch(self, batch_size: int, tier: int) -> dict:
        return collate_sm_mh([encode_sm_mh(*self.sample_sm_step(), tier) for _ in range(batch_size)])

    def sm_seq_batch(self, batch_size: int, tier: int) -> dict:
        return collate_labelled([encode_sm_seq(*self.sample_sm_step(), tier) for _ in range(batch_size)])

    def sm_seq_rev_batch(self, batch_size: int, tier: int) -> dict:
        return collate_labelled([encode_sm_seq_rev(*self.sample_sm_step(), tier) for _ in range(batch_size)])

    def sm_fork_batch(self, batch_size: int, tier: int) -> dict:
        items = []
        for _ in range(batch_size):
            inst, k = self.sample_sm_step()
            items.append(encode_sm_fork(inst, k, tier, TOKEN_S))
            items.append(encode_sm_fork(inst, k, tier, TOKEN_C))
            items.append(encode_sm_fork(inst, k, tier, TOKEN_FLAG))
        return collate_labelled(items)

    def ar_batch(self, batch_size: int, tier: int) -> dict:
        return collate_labelled([encode_ar(self.sample_instance(), tier) for _ in range(batch_size)])
