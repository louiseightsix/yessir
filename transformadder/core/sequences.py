"""Sequence container, prompt builders for Tiers 1-3, and the right-padding batch collator.
Conforms to the layouts of spec/02_mechanics.md Sections 3.1, 3.2, 4.1 and 4.2.

Tier 1 coordinate convention (both paradigms): a sum digit o_j sits at p = j; a carry digit c_j
sits at p = j, the index of the column it feeds into; the [S] query for column k sits at p = k.
"""

from dataclasses import dataclass

import torch

from transformadder.core.constants import (
    BOS,
    DELIM_A,
    DELIM_AXIOM,
    DELIM_B,
    DELIM_CARRY,
    DELIM_GENERIC,
    DELIM_OUT,
    PAD,
    ROLE_AXIOM,
    ROLE_CARRY,
    ROLE_CTRL,
    ROLE_FLAG,
    ROLE_OPERAND_A,
    ROLE_OPERAND_B,
    ROLE_OUT,
    ROLE_START,
    START,
    TOKEN_S,
)


@dataclass
class Seq:
    tokens: list[int]
    roles: list[int] | None = None   # Tier 1 only; None under NoPE
    sigs: list[int] | None = None    # Tier 1 only; None under NoPE

    def __len__(self) -> int:
        return len(self.tokens)

    def push(self, token: int, role: int, sig: int) -> "Seq":
        """Returns a new Seq with one token appended (coordinates ignored under NoPE)."""
        if self.roles is None:
            return Seq(self.tokens + [token])
        return Seq(self.tokens + [token], self.roles + [role], self.sigs + [sig])


# Slot coordinates (Tier 1; ignored under NoPE)
def sum_coords(k: int) -> tuple[int, int]:
    """[S] and o_k."""
    return ROLE_OUT, k


def carry_coords(k: int) -> tuple[int, int]:
    """[C] and c_{k+1}, emitted by column k: indexed by the column the carry feeds into."""
    return ROLE_CARRY, k + 1


def flag_coords(k: int) -> tuple[int, int]:
    """[F] trigger of column k."""
    return ROLE_FLAG, k


def trace_coords(i: int) -> tuple[int, int]:
    """Coordinates of the AR trace token at 0-indexed trace slot i. Depends only on the slot
    index, never on the content of the emitted token."""
    k = i // 4
    return sum_coords(k) if i % 4 < 2 else carry_coords(k)


def build_prefix(axiom: list[int], a: list[int], b: list[int], tier: int) -> Seq:
    """Static problem prefix shared by both paradigms: [BOS] A a b (tier-delimited)."""
    if tier == 1:
        tokens = [BOS] + axiom + a + b
        roles = (
            [ROLE_CTRL]
            + [ROLE_AXIOM] * len(axiom)
            + [ROLE_OPERAND_A] * len(a)
            + [ROLE_OPERAND_B] * len(b)
        )
        sigs = [0] + list(range(len(axiom))) + list(range(len(a))) + list(range(len(b)))
        return Seq(tokens, roles, sigs)
    if tier == 2:
        return Seq([BOS, DELIM_AXIOM] + axiom + [DELIM_A] + a + [DELIM_B] + b)
    if tier == 3:
        return Seq([BOS] + axiom + [DELIM_GENERIC] + a + [DELIM_GENERIC] + b)
    raise ValueError(f"Unknown tier: {tier}")


def build_sm_prompt(
    axiom: list[int], a: list[int], b: list[int], carry: int, out_digits: list[int], tier: int
) -> Seq:
    """SM base prompt at step k = len(out_digits): [BOS] A a b c_k OUT_k [S].

    Identical for SM-MH and SM-Seq Pass 1; always terminates on the [S] sum query.
    """
    k = len(out_digits)
    seq = build_prefix(axiom, a, b, tier)
    if tier == 1:
        seq = seq.push(carry, ROLE_CARRY, k)
        for j, o in enumerate(out_digits):
            seq = seq.push(o, ROLE_OUT, j)
        return seq.push(TOKEN_S, *sum_coords(k))
    if tier == 2:
        return Seq(seq.tokens + [DELIM_CARRY, carry, DELIM_OUT] + out_digits + [TOKEN_S])
    return Seq(seq.tokens + [DELIM_GENERIC, carry, DELIM_GENERIC] + out_digits + [TOKEN_S])


def build_ar_prompt(axiom: list[int], a: list[int], b: list[int], tier: int) -> Seq:
    """AR prompt, fed exactly once: [BOS] A a b [START]."""
    return build_prefix(axiom, a, b, tier).push(START, ROLE_START, 0)


def collate(seqs: list[Seq], device: torch.device | str | None = None) -> dict:
    """Right-pads a list of Seq into batch tensors.

    Under a causal mask no real token can attend to a later PAD, so no padding mask is needed.
    Left-padding is forbidden: it would displace [BOS] from index 0.
    """
    lengths = [len(s) for s in seqs]
    B, T = len(seqs), max(lengths)
    tier1 = seqs[0].roles is not None

    token_ids = torch.full((B, T), PAD, dtype=torch.long)
    role_ids = torch.full((B, T), ROLE_CTRL, dtype=torch.long) if tier1 else None
    sig_ids = torch.zeros((B, T), dtype=torch.long) if tier1 else None

    for i, s in enumerate(seqs):
        token_ids[i, : lengths[i]] = torch.tensor(s.tokens, dtype=torch.long)
        if tier1:
            role_ids[i, : lengths[i]] = torch.tensor(s.roles, dtype=torch.long)
            sig_ids[i, : lengths[i]] = torch.tensor(s.sigs, dtype=torch.long)

    batch = {
        "token_ids": token_ids,
        "role_ids": role_ids,
        "sig_ids": sig_ids,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
    if device is not None:
        batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
    return batch
