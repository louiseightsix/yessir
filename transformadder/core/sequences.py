"""Sequence container, prompt builders for Tiers 1-7, and the right-padding batch collator.
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
    DELIM_ORIGIN,
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
    TOKEN_C,
    TOKEN_FLAG,
    TOKEN_PROBE,
    TOKEN_S,
)


@dataclass
class Seq:
    tokens: list[int]
    roles: list[int] | None = None   # Tiers 1 and 4; None under NoPE
    sigs: list[int] | None = None    # Tier 1 only; None under NoPE

    def __len__(self) -> int:
        return len(self.tokens)

    def push(self, token: int, role: int, sig: int) -> "Seq":
        """Returns a new Seq with one token appended (coordinates ignored under NoPE)."""
        if self.roles is None:
            return Seq(self.tokens + [token])
        sigs = None if self.sigs is None else self.sigs + [sig]
        return Seq(self.tokens + [token], self.roles + [role], sigs)


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


def trace_coords(i: int, order: str = "sum_first") -> tuple[int, int]:
    """Coordinates of the AR trace token at 0-indexed trace slot i. Depends only on the slot
    index, never on the content of the emitted token."""
    k = i // 4
    if order == "carry_first":
        return carry_coords(k) if i % 4 < 2 else sum_coords(k)
    return sum_coords(k) if i % 4 < 2 else carry_coords(k)


def int_to_axiom_address(k: int, axiom: list[int]) -> list[int]:
    """Encodes non-negative column index k into little-endian base-B axiom numeral tokens."""
    base = len(axiom)
    if k == 0:
        return [axiom[0]]
    digits = []
    val = k
    while val > 0:
        digits.append(axiom[val % base])
        val //= base
    return digits


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
    if tier == 4:
        tokens = [BOS] + axiom + [DELIM_GENERIC] + a + [DELIM_GENERIC] + b
        roles = (
            [ROLE_CTRL]
            + [ROLE_AXIOM] * len(axiom)
            + [ROLE_CTRL]
            + [ROLE_OPERAND_A] * len(a)
            + [ROLE_CTRL]
            + [ROLE_OPERAND_B] * len(b)
        )
        return Seq(tokens, roles, None)
    if tier == 5:
        return Seq([BOS, DELIM_AXIOM, DELIM_ORIGIN] + axiom + [DELIM_A, DELIM_ORIGIN] + a + [DELIM_B, DELIM_ORIGIN] + b)
    if tier == 6:
        formatted_a = []
        for j, val in enumerate(a):
            formatted_a += [DELIM_GENERIC] + int_to_axiom_address(j, axiom) + [DELIM_ORIGIN, val]
        formatted_b = []
        for j, val in enumerate(b):
            formatted_b += [DELIM_GENERIC] + int_to_axiom_address(j, axiom) + [DELIM_ORIGIN, val]
        return Seq([BOS, DELIM_AXIOM] + axiom + [DELIM_A] + formatted_a + [DELIM_B] + formatted_b)
    if tier == 7:
        formatted_a = []
        for j, val in enumerate(a):
            formatted_a += [DELIM_GENERIC] + int_to_axiom_address(j, axiom) + [DELIM_ORIGIN, val]
        formatted_b = []
        for j, val in enumerate(b):
            formatted_b += [DELIM_GENERIC] + int_to_axiom_address(j, axiom) + [DELIM_ORIGIN, val]
        return Seq([BOS, DELIM_A] + formatted_a + [DELIM_B] + formatted_b)
    raise ValueError(f"Unknown tier: {tier}")


def build_sm_prompt(
    axiom: list[int],
    a: list[int],
    b: list[int],
    carry: int,
    out_digits: list[int],
    tier: int,
    query_token: int = TOKEN_S,
) -> Seq:
    """SM base prompt at step k = len(out_digits): [BOS] A a b c_k OUT_k [Q].

    Identical for SM-MH and SM-Seq Pass 1 (Q = [S]); terminates on [C] for SM-SeqRev Pass 1;
    SM-Fork builds one prompt per query, Q in ([S], [C], [F]).
    """
    k = len(out_digits)
    seq = build_prefix(axiom, a, b, tier)
    if tier == 1:
        seq = seq.push(carry, ROLE_CARRY, k)
        for j, o in enumerate(out_digits):
            seq = seq.push(o, ROLE_OUT, j)
        if query_token == TOKEN_S:
            coords = sum_coords(k)
        elif query_token == TOKEN_C:
            coords = carry_coords(k)
        elif query_token == TOKEN_FLAG:
            coords = flag_coords(k)
        else:
            raise ValueError(f"Unknown query_token: {query_token}")
        return seq.push(query_token, *coords)
    if tier == 2:
        return Seq(seq.tokens + [DELIM_CARRY, carry, DELIM_OUT] + out_digits + [query_token])
    if tier == 3:
        return Seq(seq.tokens + [DELIM_GENERIC, carry, DELIM_GENERIC] + out_digits + [query_token])
    if tier == 4:
        tokens = seq.tokens + [DELIM_GENERIC, carry, DELIM_GENERIC] + out_digits + [query_token]
        assert seq.roles is not None
        if query_token == TOKEN_S:
            query_role = ROLE_OUT
        elif query_token == TOKEN_C:
            query_role = ROLE_CARRY
        elif query_token == TOKEN_FLAG:
            query_role = ROLE_FLAG
        else:
            raise ValueError(f"Unknown query_token: {query_token}")
        roles = (
            seq.roles
            + [ROLE_CTRL, ROLE_CARRY, ROLE_CTRL]
            + [ROLE_OUT] * len(out_digits)
            + [query_role]
        )
        return Seq(tokens, roles, None)
    if tier == 5:
        return Seq(seq.tokens + [DELIM_CARRY, DELIM_ORIGIN, carry, DELIM_OUT, DELIM_ORIGIN] + out_digits + [query_token])
    if tier in (6, 7):
        formatted_out = []
        for j, val in enumerate(out_digits):
            formatted_out += [DELIM_GENERIC] + int_to_axiom_address(j, axiom) + [DELIM_ORIGIN, val]
        carry_entry = [DELIM_CARRY, DELIM_GENERIC] + int_to_axiom_address(k, axiom) + [DELIM_ORIGIN, carry]
        query_entry = [DELIM_GENERIC] + int_to_axiom_address(k, axiom) + [DELIM_ORIGIN, query_token]
        return Seq(seq.tokens + carry_entry + [DELIM_OUT] + formatted_out + query_entry)
    raise ValueError(f"Unknown tier: {tier}")


def build_ar_prompt(axiom: list[int], a: list[int], b: list[int], tier: int) -> Seq:
    """AR prompt, fed exactly once: [BOS] A a b [START]."""
    return build_prefix(axiom, a, b, tier).push(START, ROLE_START, 0)


def build_probe_prompt(axiom: list[int], a: list[int], b: list[int], query_symbol: int, tier: int) -> Seq:
    """Successor binding probe prompt: [BOS] A a b [PROBE] query_symbol.

    Conditioned on [PROBE], the model retrieves the bound alphabet rule from context
    and emits s_{i+1} = axiom[(axiom.index(query_symbol) + 1) % len(axiom)].
    """
    prefix = build_prefix(axiom, a, b, tier)
    if tier == 1:
        return prefix.push(TOKEN_PROBE, ROLE_CTRL, 0).push(query_symbol, ROLE_CTRL, 1)
    if tier == 4:
        tokens = prefix.tokens + [TOKEN_PROBE, query_symbol]
        assert prefix.roles is not None
        roles = prefix.roles + [ROLE_CTRL, ROLE_CTRL]
        return Seq(tokens, roles, None)
    return Seq(prefix.tokens + [TOKEN_PROBE, query_symbol])


def collate(seqs: list[Seq], device: torch.device | str | None = None) -> dict:
    """Right-pads a list of Seq into batch tensors.

    Under a causal mask no real token can attend to a later PAD, so no padding mask is needed.
    Left-padding is forbidden: it would displace [BOS] from index 0.
    """
    lengths = [len(s) for s in seqs]
    B, T = len(seqs), max(lengths)
    has_roles = seqs[0].roles is not None
    has_sigs = seqs[0].sigs is not None

    token_ids = torch.full((B, T), PAD, dtype=torch.long)
    role_ids = torch.full((B, T), ROLE_CTRL, dtype=torch.long) if has_roles else None
    sig_ids = torch.zeros((B, T), dtype=torch.long) if has_sigs else None

    for i, s in enumerate(seqs):
        token_ids[i, : lengths[i]] = torch.tensor(s.tokens, dtype=torch.long)
        if has_roles:
            assert s.roles is not None
            role_ids[i, : lengths[i]] = torch.tensor(s.roles, dtype=torch.long)
        if has_sigs:
            assert s.sigs is not None
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
