"""Autoregressive trace execution harness (Paradigm B: AR).
Implements the generation contract of spec/02_mechanics.md Section 4.3: plain greedy decoding
with zero state re-injection and zero numeral arithmetic.
"""

import torch

from transformadder.core.constants import FLAG_FINISHED, SIGMA_POOL, TOKEN_C, TOKEN_S
from transformadder.core.sequences import build_ar_prompt, trace_coords

_SIGMA_END = SIGMA_POOL[-1] + 1  # numerals lie in [0x00, 0xF0)


def parse_trace(trace: list[int]) -> tuple[list[int], list[int], str]:
    """Validates trace against ([S] o [C] c)* [FINISHED] and extracts digits.

    Purely structural (token classes and slot positions); no digit values are read.
    Returns (OUT, carries, status) with status in
    {"finished", "malformed", "max_tokens_exceeded"}.
    """
    if not trace or trace[-1] != FLAG_FINISHED:
        status, body = "max_tokens_exceeded", trace
    else:
        status, body = "finished", trace[:-1]

    out_digits, carries = [], []
    for i in range(0, len(body), 4):
        column = body[i:i + 4]
        well_formed = (
            len(column) == 4
            and column[0] == TOKEN_S and column[2] == TOKEN_C
            and column[1] < _SIGMA_END and column[3] < _SIGMA_END
        )
        if not well_formed:
            return out_digits, carries, "malformed"
        out_digits.append(column[1])
        carries.append(column[3])

    if status == "finished" and not out_digits:
        status = "malformed"  # bare [FINISHED]
    return out_digits, carries, status


@torch.no_grad()
def run_autoregressive_trace(model, axiom, a, b, tier_format: int, stepper=None, carries=None) -> tuple[list[int], str]:
    """Generative Driver (Paradigm B: AR). The prompt is ingested once; the model unrolls the
    scratchpad until [FINISHED] or the longest well-formed trace length is exhausted."""
    model.eval()
    prompt = build_ar_prompt(axiom, a, b, tier_format)      # [BOS] ... [START]
    budget = 4 * (max(len(a), len(b)) + 1) + 1              # longest well-formed trace
    seq, trace = prompt, []

    for i in range(budget):
        token = int(torch.argmax(model.next_logits(seq)))
        trace.append(token)
        if token == FLAG_FINISHED:
            break
        # Early break if structural trace syntax is broken (avoids up to 160 wasted forward passes)
        if i % 4 == 0 and token != TOKEN_S:
            break
        if i % 4 == 2 and token != TOKEN_C:
            break
        # Tier 1 coordinates depend only on the slot index i, never on `token`
        seq = seq.push(token, *trace_coords(i))

    out_digits, trace_carries, status = parse_trace(trace)
    if carries is not None:
        carries.extend(trace_carries)

    if stepper:
        stepper.show_trace(prompt, trace, out_digits, status)
    return out_digits, status
