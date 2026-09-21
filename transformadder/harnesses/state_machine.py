"""State Machine runtime execution harnesses (Paradigm A: SM-MH and SM-Seq).
Implements Algorithms 1-2 of spec/02_mechanics.md Section 3.3.

The harnesses contain no numeral arithmetic and no symbol translation, decode by unconstrained
argmax, and return (out_digits, status); any status other than "finished" is a failure.
Emitted carries are appended to the optional `carries` list, for logging and scoring only.
"""

import torch

from transformadder.core.constants import (
    FLAG_CLASS_FINISHED,
    FLAG_CONTINUE,
    FLAG_FINISHED,
    TOKEN_C,
    TOKEN_FLAG,
)
from transformadder.core.sequences import build_sm_prompt, carry_coords, flag_coords, sum_coords


@torch.no_grad()
def run_state_machine_mh(model, axiom, a, b, tier_format: int, stepper=None, carries=None) -> tuple[list[int], str]:
    """Multi-Head Stateful Driver (Paradigm A1: SM-MH). One forward pass per column."""
    model.eval()
    max_steps = max(len(a), len(b)) + 1            # largest legal step count
    carry, out_digits = axiom[0], []
    axiom_set = set(axiom)

    for k in range(max_steps):
        prompt = build_sm_prompt(axiom, a, b, carry, out_digits, tier_format)  # ... [S]
        logits_sum, logits_carry, logits_flag = model.forward_sm_mh_single(prompt)

        o_k = int(torch.argmax(logits_sum))
        c_next = int(torch.argmax(logits_carry))
        flag = FLAG_FINISHED if int(torch.argmax(logits_flag)) == FLAG_CLASS_FINISHED else FLAG_CONTINUE

        if stepper:
            stepper.step(k, prompt, o_k, c_next, flag, [logits_sum, logits_carry, logits_flag],
                         axiom=axiom, a=a, b=b, carry_in=carry)

        out_digits.append(o_k)
        if carries is not None:
            carries.append(c_next)
        carry = c_next

        if flag == FLAG_FINISHED:
            return out_digits, "finished"

        # Early exit on invalid / out-of-alphabet tokens (avoids 100+ wasted loops on garbage)
        if o_k not in axiom_set or c_next not in axiom_set:
            return out_digits, "malformed"

    return out_digits, "max_steps_exceeded"


@torch.no_grad()
def run_state_machine_seq(model, axiom, a, b, tier_format: int, stepper=None, carries=None) -> tuple[list[int], str]:
    """Stateless Micro-AR Driver (Paradigm A2: SM-Seq). Three forward passes per column; the
    full sequence is recomputed on each pass and transient tokens are discarded per column."""
    model.eval()
    max_steps = max(len(a), len(b)) + 1            # largest legal step count
    carry, out_digits = axiom[0], []
    axiom_set = set(axiom)

    for k in range(max_steps):
        # Micro-Pass 1: base prompt already terminates on [S]; emits o_k
        p1 = build_sm_prompt(axiom, a, b, carry, out_digits, tier_format)
        logits1 = model.next_logits(p1)
        o_k = int(torch.argmax(logits1))

        # Micro-Pass 2: emits outgoing carry c_{k+1}
        p2 = p1.push(o_k, *sum_coords(k)).push(TOKEN_C, *carry_coords(k))
        logits2 = model.next_logits(p2)
        c_next = int(torch.argmax(logits2))

        # Micro-Pass 3: emits control flag (anything other than FLAG_FINISHED continues the
        # loop and is scored as a step-transition error)
        p3 = p2.push(c_next, *carry_coords(k)).push(TOKEN_FLAG, *flag_coords(k))
        logits3 = model.next_logits(p3)
        flag = int(torch.argmax(logits3))

        if stepper:
            stepper.step(k, p1, o_k, c_next, flag, [logits1, logits2, logits3],
                         axiom=axiom, a=a, b=b, carry_in=carry)

        out_digits.append(o_k)
        if carries is not None:
            carries.append(c_next)
        carry = c_next                              # transients p1..p3 are discarded

        if flag == FLAG_FINISHED:
            return out_digits, "finished"

        # Early exit on invalid / out-of-alphabet tokens (avoids 100+ wasted micro-AR passes on garbage)
        if o_k not in axiom_set or c_next not in axiom_set:
            return out_digits, "malformed"

    return out_digits, "max_steps_exceeded"
