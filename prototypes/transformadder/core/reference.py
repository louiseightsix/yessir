"""Reference adder: the single ground-truth oracle for training targets, evaluation scoring,
and the stepper's explanatory output. Implements the transition and halting invariant of
spec/02_mechanics.md Sections 1-3.
"""

from prototypes.transformadder.core.constants import FLAG_CONTINUE, FLAG_FINISHED, TOKEN_C, TOKEN_S


def reference_step(axiom: list[int], a: list[int], b: list[int], k: int, carry_in: int) -> dict:
    """Ground-truth transition for column k given the incoming carry symbol."""
    val = {s: i for i, s in enumerate(axiom)}
    base, n = len(axiom), max(len(a), len(b))
    v_a = val[a[k]] if k < len(a) else 0
    v_b = val[b[k]] if k < len(b) else 0
    v_c = val[carry_in]
    total = v_a + v_b + v_c
    o, c_next = total % base, total // base
    finished = (k >= n - 1) and (c_next == 0)
    return {
        "k": k,
        "v_a": v_a,
        "v_b": v_b,
        "v_c": v_c,
        "total": total,
        "c_in": carry_in,
        "o": axiom[o],
        "c_out": axiom[c_next],
        "flag": FLAG_FINISHED if finished else FLAG_CONTINUE,
    }


def reference_add(axiom: list[int], a: list[int], b: list[int]) -> list[dict]:
    """Returns the K+1 ground-truth column transitions for a + b under `axiom`."""
    steps, carry, k = [], axiom[0], 0
    while True:
        step = reference_step(axiom, a, b, k, carry)
        steps.append(step)
        if step["flag"] == FLAG_FINISHED:
            return steps
        carry, k = step["c_out"], k + 1


def reference_out(steps: list[dict]) -> list[int]:
    """Complete sum sequence OUT = [o_0 .. o_K] in LSD-first order."""
    return [s["o"] for s in steps]


reference_expected_out = reference_out


def reference_trace(steps: list[dict], order: str = "sum_first") -> list[int]:
    """AR scratchpad trace:
    sum_first:   ([S] o_k [C] c_{k+1})* [FINISHED]
    carry_first: ([C] c_{k+1} [S] o_k)* [FINISHED]
    """
    trace = []
    for s in steps:
        if order == "carry_first":
            trace.extend([TOKEN_C, s["c_out"], TOKEN_S, s["o"]])
        else:
            trace.extend([TOKEN_S, s["o"], TOKEN_C, s["c_out"]])
    trace.append(FLAG_FINISHED)
    return trace
