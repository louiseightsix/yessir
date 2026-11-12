"""Shared evaluation metrics container (`EvalMetrics`) for `sir_yes_sir`."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics over `sir_yes_sir` problems (`experiments/sir_yes_sir/spec/README.md` §3).

    Attributes:
        exact_match: Fraction of problems in `0.0..1.0` where every step `k`
            in `0 .. L - 1` predicted all three outputs `(o_k, w_k, flag_k)`
            correctly.
        step_accuracy: Fraction of individual steps in `0.0..1.0` where all
            three outputs `(o_k, w_k, flag_k)` matched the ground truth.
        halt_accuracy: Fraction of individual steps in `0.0..1.0` where the
            halting output matched `flag_k` (`FLAG_CONTINUE` vs. `FLAG_FINISHED`).
    """

    exact_match: float
    step_accuracy: float
    halt_accuracy: float
