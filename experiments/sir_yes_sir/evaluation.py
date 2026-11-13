"""Shared evaluation metrics container (`EvalMetrics`) for `sir_yes_sir`."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics over `sir_yes_sir` problems (`experiments/sir_yes_sir/spec/task.md` §3).

    Attributes:
        exact_match: Fraction of problems in `0.0..1.0` where every prediction
            matches the ground truth (`o_0, ..., o_{L-1}, EOS` in `ar`;
            `(o_k, flag_k)` across all `k = 0 .. L - 1` in `cls`).
        step_accuracy: Fraction of the `L` input steps (`k` in `0 .. L - 1`)
            across all evaluated problems where the predicted symbol matches `o_k`.
        halt_accuracy: Fraction of problems in `0.0..1.0` where the model halts
            at the exact right step (no premature halt for `k < L` and `EOS` at
            `k == L` in `ar`; `FLAG_CONTINUE` for `k < L - 1` and `FLAG_FINISHED`
            at `k == L - 1` in `cls`).
    """

    exact_match: float
    step_accuracy: float
    halt_accuracy: float
