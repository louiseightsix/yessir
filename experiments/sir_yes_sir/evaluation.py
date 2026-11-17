"""Shared evaluation metrics (`EvalMetrics`) and prompt-prefix parser (`parse_prompt_prefix`) for `sir_yes_sir`."""

from collections.abc import Sequence
from dataclasses import dataclass

from yessir.special_tokens import SpecialToken


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics over `sir_yes_sir` problems (`experiments/sir_yes_sir/spec/task.md` §3).

    Attributes:
        exact_match: Fraction of problems in `0.0..1.0` where every prediction
            matches the ground truth (`o_0, ..., o_{L-1}, EOS` in `ar`;
            `(o_k, flag_k)` across all `k = 0 .. L - 1` in `cls`).
        step_accuracy: Fraction of the `L` input steps (`k` in `0 .. L - 1`)
            across all evaluated problems where step `k` was reached before
            halting and the predicted symbol matches `o_k`.
        halt_accuracy: Fraction of problems in `0.0..1.0` where the model halts
            at the exact right step (no premature halt for `k < L` and `EOS` at
            `k == L` in `ar`; `FLAG_CONTINUE` for `k < L - 1` and `FLAG_FINISHED`
            at `k == L - 1` in `cls`).
        wrap_around_accuracy: Fraction of wrap-around input steps (`x_k == s_{B-1}`,
            whose ground-truth successor is `s_0`) reached before halting where
            the predicted symbol equals `s_0`, or `None` if no evaluated step
            had `x_k == s_{B-1}`.
    """

    exact_match: float
    step_accuracy: float
    halt_accuracy: float
    wrap_around_accuracy: float | None = None


def parse_prompt_prefix(
    prefix_input_ids: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Extract `(alphabet, x)` from a `[BOS] alphabet [SEP] x [SEP]` prompt prefix."""
    sep = int(SpecialToken.SEP)
    sep_indices = [idx for idx, tok in enumerate(prefix_input_ids) if tok == sep]
    if (
        len(prefix_input_ids) < 5
        or prefix_input_ids[0] != int(SpecialToken.BOS)
        or len(sep_indices) != 2
        or sep_indices[-1] != len(prefix_input_ids) - 1
    ):
        raise ValueError(
            "expected sir_yes_sir prompt prefix [BOS] alphabet [SEP] x [SEP]"
        )
    alphabet = tuple(prefix_input_ids[1 : sep_indices[0]])
    x = tuple(prefix_input_ids[sep_indices[0] + 1 : sep_indices[1]])
    if not alphabet or not x:
        raise ValueError(
            "alphabet and x in sir_yes_sir prompt prefix must be non-empty"
        )
    return alphabet, x
