"""Shared evaluation metrics (`EvalMetrics`), 4-way first-error taxonomy (`FirstErrorType`), and variant-agnostic scoring (`StepPrediction`, `classify_first_error`, `score_predictions`) for `addition`."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique

from experiments.addition.inputs import StepOrder
from experiments.addition.reference import ColumnStep
from yessir.alphabet import SymbolSet
from yessir.vocab import Token


@unique
class FirstErrorType(StrEnum):
    """Mutually exclusive categories for the earliest prediction failure on a problem (`spec/task.md` §3)."""

    OUT_OF_ALPHABET = "out_of_alphabet"
    PRIOR_INTRUSION = "prior_intrusion"
    PROCEDURAL_ERROR = "procedural_error"
    SYNTAX_MALFORMATION = "syntax_malformation"


@dataclass(frozen=True)
class StepPrediction:
    """Teacher-forced greedy predictions extracted for column `k` (`0 <= k <= K`) of one problem.

    Separating `syntax_ok` (intra-column structural/address/query tokens such as
    `[IDX]`, `addr(k)`, `[S]`, `[C]`) from `halt_ok` (column `k`'s halting or
    step-continuation control token and absence of premature `[EOS]`) lets all
    six variants feed the exact same `score_predictions` aggregator while keeping
    `halt_accuracy` independent of non-halting header/query errors.
    """

    sum_digit: Token
    carry_out: Token
    syntax_ok: bool = True
    halt_ok: bool = True


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics and 4-way first-error rates over `addition` problems (`spec/task.md` §3).

    First-error rates use the total number of evaluated problems as their
    denominator (rather than failed problems only) so they remain well-defined
    when `exact_match == 1.0`, stay comparable across `num_eval_problems`, render
    as `0.0..1.0` bars in `web/`, and partition all evaluated problems:
    `exact_match + out_of_alphabet + prior_intrusion + procedural_error + syntax_malformation == 1.0`.
    """

    exact_match: float
    step_accuracy: float
    digit_accuracy: float
    carry_accuracy: float | None
    halt_accuracy: float
    out_of_alphabet: float = 0.0
    prior_intrusion: float = 0.0
    procedural_error: float = 0.0
    syntax_malformation: float = 0.0

    def to_dict(self, *, prefix: str = "") -> dict[str, float | None]:
        """Serialise metrics to a flat JSON-ready dict (with `exact_match` first for `web/`)."""
        return {
            f"{prefix}exact_match": self.exact_match,
            f"{prefix}step_accuracy": self.step_accuracy,
            f"{prefix}digit_accuracy": self.digit_accuracy,
            f"{prefix}carry_accuracy": self.carry_accuracy,
            f"{prefix}halt_accuracy": self.halt_accuracy,
            f"{prefix}out_of_alphabet": self.out_of_alphabet,
            f"{prefix}prior_intrusion": self.prior_intrusion,
            f"{prefix}procedural_error": self.procedural_error,
            f"{prefix}syntax_malformation": self.syntax_malformation,
        }


def is_canonical_symbol_set(alphabet: Sequence[Token]) -> bool:
    """Return True if `alphabet` consists of the canonical numeral glyphs `SymbolSet.HEX[:B]`."""
    base = len(alphabet)
    if base > len(SymbolSet.HEX):
        return False
    return set(alphabet) == set(SymbolSet.HEX[:base])


def _classify_value_mismatch(
    pred_token: Token,
    expected_token: Token,
    alphabet: Sequence[Token],
    alphabet_set: set[Token],
    *,
    canonical_symbols: bool,
) -> FirstErrorType:
    """Classify a wrong sum or carry prediction (`pred_token != expected_token`)."""
    if pred_token not in alphabet_set:
        return FirstErrorType.OUT_OF_ALPHABET

    if canonical_symbols:
        # `expected_token == alphabet[value]` where `value` is `total % B` (for `o_k`)
        # or `total // B` (for `c_{k+1}`). Under a permuted canonical alphabet,
        # emitting `SymbolSet.HEX[value]` means the model fell back to conventional
        # ASCII numeral order ('0'..'9', 'A'..'F') for that digit value.
        digit_value = alphabet.index(expected_token)
        if pred_token == SymbolSet.HEX[digit_value]:
            return FirstErrorType.PRIOR_INTRUSION

    return FirstErrorType.PROCEDURAL_ERROR


def classify_first_error(
    alphabet: Sequence[Token],
    steps: Sequence[ColumnStep],
    predictions: Sequence[StepPrediction],
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
) -> FirstErrorType | None:
    """Return the `FirstErrorType` at the earliest failing column `k`, or `None` on exact match.

    Within each column `k` (`0 <= k <= K`), checks follow the deterministic order
    of `spec/task.md` §3:
    1. Supervised trace structure/address/query tokens (`pred.syntax_ok`).
    2. Value tokens `(o_k, c_{k+1})` in `step_order` (`sum_first` or `carry_first`).
    3. Halting / step-continuation prediction (`pred.halt_ok`).
    """
    if not steps:
        raise ValueError("steps must contain at least 1 ColumnStep")
    if len(predictions) != len(steps):
        raise ValueError(
            f"predictions length ({len(predictions)}) must match steps length ({len(steps)})"
        )

    alphabet_set = set(alphabet)
    canonical_symbols = is_canonical_symbol_set(alphabet)

    for step, pred in zip(steps, predictions, strict=True):
        if not pred.syntax_ok:
            return FirstErrorType.SYNTAX_MALFORMATION

        value_pairs = (
            ((pred.sum_digit, step.sum_digit), (pred.carry_out, step.carry_out))
            if step_order == StepOrder.SUM_FIRST
            else ((pred.carry_out, step.carry_out), (pred.sum_digit, step.sum_digit))
        )
        for pred_val, exp_val in value_pairs:
            if pred_val != exp_val:
                return _classify_value_mismatch(
                    pred_val,
                    exp_val,
                    alphabet,
                    alphabet_set,
                    canonical_symbols=canonical_symbols,
                )

        if not pred.halt_ok:
            return FirstErrorType.SYNTAX_MALFORMATION

    return None


def score_predictions(
    evaluated_problems: Sequence[
        tuple[Sequence[Token], Sequence[ColumnStep], Sequence[StepPrediction]]
    ],
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
) -> EvalMetrics:
    """Compute `EvalMetrics` over `(alphabet, steps, predictions)` problem tuples.

    All six variants extract `StepPrediction`s for `k = 0 .. K` and delegate here
    so `exact_match`, `step_accuracy`, `digit_accuracy`, `carry_accuracy`,
    `halt_accuracy`, and the 4-way error breakdown share identical denominators.
    """
    num_problems = len(evaluated_problems)
    if num_problems < 1:
        raise ValueError("evaluated_problems must contain at least 1 problem")

    em_count = 0
    halt_ok_count = 0
    total_steps = 0
    step_ok_count = 0
    digit_ok_count = 0
    carry_active_steps = 0
    carry_ok_count = 0
    error_counts: dict[FirstErrorType, int] = dict.fromkeys(FirstErrorType, 0)

    for alphabet, steps, preds in evaluated_problems:
        if not alphabet:
            raise ValueError("alphabet must not be empty")
        if not steps:
            raise ValueError("steps must contain at least 1 ColumnStep")
        if len(preds) != len(steps):
            raise ValueError(
                f"predictions length ({len(preds)}) must match steps length ({len(steps)})"
            )

        s_0 = alphabet[0]
        for step, pred in zip(steps, preds, strict=True):
            total_steps += 1
            digit_ok = pred.sum_digit == step.sum_digit
            both_ok = digit_ok and (pred.carry_out == step.carry_out)
            if digit_ok:
                digit_ok_count += 1
            if both_ok:
                step_ok_count += 1
            if step.carry_in != s_0 or step.carry_out != s_0:
                carry_active_steps += 1
                if both_ok:
                    carry_ok_count += 1

        if all(p.halt_ok for p in preds):
            halt_ok_count += 1

        first_err = classify_first_error(alphabet, steps, preds, step_order=step_order)
        if first_err is None:
            em_count += 1
        else:
            error_counts[first_err] += 1

    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=step_ok_count / total_steps,
        digit_accuracy=digit_ok_count / total_steps,
        carry_accuracy=(
            carry_ok_count / carry_active_steps if carry_active_steps > 0 else None
        ),
        halt_accuracy=halt_ok_count / num_problems,
        out_of_alphabet=error_counts[FirstErrorType.OUT_OF_ALPHABET] / num_problems,
        prior_intrusion=error_counts[FirstErrorType.PRIOR_INTRUSION] / num_problems,
        procedural_error=error_counts[FirstErrorType.PROCEDURAL_ERROR] / num_problems,
        syntax_malformation=error_counts[FirstErrorType.SYNTAX_MALFORMATION]
        / num_problems,
    )
