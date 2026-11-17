"""Shared evaluation metrics (`EvalMetrics`), 4-way first-error taxonomy (`FirstErrorType`), and variant-agnostic scoring (`StepPrediction`, `classify_first_error`, `score_predictions`) for `addition`."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
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
    """Greedy predictions extracted for column `k` of one problem (`spec/task.md` §3).

    Separating `syntax_ok` (intra-column structural/address/query tokens such as
    `[IDX]`, `addr(k)`, `[S]`, `[C]`), `halt_ok` (column `k`'s halting or
    step-continuation control token and absence of premature `[EOS]`), and
    `address_ok` (`[IDX] + addr(k)` match in `addressed_ar`) lets all six variants
    feed the exact same `score_predictions` aggregator.
    """

    sum_digit: Token
    carry_out: Token
    syntax_ok: bool = True
    halt_ok: bool = True
    address_ok: bool | None = None


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics and 4-way first-error rates over `addition` problems (`spec/task.md` §3).

    All primary metrics are computed from unassisted free-running greedy rollout;
    `oracle_step_accuracy` records single-step teacher-forced column accuracy as a
    diagnostic in `eval_splits.json` / `eval.jsonl`.

    First-error rates use the total number of evaluated problems as their
    denominator (rather than failed problems only) so they remain well-defined
    when `exact_match == 1.0`, stay comparable across `num_eval_problems`, render
    as `0.0..1.0` bars in `web/`, and partition all evaluated problems:
    `exact_match + out_of_alphabet + prior_intrusion + procedural_error + syntax_malformation == 1.0`.
    """

    exact_match: float
    step_accuracy: float
    oracle_step_accuracy: float
    digit_accuracy: float
    carry_accuracy: float | None
    halt_accuracy: float
    address_accuracy: float | None
    out_of_alphabet: float
    prior_intrusion: float
    procedural_error: float
    syntax_malformation: float

    def to_dict(self, *, prefix: str = "") -> dict[str, float | None]:
        """Serialise metrics to a flat JSON-ready dict (with `exact_match` first for `web/`)."""
        # Fields come out in declaration order, so `exact_match` stays first and
        # a newly added field cannot be left out.
        return {
            f"{prefix}{field.name}": getattr(self, field.name) for field in fields(self)
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
        # Chance baseline (`spec/task.md` §3): this is one of only `B - 1` wrong
        # in-alphabet symbols, and in base 2 it is the only one, so every
        # in-alphabet error there lands in this branch.
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
    if not alphabet:
        raise ValueError("alphabet must not be empty")
    if not steps:
        raise ValueError("steps must contain at least 1 ColumnStep")

    alphabet_set = set(alphabet)
    canonical_symbols = is_canonical_symbol_set(alphabet)

    for step, pred in zip(steps, predictions, strict=False):
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

    if len(predictions) != len(steps):
        return FirstErrorType.SYNTAX_MALFORMATION

    return None


def score_predictions(
    evaluated_problems: Sequence[
        tuple[Sequence[Token], Sequence[ColumnStep], Sequence[StepPrediction]]
    ],
    *,
    step_order: StepOrder = StepOrder.SUM_FIRST,
    oracle_step_accuracy: float | None = None,
    track_addresses: bool = False,
) -> EvalMetrics:
    """Compute `EvalMetrics` over `(alphabet, steps, predictions)` problem tuples.

    `predictions` holds the variable-length sequence of `StepPrediction`s emitted
    by free-running rollout: columns missed after a premature halt (`len(preds) < K + 1`)
    count as missed steps against the `K + 1` reference denominator, and over-running
    past `K` (`len(preds) > K + 1`) fails `halt_accuracy` and `exact_match`.
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
    address_ok_count = 0
    error_counts: dict[FirstErrorType, int] = dict.fromkeys(FirstErrorType, 0)

    for alphabet, steps, preds in evaluated_problems:
        # Classify first: it also checks that this problem's `alphabet` and `steps`
        # are non-empty before any of its steps are counted.
        first_err = classify_first_error(alphabet, steps, preds, step_order=step_order)
        if first_err is None:
            em_count += 1
        else:
            error_counts[first_err] += 1

        s_0 = alphabet[0]
        for k, step in enumerate(steps):
            total_steps += 1
            is_carry_active = (step.carry_in != s_0) or (step.carry_out != s_0)
            if is_carry_active:
                carry_active_steps += 1

            if k < len(preds):
                pred = preds[k]
                digit_ok = pred.sum_digit == step.sum_digit
                both_ok = digit_ok and (pred.carry_out == step.carry_out)
                if digit_ok:
                    digit_ok_count += 1
                if both_ok:
                    step_ok_count += 1
                if is_carry_active and both_ok:
                    carry_ok_count += 1
                if track_addresses and pred.address_ok is True:
                    address_ok_count += 1

        if len(preds) == len(steps) and all(p.halt_ok for p in preds):
            halt_ok_count += 1

    rollout_step_acc = step_ok_count / total_steps
    return EvalMetrics(
        exact_match=em_count / num_problems,
        step_accuracy=rollout_step_acc,
        oracle_step_accuracy=(
            rollout_step_acc if oracle_step_accuracy is None else oracle_step_accuracy
        ),
        digit_accuracy=digit_ok_count / total_steps,
        carry_accuracy=(
            carry_ok_count / carry_active_steps if carry_active_steps > 0 else None
        ),
        halt_accuracy=halt_ok_count / num_problems,
        address_accuracy=(address_ok_count / total_steps if track_addresses else None),
        out_of_alphabet=error_counts[FirstErrorType.OUT_OF_ALPHABET] / num_problems,
        prior_intrusion=error_counts[FirstErrorType.PRIOR_INTRUSION] / num_problems,
        procedural_error=error_counts[FirstErrorType.PROCEDURAL_ERROR] / num_problems,
        syntax_malformation=error_counts[FirstErrorType.SYNTAX_MALFORMATION]
        / num_problems,
    )
