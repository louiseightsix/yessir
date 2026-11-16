"""Unit tests for `addition` shared evaluation metrics and 4-way error taxonomy (`evaluation.py`)."""

import pytest

from experiments.addition.evaluation import (
    EvalMetrics,
    FirstErrorType,
    StepPrediction,
    classify_first_error,
    score_predictions,
)
from experiments.addition.inputs import AdditionToken, StepOrder
from experiments.addition.reference import ColumnStep, reference_add
from yessir.alphabet import SymbolSet


def _perfect_preds(steps: tuple[ColumnStep, ...]) -> tuple[StepPrediction, ...]:
    return tuple(
        StepPrediction(sum_digit=s.sum_digit, carry_out=s.carry_out) for s in steps
    )


def test_perfect_predictions_score_100_percent_and_zero_errors():
    alphabet = SymbolSet.DECIMAL
    steps = reference_add(alphabet, b"789", b"52")  # 4 steps, all carry-active
    metrics = score_predictions([(alphabet, steps, _perfect_preds(steps))])

    assert metrics == EvalMetrics(
        exact_match=1.0,
        step_accuracy=1.0,
        digit_accuracy=1.0,
        carry_accuracy=1.0,
        halt_accuracy=1.0,
        out_of_alphabet=0.0,
        prior_intrusion=0.0,
        procedural_error=0.0,
        syntax_malformation=0.0,
    )
    # First key of `to_dict()` is `exact_match` (drives `headlineMetric` in `web/`).
    as_dict = metrics.to_dict()
    assert next(iter(as_dict)) == "exact_match"
    assert metrics.to_dict(prefix="ood_base3_")["ood_base3_exact_match"] == 1.0


def test_carry_accuracy_is_none_when_no_column_has_nonzero_carry():
    alphabet = SymbolSet.DECIMAL
    # 12 ('21') + 34 ('43') = 46 ('64'): c_k == '0' and c_{k+1} == '0' on both steps.
    steps = reference_add(alphabet, b"21", b"43")
    metrics = score_predictions([(alphabet, steps, _perfect_preds(steps))])

    assert metrics.exact_match == 1.0
    assert metrics.carry_accuracy is None


def test_step_digit_carry_and_halt_accuracies_distinguish_partial_errors():
    alphabet = SymbolSet.DECIMAL
    # 987 ('789') + 25 ('52') = 1012 ('2101') across k = 0, 1, 2, 3 (all 4 steps carry-active).
    steps = reference_add(alphabet, b"789", b"52")
    perfect = list(_perfect_preds(steps))

    # Problem 1: k=0 has right sum digit ('2') but wrong carry ('0' instead of '1');
    # k=1..3 match; halting is right.
    p1_preds = list(perfect)
    p1_preds[0] = StepPrediction(sum_digit=ord("2"), carry_out=ord("0"), halt_ok=True)

    # Problem 2: all digits & carries right, but premature halt on k=0 (`halt_ok=False`).
    p2_preds = list(perfect)
    p2_preds[0] = StepPrediction(
        sum_digit=steps[0].sum_digit,
        carry_out=steps[0].carry_out,
        halt_ok=False,
    )

    metrics = score_predictions(
        [(alphabet, steps, p1_preds), (alphabet, steps, p2_preds)]
    )

    # Neither problem achieved exact_match (2 failed problems: 1 procedural, 1 syntax).
    assert metrics.exact_match == 0.0
    # 8 steps total: digit_ok on all 8/8; step_ok (both o_k and c_{k+1}) on 7/8; all 8 are carry-active.
    assert metrics.digit_accuracy == 1.0
    assert metrics.step_accuracy == pytest.approx(7 / 8)
    assert metrics.carry_accuracy == pytest.approx(7 / 8)
    # Halt accuracy: Problem 1 halted properly, Problem 2 did not -> 1/2.
    assert metrics.halt_accuracy == 0.5
    assert metrics.procedural_error == 0.5
    assert metrics.syntax_malformation == 0.5
    assert metrics.out_of_alphabet == 0.0
    assert metrics.prior_intrusion == 0.0


def test_prior_intrusion_on_permuted_canonical_sum_and_carry():
    """Permuted base-4 canonical alphabet `'2031'` (`'2'=0, '0'=1, '3'=2, '1'=3`)."""
    alphabet = b"2031"
    # a = '0' (val 1), b = '3' (val 2), c_0 = '2' (val 0) -> total = 3 (v_sum=3 -> '1', v_carry=0 -> '2')
    steps = reference_add(alphabet, b"0", b"3")
    assert steps[0].sum_digit == ord("1")
    assert steps[0].carry_out == ord("2")

    # Predicting '3' (canonical glyph for value 3) instead of '1' -> PRIOR_INTRUSION.
    sum_intrusion = (StepPrediction(sum_digit=ord("3"), carry_out=ord("2")),)
    assert (
        classify_first_error(alphabet, steps, sum_intrusion)
        == FirstErrorType.PRIOR_INTRUSION
    )

    # Predicting '2' (in alphabet, value 0 != canonical '3') instead of '1' -> PROCEDURAL_ERROR.
    proc_err = (StepPrediction(sum_digit=ord("2"), carry_out=ord("2")),)
    assert (
        classify_first_error(alphabet, steps, proc_err)
        == FirstErrorType.PROCEDURAL_ERROR
    )

    # Predicting '0' (canonical glyph for carry value 0) at carry_out instead of '2' -> PRIOR_INTRUSION.
    carry_intrusion = (StepPrediction(sum_digit=ord("1"), carry_out=ord("0")),)
    assert (
        classify_first_error(alphabet, steps, carry_intrusion)
        == FirstErrorType.PRIOR_INTRUSION
    )


def test_prior_intrusion_never_fires_on_non_canonical_or_ascending_canonical():
    # Random non-canonical alphabet containing '3' by chance: must classify as PROCEDURAL_ERROR, not PRIOR_INTRUSION.
    non_canon = b"q#73"  # B=4, '3' is at index 3, wait: pick '3' at index 1 so expected index 3 is '3'? No:
    non_canon = b"q3#7"  # q=0, 3=1, #=2, 7=3 -> 1 ('3') + 2 ('#') = 3 ('7'); canonical[3] is '3'
    steps = reference_add(non_canon, b"3", b"#")
    assert steps[0].sum_digit == ord("7")
    preds = (StepPrediction(sum_digit=ord("3"), carry_out=ord("q")),)
    assert (
        classify_first_error(non_canon, steps, preds) == FirstErrorType.PROCEDURAL_ERROR
    )

    # Ascending canonical alphabet: canonical[v] == expected_token, so any wrong in-alphabet digit is PROCEDURAL_ERROR.
    canon = b"0123"
    canon_steps = reference_add(canon, b"1", b"2")
    canon_preds = (StepPrediction(sum_digit=ord("0"), carry_out=ord("0")),)
    assert (
        classify_first_error(canon, canon_steps, canon_preds)
        == FirstErrorType.PROCEDURAL_ERROR
    )


def test_emission_order_and_within_column_precedence():
    """Verify `syntax_ok` -> value tokens in `step_order` -> `halt_ok` precedence."""
    alphabet = b"0123"
    steps = reference_add(alphabet, b"1", b"1")  # expected o_0='2', c_1='0'

    # Both o_0 (procedural: '3' in alphabet) and c_1 (out_of_alphabet: '9' not in alphabet) fail:
    both_bad = (StepPrediction(sum_digit=ord("3"), carry_out=ord("9")),)
    assert (
        classify_first_error(alphabet, steps, both_bad, step_order=StepOrder.SUM_FIRST)
        == FirstErrorType.PROCEDURAL_ERROR
    )
    assert (
        classify_first_error(
            alphabet, steps, both_bad, step_order=StepOrder.CARRY_FIRST
        )
        == FirstErrorType.OUT_OF_ALPHABET
    )

    # `syntax_ok=False` takes precedence over value errors in the same column;
    # value errors take precedence over `halt_ok=False` in the same column.
    syntax_and_val_bad = (
        StepPrediction(
            sum_digit=int(AdditionToken.QUERY_SUM),
            carry_out=ord("0"),
            syntax_ok=False,
        ),
    )
    assert (
        classify_first_error(alphabet, steps, syntax_and_val_bad)
        == FirstErrorType.SYNTAX_MALFORMATION
    )

    val_and_halt_bad = (
        StepPrediction(sum_digit=ord("9"), carry_out=ord("0"), halt_ok=False),
    )
    assert (
        classify_first_error(alphabet, steps, val_and_halt_bad)
        == FirstErrorType.OUT_OF_ALPHABET
    )


def test_partition_of_unity_across_mixed_outcomes():
    alphabet = b"2031"  # permuted canonical B=4 ('2'=0, '0'=1, '3'=2, '1'=3)
    steps = reference_add(alphabet, b"0", b"3")  # o_0='1' (val 3), c_1='2' (val 0)

    problems = [
        # 1. Exact match
        (alphabet, steps, _perfect_preds(steps)),
        # 2. Out of alphabet ('9')
        (alphabet, steps, (StepPrediction(sum_digit=ord("9"), carry_out=ord("2")),)),
        # 3. Prior intrusion ('3' == canonical[3])
        (alphabet, steps, (StepPrediction(sum_digit=ord("3"), carry_out=ord("2")),)),
        # 4. Procedural error ('0' != canonical[3])
        (alphabet, steps, (StepPrediction(sum_digit=ord("0"), carry_out=ord("2")),)),
        # 5. Syntax malformation (missing halt)
        (
            alphabet,
            steps,
            (StepPrediction(sum_digit=ord("1"), carry_out=ord("2"), halt_ok=False),),
        ),
    ]
    metrics = score_predictions(problems)
    total = (
        metrics.exact_match
        + metrics.out_of_alphabet
        + metrics.prior_intrusion
        + metrics.procedural_error
        + metrics.syntax_malformation
    )
    assert total == pytest.approx(1.0)
    assert metrics.exact_match == pytest.approx(0.2)
    assert metrics.out_of_alphabet == pytest.approx(0.2)
    assert metrics.prior_intrusion == pytest.approx(0.2)
    assert metrics.procedural_error == pytest.approx(0.2)
    assert metrics.syntax_malformation == pytest.approx(0.2)


def test_score_predictions_rejects_empty_or_mismatched_inputs():
    alphabet = b"01"
    steps = reference_add(alphabet, b"0", b"0")
    with pytest.raises(ValueError, match="at least 1 problem"):
        score_predictions([])
    with pytest.raises(ValueError, match="must match steps length"):
        score_predictions([(alphabet, steps, ())])
    with pytest.raises(ValueError, match="must match steps length"):
        classify_first_error(alphabet, steps, ())
