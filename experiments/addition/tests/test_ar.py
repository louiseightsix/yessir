"""Tests for the full-trace `ar` variant (`inputs`, `dataset`, `evaluation`, `generation`)."""

import itertools
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from experiments.addition.inputs import (
    AdditionProblem,
    AdditionToken,
    EmbeddingMode,
    StepOrder,
    TokenType,
    iter_problems,
    pack_step_order,
)
from experiments.addition.reference import reference_add
from experiments.addition.variants.ar.dataset import AdditionArDataset
from experiments.addition.variants.ar.dataset import main as ar_dataset_main
from experiments.addition.variants.ar.evaluation import evaluate_ar
from experiments.addition.variants.ar.generation import rollout_ar
from experiments.addition.variants.ar.inputs import (
    build_ar_example,
    build_ar_labelled_example,
    build_ar_trace_tokens,
)
from yessir.dataset import IGNORE_INDEX
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


class _ScriptedRolloutArModel:
    """Fake `SubspaceLMHead` model returning scripted next-token argmax at `lengths - 1`."""

    def __init__(self, scripted_tokens: Sequence[Token]) -> None:
        self._tokens = list(scripted_tokens)
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")
        self.seen_examples: list[Example] = []

    def eval(self) -> "_ScriptedRolloutArModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert lengths is not None
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
        for b in range(batch_size):
            n = int(lengths[b].item())
            self.seen_examples.append(
                Example(
                    input_ids=tuple(input_ids[b, :n].tolist()),
                    token_type_ids=(
                        tuple(token_type_ids[b, :n].tolist())
                        if token_type_ids is not None
                        else None
                    ),
                    position_ids=(
                        tuple(position_ids[b, :n].tolist())
                        if position_ids is not None
                        else None
                    ),
                )
            )
            tok = self._tokens[self._cursor]
            self._cursor += 1
            logits[b, n - 1, tok] = 10.0
        return logits


class _ScriptedEvalArModel:
    """Fake `SubspaceLMHead` model placing scripted target sequences at `n - len(targets) .. n - 1`."""

    def __init__(self, scripted_targets: Sequence[Sequence[Token]]) -> None:
        self._targets = [list(seq) for seq in scripted_targets]
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")

    def eval(self) -> "_ScriptedEvalArModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert lengths is not None
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, VOCAB_SIZE)

        for b in range(batch_size):
            n = int(lengths[b].item())
            # Poison all positions (including prefix and padded tail) with PAD first.
            logits[b, :, int(SpecialToken.PAD)] = 50.0
            seq = self._targets[self._cursor]
            self._cursor += 1
            start = n - len(seq)
            for idx, tok in enumerate(seq):
                logits[b, start + idx, int(SpecialToken.PAD)] = 0.0
                logits[b, start + idx, tok] = 100.0
        return logits


def test_ar_spec_example_1_all_modes_and_prefix_consistency():
    """Verify `spec/variants/ar.md` §1–§3 Example 1 (`987 + 25 = 1012`) across all 4 modes."""
    alphabet = b"0123456789"
    a = b"789"
    b = b"52"
    bos = int(SpecialToken.BOS)
    sep = int(SpecialToken.SEP)
    eos = int(SpecialToken.EOS)
    q_s = int(AdditionToken.QUERY_SUM)
    q_c = int(AdditionToken.QUERY_CARRY)

    prefix_ids = (bos, *alphabet, sep, *a, sep, *b, sep)
    prefix_tt = (
        TokenType.SPECIAL,
        *([TokenType.ALPHABET] * 10),
        TokenType.SPECIAL,
        *([TokenType.OPERAND_A] * 3),
        TokenType.SPECIAL,
        *([TokenType.OPERAND_B] * 2),
        TokenType.SPECIAL,
    )
    prefix_pos = (0, *range(10), 0, *range(3), 0, *range(2), 0)
    p_len = len(prefix_ids)
    assert p_len == 10 + 3 + 2 + 4 == 19

    # 1. sum_first, use_query_tokens=False -> 2 1  1 1  0 1  1 0  [EOS]
    sf_noq = build_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.SUM_FIRST, use_query_tokens=False
    )
    expected_sf_trace = tuple(b"21110110")
    assert sf_noq.example.input_ids == (*prefix_ids, *expected_sf_trace)
    assert sf_noq.example.token_type_ids == (
        *prefix_tt,
        *(
            [
                TokenType.OUTPUT,
                TokenType.CARRY,
            ]
            * 4
        ),
    )
    assert sf_noq.example.position_ids == (*prefix_pos, 0, 1, 1, 2, 2, 3, 3, 4)
    assert sf_noq.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_sf_trace, eos)

    # 2. carry_first, use_query_tokens=False -> 1 2  1 1  1 0  0 1  [EOS]
    cf_noq = build_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.CARRY_FIRST, use_query_tokens=False
    )
    expected_cf_trace = tuple(b"12111001")
    assert cf_noq.example.input_ids == (*prefix_ids, *expected_cf_trace)
    assert cf_noq.example.token_type_ids == (
        *prefix_tt,
        *(
            [
                TokenType.CARRY,
                TokenType.OUTPUT,
            ]
            * 4
        ),
    )
    assert cf_noq.example.position_ids == (*prefix_pos, 1, 0, 2, 1, 3, 2, 4, 3)
    assert cf_noq.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_cf_trace, eos)

    # 3. sum_first, use_query_tokens=True
    sf_q = build_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.SUM_FIRST, use_query_tokens=True
    )
    expected_sf_q_trace = (
        q_s,
        ord("2"),
        q_c,
        ord("1"),
        q_s,
        ord("1"),
        q_c,
        ord("1"),
        q_s,
        ord("0"),
        q_c,
        ord("1"),
        q_s,
        ord("1"),
        q_c,
        ord("0"),
    )
    assert sf_q.example.input_ids == (*prefix_ids, *expected_sf_q_trace)
    assert sf_q.example.position_ids == (
        *prefix_pos,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
    )
    assert sf_q.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_sf_q_trace, eos)

    # 4. carry_first, use_query_tokens=True
    cf_q = build_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.CARRY_FIRST, use_query_tokens=True
    )
    expected_cf_q_trace = (
        q_c,
        ord("1"),
        q_s,
        ord("2"),
        q_c,
        ord("1"),
        q_s,
        ord("1"),
        q_c,
        ord("1"),
        q_s,
        ord("0"),
        q_c,
        ord("0"),
        q_s,
        ord("1"),
    )
    assert cf_q.example.input_ids == (*prefix_ids, *expected_cf_q_trace)
    assert cf_q.example.position_ids == (
        *prefix_pos,
        0,
        1,
        0,
        0,
        1,
        2,
        1,
        1,
        2,
        3,
        2,
        2,
        3,
        4,
        3,
        3,
    )
    assert cf_q.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_cf_q_trace, eos)

    # Verify rollout's incremental prefixes match slices of the full teacher-forced Example!
    steps = reference_add(alphabet, a, b)
    for order in StepOrder:
        for use_q in (False, True):
            full = build_ar_labelled_example(
                alphabet, a, b, steps=steps, step_order=order, use_query_tokens=use_q
            )
            trace = build_ar_trace_tokens(
                steps, step_order=order, use_query_tokens=use_q
            )
            for t_len in range(len(trace) + 1):
                partial = build_ar_example(
                    alphabet,
                    a,
                    b,
                    trace[:t_len],
                    step_order=order,
                    use_query_tokens=use_q,
                )
                end = p_len + t_len
                assert partial.input_ids == full.example.input_ids[:end]
                assert (
                    partial.token_type_ids == full.example.token_type_ids[:end]  # ty: ignore[not-subscriptable]
                )
                assert (
                    partial.position_ids == full.example.position_ids[:end]  # ty: ignore[not-subscriptable]
                )


def test_ar_spec_example_2_and_embedding_modes():
    """Verify `spec/variants/ar.md` §2 Example 2 (`alphabet = b'q#7'`, `5 + 1 = 6`)."""
    alphabet = b"q#7"
    a = b"7#"
    b = b"#"
    eos = int(SpecialToken.EOS)

    sf = build_ar_labelled_example(alphabet, a, b, step_order=StepOrder.SUM_FIRST)
    assert sf.labels[-5:] == (ord("q"), ord("#"), ord("7"), ord("q"), eos)

    cf = build_ar_labelled_example(alphabet, a, b, step_order=StepOrder.CARRY_FIRST)
    assert cf.labels[-5:] == (ord("#"), ord("q"), ord("q"), ord("7"), eos)

    ex_tt_nope = build_ar_example(
        alphabet, a, b, b"q#", embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE
    )
    assert ex_tt_nope.token_type_ids is not None
    assert ex_tt_nope.position_ids is None

    ex_nope = build_ar_example(alphabet, a, b, b"q#", embedding_mode=EmbeddingMode.NOPE)
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_addition_ar_dataset_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    ds = AdditionArDataset(
        bases=(3,),
        min_length=2,
        max_length=2,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        seed=11,
    )
    first_prob = next(iter_problems(bases=(3,), min_length=2, max_length=2, seed=11))
    expected = build_ar_labelled_example(
        first_prob.alphabet,
        first_prob.a,
        first_prob.b,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    assert next(iter(ds)) == expected

    dump_path = tmp_path / "ar_dump.jsonl"
    ar_dataset_main(
        [
            "--bases",
            "2",
            "4",
            "--min-length",
            "1",
            "--max-length",
            "3",
            "--step-order",
            "carry_first",
            "--use-query-tokens",
            "--num-examples",
            "3",
            "--out",
            str(dump_path),
            "--print",
        ]
    )
    assert len(dump_path.read_text().splitlines()) == 3
    out = capsys.readouterr().out
    assert "Example 0:" in out


def test_rollout_ar_stops_on_eos_and_caps_at_max_tokens():
    alphabet = b"q#7"
    a = b"7#"
    b = b"#"
    eos = int(SpecialToken.EOS)

    scripted = [ord("q"), ord("#"), ord("7"), ord("q"), eos, ord("q")]
    model = _ScriptedRolloutArModel(scripted)
    emitted = rollout_ar(model, alphabet, a, b)  # ty: ignore[invalid-argument-type]
    assert model.training is False
    assert emitted == scripted[:5]
    assert len(model.seen_examples) == 5

    # Premature EOS at step 0 stops immediately.
    early = _ScriptedRolloutArModel([eos, ord("q")])
    assert rollout_ar(early, alphabet, a, b) == [eos]  # ty: ignore[invalid-argument-type]
    assert len(early.seen_examples) == 1

    # When EOS is never emitted, caps at 2 * (max(2, 1) + 1) + 1 == 7 tokens (or 4 * 3 + 1 == 13 with query tokens).
    never_noq = _ScriptedRolloutArModel([ord("q")] * 10)
    assert len(rollout_ar(never_noq, alphabet, a, b, use_query_tokens=False)) == 7  # ty: ignore[invalid-argument-type]

    never_q = _ScriptedRolloutArModel([ord("q")] * 20)
    assert len(rollout_ar(never_q, alphabet, a, b, use_query_tokens=True)) == 13  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("step_order", list(StepOrder))
@pytest.mark.parametrize("use_query_tokens", [False, True])
def test_evaluate_ar_all_modes_and_batch_size_invariance(
    step_order: StepOrder, use_query_tokens: bool
):
    problems = list(
        itertools.islice(
            iter_problems(
                bases=(2, 10), min_length=1, max_length=4, p_prior=1.0, seed=5
            ),
            2,
        )
    )
    s0 = reference_add(problems[0].alphabet, problems[0].a, problems[0].b)
    s1 = reference_add(problems[1].alphabet, problems[1].a, problems[1].b)
    eos = int(SpecialToken.EOS)

    t0 = [
        *build_ar_trace_tokens(
            s0, step_order=step_order, use_query_tokens=use_query_tokens
        ),
        eos,
    ]
    t1 = [
        *build_ar_trace_tokens(
            s1, step_order=step_order, use_query_tokens=use_query_tokens
        ),
        eos,
    ]
    # Corrupt sum_digit at column 0 of problem 1 while keeping query tokens & EOS intact.
    wrong_digit = (
        problems[1].alphabet[0]
        if s1[0].sum_digit != problems[1].alphabet[0]
        else problems[1].alphabet[1]
    )
    sum_idx_in_col0 = (
        (1 if step_order == StepOrder.SUM_FIRST else 3)
        if use_query_tokens
        else (0 if step_order == StepOrder.SUM_FIRST else 1)
    )
    t1[sum_idx_in_col0] = wrong_digit

    m_b1 = evaluate_ar(
        _ScriptedEvalArModel([t0, t1]),  # ty: ignore[invalid-argument-type]
        problems,
        num_problems=2,
        batch_size=1,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
    )
    m_b5 = evaluate_ar(
        _ScriptedEvalArModel([t0, t1]),  # ty: ignore[invalid-argument-type]
        problems,
        num_problems=2,
        batch_size=5,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
    )
    assert m_b1 == m_b5
    assert m_b1.exact_match == pytest.approx(0.5)
    assert m_b1.halt_accuracy == pytest.approx(1.0)
    total_steps = len(s0) + len(s1)
    assert m_b1.step_accuracy == pytest.approx((total_steps - 1) / total_steps)


def test_evaluate_ar_query_syntax_premature_eos_and_step_order_taxonomy():
    prob = AdditionProblem(alphabet=tuple(b"q#7"), a=tuple(b"7"), b=tuple(b"#"))
    steps = reference_add(prob.alphabet, prob.a, prob.b)
    eos = int(SpecialToken.EOS)

    # 1. Wrong query token [S] -> syntax_malformation=1.0, exact_match=0.0, but step_accuracy=1.0 & halt_accuracy=1.0!
    q_targets = [
        *build_ar_trace_tokens(
            steps, step_order=StepOrder.SUM_FIRST, use_query_tokens=True
        ),
        eos,
    ]
    bad_q = list(q_targets)
    bad_q[0] = int(AdditionToken.QUERY_CARRY)  # expected [S]
    m_bad_q = evaluate_ar(
        _ScriptedEvalArModel([bad_q]),  # ty: ignore[invalid-argument-type]
        [prob],
        num_problems=1,
        step_order=StepOrder.SUM_FIRST,
        use_query_tokens=True,
    )
    assert m_bad_q.exact_match == 0.0
    assert m_bad_q.step_accuracy == 1.0
    assert m_bad_q.halt_accuracy == 1.0
    assert m_bad_q.syntax_malformation == 1.0

    # 2. Premature EOS inside column 0 -> halt_accuracy=0.0, out_of_alphabet=1.0.
    noq_targets = [
        *build_ar_trace_tokens(
            steps, step_order=StepOrder.SUM_FIRST, use_query_tokens=False
        ),
        eos,
    ]
    premature = list(noq_targets)
    premature[0] = eos
    m_prem = evaluate_ar(
        _ScriptedEvalArModel([premature]),  # ty: ignore[invalid-argument-type]
        [prob],
        num_problems=1,
    )
    assert m_prem.halt_accuracy == 0.0
    assert m_prem.out_of_alphabet == 1.0

    # 3. `step_order` determines first-error classification when both o_0 (in-alphabet wrong)
    # and c_1 (out-of-alphabet 255) fail in column 0:
    # - Under SUM_FIRST, o_0 is checked first -> procedural_error=1.0.
    # - Under CARRY_FIRST, c_1 is checked first -> out_of_alphabet=1.0.
    wrong_in_alpha = ord("7")  # ground-truth o_0 is 'q' (2 + 1 = 3 = 10_3)
    out_of_alpha = 255
    for order, expected_proc, expected_ooa in (
        (StepOrder.SUM_FIRST, 1.0, 0.0),
        (StepOrder.CARRY_FIRST, 0.0, 1.0),
    ):
        seq = [
            *pack_step_order(wrong_in_alpha, out_of_alpha, order),
            *pack_step_order(steps[1].sum_digit, steps[1].carry_out, order),
            eos,
        ]
        m_ord = evaluate_ar(
            _ScriptedEvalArModel([seq]),  # ty: ignore[invalid-argument-type]
            [prob],
            num_problems=1,
            step_order=order,
        )
        assert m_ord.procedural_error == expected_proc
        assert m_ord.out_of_alphabet == expected_ooa


def test_evaluate_and_rollout_ar_with_real_transformer_and_validation():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    problems = iter_problems(bases=(2, 4), min_length=1, max_length=3, seed=77)
    metrics = evaluate_ar(
        model,
        problems,
        num_problems=4,
        batch_size=2,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    assert 0.0 <= metrics.exact_match <= 1.0
    assert 0.0 <= metrics.step_accuracy <= 1.0
    assert 0.0 <= metrics.halt_accuracy <= 1.0

    emitted = rollout_ar(
        model,
        b"01",
        b"10",
        b"1",
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    assert 1 <= len(emitted) <= 13

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_ar(model, problems, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_ar(model, problems, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="exhausted early"):
        evaluate_ar(model, [], num_problems=1)

    cls_model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    with pytest.raises(ValueError, match="SubspaceLMHead"):
        evaluate_ar(
            cls_model,
            iter_problems(bases=(2,), min_length=1, max_length=1, seed=1),
            num_problems=1,
        )
    with pytest.raises(ValueError, match="SubspaceLMHead"):
        rollout_ar(cls_model, b"01", b"0", b"1")
