"""Tests for the `step_ar` variant (`inputs`, `dataset`, `evaluation`, `generation`)."""

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import torch

from experiments.addition.inputs import (
    AdditionToken,
    EmbeddingMode,
    StepOrder,
    TokenType,
    iter_problems,
)
from experiments.addition.reference import reference_add
from experiments.addition.variants.step_ar.dataset import AdditionStepArDataset
from experiments.addition.variants.step_ar.dataset import main as step_ar_dataset_main
from experiments.addition.variants.step_ar.evaluation import evaluate_step_ar
from experiments.addition.variants.step_ar.generation import rollout_step_ar
from experiments.addition.variants.step_ar.inputs import (
    build_step_ar_example,
    build_step_ar_step_labelled_example,
    pack_step_order,
)
from yessir.dataset import IGNORE_INDEX
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


class _ScriptedRolloutStepArModel:
    """Fake `SubspaceLMHead` model returning scripted next-token argmax at `lengths - 1`."""

    def __init__(self, scripted_tokens: Sequence[Token]) -> None:
        self._tokens = list(scripted_tokens)
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")
        self.seen_examples: list[Example] = []

    def eval(self) -> "_ScriptedRolloutStepArModel":
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


class _PrefixLookupLMModel:
    """Causal `SubspaceLMHead` fake mapping unpadded prefixes `input_ids[:pos+1]` to next token."""

    def __init__(
        self,
        table: Mapping[tuple[int, ...], Token],
        *,
        default: Token = int(SpecialToken.FLAG_FINISHED),
    ) -> None:
        self._table = dict(table)
        self._default = default
        self.training = True
        self.device = torch.device("cpu")

    def eval(self) -> "_PrefixLookupLMModel":
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
            logits[b, :, int(SpecialToken.PAD)] = 50.0
            row = tuple(input_ids[b, :n].tolist())
            for pos in range(n):
                tok = self._table.get(row[: pos + 1], self._default)
                logits[b, pos, int(SpecialToken.PAD)] = 0.0
                logits[b, pos, tok] = 100.0
        return logits


def test_step_ar_spec_example_1_all_modes_and_prefix_consistency():
    """Verify `spec/variants/step_ar.md` §1–§3 Example 1 (`987 + 25 = 1012`) across all 4 modes."""
    alphabet = b"0123456789"
    a = b"789"
    b = b"52"
    bos = int(SpecialToken.BOS)
    sep = int(SpecialToken.SEP)
    q_s = int(AdditionToken.QUERY_SUM)
    q_c = int(AdditionToken.QUERY_CARRY)
    q_f = int(AdditionToken.QUERY_FLAG)
    cont = int(SpecialToken.FLAG_CONTINUE)
    fin = int(SpecialToken.FLAG_FINISHED)

    expected_steps = [
        (ord("0"), b"", ord("2"), ord("1"), cont),
        (ord("1"), b"2", ord("1"), ord("1"), cont),
        (ord("1"), b"21", ord("0"), ord("1"), cont),
        (ord("1"), b"210", ord("1"), ord("0"), fin),
    ]

    for k, (carry_in, out_prefix, o_k, c_next, flag_k) in enumerate(expected_steps):
        base_ids = (bos, *alphabet, sep, *a, sep, *b, sep, carry_in, sep, *out_prefix)
        base_tt = (
            TokenType.SPECIAL,
            *([TokenType.ALPHABET] * 10),
            TokenType.SPECIAL,
            *([TokenType.OPERAND_A] * 3),
            TokenType.SPECIAL,
            *([TokenType.OPERAND_B] * 2),
            TokenType.SPECIAL,
            TokenType.CARRY,
            TokenType.SPECIAL,
            *([TokenType.OUTPUT] * k),
        )
        base_pos = (0, *range(10), 0, *range(3), 0, *range(2), 0, k, 0, *range(k))

        # 1. sum_first, use_query_tokens=False
        sf_noq = build_step_ar_step_labelled_example(
            alphabet, a, b, k, step_order=StepOrder.SUM_FIRST, use_query_tokens=False
        )
        assert sf_noq.example.input_ids == (*base_ids, sep, o_k, c_next)
        assert sf_noq.example.token_type_ids == (
            *base_tt,
            TokenType.SPECIAL,
            TokenType.OUTPUT,
            TokenType.CARRY,
        )
        assert sf_noq.example.position_ids == (*base_pos, k, k, k + 1)
        assert sf_noq.labels == (IGNORE_INDEX,) * len(base_ids) + (o_k, c_next, flag_k)

        # 2. carry_first, use_query_tokens=False
        cf_noq = build_step_ar_step_labelled_example(
            alphabet, a, b, k, step_order=StepOrder.CARRY_FIRST, use_query_tokens=False
        )
        assert cf_noq.example.input_ids == (*base_ids, sep, c_next, o_k)
        assert cf_noq.example.token_type_ids == (
            *base_tt,
            TokenType.SPECIAL,
            TokenType.CARRY,
            TokenType.OUTPUT,
        )
        assert cf_noq.example.position_ids == (*base_pos, k, k + 1, k)
        assert cf_noq.labels == (IGNORE_INDEX,) * len(base_ids) + (c_next, o_k, flag_k)

        # 3. sum_first, use_query_tokens=True
        sf_q = build_step_ar_step_labelled_example(
            alphabet, a, b, k, step_order=StepOrder.SUM_FIRST, use_query_tokens=True
        )
        assert sf_q.example.input_ids == (*base_ids, q_s, o_k, q_c, c_next, q_f)
        assert sf_q.example.token_type_ids == (
            *base_tt,
            TokenType.SPECIAL,
            TokenType.OUTPUT,
            TokenType.SPECIAL,
            TokenType.CARRY,
            TokenType.SPECIAL,
        )
        assert sf_q.example.position_ids == (*base_pos, k, k, k, k + 1, k)
        assert sf_q.labels == (IGNORE_INDEX,) * len(base_ids) + (
            o_k,
            IGNORE_INDEX,
            c_next,
            IGNORE_INDEX,
            flag_k,
        )

        # 4. carry_first, use_query_tokens=True
        cf_q = build_step_ar_step_labelled_example(
            alphabet, a, b, k, step_order=StepOrder.CARRY_FIRST, use_query_tokens=True
        )
        assert cf_q.example.input_ids == (*base_ids, q_c, c_next, q_s, o_k, q_f)
        assert cf_q.example.token_type_ids == (
            *base_tt,
            TokenType.SPECIAL,
            TokenType.CARRY,
            TokenType.SPECIAL,
            TokenType.OUTPUT,
            TokenType.SPECIAL,
        )
        assert cf_q.example.position_ids == (*base_pos, k, k + 1, k, k, k)
        assert cf_q.labels == (IGNORE_INDEX,) * len(base_ids) + (
            c_next,
            IGNORE_INDEX,
            o_k,
            IGNORE_INDEX,
            flag_k,
        )

        # Verify rollout's 3 incremental step prompts are exact prefixes of the teacher-forced sequence!
        for order in StepOrder:
            t_0, t_1 = pack_step_order(o_k, c_next, order)
            for use_q in (False, True):
                full_ex = build_step_ar_step_labelled_example(
                    alphabet, a, b, k, step_order=order, use_query_tokens=use_q
                )
                sup_indices = [
                    i for i, lbl in enumerate(full_ex.labels) if lbl != IGNORE_INDEX
                ]
                assert len(sup_indices) == 3
                for pass_idx, partial_tokens in enumerate(((), (t_0,), (t_0, t_1))):
                    partial_ex = build_step_ar_example(
                        alphabet,
                        a,
                        b,
                        carry_in,
                        out_prefix,
                        partial_tokens,
                        step_order=order,
                        use_query_tokens=use_q,
                    )
                    end = sup_indices[pass_idx] + 1
                    assert partial_ex.input_ids == full_ex.example.input_ids[:end]
                    assert (
                        partial_ex.token_type_ids
                        == full_ex.example.token_type_ids[:end]  # ty: ignore[not-subscriptable]
                    )
                    assert (
                        partial_ex.position_ids == full_ex.example.position_ids[:end]  # ty: ignore[not-subscriptable]
                    )

    with pytest.raises(IndexError, match="Step index k=4"):
        build_step_ar_step_labelled_example(alphabet, a, b, 4)
    with pytest.raises(ValueError, match="at most 2 tokens"):
        build_step_ar_example(alphabet, a, b, ord("0"), (), (1, 2, 3))


def test_step_ar_spec_example_2_and_embedding_modes():
    """Verify `spec/variants/step_ar.md` §2 Example 2 (`alphabet = b'q#7'`, `5 + 1 = 6`)."""
    alphabet = b"q#7"
    a = b"7#"
    b = b"#"
    cont = int(SpecialToken.FLAG_CONTINUE)
    fin = int(SpecialToken.FLAG_FINISHED)

    sf0 = build_step_ar_step_labelled_example(
        alphabet, a, b, 0, step_order=StepOrder.SUM_FIRST
    )
    assert sf0.labels[-3:] == (ord("q"), ord("#"), cont)

    cf0 = build_step_ar_step_labelled_example(
        alphabet, a, b, 0, step_order=StepOrder.CARRY_FIRST
    )
    assert cf0.labels[-3:] == (ord("#"), ord("q"), cont)

    sf1 = build_step_ar_step_labelled_example(
        alphabet, a, b, 1, step_order=StepOrder.SUM_FIRST
    )
    assert sf1.labels[-3:] == (ord("7"), ord("q"), fin)

    cf1 = build_step_ar_step_labelled_example(
        alphabet, a, b, 1, step_order=StepOrder.CARRY_FIRST
    )
    assert cf1.labels[-3:] == (ord("q"), ord("7"), fin)

    ex_tt_nope = build_step_ar_example(
        alphabet, a, b, ord("q"), embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE
    )
    assert ex_tt_nope.token_type_ids is not None
    assert ex_tt_nope.position_ids is None

    ex_nope = build_step_ar_example(
        alphabet, a, b, ord("q"), embedding_mode=EmbeddingMode.NOPE
    )
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_addition_step_ar_dataset_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(ValueError, match="p_terminal must be in 0.0..1.0"):
        AdditionStepArDataset(p_terminal=-0.2)

    ds = AdditionStepArDataset(
        bases=(3,),
        min_length=2,
        max_length=2,
        all_steps=True,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        seed=11,
    )
    first_prob = next(iter_problems(bases=(3,), min_length=2, max_length=2, seed=11))
    oracle = reference_add(first_prob.alphabet, first_prob.a, first_prob.b)
    items = list(itertools.islice(ds, len(oracle)))
    for st, item in zip(oracle, items, strict=True):
        non_ignored = [lbl for lbl in item.labels if lbl != IGNORE_INDEX]
        assert non_ignored == [st.carry_out, st.sum_digit, int(st.flag)]

    dump_path = tmp_path / "step_ar_dump.jsonl"
    step_ar_dataset_main(
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


def test_rollout_step_ar_unpacks_step_order_and_inserts_query_tokens():
    alphabet = b"012"
    a = b"21"
    b = b"1"
    # Under CARRY_FIRST + use_query_tokens=True:
    # Step 0 emits (c_1=SEP, o_0=255, flag_0=FLAG_CONTINUE) -> unpacked (o_0=255, c_1=SEP).
    # Step 1 emits (c_2='0', o_1='1', flag_1=FLAG_FINISHED) -> unpacked (o_1='1', c_2='0'), stops.
    scripted = [
        int(SpecialToken.SEP),
        255,
        int(SpecialToken.FLAG_CONTINUE),
        ord("0"),
        ord("1"),
        int(SpecialToken.FLAG_FINISHED),
    ]
    model = _ScriptedRolloutStepArModel(scripted)
    emitted = rollout_step_ar(
        model,  # ty: ignore[invalid-argument-type]
        alphabet,
        a,
        b,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    assert model.training is False
    assert emitted == [
        (255, int(SpecialToken.SEP), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
    ]
    assert len(model.seen_examples) == 6
    assert model.seen_examples[0] == build_step_ar_example(
        alphabet,
        a,
        b,
        ord("0"),
        (),
        (),
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    assert model.seen_examples[3] == build_step_ar_example(
        alphabet,
        a,
        b,
        int(SpecialToken.SEP),
        (255,),
        (),
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )

    # Caps at max(len(a), len(b)) + 1 == 3 steps (9 forward passes) if FLAG_FINISHED never emitted.
    never_halt = _ScriptedRolloutStepArModel(
        [ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)] * 4
    )
    capped = rollout_step_ar(never_halt, alphabet, a, b)  # ty: ignore[invalid-argument-type]
    assert len(capped) == 3
    assert len(never_halt.seen_examples) == 9


@pytest.mark.parametrize("step_order", list(StepOrder))
@pytest.mark.parametrize("use_query_tokens", [False, True])
def test_evaluate_step_ar_all_modes_rollout_cascade_and_batch_size_invariance(
    step_order: StepOrder, use_query_tokens: bool
):
    from experiments.addition.inputs import AdditionProblem

    prob0 = AdditionProblem(alphabet=tuple(b"01"), a=tuple(b"101"), b=tuple(b"010"))
    prob1 = AdditionProblem(
        alphabet=tuple(b"0123456789"), a=tuple(b"99"), b=tuple(b"1")
    )
    s0 = reference_add(prob0.alphabet, prob0.a, prob0.b)
    s1 = reference_add(prob1.alphabet, prob1.a, prob1.b)

    def _register_step(
        table: dict[tuple[int, ...], int],
        prob: AdditionProblem,
        carry_in: int,
        out_prefix: tuple[int, ...],
        t0: int,
        t1: int,
        flag_tok: int,
    ) -> None:
        for partial, nxt in (
            ((), t0),
            ((t0,), t1),
            ((t0, t1), flag_tok),
        ):
            ex = build_step_ar_example(
                prob.alphabet,
                prob.a,
                prob.b,
                carry_in,
                out_prefix,
                partial,
                step_order=step_order,
                use_query_tokens=use_query_tokens,
            )
            table[ex.input_ids] = nxt

    table: dict[tuple[int, ...], int] = {}
    out0 = tuple(st.sum_digit for st in s0)
    for k, st in enumerate(s0):
        t0, t1 = pack_step_order(st.sum_digit, st.carry_out, step_order)
        _register_step(table, prob0, st.carry_in, out0[:k], t0, t1, int(st.flag))

    out1 = tuple(st.sum_digit for st in s1)
    for k, st in enumerate(s1):
        t0, t1 = pack_step_order(st.sum_digit, st.carry_out, step_order)
        _register_step(table, prob1, st.carry_in, out1[:k], t0, t1, int(st.flag))

    # Override k=0 of prob1 to drop carry_out ('0' instead of '1').
    # Under CARRY_FIRST, also verify E-2: if oracle c_1='1' were injected, o_0 would be '0',
    # whereas conditioning on the model's own wrong c_1='0' emits wrong o_0='9'!
    if step_order == StepOrder.CARRY_FIRST:
        _register_step(
            table,
            prob1,
            ord("0"),
            (),
            ord("0"),
            ord("9"),
            int(SpecialToken.FLAG_CONTINUE),
        )
        rollout_o0 = ord("9")
    else:
        _register_step(
            table,
            prob1,
            ord("0"),
            (),
            ord("0"),
            ord("0"),
            int(SpecialToken.FLAG_CONTINUE),
        )
        rollout_o0 = ord("0")

    t0_r1, t1_r1 = pack_step_order(ord("9"), ord("0"), step_order)
    _register_step(
        table,
        prob1,
        ord("0"),
        (rollout_o0,),
        t0_r1,
        t1_r1,
        int(SpecialToken.FLAG_FINISHED),
    )

    m_b1 = evaluate_step_ar(
        _PrefixLookupLMModel(table),  # ty: ignore[invalid-argument-type]
        [prob0, prob1],
        num_problems=2,
        batch_size=1,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
    )
    m_b5 = evaluate_step_ar(
        _PrefixLookupLMModel(table),  # ty: ignore[invalid-argument-type]
        [prob0, prob1],
        num_problems=2,
        batch_size=5,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
    )
    assert m_b1 == m_b5
    assert m_b1.exact_match == pytest.approx(0.5)
    assert m_b1.halt_accuracy == pytest.approx(0.5)
    total_steps = len(s0) + len(s1)
    assert m_b1.step_accuracy == pytest.approx(len(s0) / total_steps)
    assert m_b1.oracle_step_accuracy == pytest.approx((total_steps - 1) / total_steps)
    expected_digits = len(s0) + (0 if step_order == StepOrder.CARRY_FIRST else 1)
    assert m_b1.digit_accuracy == pytest.approx(expected_digits / total_steps)


def test_evaluate_and_rollout_step_ar_with_real_transformer_and_validation():
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
    metrics = evaluate_step_ar(
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

    steps = rollout_step_ar(
        model,
        b"01",
        b"10",
        b"1",
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    assert 1 <= len(steps) <= 3

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_step_ar(model, problems, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_step_ar(model, problems, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="exhausted early"):
        evaluate_step_ar(model, [], num_problems=1)

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
        evaluate_step_ar(
            cls_model,
            iter_problems(bases=(2,), min_length=1, max_length=1, seed=1),
            num_problems=1,
        )
    with pytest.raises(ValueError, match="SubspaceLMHead"):
        rollout_step_ar(cls_model, b"01", b"0", b"1")
