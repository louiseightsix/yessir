"""Tests for the `cls` variant (`inputs`, `dataset`, `evaluation`, `generation`)."""

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import torch

from experiments.addition.inputs import EmbeddingMode, TokenType, iter_problems
from experiments.addition.reference import reference_add
from experiments.addition.variants.cls.dataset import AdditionClsDataset
from experiments.addition.variants.cls.dataset import main as cls_dataset_main
from experiments.addition.variants.cls.evaluation import evaluate_cls
from experiments.addition.variants.cls.generation import rollout_cls
from experiments.addition.variants.cls.inputs import (
    build_cls_example,
    build_cls_step_labelled_example,
)
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


class _ScriptedClsModel:
    """Fake 3-head classification model returning scripted `(o_k, c_{k+1}, flag_k)` per item."""

    def __init__(self, scripted_steps: Sequence[tuple[Token, Token, Token]]) -> None:
        self._steps = list(scripted_steps)
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")
        self.seen_examples: list[Example] = []

    def eval(self) -> "_ScriptedClsModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert lengths is not None
        batch_size = input_ids.shape[0]
        logits_o = torch.zeros(batch_size, VOCAB_SIZE)
        logits_c = torch.zeros(batch_size, VOCAB_SIZE)
        logits_f = torch.zeros(batch_size, VOCAB_SIZE)

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
            o_k, c_next, flag_k = self._steps[self._cursor]
            self._cursor += 1
            logits_o[b, o_k] = 10.0
            logits_c[b, c_next] = 10.0
            logits_f[b, flag_k] = 10.0

        return logits_o, logits_c, logits_f


def test_cls_spec_example_1_and_position_coupling():
    """Verify `spec/variants/cls.md` §1–§2 Example 1 (`987 + 25 = 1012` in base 10)."""
    alphabet = b"0123456789"
    a = b"789"
    b = b"52"
    bos = int(SpecialToken.BOS)
    sep = int(SpecialToken.SEP)
    cont = int(SpecialToken.FLAG_CONTINUE)
    fin = int(SpecialToken.FLAG_FINISHED)

    expected_steps = [
        (ord("0"), b"", (ord("2"), ord("1"), cont)),
        (ord("1"), b"2", (ord("1"), ord("1"), cont)),
        (ord("1"), b"21", (ord("0"), ord("1"), cont)),
        (ord("1"), b"210", (ord("1"), ord("0"), fin)),
    ]

    for k, (carry_in, out_prefix, expected_labels) in enumerate(expected_steps):
        labelled = build_cls_step_labelled_example(alphabet, a, b, k)
        assert labelled.labels == expected_labels
        ex = labelled.example

        expected_ids = (
            bos,
            *alphabet,
            sep,
            *a,
            sep,
            *b,
            sep,
            carry_in,
            sep,
            *out_prefix,
        )
        expected_tt = (
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
        expected_pos = (
            0,
            *range(10),
            0,
            *range(3),
            0,
            *range(2),
            0,
            k,
            0,
            *range(1, k + 1),
        )
        assert ex.input_ids == expected_ids
        assert ex.token_type_ids == expected_tt
        assert ex.position_ids == expected_pos
        # Key invariant: predicting final position, incoming carry c_k, and column k share position_id == k.
        assert ex.position_ids[-1] == k

    with pytest.raises(IndexError, match="Step index k=4"):
        build_cls_step_labelled_example(alphabet, a, b, 4)


def test_cls_spec_example_2_and_embedding_modes():
    """Verify `spec/variants/cls.md` §2 Example 2 (`alphabet = b'q#7'`, `5 + 1 = 6`)."""
    alphabet = b"q#7"
    a = b"7#"
    b = b"#"
    cont = int(SpecialToken.FLAG_CONTINUE)
    fin = int(SpecialToken.FLAG_FINISHED)

    step0 = build_cls_step_labelled_example(alphabet, a, b, 0)
    assert step0.labels == (ord("q"), ord("#"), cont)

    step1 = build_cls_step_labelled_example(alphabet, a, b, 1)
    assert step1.labels == (ord("7"), ord("q"), fin)

    ex_tt_nope = build_cls_example(
        alphabet,
        a,
        b,
        ord("#"),
        b"q",
        embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE,
    )
    assert ex_tt_nope.token_type_ids is not None
    assert ex_tt_nope.position_ids is None

    ex_nope = build_cls_example(
        alphabet,
        a,
        b,
        ord("#"),
        b"q",
        embedding_mode=EmbeddingMode.NOPE,
    )
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_addition_cls_dataset_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(ValueError, match="p_terminal must be in 0.0..1.0"):
        AdditionClsDataset(p_terminal=-0.1)

    ds1 = AdditionClsDataset(bases=(3,), min_length=2, max_length=2, seed=11)
    ds2 = AdditionClsDataset(bases=(3,), min_length=2, max_length=2, seed=11)
    items1 = list(itertools.islice(ds1, 6))
    items2 = list(itertools.islice(ds2, 6))
    assert items1 == items2

    # With all_steps=True on a fixed problem stream, steps start at k=0 (ending in SEP)
    # and end with FLAG_FINISHED before the next problem begins.
    ds_all = AdditionClsDataset(
        bases=(3,), min_length=2, max_length=2, all_steps=True, seed=11
    )
    first_prob = next(iter_problems(bases=(3,), min_length=2, max_length=2, seed=11))
    oracle = reference_add(first_prob.alphabet, first_prob.a, first_prob.b)
    all_items = list(itertools.islice(ds_all, len(oracle)))
    assert [it.labels for it in all_items] == [
        (s.sum_digit, s.carry_out, int(s.flag)) for s in oracle
    ]

    dump_path = tmp_path / "cls_dump.jsonl"
    cls_dataset_main(
        [
            "--bases",
            "2",
            "4",
            "--min-length",
            "1",
            "--max-length",
            "3",
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


def test_rollout_cls_feeds_back_predictions_and_handles_stopping():
    alphabet = b"012"
    a = b"21"
    b = b"1"
    # Even if an untrained model emits out-of-alphabet or special-token bytes (e.g. 255 or SEP),
    # rollout_cls feeds them back into carry/output without crashing until FLAG_FINISHED or cap.
    scripted = [
        (255, int(SpecialToken.SEP), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
        (ord("0"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedClsModel(scripted)
    emitted = rollout_cls(model, alphabet, a, b)  # ty: ignore[invalid-argument-type]
    assert model.training is False
    assert emitted == scripted[:2]
    assert model.seen_examples == [
        build_cls_example(alphabet, a, b, ord("0"), ()),
        build_cls_example(alphabet, a, b, int(SpecialToken.SEP), (255,)),
    ]

    # When FLAG_FINISHED is never emitted, rollout stops at max(len(a), len(b)) + 1 == 3 steps.
    never_halt = _ScriptedClsModel(
        [
            (ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)),
            (ord("2"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
            (ord("0"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
            (ord("1"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        ]
    )
    capped = rollout_cls(never_halt, alphabet, a, b)  # ty: ignore[invalid-argument-type]
    assert len(capped) == 3


class _LookupClsModel:
    """Stateless 3-head fake mapping unpadded `input_ids` to `(o_k, c_{k+1}, flag_k)`."""

    def __init__(
        self,
        table: Mapping[tuple[int, ...], tuple[Token, Token, Token]],
        *,
        default: tuple[Token, Token, Token] = (0, 0, int(SpecialToken.FLAG_FINISHED)),
    ) -> None:
        self._table = dict(table)
        self._default = default
        self.training = True
        self.device = torch.device("cpu")

    def eval(self) -> "_LookupClsModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert lengths is not None
        batch_size = input_ids.shape[0]
        logits_o = torch.zeros(batch_size, VOCAB_SIZE)
        logits_c = torch.zeros(batch_size, VOCAB_SIZE)
        logits_f = torch.zeros(batch_size, VOCAB_SIZE)
        for b in range(batch_size):
            n = int(lengths[b].item())
            key = tuple(input_ids[b, :n].tolist())
            o_k, c_next, flag_k = self._table.get(key, self._default)
            logits_o[b, o_k] = 10.0
            logits_c[b, c_next] = 10.0
            logits_f[b, flag_k] = 10.0
        return logits_o, logits_c, logits_f


def test_evaluate_cls_rollout_cascade_oracle_diagnostic_and_batch_invariance():
    alphabet = b"0123456789"
    # Problem 0: 21 + 34 = 55 ('12' + '43' = '55', 2 columns, no carry) -> solved.
    # Problem 1: 99 + 1 = 100 ('99' + '1' = '001', 3 carry-active columns):
    #   - At k=0, model predicts o_0='0', c_1='0' (drops the carry!).
    #   - In free-running rollout, k=1 receives c_1='0' ('9' + '0' + '0' -> '9', c_2='0', FLAG_FINISHED),
    #     halting prematurely at k=1! Rollout gets 0/3 columns right and fails halt_accuracy.
    #   - Under oracle prefix, k=1 and k=2 have exact single-step rules -> oracle_step_accuracy gets 2/3 on Problem 1!
    p0 = next(
        iter_problems(bases=(10,), min_length=2, max_length=2, p_prior=1.0, seed=1)
    )
    from experiments.addition.inputs import AdditionProblem

    prob0 = AdditionProblem(alphabet=tuple(alphabet), a=tuple(b"12"), b=tuple(b"43"))
    prob1 = AdditionProblem(alphabet=tuple(alphabet), a=tuple(b"99"), b=tuple(b"1"))
    assert p0.alphabet == prob0.alphabet
    s0 = reference_add(prob0.alphabet, prob0.a, prob0.b)
    s1 = reference_add(prob1.alphabet, prob1.a, prob1.b)
    assert len(s0) == 2 and len(s1) == 3

    table: dict[tuple[int, ...], tuple[int, int, int]] = {}
    out0 = tuple(st.sum_digit for st in s0)
    for k, st in enumerate(s0):
        ex = build_cls_example(prob0.alphabet, prob0.a, prob0.b, st.carry_in, out0[:k])
        table[ex.input_ids] = (st.sum_digit, st.carry_out, int(st.flag))

    out1 = tuple(st.sum_digit for st in s1)
    for k, st in enumerate(s1):
        ex = build_cls_example(prob1.alphabet, prob1.a, prob1.b, st.carry_in, out1[:k])
        table[ex.input_ids] = (st.sum_digit, st.carry_out, int(st.flag))
    # Override k=0 of prob1 to drop carry_out ('0' instead of '1'), and script rollout k=1 with carry='0'.
    ex_p1_k0 = build_cls_example(prob1.alphabet, prob1.a, prob1.b, ord("0"), ())
    table[ex_p1_k0.input_ids] = (ord("0"), ord("0"), int(SpecialToken.FLAG_CONTINUE))
    ex_p1_rollout_k1 = build_cls_example(
        prob1.alphabet, prob1.a, prob1.b, ord("0"), (ord("0"),)
    )
    table[ex_p1_rollout_k1.input_ids] = (
        ord("9"),
        ord("0"),
        int(SpecialToken.FLAG_FINISHED),
    )

    m_b1 = evaluate_cls(
        _LookupClsModel(table),  # ty: ignore[invalid-argument-type]
        [prob0, prob1],
        num_problems=2,
        batch_size=1,
    )
    m_b3 = evaluate_cls(
        _LookupClsModel(table),  # ty: ignore[invalid-argument-type]
        [prob0, prob1],
        num_problems=2,
        batch_size=3,
    )
    assert m_b1 == m_b3
    assert m_b1.exact_match == pytest.approx(0.5)
    assert m_b1.halt_accuracy == pytest.approx(0.5)
    # Total reference steps = 2 + 3 = 5. Rollout gets 2/2 on prob0 and 0/3 on prob1 -> 2/5;
    # oracle single-step gets 2/2 on prob0 and 2/3 (k=1, k=2) on prob1 -> 4/5!
    assert m_b1.step_accuracy == pytest.approx(2 / 5)
    assert m_b1.oracle_step_accuracy == pytest.approx(4 / 5)
    assert m_b1.carry_accuracy == pytest.approx(0.0)
    assert m_b1.address_accuracy is None


def test_evaluate_and_rollout_cls_with_real_transformer_and_validation():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    problems = iter_problems(bases=(2, 4), min_length=1, max_length=3, seed=77)
    metrics = evaluate_cls(model, problems, num_problems=4, batch_size=3)
    assert 0.0 <= metrics.exact_match <= 1.0
    assert 0.0 <= metrics.step_accuracy <= 1.0
    assert 0.0 <= metrics.halt_accuracy <= 1.0

    steps = rollout_cls(model, b"01", b"10", b"1")
    assert 1 <= len(steps) <= 3

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_cls(model, problems, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_cls(model, problems, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="exhausted early"):
        evaluate_cls(model, [], num_problems=1)

    lm_model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    with pytest.raises(ValueError, match="3-head ParallelClassificationHead"):
        evaluate_cls(
            lm_model,
            iter_problems(bases=(2,), min_length=1, max_length=1, seed=1),
            num_problems=1,
        )
    with pytest.raises(ValueError, match="3-head ParallelClassificationHead"):
        rollout_cls(lm_model, b"01", b"0", b"1")
