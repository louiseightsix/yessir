"""Tests for the `step_fork` variant (`inputs`, `dataset`, `evaluation`, `generation`)."""

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import torch

from experiments.addition.inputs import (
    AdditionToken,
    EmbeddingMode,
    TokenType,
    iter_problems,
)
from experiments.addition.reference import reference_add
from experiments.addition.variants.step_fork.dataset import AdditionStepForkDataset
from experiments.addition.variants.step_fork.dataset import (
    main as step_fork_dataset_main,
)
from experiments.addition.variants.step_fork.evaluation import evaluate_step_fork
from experiments.addition.variants.step_fork.generation import rollout_step_fork
from experiments.addition.variants.step_fork.inputs import (
    STEP_FORK_QUERY_TOKENS,
    build_step_fork_branch_example,
    build_step_fork_examples,
    build_step_fork_step_labelled_examples,
)
from yessir.dataset import IGNORE_INDEX
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


class _ScriptedStepForkModel:
    """Fake `SubspaceLMHead` model returning scripted argmax tokens at each row's `lengths - 1`."""

    def __init__(self, scripted_tokens: Sequence[Token]) -> None:
        self._tokens = list(scripted_tokens)
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")
        self.seen_batches: list[list[Example]] = []

    def eval(self) -> "_ScriptedStepForkModel":
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
        batch_examples: list[Example] = []

        for b in range(batch_size):
            n = int(lengths[b].item())
            batch_examples.append(
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
            # Put a bogus logit at any padded tail (`seq_len - 1` when `n < seq_len`)
            # so padding-sensitive code that indexes `-1` instead of `lengths - 1` fails.
            if n < seq_len:
                logits[b, seq_len - 1, int(SpecialToken.PAD)] = 100.0
            logits[b, n - 1, tok] = 10.0

        self.seen_batches.append(batch_examples)
        return logits


def test_step_fork_spec_example_1_and_position_coupling():
    """Verify `spec/variants/step_fork.md` §1–§2 Example 1 (`987 + 25 = 1012` in base 10)."""
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

    for k, (carry_in, out_prefix, expected_targets) in enumerate(expected_steps):
        branches = build_step_fork_step_labelled_examples(alphabet, a, b, k)
        assert len(branches) == 3

        for q_tok, target, labelled in zip(
            STEP_FORK_QUERY_TOKENS, expected_targets, branches, strict=True
        ):
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
                int(q_tok),
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
                TokenType.SPECIAL,
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
                *range(k),
                k,
            )
            assert ex.input_ids == expected_ids
            assert ex.token_type_ids == expected_tt
            assert ex.position_ids == expected_pos
            # Trailing query token [S]/[C]/[F] sits at position_id == k; o_j sits at j (0..k-1).
            assert ex.position_ids[-1] == k
            assert labelled.labels == (IGNORE_INDEX,) * (len(ex) - 1) + (target,)

    with pytest.raises(IndexError, match="Step index k=4"):
        build_step_fork_step_labelled_examples(alphabet, a, b, 4)


def test_step_fork_spec_example_2_branch_builder_and_embedding_modes():
    """Verify `spec/variants/step_fork.md` §2 Example 2 (`alphabet = b'q#7'`, `5 + 1 = 6`)."""
    alphabet = b"q#7"
    a = b"7#"
    b = b"#"
    cont = int(SpecialToken.FLAG_CONTINUE)
    fin = int(SpecialToken.FLAG_FINISHED)

    s0, c0, f0 = build_step_fork_step_labelled_examples(alphabet, a, b, 0)
    assert (s0.labels[-1], c0.labels[-1], f0.labels[-1]) == (ord("q"), ord("#"), cont)

    s1, c1, f1 = build_step_fork_step_labelled_examples(alphabet, a, b, 1)
    assert (s1.labels[-1], c1.labels[-1], f1.labels[-1]) == (ord("7"), ord("q"), fin)

    single_c1 = build_step_fork_branch_example(
        alphabet, a, b, ord("#"), b"q", query_token=AdditionToken.QUERY_CARRY
    )
    assert single_c1 == c1.example

    with pytest.raises(ValueError, match="query_token must be one of"):
        build_step_fork_branch_example(
            alphabet,
            a,
            b,
            ord("#"),
            b"q",
            query_token=SpecialToken.SEP,  # ty: ignore[invalid-argument-type]
        )

    ex_tt_nope, _, _ = build_step_fork_examples(
        alphabet, a, b, ord("#"), b"q", embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE
    )
    assert ex_tt_nope.token_type_ids is not None
    assert ex_tt_nope.position_ids is None

    ex_nope, _, _ = build_step_fork_examples(
        alphabet, a, b, ord("#"), b"q", embedding_mode=EmbeddingMode.NOPE
    )
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_addition_step_fork_dataset_branch_grouping_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(ValueError, match="p_terminal must be in 0.0..1.0"):
        AdditionStepForkDataset(p_terminal=1.1)

    # Slicing `3 * batch_size` consecutive items yields `[S], [C], [F]` triples per transition.
    ds = AdditionStepForkDataset(bases=(3,), min_length=2, max_length=2, seed=11)
    batch_items = list(itertools.islice(ds, 3 * 4))
    for step_idx in range(4):
        triple = batch_items[3 * step_idx : 3 * step_idx + 3]
        assert tuple(it.example.input_ids[-1] for it in triple) == tuple(
            int(q) for q in STEP_FORK_QUERY_TOKENS
        )
        # All 3 branches in a triple share identical prefix tokens prior to the query token.
        assert (
            triple[0].example.input_ids[:-1]
            == triple[1].example.input_ids[:-1]
            == triple[2].example.input_ids[:-1]
        )

    ds_all = AdditionStepForkDataset(
        bases=(3,), min_length=2, max_length=2, all_steps=True, seed=11
    )
    first_prob = next(iter_problems(bases=(3,), min_length=2, max_length=2, seed=11))
    oracle = reference_add(first_prob.alphabet, first_prob.a, first_prob.b)
    all_items = list(itertools.islice(ds_all, 3 * len(oracle)))
    expected_targets = [
        tok for s in oracle for tok in (s.sum_digit, s.carry_out, int(s.flag))
    ]
    assert [it.labels[-1] for it in all_items] == expected_targets

    dump_path = tmp_path / "step_fork_dump.jsonl"
    step_fork_dataset_main(
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


def test_rollout_step_fork_batches_three_branches_per_step():
    alphabet = b"012"
    a = b"21"
    b = b"1"
    # Step 0 predicts out-of-alphabet/special bytes `(255, SEP, FLAG_CONTINUE)`;
    # step 1 predicts `('1', '0', FLAG_FINISHED)` and stops immediately.
    scripted_tokens = [
        255,
        int(SpecialToken.SEP),
        int(SpecialToken.FLAG_CONTINUE),
        ord("1"),
        ord("0"),
        int(SpecialToken.FLAG_FINISHED),
    ]
    model = _ScriptedStepForkModel(scripted_tokens)
    emitted = rollout_step_fork(model, alphabet, a, b)  # ty: ignore[invalid-argument-type]
    assert model.training is False
    assert emitted == [
        (255, int(SpecialToken.SEP), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
    ]
    assert len(model.seen_batches) == 2
    assert model.seen_batches[0] == list(
        build_step_fork_examples(alphabet, a, b, ord("0"), ())
    )
    assert model.seen_batches[1] == list(
        build_step_fork_examples(alphabet, a, b, int(SpecialToken.SEP), (255,))
    )

    # Caps at `max(len(a), len(b)) + 1 == 3` steps when `FLAG_FINISHED` is never emitted.
    never_halt = _ScriptedStepForkModel(
        [ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)] * 4
    )
    capped = rollout_step_fork(never_halt, alphabet, a, b)  # ty: ignore[invalid-argument-type]
    assert len(capped) == 3


class _LookupLMModel:
    """Stateless `SubspaceLMHead` fake mapping unpadded prefixes to next-token argmax."""

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

    def eval(self) -> "_LookupLMModel":
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
            if n < seq_len:
                logits[b, seq_len - 1, int(SpecialToken.PAD)] = 100.0
            key = tuple(input_ids[b, :n].tolist())
            logits[b, n - 1, self._table.get(key, self._default)] = 10.0
        return logits


def test_evaluate_step_fork_rollout_cascade_padding_and_batch_invariance():
    from experiments.addition.inputs import AdditionProblem

    # Mix problems of different operand lengths so padded tail positions get poisoned with PAD.
    prob0 = AdditionProblem(alphabet=tuple(b"01"), a=tuple(b"101"), b=tuple(b"010"))
    prob1 = AdditionProblem(
        alphabet=tuple(b"0123456789"), a=tuple(b"99"), b=tuple(b"1")
    )
    s0 = reference_add(prob0.alphabet, prob0.a, prob0.b)
    s1 = reference_add(prob1.alphabet, prob1.a, prob1.b)

    table: dict[tuple[int, ...], int] = {}
    out0 = tuple(st.sum_digit for st in s0)
    for k, st in enumerate(s0):
        for ex, target in zip(
            build_step_fork_examples(
                prob0.alphabet, prob0.a, prob0.b, st.carry_in, out0[:k]
            ),
            (st.sum_digit, st.carry_out, int(st.flag)),
            strict=True,
        ):
            table[ex.input_ids] = target

    out1 = tuple(st.sum_digit for st in s1)
    for k, st in enumerate(s1):
        for ex, target in zip(
            build_step_fork_examples(
                prob1.alphabet, prob1.a, prob1.b, st.carry_in, out1[:k]
            ),
            (st.sum_digit, st.carry_out, int(st.flag)),
            strict=True,
        ):
            table[ex.input_ids] = target

    # Drop carry at k=0 of prob1 ('0' instead of '1'), causing rollout cascade & early halt at k=1.
    ex_s0, ex_c0, ex_f0 = build_step_fork_examples(
        prob1.alphabet, prob1.a, prob1.b, ord("0"), ()
    )
    table[ex_s0.input_ids] = ord("0")
    table[ex_c0.input_ids] = ord("0")
    table[ex_f0.input_ids] = int(SpecialToken.FLAG_CONTINUE)
    for ex, target in zip(
        build_step_fork_examples(
            prob1.alphabet, prob1.a, prob1.b, ord("0"), (ord("0"),)
        ),
        (ord("9"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
        strict=True,
    ):
        table[ex.input_ids] = target

    m_b1 = evaluate_step_fork(
        _LookupLMModel(table),  # ty: ignore[invalid-argument-type]
        [prob0, prob1],
        num_problems=2,
        batch_size=1,
    )
    m_b5 = evaluate_step_fork(
        _LookupLMModel(table),  # ty: ignore[invalid-argument-type]
        [prob0, prob1],
        num_problems=2,
        batch_size=5,
    )
    assert m_b1 == m_b5
    assert m_b1.exact_match == pytest.approx(0.5)
    assert m_b1.halt_accuracy == pytest.approx(0.5)
    total_steps = len(s0) + len(s1)
    assert m_b1.step_accuracy == pytest.approx(len(s0) / total_steps)
    assert m_b1.oracle_step_accuracy == pytest.approx((total_steps - 1) / total_steps)


def test_evaluate_and_rollout_step_fork_with_real_transformer_and_validation():
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
    metrics = evaluate_step_fork(model, problems, num_problems=4, batch_size=2)
    assert 0.0 <= metrics.exact_match <= 1.0
    assert 0.0 <= metrics.step_accuracy <= 1.0
    assert 0.0 <= metrics.halt_accuracy <= 1.0

    steps = rollout_step_fork(model, b"01", b"10", b"1")
    assert 1 <= len(steps) <= 3

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_step_fork(model, problems, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_step_fork(model, problems, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="exhausted early"):
        evaluate_step_fork(model, [], num_problems=1)

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
        evaluate_step_fork(
            cls_model,
            iter_problems(bases=(2,), min_length=1, max_length=1, seed=1),
            num_problems=1,
        )
    with pytest.raises(ValueError, match="SubspaceLMHead"):
        rollout_step_fork(cls_model, b"01", b"0", b"1")
    assert AdditionStepForkDataset.BRANCHES_PER_STEP == 3


@pytest.mark.parametrize("embedding_mode", list(EmbeddingMode))
def test_partially_trained_step_fork_transformer_exact_match_matches_rollout(
    embedding_mode: EmbeddingMode,
):
    from yessir.training import train

    seed_everything(7)
    use_tt = embedding_mode != EmbeddingMode.NOPE
    d_pos = 8 if embedding_mode == EmbeddingMode.TOKEN_TYPE_PE else 0
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=2,
        n_heads=4,
        d_ff=32,
        use_token_type=use_tt,
        d_position=d_pos,
        d_scratch=16,
    )
    train(
        model,
        AdditionStepForkDataset(
            bases=(2, 3),
            min_length=1,
            max_length=1,
            p_prior=1.0,
            embedding_mode=embedding_mode,
            seed=7,
        ),
        max_steps=35,
        batch_size=24,
        lr=1e-3,
    )
    problems = list(
        itertools.islice(
            iter_problems(
                bases=(2, 3), min_length=1, max_length=2, p_prior=1.0, seed=99
            ),
            30,
        )
    )
    metrics = evaluate_step_fork(
        model, problems, num_problems=30, batch_size=7, embedding_mode=embedding_mode
    )
    assert 0.0 < metrics.exact_match < 1.0
    assert metrics == evaluate_step_fork(
        model, problems, num_problems=30, batch_size=1, embedding_mode=embedding_mode
    )

    rollout_matches = 0
    for p in problems:
        expected = [
            (st.sum_digit, st.carry_out, int(st.flag))
            for st in reference_add(p.alphabet, p.a, p.b)
        ]
        single_rollout = rollout_step_fork(
            model, p.alphabet, p.a, p.b, embedding_mode=embedding_mode
        )
        single_metrics = evaluate_step_fork(
            model, [p], num_problems=1, batch_size=1, embedding_mode=embedding_mode
        )
        is_match = single_rollout == expected
        assert single_metrics.exact_match == (1.0 if is_match else 0.0)
        if is_match:
            rollout_matches += 1
    assert rollout_matches / len(problems) == pytest.approx(metrics.exact_match)
