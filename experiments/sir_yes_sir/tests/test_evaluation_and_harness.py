"""Tests for `experiments/sir_yes_sir/harness.py` and `experiments/sir_yes_sir/evaluation.py`."""

import itertools
from collections.abc import Sequence

import pytest
import torch

from experiments.sir_yes_sir.dataset import SirYesSirDataset
from experiments.sir_yes_sir.evaluation import EvalMetrics, evaluate_sm_mh
from experiments.sir_yes_sir.harness import run_sm_mh
from experiments.sir_yes_sir.inputs import build_sm_example
from experiments.sir_yes_sir.reference import reference_step
from yessir.dataset import LabelledExample
from yessir.example import Example
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


class _ScriptedModel:
    """Fake model returning scripted `(o_k, w_k, flag_k)` argmax predictions per step."""

    def __init__(self, scripted_steps: Sequence[Sequence[Token]]) -> None:
        self._steps = [tuple(step) for step in scripted_steps]
        self._cursor = 0
        self.training = True

    def eval(self) -> "_ScriptedModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_ids.size(0)
        logits_o = torch.zeros(batch_size, VOCAB_SIZE)
        logits_w = torch.zeros(batch_size, VOCAB_SIZE)
        logits_f = torch.zeros(batch_size, VOCAB_SIZE)

        for b in range(batch_size):
            o_k, w_k, flag_k = self._steps[self._cursor]
            self._cursor += 1
            logits_o[b, o_k] = 10.0
            logits_w[b, w_k] = 10.0
            logits_f[b, flag_k] = 10.0

        return logits_o, logits_w, logits_f


def test_run_sm_mh_reproduces_spec_example_1_when_predictions_match_oracle():
    alphabet = b"0123"
    x = b"130"
    oracle_steps = [
        (ord("2"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
    ]
    model = _ScriptedModel(oracle_steps)

    emitted = run_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert model.training is False
    assert emitted == oracle_steps


def test_run_sm_mh_stops_early_on_premature_flag_finished():
    alphabet = b"0123"
    x = b"130"
    scripted = [
        (ord("2"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    emitted = run_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert emitted == [scripted[0]]


def test_run_sm_mh_stops_early_on_out_of_alphabet_primary_symbol():
    alphabet = b"0123"
    x = b"130"
    scripted = [
        (ord("9"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_FINISHED)),
    ]
    model = _ScriptedModel(scripted)

    emitted = run_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert emitted == [scripted[0]]


def test_run_sm_mh_stops_at_len_x_when_flag_finished_never_emitted():
    alphabet = b"0123"
    x = b"13"
    scripted = [
        (ord("2"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    emitted = run_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert len(emitted) == 2
    assert emitted == scripted


def test_evaluate_sm_mh_computes_exact_match_step_and_halt_accuracy():
    dataset = SirYesSirDataset(alphabet_size=4, x_length=2, seed=7)
    # 2 problems of length 2 = 4 steps total
    items = list(itertools.islice(dataset, 4))

    # Problem 0 (steps 0, 1): both steps completely right -> problem 0 is EM
    # Problem 1 (steps 2, 3):
    #   step 2: wrong o_k, right w_k, right flag_k (step_ok=False, halt_ok=True)
    #   step 3: right o_k, right w_k, wrong flag_k (step_ok=False, halt_ok=False)
    scripted = [
        items[0].labels,
        items[1].labels,
        ((items[2].labels[0] + 1) % 240, items[2].labels[1], items[2].labels[2]),
        (items[3].labels[0], items[3].labels[1], int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    metrics = evaluate_sm_mh(
        model,  # ty: ignore[invalid-argument-type]
        items,
        num_problems=2,
        batch_size=2,
    )
    assert model.training is False
    assert metrics == EvalMetrics(
        exact_match=0.5,
        step_accuracy=0.5,
        halt_accuracy=0.75,
    )


def test_evaluate_sm_mh_with_real_transformer_and_batch_size_invariance():
    seed_everything(0)
    model = Transformer(
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = SirYesSirDataset(alphabet_size=4, x_length=3, seed=99)

    m_batched = evaluate_sm_mh(model, dataset, num_problems=4, batch_size=5)
    m_single = evaluate_sm_mh(model, dataset, num_problems=4, batch_size=1)
    assert m_batched == m_single
    assert 0.0 <= m_batched.exact_match <= 1.0
    assert 0.0 <= m_batched.step_accuracy <= 1.0
    assert 0.0 <= m_batched.halt_accuracy <= 1.0


def test_evaluate_sm_mh_validation_errors():
    seed_everything(0)
    model = Transformer(
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = SirYesSirDataset(alphabet_size=4, x_length=2, seed=0)

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_sm_mh(model, dataset, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_sm_mh(model, dataset, num_problems=1, batch_size=0)

    # Mid-problem stream (consumed step k=0 of problem 0, so next item is k=1)
    stream = iter(dataset)
    next(stream)
    with pytest.raises(ValueError, match="stream must start at step k=0"):
        evaluate_sm_mh(model, stream, num_problems=1)

    # Wrong number of labels
    ex_k0 = build_sm_example(b"01", b"0", ())
    with pytest.raises(ValueError, match="expected 3 labels"):
        evaluate_sm_mh(
            model,
            [LabelledExample(example=ex_k0, labels=(ord("1"), ord("0")))],
            num_problems=1,
        )

    # Invalid halting flag label
    with pytest.raises(ValueError, match="expected flag label to be FLAG_CONTINUE"):
        evaluate_sm_mh(
            model,
            [LabelledExample(example=ex_k0, labels=(ord("1"), ord("0"), 0x05))],
            num_problems=1,
        )

    # Dataset exhausted before num_problems complete problems
    o_0, w_0, f_0 = reference_step(b"01", b"01", 0)
    unfinished = [LabelledExample(example=ex_k0, labels=(o_0, w_0, int(f_0)))]
    with pytest.raises(ValueError, match="dataset exhausted early"):
        evaluate_sm_mh(model, unfinished, num_problems=1)

    # Short/malformed Example at start of problem
    short_ex = Example(input_ids=(int(SpecialToken.STEP),))
    with pytest.raises(ValueError, match="stream must start at step k=0"):
        evaluate_sm_mh(
            model,
            [
                LabelledExample(
                    example=short_ex,
                    labels=(ord("1"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
                )
            ],
            num_problems=1,
        )
