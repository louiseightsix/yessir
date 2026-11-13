"""Tests for `experiments/sir_yes_sir/variants/*/harness.py` and `experiments/sir_yes_sir/variants/*/evaluation.py`."""

import itertools
from collections.abc import Sequence

import pytest
import torch

from experiments.sir_yes_sir.evaluation import EvalMetrics
from experiments.sir_yes_sir.reference import reference_step
from experiments.sir_yes_sir.variants.ar.dataset import SirYesSirArDataset
from experiments.sir_yes_sir.variants.ar.evaluation import evaluate_ar
from experiments.sir_yes_sir.variants.ar.harness import parse_ar_trace, rollout_ar
from experiments.sir_yes_sir.variants.ar.inputs import StepOrder, build_ar_example
from experiments.sir_yes_sir.variants.sm_mh.dataset import SirYesSirSmMhDataset
from experiments.sir_yes_sir.variants.sm_mh.evaluation import evaluate_sm_mh
from experiments.sir_yes_sir.variants.sm_mh.harness import rollout_sm_mh
from experiments.sir_yes_sir.variants.sm_mh.inputs import build_sm_example
from yessir.dataset import LabelledExample
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


def _parse_sm_input_ids(
    seq: Sequence[Token],
) -> tuple[tuple[Token, ...], tuple[Token, ...], tuple[Token, ...]]:
    """Extract `(alphabet, x, out_history)` from a state-machine `input_ids` sequence."""
    sep_indices = [i for i, tok in enumerate(seq) if tok == SpecialToken.SEP]
    alphabet = tuple(seq[1 : sep_indices[0]])
    x = tuple(seq[sep_indices[0] + 1 : sep_indices[1]])
    out_history = tuple(seq[sep_indices[1] + 1 : -1])
    return alphabet, x, out_history


class _ScriptedModel:
    """Fake model returning scripted `(o_k, w_k, flag_k)` argmax predictions per step and recording inputs."""

    def __init__(self, scripted_steps: Sequence[Sequence[Token]]) -> None:
        self._steps = [tuple(step) for step in scripted_steps]
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")
        self.seen_examples: list[Example] = []

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
            o_k, w_k, flag_k = self._steps[self._cursor]
            self._cursor += 1
            logits_o[b, o_k] = 10.0
            logits_w[b, w_k] = 10.0
            logits_f[b, flag_k] = 10.0

        return logits_o, logits_w, logits_f


class _WrapBlindModel:
    """Input-dependent fake model that follows `reference_step` except when `x_k` is the last alphabet symbol
    or when `output_history` already contains a wrong symbol."""

    def __init__(self) -> None:
        self.training = True
        self.device = torch.device("cpu")

    def eval(self) -> "_WrapBlindModel":
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
            seq = input_ids[b, : int(lengths[b].item())].tolist()
            alphabet, x, out_history = _parse_sm_input_ids(seq)
            k = len(out_history)

            o_k, w_k, flag_k = reference_step(alphabet, x, k)
            ref_history = tuple(reference_step(alphabet, x, j)[0] for j in range(k))

            # Fail whenever x_k wraps around (x_k == alphabet[-1]) or prior history is already wrong
            if x[k] == alphabet[-1] or out_history != ref_history:
                o_k = alphabet[1]

            logits_o[b, o_k] = 10.0
            logits_w[b, w_k] = 10.0
            logits_f[b, int(flag_k)] = 10.0

        return logits_o, logits_w, logits_f


def test_rollout_sm_mh_feeds_back_models_own_emitted_outputs():
    alphabet = b"0123"
    x = b"130"
    # Ground-truth o_0 is '2' and o_1 is '0'; script wrong in-alphabet symbols '3' then '1'
    # so the model's history diverges from the reference history.
    scripted = [
        (ord("3"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), ord("1"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
    ]
    model = _ScriptedModel(scripted)

    emitted = rollout_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert model.training is False
    assert emitted == scripted
    assert model.seen_examples == [
        build_sm_example(alphabet, x, ()),
        build_sm_example(alphabet, x, (ord("3"),)),
        build_sm_example(alphabet, x, (ord("3"), ord("1"))),
    ]


def test_rollout_sm_mh_stops_early_on_premature_flag_finished():
    alphabet = b"0123"
    x = b"130"
    scripted = [
        (ord("2"), ord("0"), int(SpecialToken.FLAG_FINISHED)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    emitted = rollout_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert emitted == [scripted[0]]
    assert len(model.seen_examples) == 1


def test_rollout_sm_mh_stops_early_on_out_of_alphabet_primary_symbol():
    alphabet = b"0123"
    x = b"130"
    scripted = [
        (ord("9"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_FINISHED)),
    ]
    model = _ScriptedModel(scripted)

    emitted = rollout_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert emitted == [scripted[0]]
    assert len(model.seen_examples) == 1


def test_rollout_sm_mh_stops_at_len_x_when_flag_finished_never_emitted():
    alphabet = b"0123"
    x = b"13"
    scripted = [
        (ord("2"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    emitted = rollout_sm_mh(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert len(emitted) == 2
    assert emitted == scripted


def test_evaluate_sm_mh_computes_exact_match_step_and_halt_accuracy():
    dataset = SirYesSirSmMhDataset(alphabet_size=4, x_length=2, seed=7)
    # 2 problems of length 2 = 4 steps total
    items = list(itertools.islice(dataset, 4))

    # Problem 0 (steps 0, 1): both steps completely right -> problem 0 is EM
    # Problem 1 (steps 2, 3):
    #   step 2: wrong o_k, right w_k, right flag_k (step_ok=False, halt_ok=True)
    #   step 3: right o_k, right w_k, wrong flag_k (step_ok=False, halt_ok=False)
    scripted = [
        items[0].labels,
        items[1].labels,
        (
            (items[2].labels[0] + 1) % VOCAB_SIZE,
            items[2].labels[1],
            items[2].labels[2],
        ),
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


def test_teacher_forced_and_free_running_exact_match_agree_and_batch_size_invariance():
    num_problems = 20
    alphabet_size = 4
    x_length = 3
    seed = 123

    dataset = SirYesSirSmMhDataset(
        alphabet_size=alphabet_size, x_length=x_length, seed=seed
    )
    model = _WrapBlindModel()

    metrics_b1 = evaluate_sm_mh(
        model,  # ty: ignore[invalid-argument-type]
        dataset,
        num_problems=num_problems,
        batch_size=1,
    )
    metrics_b7 = evaluate_sm_mh(
        model,  # ty: ignore[invalid-argument-type]
        dataset,
        num_problems=num_problems,
        batch_size=7,
    )
    assert metrics_b1 == metrics_b7
    assert 0.0 < metrics_b1.exact_match < 1.0

    # Run free-running `rollout_sm_mh` on the exact same `num_problems` problems extracted from `dataset`
    all_items = list(itertools.islice(dataset, num_problems * x_length))
    free_running_em_count = 0
    for prob_idx in range(num_problems):
        prob_items = all_items[prob_idx * x_length : (prob_idx + 1) * x_length]
        alphabet, x, out_history = _parse_sm_input_ids(prob_items[0].example.input_ids)
        assert out_history == ()
        expected_steps = [item.labels for item in prob_items]
        emitted_steps = rollout_sm_mh(
            model,  # ty: ignore[invalid-argument-type]
            alphabet,
            x,
        )
        if emitted_steps == expected_steps:
            free_running_em_count += 1

    assert free_running_em_count / num_problems == pytest.approx(metrics_b1.exact_match)


def test_evaluate_sm_mh_runs_real_transformer_end_to_end():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.MULTITASK,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = SirYesSirSmMhDataset(alphabet_size=4, x_length=3, seed=99)
    metrics = evaluate_sm_mh(model, dataset, num_problems=3, batch_size=4)
    assert 0.0 <= metrics.exact_match <= 1.0
    assert 0.0 <= metrics.step_accuracy <= 1.0
    assert 0.0 <= metrics.halt_accuracy <= 1.0


def test_evaluate_sm_mh_validation_errors():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.MULTITASK,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = SirYesSirSmMhDataset(alphabet_size=4, x_length=2, seed=0)

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
    unfinished = [
        LabelledExample(
            example=ex_k0,
            labels=(ord("1"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        )
    ]
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


class _ScriptedArModel:
    """Fake sequence-level model returning scripted next-token argmax predictions."""

    def __init__(self, scripted_tokens: Sequence[Token]) -> None:
        self._tokens = list(scripted_tokens)
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")
        self.seen_examples: list[Example] = []

    def eval(self) -> "_ScriptedArModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
        for b in range(batch_size):
            n = int(lengths[b].item()) if lengths is not None else seq_len
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


def test_parse_ar_trace_and_rollout_ar():
    alphabet = b"0123"
    x = b"13"

    # Script 2 steps in WRAP_FIRST order: (w_0='0', o_0='2', CONTINUE), (w_1='1', o_1='0', FINISHED)
    scripted = [
        ord("0"),
        ord("2"),
        int(SpecialToken.FLAG_CONTINUE),
        ord("1"),
        ord("0"),
        int(SpecialToken.FLAG_FINISHED),
    ]
    model = _ScriptedArModel(scripted)
    steps = rollout_ar(
        model,  # ty: ignore[invalid-argument-type]
        alphabet,
        x,
        step_order=StepOrder.WRAP_FIRST,
    )
    assert model.training is False
    # Returned tuples are canonical (o_k, w_k, flag_k)
    assert steps == [
        (ord("2"), ord("0"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), ord("1"), int(SpecialToken.FLAG_FINISHED)),
    ]
    assert model.seen_examples[0] == build_ar_example(alphabet, x, ())
    assert model.seen_examples[1] == build_ar_example(alphabet, x, (ord("0"),))

    # Premature FLAG_FINISHED at the end of step 0 stops generation after 3 tokens
    early_model = _ScriptedArModel(
        [ord("2"), ord("0"), int(SpecialToken.FLAG_FINISHED), ord("0")]
    )
    early_steps = rollout_ar(
        early_model,  # ty: ignore[invalid-argument-type]
        alphabet,
        x,
        step_order=StepOrder.OUT_FIRST,
    )
    assert early_steps == [(ord("2"), ord("0"), int(SpecialToken.FLAG_FINISHED))]
    assert len(early_model.seen_examples) == 3

    # Trailing incomplete step tokens are ignored by parse_ar_trace
    assert parse_ar_trace([ord("2"), ord("0")]) == []


def test_evaluate_ar_end_to_end_and_validation():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.TIED_IDENTITY,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = SirYesSirArDataset(alphabet_size=4, x_length=2, seed=99)
    metrics = evaluate_ar(model, dataset, num_problems=3, batch_size=2)
    assert 0.0 <= metrics.exact_match <= 1.0
    assert 0.0 <= metrics.step_accuracy <= 1.0
    assert 0.0 <= metrics.halt_accuracy <= 1.0

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_ar(model, dataset, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_ar(model, dataset, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="dataset exhausted early"):
        evaluate_ar(model, [], num_problems=1)
