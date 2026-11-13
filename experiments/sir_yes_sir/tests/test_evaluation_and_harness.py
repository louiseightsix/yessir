"""Tests for `experiments/sir_yes_sir/variants/*/harness.py` and `experiments/sir_yes_sir/variants/*/evaluation.py`."""

import itertools
from collections.abc import Sequence

import pytest
import torch

from experiments.sir_yes_sir.evaluation import EvalMetrics
from experiments.sir_yes_sir.inputs import build_example
from experiments.sir_yes_sir.reference import reference_step
from experiments.sir_yes_sir.variants.ar.dataset import SirYesSirArDataset
from experiments.sir_yes_sir.variants.ar.evaluation import evaluate_ar
from experiments.sir_yes_sir.variants.ar.harness import rollout_ar
from experiments.sir_yes_sir.variants.cls.dataset import SirYesSirClsDataset
from experiments.sir_yes_sir.variants.cls.evaluation import evaluate_cls
from experiments.sir_yes_sir.variants.cls.harness import rollout_cls
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


def _parse_example_input_ids(
    seq: Sequence[Token],
) -> tuple[tuple[Token, ...], tuple[Token, ...], tuple[Token, ...]]:
    """Extract `(alphabet, x, output)` from a shared `sir_yes_sir` `input_ids` sequence."""
    sep_indices = [i for i, tok in enumerate(seq) if tok == SpecialToken.SEP]
    alphabet = tuple(seq[1 : sep_indices[0]])
    x = tuple(seq[sep_indices[0] + 1 : sep_indices[1]])
    output = tuple(seq[sep_indices[1] + 1 :])
    return alphabet, x, output


class _ScriptedModel:
    """Fake model returning scripted `(o_k, flag_k)` argmax predictions per step and recording inputs."""

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
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if lengths is None:
            logits_o = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
            logits_f = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
            for b in range(batch_size):
                seq = input_ids[b].tolist()
                alphabet, x, _ = _parse_example_input_ids(seq)
                prefix_len = 1 + len(alphabet) + 1 + len(x) + 1
                for k in range(len(x)):
                    o_k, flag_k = self._steps[self._cursor]
                    self._cursor += 1
                    logits_o[b, prefix_len - 1 + k, o_k] = 10.0
                    logits_f[b, prefix_len - 1 + k, flag_k] = 10.0
            return logits_o, logits_f

        logits_o = torch.zeros(batch_size, VOCAB_SIZE)
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
            o_k, flag_k = self._steps[self._cursor]
            self._cursor += 1
            logits_o[b, o_k] = 10.0
            logits_f[b, flag_k] = 10.0

        return logits_o, logits_f


class _WrapBlindModel:
    """Causal fake model that follows `reference_step` at query position `P - 1 + k`
    except when `x_k` is the last alphabet symbol (`alphabet[-1]`) or when the prefix
    `output[:k]` already contains a wrong symbol."""

    def __init__(self) -> None:
        self.training = True
        self.device = torch.device("cpu")

    def eval(self) -> "_WrapBlindModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if lengths is None:
            logits_o = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
            logits_f = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
            for b in range(batch_size):
                seq = input_ids[b].tolist()
                alphabet, x, _ = _parse_example_input_ids(seq)
                prefix_len = 1 + len(alphabet) + 1 + len(x) + 1
                for k in range(len(x)):
                    o_k, flag_k = reference_step(alphabet, x, k)
                    ref_history = tuple(
                        reference_step(alphabet, x, j)[0] for j in range(k)
                    )
                    seen_history = tuple(seq[prefix_len : prefix_len + k])
                    if x[k] == alphabet[-1] or seen_history != ref_history:
                        o_k = alphabet[1]
                    logits_o[b, prefix_len - 1 + k, o_k] = 10.0
                    logits_f[b, prefix_len - 1 + k, int(flag_k)] = 10.0
            return logits_o, logits_f

        logits_o = torch.zeros(batch_size, VOCAB_SIZE)
        logits_f = torch.zeros(batch_size, VOCAB_SIZE)

        for b in range(batch_size):
            seq = input_ids[b, : int(lengths[b].item())].tolist()
            alphabet, x, output = _parse_example_input_ids(seq)
            k = len(output)

            o_k, flag_k = reference_step(alphabet, x, k)
            ref_history = tuple(reference_step(alphabet, x, j)[0] for j in range(k))

            # Fail whenever x_k wraps around (x_k == alphabet[-1]) or prior history is already wrong
            if x[k] == alphabet[-1] or output != ref_history:
                o_k = alphabet[1]

            logits_o[b, o_k] = 10.0
            logits_f[b, int(flag_k)] = 10.0

        return logits_o, logits_f


def test_rollout_cls_feeds_back_models_own_emitted_outputs():
    alphabet = b"0123"
    x = b"130"
    # Ground-truth o_0 is '2' and o_1 is '0'; script wrong symbols '3' then '1'
    # so the model's history diverges from the reference history.
    scripted = [
        (ord("3"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("1"), int(SpecialToken.FLAG_FINISHED)),
    ]
    model = _ScriptedModel(scripted)

    emitted = rollout_cls(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert model.training is False
    assert emitted == scripted
    assert model.seen_examples == [
        build_example(alphabet, x, ()),
        build_example(alphabet, x, (ord("3"),)),
        build_example(alphabet, x, (ord("3"), ord("1"))),
    ]


def test_rollout_cls_stops_early_on_premature_flag_finished():
    alphabet = b"0123"
    x = b"130"
    scripted = [
        (ord("2"), int(SpecialToken.FLAG_FINISHED)),
        (ord("0"), int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    emitted = rollout_cls(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert emitted == [scripted[0]]
    assert len(model.seen_examples) == 1


def test_rollout_cls_stops_at_len_x_when_flag_finished_never_emitted():
    alphabet = b"0123"
    x = b"13"
    scripted = [
        (ord("2"), int(SpecialToken.FLAG_CONTINUE)),
        (ord("0"), int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    emitted = rollout_cls(model, alphabet, x)  # ty: ignore[invalid-argument-type]
    assert len(emitted) == 2
    assert emitted == scripted


def test_evaluate_cls_computes_exact_match_step_and_halt_accuracy():
    dataset = SirYesSirClsDataset(alphabet_size=4, x_length=2, seed=7)
    # 2 problems of length L = 2 (each item is one problem with 2 non-ignored steps)
    items = list(itertools.islice(dataset, 2))
    p0_steps = [
        s for s in items[0].labels if isinstance(s, tuple) and s[0] != IGNORE_INDEX
    ]
    p1_steps = [
        s for s in items[1].labels if isinstance(s, tuple) and s[0] != IGNORE_INDEX
    ]

    # Problem 0 (steps 0, 1): both steps completely right -> EM=True, steps=2/2, halt=True
    # Problem 1 (steps 0, 1):
    #   step 0: wrong o_0, right flag_0 (step_ok=False)
    #   step 1: right o_1, wrong flag_1 (step_ok=True, halt_ok=False for problem 1)
    scripted = [
        p0_steps[0],
        p0_steps[1],
        ((p1_steps[0][0] + 1) % VOCAB_SIZE, p1_steps[0][1]),
        (p1_steps[1][0], int(SpecialToken.FLAG_CONTINUE)),
    ]
    model = _ScriptedModel(scripted)

    metrics = evaluate_cls(
        model,  # ty: ignore[invalid-argument-type]
        items,
        num_problems=2,
        batch_size=2,
    )
    assert model.training is False
    assert metrics == EvalMetrics(
        exact_match=0.5,
        step_accuracy=0.75,
        halt_accuracy=0.5,
    )


def test_teacher_forced_and_free_running_exact_match_agree_and_batch_size_invariance():
    num_problems = 20
    alphabet_size = 4
    x_length = 3
    seed = 123

    dataset = SirYesSirClsDataset(
        alphabet_size=alphabet_size, x_length=x_length, seed=seed
    )
    model = _WrapBlindModel()

    metrics_b1 = evaluate_cls(
        model,  # ty: ignore[invalid-argument-type]
        dataset,
        num_problems=num_problems,
        batch_size=1,
    )
    metrics_b7 = evaluate_cls(
        model,  # ty: ignore[invalid-argument-type]
        dataset,
        num_problems=num_problems,
        batch_size=7,
    )
    assert metrics_b1 == metrics_b7
    assert 0.0 < metrics_b1.exact_match < 1.0

    # Run free-running `rollout_cls` on the exact same `num_problems` problems
    all_items = list(itertools.islice(dataset, num_problems))
    free_running_em_count = 0
    for item in all_items:
        alphabet, x, _ = _parse_example_input_ids(item.example.input_ids)
        expected_steps = [
            s for s in item.labels if isinstance(s, tuple) and s[0] != IGNORE_INDEX
        ]
        emitted_steps = rollout_cls(
            model,  # ty: ignore[invalid-argument-type]
            alphabet,
            x,
        )
        if emitted_steps == expected_steps:
            free_running_em_count += 1

    assert free_running_em_count / num_problems == pytest.approx(metrics_b1.exact_match)


def test_evaluate_cls_runs_real_transformer_end_to_end():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = SirYesSirClsDataset(alphabet_size=4, x_length=3, seed=99)
    metrics = evaluate_cls(model, dataset, num_problems=3, batch_size=4)
    assert 0.0 <= metrics.exact_match <= 1.0
    assert 0.0 <= metrics.step_accuracy <= 1.0
    assert 0.0 <= metrics.halt_accuracy <= 1.0


def test_evaluate_cls_validation_errors():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        d_position=8,
        d_scratch=0,
    )
    dataset = SirYesSirClsDataset(alphabet_size=4, x_length=2, seed=0)

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_cls(model, dataset, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_cls(model, dataset, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="dataset exhausted early"):
        evaluate_cls(model, [], num_problems=1)

    ex = build_example(b"01", b"0")
    # Label length mismatch
    with pytest.raises(ValueError, match="expected sequence-length labels"):
        evaluate_cls(
            model,
            [
                LabelledExample(
                    example=ex, labels=((ord("1"), int(SpecialToken.FLAG_FINISHED)),)
                )
            ],
            num_problems=1,
        )

    # 1D labels instead of 2-tuple per position
    with pytest.raises(ValueError, match="expected per-position 2-tuple labels"):
        evaluate_cls(
            model,
            [LabelledExample(example=ex, labels=(IGNORE_INDEX,) * len(ex))],
            num_problems=1,
        )

    # All positions ignored
    with pytest.raises(ValueError, match="at least 1 non-ignored step"):
        evaluate_cls(
            model,
            [
                LabelledExample(
                    example=ex, labels=((IGNORE_INDEX, IGNORE_INDEX),) * len(ex)
                )
            ],
            num_problems=1,
        )

    # Final step flag not FLAG_FINISHED
    bad_final_labels = ((IGNORE_INDEX, IGNORE_INDEX),) * (len(ex) - 1) + (
        (ord("1"), int(SpecialToken.FLAG_CONTINUE)),
    )
    with pytest.raises(
        ValueError, match="expected final cls step flag to be FLAG_FINISHED"
    ):
        evaluate_cls(
            model,
            [LabelledExample(example=ex, labels=bad_final_labels)],
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


def test_rollout_ar_stops_on_eos_and_caps_at_len_x_plus_one():
    alphabet = b"0123"
    x = b"13"

    # Script 2 successor symbols ('2', '0') followed by EOS
    scripted = [ord("2"), ord("0"), int(SpecialToken.EOS)]
    model = _ScriptedArModel(scripted)
    emitted = rollout_ar(
        model,  # ty: ignore[invalid-argument-type]
        alphabet,
        x,
    )
    assert model.training is False
    assert emitted == scripted
    assert model.seen_examples == [
        build_example(alphabet, x, ()),
        build_example(alphabet, x, (ord("2"),)),
        build_example(alphabet, x, (ord("2"), ord("0"))),
    ]

    # Premature EOS at step 0 stops generation immediately
    early_model = _ScriptedArModel([int(SpecialToken.EOS), ord("2")])
    early_emitted = rollout_ar(
        early_model,  # ty: ignore[invalid-argument-type]
        alphabet,
        x,
    )
    assert early_emitted == [int(SpecialToken.EOS)]
    assert len(early_model.seen_examples) == 1

    # When EOS is never emitted, stops after L + 1 tokens
    never_halt_model = _ScriptedArModel([ord("2"), ord("0"), ord("1"), ord("3")])
    never_halt_emitted = rollout_ar(
        never_halt_model,  # ty: ignore[invalid-argument-type]
        alphabet,
        x,
    )
    assert never_halt_emitted == [ord("2"), ord("0"), ord("1")]
    assert len(never_halt_model.seen_examples) == 3


def test_evaluate_ar_metrics_and_validation():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
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

    # Verify exact_match, step_accuracy (per L step), and halt_accuracy (per problem)
    # across 2 problems of L = 2 (targets of length L + 1 = 3):
    # - Problem 0: perfect [o_0, o_1, EOS] -> EM=True, steps=2/2, halt=True
    # - Problem 1: [o_0, wrong_non_eos, wrong_non_eos] -> EM=False, steps=1/2, halt=False
    items = list(itertools.islice(dataset, 2))
    t0 = [t for t in items[0].labels if isinstance(t, int) and t != IGNORE_INDEX]
    t1 = [t for t in items[1].labels if isinstance(t, int) and t != IGNORE_INDEX]

    class _BatchArModel:
        training = True
        device = torch.device("cpu")

        def eval(self):
            self.training = False
            return self

        def __call__(
            self, input_ids, lengths=None, token_type_ids=None, position_ids=None
        ):
            b_sz, seq_len = input_ids.shape
            logits = torch.zeros(b_sz, seq_len, VOCAB_SIZE)
            # Supervised positions are the last 3 positions of each unpadded sequence
            preds_per_prob = [
                t0,
                [t1[0], (t1[1] + 1) % 4, ord("0")],
            ]
            for b in range(b_sz):
                for idx, tok in enumerate(preds_per_prob[b]):
                    logits[b, seq_len - 3 + idx, tok] = 10.0
            return logits

    custom_metrics = evaluate_ar(
        _BatchArModel(),  # ty: ignore[invalid-argument-type]
        items,
        num_problems=2,
        batch_size=2,
    )
    assert custom_metrics == EvalMetrics(
        exact_match=0.5,
        step_accuracy=0.75,
        halt_accuracy=0.5,
    )

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_ar(model, dataset, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_ar(model, dataset, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="dataset exhausted early"):
        evaluate_ar(model, [], num_problems=1)
