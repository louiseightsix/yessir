"""End-to-end CPU convergence smoke tests for the `addition` experiment."""

import pytest

from experiments.addition.inputs import StepOrder, iter_problems
from experiments.addition.reference import reference_add
from experiments.addition.variants.addressed_ar.dataset import (
    AdditionAddressedArDataset,
)
from experiments.addition.variants.addressed_ar.evaluation import evaluate_addressed_ar
from experiments.addition.variants.addressed_ar.generation import (
    parse_addressed_ar_trace,
    rollout_addressed_ar,
)
from experiments.addition.variants.addressed_ar.inputs import (
    build_addressed_ar_trace_tokens,
)
from experiments.addition.variants.addressed_step_ar.dataset import (
    AdditionAddressedStepArDataset,
)
from experiments.addition.variants.addressed_step_ar.evaluation import (
    evaluate_addressed_step_ar,
)
from experiments.addition.variants.addressed_step_ar.generation import (
    rollout_addressed_step_ar,
)
from experiments.addition.variants.ar.dataset import AdditionArDataset
from experiments.addition.variants.ar.evaluation import evaluate_ar
from experiments.addition.variants.ar.generation import rollout_ar
from experiments.addition.variants.ar.inputs import build_ar_trace_tokens
from experiments.addition.variants.cls.dataset import AdditionClsDataset
from experiments.addition.variants.cls.evaluation import evaluate_cls
from experiments.addition.variants.cls.generation import rollout_cls
from experiments.addition.variants.step_ar.dataset import AdditionStepArDataset
from experiments.addition.variants.step_ar.evaluation import evaluate_step_ar
from experiments.addition.variants.step_ar.generation import rollout_step_ar
from experiments.addition.variants.step_fork.dataset import AdditionStepForkDataset
from experiments.addition.variants.step_fork.evaluation import evaluate_step_fork
from experiments.addition.variants.step_fork.generation import rollout_step_fork
from yessir.alphabet import SymbolSet
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything, train
from yessir.vocab import VOCAB_SIZE

# Training outcomes depend on floating-point details that differ across CPU
# architectures, so these tests are excluded from CI (see `pyproject.toml`).
pytestmark = pytest.mark.convergence


def test_addition_cls_cpu_convergence():
    """Train a tiny `Transformer` on CPU under `cls` and verify >98% exact match.

    Why `bases=(2,)` over `symbols=SymbolSet.DECIMAL[:2]` (`min_length=1, max_length=2`):
    - Unlike cyclic successor (where `B = 2` swaps `'0' <-> '1'` identically under
      both `[0, 1]` and `[1, 0]`), binary addition is asymmetric (`s_0 + s_0 = s_0`
      carry `s_0` vs `s_1 + s_1 = s_0` carry `s_1`). Each problem samples either
      `b"01"` (`'0' = 0, '1' = 1`) or `b"10"` (`'1' = 0, '0' = 1`), so every byte's
      role as additive identity vs carry generator depends on the in-context alphabet.
    - Restricting `symbols` to 2 byte tokens keeps the untied 3-head
      `ParallelClassificationHead` fast enough to converge on CPU in ~25 seconds
      while exercising variable lengths (`1` vs `2`), multi-column carry chains,
      MSD carry-out (`k = K`), and halting.
    """
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:2]
    bases = (2,)
    min_length = 1
    max_length = 2

    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=2,
        n_heads=8,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )

    train_dataset = AdditionClsDataset(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        all_steps=True,
        seed=42,
    )
    eval_problems = iter_problems(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, max_steps=750, batch_size=24, lr=8e-4)

    metrics = evaluate_cls(model, eval_problems, num_problems=100)
    assert metrics.exact_match > 0.98
    assert metrics.step_accuracy > 0.99
    assert metrics.halt_accuracy > 0.99

    # Verify free-running `rollout_cls` on both reversed 2-symbol alphabets
    # (`b"01"` where '0'=0,'1'=1, and `b"10"` where '1'=0,'0'=1), including
    # a 2-column carry chain (`11_2 + 1_2 = 100_2`) that overflows into column K.
    for alphabet in (b"01", b"10"):
        s0, s1 = alphabet[0], alphabet[1]
        for a, b in (
            (bytes([s1, s0]), bytes([s1])),
            (bytes([s1, s1]), bytes([s1])),
        ):
            expected = [
                (step.sum_digit, step.carry_out, int(step.flag))
                for step in reference_add(alphabet, a, b)
            ]
            assert rollout_cls(model, alphabet, a, b) == expected


def test_addition_step_fork_cpu_convergence():
    """Train a tiny `SubspaceLMHead` `Transformer` on CPU under `step_fork` and verify >=98% exact match.

    Why `n_layers=3` (vs `n_layers=2` for `cls`):
    - With the parameter-free `SubspaceLMHead` (`RMSNorm(x)[:, :, :256]`), the
      trailing query token (`[S]`/`[C]`/`[F]`) needs three attention hops to write
      the target byte into coordinates `0..255`: (1) fetch `(a_k, b_k, c_k)` at
      `position_id == k`, (2) match them against `alphabet` to compute the target
      digit/carry index, and (3) attend back to `alphabet` to copy `s_{idx}`.
    """
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:2]
    bases = (2,)
    min_length = 1
    max_length = 2

    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=3,
        n_heads=8,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )

    train_dataset = AdditionStepForkDataset(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        all_steps=True,
        seed=42,
    )
    eval_problems = iter_problems(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        seed=999,
    )

    # `batch_size = 3 * 12 = 36` collates `[S], [C], [F]` branches for 12 column transitions per step.
    train(model, train_dataset, max_steps=900, batch_size=3 * 12, lr=8e-4)

    metrics = evaluate_step_fork(model, eval_problems, num_problems=100)
    assert metrics.exact_match >= 0.98
    assert metrics.step_accuracy >= 0.99
    assert metrics.halt_accuracy >= 0.99

    for alphabet in (b"01", b"10"):
        s0, s1 = alphabet[0], alphabet[1]
        for a, b in (
            (bytes([s1, s0]), bytes([s1])),
            (bytes([s1, s1]), bytes([s1])),
        ):
            expected = [
                (step.sum_digit, step.carry_out, int(step.flag))
                for step in reference_add(alphabet, a, b)
            ]
            assert rollout_step_fork(model, alphabet, a, b) == expected


def test_addition_step_ar_cpu_convergence():
    """Train a tiny `SubspaceLMHead` `Transformer` on CPU under `step_ar` (`carry_first`) and verify >=98% exact match."""
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:2]
    bases = (2,)
    min_length = 1
    max_length = 2
    step_order = StepOrder.CARRY_FIRST

    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=3,
        n_heads=8,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )

    train_dataset = AdditionStepArDataset(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        step_order=step_order,
        seed=42,
    )
    eval_problems = iter_problems(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, max_steps=1000, batch_size=24, lr=6e-4)

    metrics = evaluate_step_ar(
        model, eval_problems, num_problems=100, step_order=step_order
    )
    assert metrics.exact_match >= 0.98
    assert metrics.step_accuracy >= 0.99
    assert metrics.halt_accuracy >= 0.99

    for alphabet in (b"01", b"10"):
        s0, s1 = alphabet[0], alphabet[1]
        for a, b in (
            (bytes([s1, s0]), bytes([s1])),
            (bytes([s1, s1]), bytes([s1])),
        ):
            expected = [
                (step.sum_digit, step.carry_out, int(step.flag))
                for step in reference_add(alphabet, a, b)
            ]
            assert (
                rollout_step_ar(model, alphabet, a, b, step_order=step_order)
                == expected
            )


def test_addition_ar_cpu_convergence():
    """Train a tiny pure-LM (`input_ids`-only) `SubspaceLMHead` `Transformer` on CPU under full-trace `ar` (`carry_first`) and verify >=98% exact match."""
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:2]
    bases = (2,)
    min_length = 1
    max_length = 2
    step_order = StepOrder.CARRY_FIRST

    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=4,
        n_heads=8,
        d_ff=64,
        use_token_type=False,
        d_position=0,
        d_scratch=40,
    )

    train_dataset = AdditionArDataset(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        step_order=step_order,
        seed=42,
    )
    eval_problems = iter_problems(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, max_steps=1500, batch_size=24, lr=5e-4)

    metrics = evaluate_ar(model, eval_problems, num_problems=100, step_order=step_order)
    assert metrics.exact_match >= 0.98
    assert metrics.step_accuracy >= 0.99
    assert metrics.halt_accuracy >= 0.99

    for alphabet in (b"01", b"10"):
        s0, s1 = alphabet[0], alphabet[1]
        for a, b in (
            (bytes([s1, s0]), bytes([s1])),
            (bytes([s1, s1]), bytes([s1])),
        ):
            steps = reference_add(alphabet, a, b)
            expected = [
                *build_ar_trace_tokens(steps, step_order=step_order),
                int(SpecialToken.EOS),
            ]
            assert rollout_ar(model, alphabet, a, b, step_order=step_order) == expected


def test_addition_addressed_step_ar_cpu_convergence():
    """Train a tiny `SubspaceLMHead` `Transformer` on CPU under `addressed_step_ar` (default `nope`, `carry_first`, `use_query_tokens=True`) and verify >=98% exact match."""
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:2]
    bases = (2,)
    min_length = 1
    max_length = 2
    step_order = StepOrder.CARRY_FIRST
    use_query_tokens = True

    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=4,
        n_heads=8,
        d_ff=64,
        use_token_type=False,
        d_position=0,
        d_scratch=40,
    )

    train_dataset = AdditionAddressedStepArDataset(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
        seed=42,
    )
    eval_problems = iter_problems(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, max_steps=2000, batch_size=32, lr=3e-4)

    metrics = evaluate_addressed_step_ar(
        model,
        eval_problems,
        num_problems=100,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
        embedding_mode=AdditionAddressedStepArDataset()._embedding_mode,
    )
    assert metrics.exact_match >= 0.98
    assert metrics.step_accuracy >= 0.99
    assert metrics.halt_accuracy >= 0.99

    for alphabet in (b"01", b"10"):
        s0, s1 = alphabet[0], alphabet[1]
        for a, b in (
            (bytes([s1, s0]), bytes([s1])),
            (bytes([s1, s1]), bytes([s1])),
        ):
            expected = [
                (step.sum_digit, step.carry_out, int(step.flag))
                for step in reference_add(alphabet, a, b)
            ]
            assert (
                rollout_addressed_step_ar(
                    model,
                    alphabet,
                    a,
                    b,
                    step_order=step_order,
                    use_query_tokens=use_query_tokens,
                    embedding_mode=AdditionAddressedStepArDataset()._embedding_mode,
                )
                == expected
            )


def test_addition_addressed_ar_cpu_convergence():
    """Train a tiny pure-LM (`input_ids`-only) `SubspaceLMHead` `Transformer` on CPU under full-trace `addressed_ar` (`carry_first`) and verify >=98% exact match."""
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:2]
    bases = (2,)
    min_length = 1
    max_length = 2
    step_order = StepOrder.CARRY_FIRST

    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=4,
        n_heads=8,
        d_ff=64,
        use_token_type=False,
        d_position=0,
        d_scratch=40,
    )

    train_dataset = AdditionAddressedArDataset(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        step_order=step_order,
        seed=42,
    )
    eval_problems = iter_problems(
        bases=bases,
        min_length=min_length,
        max_length=max_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, max_steps=900, batch_size=24, lr=5e-4)

    metrics = evaluate_addressed_ar(
        model, eval_problems, num_problems=100, step_order=step_order
    )
    assert metrics.exact_match >= 0.98
    assert metrics.step_accuracy >= 0.99
    assert metrics.halt_accuracy >= 0.99

    for alphabet in (b"01", b"10"):
        s0, s1 = alphabet[0], alphabet[1]
        for a, b in (
            (bytes([s1, s0]), bytes([s1])),
            (bytes([s1, s1]), bytes([s1])),
        ):
            steps = reference_add(alphabet, a, b)
            expected = [
                *build_addressed_ar_trace_tokens(
                    steps, alphabet, step_order=step_order
                ),
                int(SpecialToken.EOS),
            ]
            emitted = rollout_addressed_ar(model, alphabet, a, b, step_order=step_order)
            assert emitted == expected
            parsed, well_formed = parse_addressed_ar_trace(
                emitted, step_order=step_order
            )
            assert well_formed is True
            assert [(col.sum_digit, col.carry_out) for col in parsed] == [
                (st.sum_digit, st.carry_out) for st in steps
            ]
