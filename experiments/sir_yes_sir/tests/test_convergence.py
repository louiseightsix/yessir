"""End-to-end CPU convergence smoke test for the `sir_yes_sir` experiment."""

import pytest

from experiments.sir_yes_sir.reference import reference_step
from experiments.sir_yes_sir.variants.ar.dataset import SirYesSirArDataset
from experiments.sir_yes_sir.variants.ar.evaluation import evaluate_ar
from experiments.sir_yes_sir.variants.ar.generation import greedy_generate_ar
from experiments.sir_yes_sir.variants.cls.dataset import SirYesSirClsDataset
from experiments.sir_yes_sir.variants.cls.evaluation import evaluate_cls
from experiments.sir_yes_sir.variants.cls.generation import greedy_generate_cls
from yessir.alphabet import SymbolSet
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything, train
from yessir.vocab import VOCAB_SIZE

# Training outcomes depend on floating-point details that differ across CPU
# architectures, so these tests are excluded from CI (see `pyproject.toml`).
pytestmark = pytest.mark.convergence


def test_sir_yes_sir_cls_cpu_convergence():
    """Train a tiny `Transformer` on CPU and verify >99% in-distribution exact match.

    Why `symbols=SymbolSet.DECIMAL[:3]` with `alphabet_size=3`:
    - `B = 3` is the smallest alphabet size where a model must read the in-context
      alphabet ordering to predict `o_k` (for `B = 2` over `{a, b}`, `o_k` is
      always the other symbol under both `[a, b]` and `[b, a]`). With `B = 3`, each
      problem samples one of the `3! = 6` permutations of `('0', '1', '2')`, so for
      any `x_k` its successor `o_k` depends on the in-context alphabet order.
    - Restricting `symbols` to 3 byte tokens rather than all 240 bytes in
      `SymbolSet.ALL` keeps the untied `ParallelClassificationHead` sample complexity
      small enough to converge on CPU within a test run.
    """
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:3]
    alphabet_size = 3
    x_length = 2

    model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE),
        n_layers=2,
        n_heads=8,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )

    train_dataset = SirYesSirClsDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        symbols=symbols,
        seed=42,
    )
    eval_dataset = SirYesSirClsDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, max_steps=500, batch_size=24, lr=8e-4)

    metrics = evaluate_cls(model, eval_dataset, num_problems=100)
    assert metrics.exact_match > 0.99
    assert metrics.step_accuracy > 0.99
    assert metrics.halt_accuracy > 0.99

    # Also verify free-running `greedy_generate_cls` on two reversed 3-symbol permutations
    # (`b"012"` and `b"210"`), where every symbol's `o_k` differs.
    for alphabet in (b"012", b"210"):
        for x in (b"01", b"12"):
            expected = [
                tuple(int(t) for t in reference_step(alphabet, x, k))
                for k in range(len(x))
            ]
            assert greedy_generate_cls(model, alphabet, x) == expected


def test_sir_yes_sir_ar_cpu_convergence():
    """Train a tiny `SubspaceLMHead` `Transformer` on CPU under the `ar` variant and verify >99% exact match."""
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:3]
    alphabet_size = 3
    x_length = 2

    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=2,
        n_heads=8,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )

    train_dataset = SirYesSirArDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        symbols=symbols,
        seed=42,
    )
    eval_dataset = SirYesSirArDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, max_steps=500, batch_size=16, lr=1e-3)

    metrics = evaluate_ar(model, eval_dataset, num_problems=100)
    assert metrics.exact_match > 0.99
    assert metrics.step_accuracy > 0.99
    assert metrics.halt_accuracy > 0.99

    for alphabet in (b"012", b"210"):
        for x in (b"01", b"12"):
            expected = [
                *(reference_step(alphabet, x, k)[0] for k in range(len(x))),
                int(SpecialToken.EOS),
            ]
            assert greedy_generate_ar(model, alphabet, x) == expected
