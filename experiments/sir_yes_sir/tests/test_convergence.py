"""End-to-end CPU convergence smoke test for the `sir_yes_sir` experiment."""

from experiments.sir_yes_sir.reference import reference_step
from experiments.sir_yes_sir.variants.ar.dataset import SirYesSirArDataset
from experiments.sir_yes_sir.variants.ar.evaluation import evaluate_ar
from experiments.sir_yes_sir.variants.ar.harness import run_ar
from experiments.sir_yes_sir.variants.sm_mh.dataset import SirYesSirDataset
from experiments.sir_yes_sir.variants.sm_mh.evaluation import evaluate_sm_mh
from experiments.sir_yes_sir.variants.sm_mh.harness import run_sm_mh
from yessir.alphabet import SymbolSet
from yessir.model import Transformer
from yessir.training import seed_everything, train
from yessir.vocab import VOCAB_SIZE


def test_sir_yes_sir_sm_mh_cpu_convergence():
    """Train a tiny `Transformer` on CPU and verify >99% in-distribution exact match.

    Why `symbols=SymbolSet.DECIMAL[:3]` with `alphabet_size=3`:
    - `B = 3` is the smallest alphabet size where a model must read the in-context
      alphabet ordering to predict `(o_k, w_k)` (for `B = 2` over `{a, b}`, `o_k` is
      always the other symbol and `w_k` is always `x_k` under both `[a, b]` and
      `[b, a]`). With `B = 3`, each problem samples one of the `3! = 6` permutations
      of `('0', '1', '2')`, so for any `x_k` its successor `o_k` and wrap symbol
      `w_k` depend on the in-context alphabet order.
    - Restricting `symbols` to 3 byte tokens rather than all 240 bytes in
      `SymbolSet.ALL` keeps the untied `MultiTaskHead` sample complexity small
      enough to converge on CPU within a test run.
    """
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:3]
    alphabet_size = 3
    x_length = 2

    model = Transformer(
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=2,
        n_heads=8,
        d_ff=64,
        d_position=8,
        d_scratch=16,
    )

    train_dataset = SirYesSirDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        symbols=symbols,
        seed=42,
    )
    eval_dataset = SirYesSirDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        symbols=symbols,
        seed=999,
    )

    train(model, train_dataset, steps=500, batch_size=24, lr=8e-4)

    metrics = evaluate_sm_mh(model, eval_dataset, num_problems=100)
    assert metrics.exact_match > 0.99
    assert metrics.step_accuracy > 0.99
    assert metrics.halt_accuracy > 0.99

    # Also verify free-running `run_sm_mh` on two reversed 3-symbol permutations
    # (`b"012"` and `b"210"`), where every symbol's `o_k` and `w_k` both differ.
    for alphabet in (b"012", b"210"):
        for x in (b"01", b"12"):
            expected = [
                tuple(int(t) for t in reference_step(alphabet, x, k))
                for k in range(len(x))
            ]
            assert run_sm_mh(model, alphabet, x) == expected


def test_sir_yes_sir_ar_cpu_convergence():
    """Train a tiny `TiedIdentityHead` `Transformer` on CPU under the `ar` variant and verify >99% exact match."""
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:3]
    alphabet_size = 3
    x_length = 2

    model = Transformer(
        num_classes=None,
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

    train(model, train_dataset, steps=500, batch_size=16, lr=1e-3)

    metrics = evaluate_ar(model, eval_dataset, num_problems=100)
    assert metrics.exact_match > 0.99
    assert metrics.step_accuracy > 0.99
    assert metrics.halt_accuracy > 0.99

    for alphabet in (b"012", b"210"):
        for x in (b"01", b"12"):
            expected = [
                tuple(int(t) for t in reference_step(alphabet, x, k))
                for k in range(len(x))
            ]
            assert run_ar(model, alphabet, x) == expected
