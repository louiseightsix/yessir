"""End-to-end CPU convergence smoke test for the `sir_yes_sir` experiment."""

from experiments.sir_yes_sir.dataset import SirYesSirDataset
from experiments.sir_yes_sir.evaluation import evaluate_sm_mh
from experiments.sir_yes_sir.harness import run_sm_mh
from experiments.sir_yes_sir.reference import reference_step
from yessir.alphabet import SymbolSet
from yessir.model import Transformer
from yessir.training import seed_everything, train
from yessir.vocab import VOCAB_SIZE


def test_sir_yes_sir_sm_mh_cpu_convergence():
    """Train a tiny `Transformer` on CPU and verify >99% in-distribution exact match.

    Why `symbols=SymbolSet.DECIMAL[:2]` with `alphabet_size=2`:
    - Each problem randomly permutes the alphabet into either `['0', '1']` or `['1', '0']`.
      In `['0', '1']`, `'0'` is `s_0` (no-wrap symbol) and `'1'` is `s_1` (wrap symbol);
      in `['1', '0']`, their roles are reversed. An alphabet-blind model cannot tell which
      symbol is `s_0` vs. `s_1` and can score at most 50% step accuracy (12.5% exact match
      over `x_length = 3`). Achieving >99% exact match therefore proves the model attends
      to the in-context alphabet and binds symbols per instance.
    - Restricting the symbol pool to 2 byte tokens instead of all 240 bytes in
      `SymbolSet.ALL` lets an untied `MultiTaskHead` + `SubspaceEmbedding` converge in
      150 CPU steps (~4.5s) rather than requiring thousands of steps to populate 240
      independent output-head rows.
    """
    seed_everything(42)

    symbols = SymbolSet.DECIMAL[:2]
    alphabet_size = 2
    x_length = 3

    model = Transformer(
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=2,
        n_heads=4,
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

    train(model, train_dataset, steps=150, batch_size=24, lr=1e-3)

    metrics = evaluate_sm_mh(model, eval_dataset, num_problems=100)
    assert metrics.exact_match > 0.99
    assert metrics.step_accuracy > 0.99
    assert metrics.halt_accuracy > 0.99

    # Also verify free-running `run_sm_mh` on both alphabet permutations
    for alphabet in (b"01", b"10"):
        x = b"101"
        expected = [
            tuple(int(t) for t in reference_step(alphabet, x, k)) for k in range(len(x))
        ]
        assert run_sm_mh(model, alphabet, x) == expected
