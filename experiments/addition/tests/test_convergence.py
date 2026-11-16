"""End-to-end CPU convergence smoke tests for the `addition` experiment."""

import pytest

from experiments.addition.inputs import iter_problems
from experiments.addition.reference import reference_add
from experiments.addition.variants.cls.dataset import AdditionClsDataset
from experiments.addition.variants.cls.evaluation import evaluate_cls
from experiments.addition.variants.cls.generation import rollout_cls
from yessir.alphabet import SymbolSet
from yessir.heads import HeadType
from yessir.model import Transformer
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

    train(model, train_dataset, max_steps=600, batch_size=24, lr=1e-3)

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
