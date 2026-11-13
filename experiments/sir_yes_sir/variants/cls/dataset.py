"""Data generator (`SirYesSirClsDataset`) for the `cls` variant of `sir_yes_sir`."""

import argparse
import random
from collections.abc import Iterator, Sequence

from experiments.sir_yes_sir.config import ClsTrainingMode
from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    add_common_dataset_args,
    iter_problems,
    resolve_symbol_pool,
)
from experiments.sir_yes_sir.variants.cls.inputs import (
    build_cls_labelled_example,
    build_cls_step_labelled_example,
)
from yessir.alphabet import SymbolSet
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import TokenSequence


class SirYesSirClsDataset:
    """Infinite deterministic stream of parallel-classification `LabelledExample`s for `sir_yes_sir`.

    How examples are emitted per problem `(alphabet, x)` (`training_mode`):
    - `ClsTrainingMode.SINGLE_PASS` (default): yields one sequence-labelled
      `LabelledExample` per problem (`build_cls_labelled_example`) supervising all
      `L` query positions in a single teacher-forced forward pass.
    - `ClsTrainingMode.ALL_STEPS`: yields `L` separate single-step `LabelledExample`s
      per problem (`k = 0 .. L - 1`) reading the final position `lengths - 1`.
    - `ClsTrainingMode.RANDOM_STEP`: yields one randomly sampled single-step
      `LabelledExample` (`k` in `0 .. L - 1`) per problem.
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 4,
        x_length: int = 3,
        symbols: TokenSequence = SymbolSet.ALL,
        permute_alphabet: bool = True,
        training_mode: ClsTrainingMode = ClsTrainingMode.SINGLE_PASS,
        embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
        seed: int = 42,
    ) -> None:
        self._symbol_pool = resolve_symbol_pool(alphabet_size, x_length, symbols)
        self._alphabet_size = alphabet_size
        self._x_length = x_length
        self._permute_alphabet = permute_alphabet
        self._training_mode = ClsTrainingMode(training_mode)
        self._embedding_mode = EmbeddingMode(embedding_mode)
        self._seed = seed

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield `LabelledExample`s deterministically from `self._seed`."""
        step_rng = random.Random(self._seed)
        for alphabet, x in iter_problems(
            alphabet_size=self._alphabet_size,
            x_length=self._x_length,
            symbol_pool=self._symbol_pool,
            seed=self._seed,
            permute_alphabet=self._permute_alphabet,
        ):
            if self._training_mode == ClsTrainingMode.SINGLE_PASS:
                yield build_cls_labelled_example(
                    alphabet,
                    x,
                    embedding_mode=self._embedding_mode,
                )
            elif self._training_mode == ClsTrainingMode.ALL_STEPS:
                for k in range(self._x_length):
                    yield build_cls_step_labelled_example(
                        alphabet,
                        x,
                        k,
                        embedding_mode=self._embedding_mode,
                    )
            else:
                k = step_rng.randrange(self._x_length)
                yield build_cls_step_labelled_example(
                    alphabet,
                    x,
                    k,
                    embedding_mode=self._embedding_mode,
                )


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    add_common_dataset_args(parser)
    parser.add_argument(
        "--training-mode",
        type=ClsTrainingMode,
        choices=tuple(ClsTrainingMode),
        default=ClsTrainingMode.SINGLE_PASS,
        help="Training example construction mode for cls (default: single_pass)",
    )


def _build_from_args(args: argparse.Namespace) -> SirYesSirClsDataset:
    return SirYesSirClsDataset(
        alphabet_size=args.alphabet_size,
        x_length=args.x_length,
        permute_alphabet=args.permute_alphabet,
        training_mode=args.training_mode,
        embedding_mode=args.embedding_mode,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Generate, print, or dump `sir_yes_sir` (`cls`) training examples."""
    run_inspect_cli(
        description="Generate, print, or dump sir_yes_sir (cls) training examples.",
        add_dataset_args=_add_cli_args,
        build_dataset=_build_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
