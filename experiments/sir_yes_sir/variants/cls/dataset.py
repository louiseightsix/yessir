"""Data generator (`SirYesSirClsDataset`) for the `cls` variant of `sir_yes_sir`."""

import argparse
from collections.abc import Iterator, Sequence

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    add_common_dataset_args,
    iter_problems,
    resolve_symbol_pool,
)
from experiments.sir_yes_sir.variants.cls.inputs import build_cls_labelled_example
from yessir.alphabet import SymbolSet
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import TokenSequence


class SirYesSirClsDataset:
    """Infinite deterministic stream of single-pass parallel-classification `LabelledExample`s for `sir_yes_sir`.

    Each yielded `LabelledExample` is one complete problem `(alphabet, x)` with
    `(o_0, ..., o_{L-2})` appended to the prompt prefix and per-position `(o_k, flag_k)`
    labels at query indices `P - 1 .. P + L - 2` (and `(IGNORE_INDEX, IGNORE_INDEX)`
    at earlier prompt positions).
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 4,
        x_length: int = 3,
        symbols: TokenSequence = SymbolSet.ALL,
        embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
        seed: int = 42,
    ) -> None:
        self._symbol_pool = resolve_symbol_pool(alphabet_size, x_length, symbols)
        self._alphabet_size = alphabet_size
        self._x_length = x_length
        self._embedding_mode = EmbeddingMode(embedding_mode)
        self._seed = seed

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield one sequence-labelled `LabelledExample` per problem from `self._seed`."""
        for alphabet, x in iter_problems(
            alphabet_size=self._alphabet_size,
            x_length=self._x_length,
            symbol_pool=self._symbol_pool,
            seed=self._seed,
        ):
            yield build_cls_labelled_example(
                alphabet,
                x,
                embedding_mode=self._embedding_mode,
            )


def _build_from_args(args: argparse.Namespace) -> SirYesSirClsDataset:
    return SirYesSirClsDataset(
        alphabet_size=args.alphabet_size,
        x_length=args.x_length,
        embedding_mode=args.embedding_mode,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Generate, print, or dump `sir_yes_sir` (`cls`) training examples."""
    run_inspect_cli(
        description="Generate, print, or dump sir_yes_sir (cls) training examples.",
        add_dataset_args=add_common_dataset_args,
        build_dataset=_build_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
