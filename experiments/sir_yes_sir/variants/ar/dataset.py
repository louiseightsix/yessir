"""Data generator (`SirYesSirArDataset`) for the `ar` variant of `sir_yes_sir`."""

import argparse
import random
from collections.abc import Iterator, Sequence

from experiments.sir_yes_sir.inputs import EmbeddingMode
from experiments.sir_yes_sir.variants.ar.inputs import (
    StepOrder,
    build_ar_labelled_example,
)
from yessir.alphabet import SymbolSet, create_alphabet
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import TokenSequence


class SirYesSirArDataset:
    """Infinite deterministic stream of autoregressive (`ar`) `LabelledExample`s for `sir_yes_sir`.

    Each yielded `LabelledExample` is one complete problem `(alphabet, x)` with
    `trace[:-1]` appended to the prompt prefix and next-token `labels` masked with
    `IGNORE_INDEX = -100` prior to the second `[SEP]`.
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 4,
        x_length: int = 3,
        symbols: TokenSequence = SymbolSet.ALL,
        step_order: StepOrder = StepOrder.OUT_FIRST,
        embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
        seed: int = 42,
    ) -> None:
        symbol_pool = create_alphabet(symbols)
        if not (2 <= alphabet_size <= len(symbol_pool)):
            raise ValueError(
                f"alphabet_size must be in 2..{len(symbol_pool)}, got {alphabet_size}"
            )
        if x_length < 1:
            raise ValueError(f"x_length must be >= 1, got {x_length}")

        self._alphabet_size = alphabet_size
        self._x_length = x_length
        self._symbol_pool = symbol_pool
        self._step_order = StepOrder(step_order)
        self._embedding_mode = EmbeddingMode(embedding_mode)
        self._seed = seed

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield one sequence-labelled `LabelledExample` per problem from `self._seed`."""
        rng = random.Random(self._seed)
        while True:
            alphabet = create_alphabet(
                self._symbol_pool, sample_size=self._alphabet_size, rng=rng
            )
            x = tuple(rng.choice(alphabet) for _ in range(self._x_length))
            yield build_ar_labelled_example(
                alphabet,
                x,
                step_order=self._step_order,
                embedding_mode=self._embedding_mode,
            )


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-B",
        "--alphabet-size",
        type=int,
        default=4,
        help="Alphabet size B (default: 4)",
    )
    parser.add_argument(
        "-L",
        "--x-length",
        type=int,
        default=3,
        help="Length L of input sequence x (default: 3)",
    )
    parser.add_argument(
        "--step-order",
        type=StepOrder,
        choices=tuple(StepOrder),
        default=StepOrder.OUT_FIRST,
        help="Within-step token emission order (default: out_first)",
    )
    parser.add_argument(
        "--embedding-mode",
        type=EmbeddingMode,
        choices=tuple(EmbeddingMode),
        default=EmbeddingMode.TOKEN_TYPE_PE,
        help="Embedding mode for token_type_ids and position_ids (default: token_type_pe)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )


def _build_from_args(args: argparse.Namespace) -> SirYesSirArDataset:
    return SirYesSirArDataset(
        alphabet_size=args.alphabet_size,
        x_length=args.x_length,
        step_order=args.step_order,
        embedding_mode=args.embedding_mode,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Generate, print, or dump `sir_yes_sir` (`ar`) training examples."""
    run_inspect_cli(
        description="Generate, print, or dump sir_yes_sir (ar) training examples.",
        add_dataset_args=_add_cli_args,
        build_dataset=_build_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
