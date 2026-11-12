"""Data generator for the `sir_yes_sir` experiment (`SirYesSirDataset`)."""

import argparse
import random
from collections.abc import Iterator, Sequence

from experiments.sir_yes_sir.inputs import EmbeddingMode, build_sm_example
from experiments.sir_yes_sir.reference import reference_step
from yessir.alphabet import SymbolSet, create_alphabet
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import Token


class SirYesSirDataset:
    """Infinite deterministic stream of state-machine `LabelledExample`s for `sir_yes_sir`.

    Each problem samples a fresh alphabet of size `alphabet_size` from `SymbolSet.ALL`
    and a random input sequence `x` of length `x_length` over that alphabet, then
    yields one `LabelledExample` for each step `k = 0 .. x_length - 1`.
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 4,
        x_length: int = 3,
        embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
        seed: int = 42,
    ) -> None:
        if not (2 <= alphabet_size <= len(SymbolSet.ALL)):
            raise ValueError(
                f"alphabet_size must be in 2..{len(SymbolSet.ALL)}, got {alphabet_size}"
            )
        if x_length < 1:
            raise ValueError(f"x_length must be >= 1, got {x_length}")

        self._alphabet_size = alphabet_size
        self._x_length = x_length
        self._embedding_mode = embedding_mode
        self._seed = seed

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield `LabelledExample`s deterministically from `self._seed`."""
        rng = random.Random(self._seed)
        while True:
            alphabet = create_alphabet(
                SymbolSet.ALL, sample_size=self._alphabet_size, rng=rng
            )
            x = tuple(rng.choice(alphabet) for _ in range(self._x_length))

            output_history: list[Token] = []
            for k in range(self._x_length):
                o_k, w_k, flag_k = reference_step(alphabet, x, k)
                example = build_sm_example(
                    alphabet,
                    x,
                    output_history,
                    embedding_mode=self._embedding_mode,
                )
                yield LabelledExample(
                    example=example,
                    labels=(o_k, w_k, int(flag_k)),
                )
                output_history.append(o_k)


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


def _build_from_args(args: argparse.Namespace) -> SirYesSirDataset:
    return SirYesSirDataset(
        alphabet_size=args.alphabet_size,
        x_length=args.x_length,
        embedding_mode=args.embedding_mode,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Generate, print, or dump `sir_yes_sir` training examples."""
    run_inspect_cli(
        description="Generate, print, or dump sir_yes_sir training examples.",
        add_dataset_args=_add_cli_args,
        build_dataset=_build_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
