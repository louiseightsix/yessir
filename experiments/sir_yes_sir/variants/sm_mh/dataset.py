"""Data generator (`SirYesSirDataset`) for the `sm_mh` variant of `sir_yes_sir`."""

import argparse
from collections.abc import Iterator, Sequence

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    add_common_dataset_args,
    iter_problems,
    validate_stream_params,
)
from experiments.sir_yes_sir.reference import reference_step
from experiments.sir_yes_sir.variants.sm_mh.inputs import build_sm_example
from yessir.alphabet import SymbolSet
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import Token, TokenSequence


class SirYesSirDataset:
    """Infinite deterministic stream of state-machine `LabelledExample`s for `sir_yes_sir`.

    Each problem samples a fresh alphabet of size `alphabet_size` from `symbols`
    (defaulting to `SymbolSet.ALL`) and a random input sequence `x` of length
    `x_length` over that alphabet, then yields one `LabelledExample` for each
    step `k = 0 .. x_length - 1`.
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
        self._symbol_pool = validate_stream_params(alphabet_size, x_length, symbols)
        self._alphabet_size = alphabet_size
        self._x_length = x_length
        self._embedding_mode = embedding_mode
        self._seed = seed

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield `LabelledExample`s deterministically from `self._seed`."""
        for alphabet, x in iter_problems(
            alphabet_size=self._alphabet_size,
            x_length=self._x_length,
            symbol_pool=self._symbol_pool,
            seed=self._seed,
        ):
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


def _build_from_args(args: argparse.Namespace) -> SirYesSirDataset:
    return SirYesSirDataset(
        alphabet_size=args.alphabet_size,
        x_length=args.x_length,
        embedding_mode=args.embedding_mode,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Generate, print, or dump `sir_yes_sir` (`sm_mh`) training examples."""
    run_inspect_cli(
        description="Generate, print, or dump sir_yes_sir (sm_mh) training examples.",
        add_dataset_args=add_common_dataset_args,
        build_dataset=_build_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
