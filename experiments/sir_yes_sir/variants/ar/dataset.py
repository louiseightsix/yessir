"""Data generator (`SirYesSirArDataset`) for the `ar` variant of `sir_yes_sir`."""

import argparse
from collections.abc import Iterator, Sequence

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    add_common_dataset_args,
    iter_problems,
    validate_stream_params,
)
from experiments.sir_yes_sir.variants.ar.inputs import (
    StepOrder,
    build_ar_labelled_example,
)
from yessir.alphabet import SymbolSet
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
        self._symbol_pool = validate_stream_params(alphabet_size, x_length, symbols)
        self._alphabet_size = alphabet_size
        self._x_length = x_length
        self._step_order = StepOrder(step_order)
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
            yield build_ar_labelled_example(
                alphabet,
                x,
                step_order=self._step_order,
                embedding_mode=self._embedding_mode,
            )


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    add_common_dataset_args(parser)
    parser.add_argument(
        "--step-order",
        type=StepOrder,
        choices=tuple(StepOrder),
        default=StepOrder.OUT_FIRST,
        help="Within-step token emission order (default: out_first)",
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
