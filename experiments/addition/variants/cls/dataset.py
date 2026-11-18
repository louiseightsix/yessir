"""Training data generator (`AdditionClsDataset`) for the `cls` variant of `addition`."""

import argparse
from collections.abc import Iterator, Sequence

from experiments.addition.inputs import (
    ADDITION_TOKEN_LABELS,
    EmbeddingMode,
    add_common_dataset_args,
    iter_problem_steps,
    validate_sampling_params,
)
from experiments.addition.variants.cls.inputs import build_cls_step_labelled_example
from yessir.alphabet import SymbolSet, resolve_symbol_pool_slice
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import TokenSequence


class AdditionClsDataset:
    """Infinite deterministic stream of single-step `cls` `LabelledExample`s.

    By default (`all_steps=False`), each item draws a fresh problem `(alphabet, a, b)`
    and samples a single column step `k` (choosing the terminal step `k = K` with
    probability `p_terminal` and a uniform `k in 0..K` otherwise; `spec/task.md` §1).
    When `all_steps=True`, yields all `K + 1` steps `k = 0..K` of each problem
    sequentially (`studies/addition/step-sampling.json`).
    """

    def __init__(
        self,
        *,
        bases: Sequence[int] = (2, 4, 8, 10),
        min_length: int = 1,
        max_length: int = 20,
        symbols: TokenSequence = SymbolSet.ALL,
        p_prior: float = 0.0,
        permuted_canonical: bool = False,
        p_chain: float = 0.25,
        p_terminal: float = 0.2,
        all_steps: bool = False,
        embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= p_terminal <= 1.0):
            raise ValueError(f"p_terminal must be in 0.0..1.0, got {p_terminal}")
        bases_tuple, symbol_pool = validate_sampling_params(
            bases=bases,
            min_length=min_length,
            max_length=max_length,
            symbols=symbols,
            p_prior=p_prior,
            permuted_canonical=permuted_canonical,
            p_chain=p_chain,
        )
        self._bases = bases_tuple
        self._min_length = min_length
        self._max_length = max_length
        self._symbol_pool = symbol_pool
        self._p_prior = p_prior
        self._permuted_canonical = permuted_canonical
        self._p_chain = p_chain
        self._p_terminal = p_terminal
        self._all_steps = all_steps
        self._embedding_mode = EmbeddingMode(embedding_mode)
        self._seed = seed

    def __iter__(self) -> Iterator[LabelledExample]:
        for problem, steps, k in iter_problem_steps(
            bases=self._bases,
            min_length=self._min_length,
            max_length=self._max_length,
            symbols=self._symbol_pool,
            p_prior=self._p_prior,
            permuted_canonical=self._permuted_canonical,
            p_chain=self._p_chain,
            p_terminal=self._p_terminal,
            all_steps=self._all_steps,
            seed=self._seed,
        ):
            yield build_cls_step_labelled_example(
                problem.alphabet,
                problem.a,
                problem.b,
                k,
                steps=steps,
                embedding_mode=self._embedding_mode,
            )


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    add_common_dataset_args(parser)
    parser.add_argument(
        "--p-terminal",
        type=float,
        default=0.2,
        help="Probability of sampling the terminal step k = K (default: 0.2)",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        default=False,
        help="Yield all K + 1 steps of each problem sequentially (default: False)",
    )


def _build_from_args(args: argparse.Namespace) -> AdditionClsDataset:
    return AdditionClsDataset(
        bases=args.bases,
        min_length=args.min_length,
        max_length=args.max_length,
        symbols=resolve_symbol_pool_slice(args.symbol_set, args.symbol_pool_size),
        p_prior=args.p_prior,
        permuted_canonical=args.permuted_canonical,
        p_chain=args.p_chain,
        p_terminal=args.p_terminal,
        all_steps=args.all_steps,
        embedding_mode=args.embedding_mode,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Generate, print, or dump `addition` (`cls`) training examples."""
    run_inspect_cli(
        description="Generate, print, or dump addition (cls) training examples.",
        add_dataset_args=_add_cli_args,
        build_dataset=_build_from_args,
        token_labels=ADDITION_TOKEN_LABELS,
        argv=argv,
    )


if __name__ == "__main__":
    main()
