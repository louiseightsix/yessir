"""Training data generator (`AdditionStepForkDataset`) for the `step_fork` variant of `addition`."""

import argparse
from collections.abc import Iterator, Sequence

from experiments.addition.inputs import (
    ADDITION_TOKEN_LABELS,
    EmbeddingMode,
    add_common_dataset_args,
    iter_problem_steps,
    validate_sampling_params,
)
from experiments.addition.variants.step_fork.inputs import (
    build_step_fork_step_labelled_examples,
)
from yessir.alphabet import SymbolSet, resolve_symbol_pool_slice
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import TokenSequence


class AdditionStepForkDataset:
    """Infinite deterministic stream of branched `step_fork` `LabelledExample`s.

    For each sampled column transition `(problem, steps, k)`, yields its
    `BRANCHES_PER_STEP = 3` parallel query branches (`[S]`, `[C]`, `[F]`)
    consecutively (`spec/variants/step_fork.md` §3).

    Batching contract: callers of `train(model, dataset, batch_size=...)` should
    pass a multiple of `BRANCHES_PER_STEP` (`3 * transitions_per_step`, as
    `run_addition_step_fork` does via `3 * config.batch_size`) so every column
    transition's `([S], [C], [F])` triple stays in one mini-batch rather than
    splitting across optimiser steps when `batch_size % 3 != 0`.
    """

    BRANCHES_PER_STEP: int = 3

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
            yield from build_step_fork_step_labelled_examples(
                problem.alphabet,
                problem.a,
                problem.b,
                k,
                steps=steps,
                embedding_mode=self._embedding_mode,
                validate=False,
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


def _build_from_args(args: argparse.Namespace) -> AdditionStepForkDataset:
    return AdditionStepForkDataset(
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
    """Generate, print, or dump `addition` (`step_fork`) training examples."""
    run_inspect_cli(
        description="Generate, print, or dump addition (step_fork) training examples.",
        add_dataset_args=_add_cli_args,
        build_dataset=_build_from_args,
        token_labels=ADDITION_TOKEN_LABELS,
        argv=argv,
    )


if __name__ == "__main__":
    main()
