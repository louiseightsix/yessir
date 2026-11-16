"""Training data generator (`AdditionStepArDataset`) for the `step_ar` variant of `addition`."""

import argparse
from collections.abc import Iterator, Sequence

from experiments.addition.inputs import (
    EmbeddingMode,
    StepOrder,
    add_common_dataset_args,
    iter_problem_steps,
    validate_sampling_params,
)
from experiments.addition.variants.step_ar.inputs import (
    build_step_ar_step_labelled_example,
)
from yessir.alphabet import SymbolSet
from yessir.dataset import LabelledExample
from yessir.inspect_data import run_inspect_cli
from yessir.vocab import TokenSequence


class AdditionStepArDataset:
    """Infinite deterministic stream of single-step `step_ar` `LabelledExample`s.

    Each yielded `LabelledExample` supervises the 3 sequential next-token targets
    `(t_0, t_1, flag_k)` of one sampled column step `k` (`spec/variants/step_ar.md` §3).
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
        step_order: StepOrder = StepOrder.SUM_FIRST,
        use_query_tokens: bool = False,
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
        self._step_order = StepOrder(step_order)
        self._use_query_tokens = use_query_tokens
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
            yield build_step_ar_step_labelled_example(
                problem.alphabet,
                problem.a,
                problem.b,
                k,
                steps=steps,
                step_order=self._step_order,
                use_query_tokens=self._use_query_tokens,
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
    parser.add_argument(
        "--step-order",
        type=StepOrder,
        choices=tuple(StepOrder),
        default=StepOrder.SUM_FIRST,
        help="Within-step emission order: sum_first or carry_first (default: sum_first)",
    )
    parser.add_argument(
        "--use-query-tokens",
        action="store_true",
        default=False,
        help="Insert explicit [S], [C], [F] query tokens before each step target (default: False)",
    )


def _build_from_args(args: argparse.Namespace) -> AdditionStepArDataset:
    return AdditionStepArDataset(
        bases=args.bases,
        min_length=args.min_length,
        max_length=args.max_length,
        p_prior=args.p_prior,
        permuted_canonical=args.permuted_canonical,
        p_chain=args.p_chain,
        p_terminal=args.p_terminal,
        all_steps=args.all_steps,
        step_order=args.step_order,
        use_query_tokens=args.use_query_tokens,
        embedding_mode=args.embedding_mode,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Generate, print, or dump `addition` (`step_ar`) training examples."""
    run_inspect_cli(
        description="Generate, print, or dump addition (step_ar) training examples.",
        add_dataset_args=_add_cli_args,
        build_dataset=_build_from_args,
        argv=argv,
    )


if __name__ == "__main__":
    main()
