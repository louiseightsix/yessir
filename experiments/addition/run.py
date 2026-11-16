"""Top-level `Experiment` descriptor (`ADDITION_EXPERIMENT`) and CLI entry point for `addition`."""

from collections.abc import Sequence

from experiments.addition.config import ADDITION_DESCRIPTION
from experiments.addition.variants.cls.run import CLS_VARIANT
from yessir.experiment import Experiment, run_experiment_cli

ADDITION_EXPERIMENT = Experiment(
    name="addition",
    description=ADDITION_DESCRIPTION,
    variants=(CLS_VARIANT,),
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run `addition` from the command line."""
    run_experiment_cli(ADDITION_EXPERIMENT, argv)


if __name__ == "__main__":
    main()
