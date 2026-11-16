"""Top-level `Experiment` descriptor (`ADDITION_EXPERIMENT`) and CLI entry point for `addition`."""

from collections.abc import Sequence

from experiments.addition.config import ADDITION_DESCRIPTION
from experiments.addition.variants.ar.run import AR_VARIANT
from experiments.addition.variants.cls.run import CLS_VARIANT
from experiments.addition.variants.step_ar.run import STEP_AR_VARIANT
from experiments.addition.variants.step_fork.run import STEP_FORK_VARIANT
from yessir.experiment import Experiment, run_experiment_cli

ADDITION_EXPERIMENT = Experiment(
    name="addition",
    description=ADDITION_DESCRIPTION,
    variants=(CLS_VARIANT, STEP_FORK_VARIANT, STEP_AR_VARIANT, AR_VARIANT),
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run `addition` from the command line."""
    run_experiment_cli(ADDITION_EXPERIMENT, argv)


if __name__ == "__main__":
    main()
