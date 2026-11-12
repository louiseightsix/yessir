"""Central experiment registry and CLI entry point (`python -m experiments`)."""

from collections.abc import Sequence

from experiments.sir_yes_sir.run import SIR_YES_SIR_EXPERIMENT
from yessir.experiment import Experiment, run_experiments_cli

EXPERIMENTS: tuple[Experiment, ...] = (SIR_YES_SIR_EXPERIMENT,)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch CLI arguments to the selected experiment in `EXPERIMENTS`."""
    run_experiments_cli(EXPERIMENTS, argv)


if __name__ == "__main__":
    main()
