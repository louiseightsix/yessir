"""Top-level `Experiment` descriptor (`SIR_YES_SIR_EXPERIMENT`) and CLI entry point for `sir_yes_sir`."""

from collections.abc import Sequence

from experiments.sir_yes_sir.config import SIR_YES_SIR_DESCRIPTION
from experiments.sir_yes_sir.variants.sm_mh.run import SM_MH_VARIANT
from yessir.experiment import Experiment, run_experiment_cli

SIR_YES_SIR_EXPERIMENT = Experiment(
    name="sir_yes_sir",
    description=SIR_YES_SIR_DESCRIPTION,
    variants=(SM_MH_VARIANT,),
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run `sir_yes_sir` from the command line."""
    run_experiment_cli(SIR_YES_SIR_EXPERIMENT, argv)


if __name__ == "__main__":
    main()
