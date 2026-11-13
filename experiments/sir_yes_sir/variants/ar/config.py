"""Configuration schema (`SirYesSirArConfig`) for the `ar` variant of `sir_yes_sir`."""

from dataclasses import dataclass

from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.sir_yes_sir.variants.ar.inputs import StepOrder

AR_DESCRIPTION = "Autoregressive trace harness read through SubspaceLMHead."


@dataclass(frozen=True, kw_only=True)
class SirYesSirArConfig(SirYesSirConfig):
    """Configuration for an autoregressive (`ar`) `sir_yes_sir` run.

    Inherits all task, trunk, and training settings from `SirYesSirConfig` and adds
    `step_order` (`out_first` vs. `wrap_first`) for within-step token emission order.
    """

    step_order: StepOrder = StepOrder.OUT_FIRST

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "step_order", StepOrder(self.step_order))
