"""Print each experiment's description, variants, default configs, and field specs as JSON.

Run from the repo root with plain `python3 -m experiments.web_export`. It
imports only the config classes and `yessir.tui.build_field_specs`, which need
nothing beyond the standard library, so it works without the project's venv
(and without torch).
"""

import json
from typing import Any

from experiments.addition.config import (
    ADDITION_DESCRIPTION,
    ADDRESSED_AR_DESCRIPTION,
    ADDRESSED_STEP_AR_DESCRIPTION,
    STEP_AR_DESCRIPTION,
    STEP_FORK_DESCRIPTION,
    AdditionAddressedArConfig,
    AdditionAddressedStepArConfig,
    AdditionArConfig,
    AdditionClsConfig,
    AdditionStepArConfig,
    AdditionStepForkConfig,
)
from experiments.addition.config import (
    AR_DESCRIPTION as ADDITION_AR_DESCRIPTION,
)
from experiments.addition.config import (
    CLS_DESCRIPTION as ADDITION_CLS_DESCRIPTION,
)
from experiments.sir_yes_sir.config import (
    AR_DESCRIPTION,
    CLS_DESCRIPTION,
    SIR_YES_SIR_DESCRIPTION,
    SirYesSirClsConfig,
    SirYesSirConfig,
)
from yessir.experiment import ExperimentConfig
from yessir.tui import build_field_specs

# Not taken from the `EXPERIMENTS` registry in `experiments/__main__.py`:
# that imports each experiment's `run.py`, which imports torch.
# `test_web_export.py` verifies both dicts stay in sync with `EXPERIMENTS`.
VARIANTS: dict[str, tuple[tuple[str, str, type[ExperimentConfig]], ...]] = {
    "sir_yes_sir": (
        ("cls", CLS_DESCRIPTION, SirYesSirClsConfig),
        ("ar", AR_DESCRIPTION, SirYesSirConfig),
    ),
    "addition": (
        ("cls", ADDITION_CLS_DESCRIPTION, AdditionClsConfig),
        ("step_fork", STEP_FORK_DESCRIPTION, AdditionStepForkConfig),
        ("step_ar", STEP_AR_DESCRIPTION, AdditionStepArConfig),
        ("ar", ADDITION_AR_DESCRIPTION, AdditionArConfig),
        (
            "addressed_step_ar",
            ADDRESSED_STEP_AR_DESCRIPTION,
            AdditionAddressedStepArConfig,
        ),
        (
            "addressed_ar",
            ADDRESSED_AR_DESCRIPTION,
            AdditionAddressedArConfig,
        ),
    ),
}
DESCRIPTIONS: dict[str, str] = {
    "sir_yes_sir": SIR_YES_SIR_DESCRIPTION,
    "addition": ADDITION_DESCRIPTION,
}


def export_schemas() -> dict[str, dict[str, Any]]:
    """Build the JSON-serialisable schema dictionary for every registered experiment."""
    return {
        name: {
            "description": DESCRIPTIONS[name],
            "variants": [
                {
                    "name": v_name,
                    "description": v_desc,
                    "default_config": v_cls().to_dict(),
                    "fields": [spec.to_dict() for spec in build_field_specs(v_cls)],
                }
                for v_name, v_desc, v_cls in variants
            ],
        }
        for name, variants in VARIANTS.items()
    }


def main() -> None:
    print(json.dumps(export_schemas(), indent=2))


if __name__ == "__main__":
    main()
