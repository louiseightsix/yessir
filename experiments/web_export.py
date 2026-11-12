"""Print each experiment's description, variants, default configs, and field specs as JSON.

Run from the repo root with plain `python3 -m experiments.web_export`. It
imports only the config classes and `yessir.tui.build_field_specs`, which need
nothing beyond the standard library, so it works without the project's venv
(and without torch).
"""

import json
from typing import Any

from experiments.sir_yes_sir.config import (
    SIR_YES_SIR_DESCRIPTION,
    SM_MH_DESCRIPTION,
    SirYesSirConfig,
)
from experiments.sir_yes_sir.variants.ar.config import (
    AR_DESCRIPTION,
    SirYesSirArConfig,
)
from yessir.experiment import ExperimentConfig
from yessir.tui import build_field_specs

# Not taken from the `EXPERIMENTS` registry in `experiments/__main__.py`:
# that imports each experiment's `run.py`, which imports torch.
# `test_web_export.py` verifies both dicts stay in sync with `EXPERIMENTS`.
VARIANTS: dict[str, tuple[tuple[str, str, type[ExperimentConfig]], ...]] = {
    "sir_yes_sir": (
        ("sm_mh", SM_MH_DESCRIPTION, SirYesSirConfig),
        ("ar", AR_DESCRIPTION, SirYesSirArConfig),
    ),
}
DESCRIPTIONS: dict[str, str] = {"sir_yes_sir": SIR_YES_SIR_DESCRIPTION}


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
