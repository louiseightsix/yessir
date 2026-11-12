"""Print each experiment's description, default config, and field specs as JSON.

Run from the repo root with plain `python3 -m experiments.web_export`. It
imports only the config classes and `yessir.tui.build_field_specs`, which need
nothing beyond the standard library, so it works without the project's venv
(and without torch).
"""

import json
from typing import Any

from experiments.sir_yes_sir.config import SirYesSirConfig
from yessir.experiment import ExperimentConfig
from yessir.tui import build_field_specs

# Not taken from the `EXPERIMENTS` registry in `experiments/__main__.py`:
# that imports each experiment's `run.py`, which imports torch.
# `test_web_export.py` verifies both dicts stay in sync with `EXPERIMENTS`.
CONFIGS: dict[str, type[ExperimentConfig]] = {"sir_yes_sir": SirYesSirConfig}
DESCRIPTIONS: dict[str, str] = {
    "sir_yes_sir": "In-context successor (next symbol mod B, no carry) smoke-test experiment.",
}


def export_schemas() -> dict[str, dict[str, Any]]:
    """Build the JSON-serialisable schema dictionary for every registered experiment."""
    return {
        name: {
            "description": DESCRIPTIONS[name],
            "default_config": cls().to_dict(),
            "fields": [spec.to_dict() for spec in build_field_specs(cls)],
        }
        for name, cls in CONFIGS.items()
    }


def main() -> None:
    print(json.dumps(export_schemas(), indent=2))


if __name__ == "__main__":
    main()
