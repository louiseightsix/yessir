"""Print each experiment's default config as JSON, for the web UI build.

Run from the repo root with plain `python3 -m experiments.web_export`. It
imports only the config classes, which need nothing beyond the standard
library, so it works without the project's venv (and without torch).
"""

import json

from experiments.sir_yes_sir.config import SirYesSirConfig

# Not taken from the `EXPERIMENTS` registry in `experiments/__main__.py`:
# that imports each experiment's `run.py`, which imports torch.
CONFIGS = {"sir_yes_sir": SirYesSirConfig}


def main() -> None:
    print(
        json.dumps({name: cls().to_dict() for name, cls in CONFIGS.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
