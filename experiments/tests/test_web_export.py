"""Tests for `experiments.web_export`."""

import json
import subprocess
import sys

from experiments.__main__ import EXPERIMENTS
from experiments.addition.config import (
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
    SirYesSirClsConfig,
    SirYesSirConfig,
)
from experiments.web_export import DESCRIPTIONS, VARIANTS
from yessir.tui import build_field_specs


def test_web_export_prints_schemas_with_stdlib_only():
    # `-S` disables `site` so venv (`.venv-darwin` / `.venv-linux`) third-party
    # packages (torch, numpy, petname) are not on `sys.path`, matching the plain
    # `python3` Cloudflare build.
    code = (
        "import runpy, sys\n"
        "runpy.run_module('experiments.web_export', run_name='__main__')\n"
        "assert 'torch' not in sys.modules, 'web_export loaded torch'\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], check=True, capture_output=True, text=True
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "sir_yes_sir": {
            "description": DESCRIPTIONS["sir_yes_sir"],
            "variants": [
                {
                    "name": "cls",
                    "description": CLS_DESCRIPTION,
                    "default_config": SirYesSirClsConfig().to_dict(),
                    "fields": [
                        spec.to_dict() for spec in build_field_specs(SirYesSirClsConfig)
                    ],
                },
                {
                    "name": "ar",
                    "description": AR_DESCRIPTION,
                    "default_config": SirYesSirConfig().to_dict(),
                    "fields": [
                        spec.to_dict() for spec in build_field_specs(SirYesSirConfig)
                    ],
                },
            ],
        },
        "addition": {
            "description": DESCRIPTIONS["addition"],
            "variants": [
                {
                    "name": "cls",
                    "description": ADDITION_CLS_DESCRIPTION,
                    "default_config": AdditionClsConfig().to_dict(),
                    "fields": [
                        spec.to_dict() for spec in build_field_specs(AdditionClsConfig)
                    ],
                },
                {
                    "name": "step_fork",
                    "description": STEP_FORK_DESCRIPTION,
                    "default_config": AdditionStepForkConfig().to_dict(),
                    "fields": [
                        spec.to_dict()
                        for spec in build_field_specs(AdditionStepForkConfig)
                    ],
                },
                {
                    "name": "step_ar",
                    "description": STEP_AR_DESCRIPTION,
                    "default_config": AdditionStepArConfig().to_dict(),
                    "fields": [
                        spec.to_dict()
                        for spec in build_field_specs(AdditionStepArConfig)
                    ],
                },
                {
                    "name": "ar",
                    "description": ADDITION_AR_DESCRIPTION,
                    "default_config": AdditionArConfig().to_dict(),
                    "fields": [
                        spec.to_dict() for spec in build_field_specs(AdditionArConfig)
                    ],
                },
                {
                    "name": "addressed_step_ar",
                    "description": ADDRESSED_STEP_AR_DESCRIPTION,
                    "default_config": AdditionAddressedStepArConfig().to_dict(),
                    "fields": [
                        spec.to_dict()
                        for spec in build_field_specs(AdditionAddressedStepArConfig)
                    ],
                },
                {
                    "name": "addressed_ar",
                    "description": ADDRESSED_AR_DESCRIPTION,
                    "default_config": AdditionAddressedArConfig().to_dict(),
                    "fields": [
                        spec.to_dict()
                        for spec in build_field_specs(AdditionAddressedArConfig)
                    ],
                },
            ],
        },
    }


def test_web_export_matches_experiment_registry():
    assert VARIANTS == {
        exp.name: tuple((v.name, v.description, v.config_cls) for v in exp.variants)
        for exp in EXPERIMENTS
    }
    assert DESCRIPTIONS == {exp.name: exp.description for exp in EXPERIMENTS}
