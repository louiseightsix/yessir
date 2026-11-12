"""Tests for `experiments.web_export`."""

import json
import subprocess
import sys

from experiments.__main__ import EXPERIMENTS
from experiments.sir_yes_sir.config import SirYesSirConfig
from experiments.web_export import CONFIGS


def test_web_export_prints_default_configs_with_stdlib_only():
    # `-S` disables `site` so `.venv` third-party packages (torch, numpy, petname)
    # are not on `sys.path`, matching the plain `python3` Cloudflare build.
    code = (
        "import runpy, sys\n"
        "runpy.run_module('experiments.web_export', run_name='__main__')\n"
        "assert 'torch' not in sys.modules, 'web_export loaded torch'\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == {"sir_yes_sir": SirYesSirConfig().to_dict()}


def test_web_export_matches_experiment_registry():
    assert CONFIGS == {exp.name: exp.config_cls for exp in EXPERIMENTS}

