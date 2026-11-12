"""Tests for `experiments.web_export`."""

import json
import subprocess
import sys

from experiments.sir_yes_sir.config import SirYesSirConfig


def test_web_export_prints_default_configs_without_loading_torch():
    # A fresh interpreter, as in the web build. The trailing assert checks
    # torch stayed unloaded after the export ran.
    code = (
        "import runpy, sys\n"
        "runpy.run_module('experiments.web_export', run_name='__main__')\n"
        "assert 'torch' not in sys.modules, 'web_export loaded torch'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == {"sir_yes_sir": SirYesSirConfig().to_dict()}
