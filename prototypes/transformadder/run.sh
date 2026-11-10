#!/usr/bin/env bash
# ==============================================================================
# TransformAdder prototype: interactive experiment setup & execution.
#
# Runs the configurator inside the repo's uv-managed environment (created and
# synced on first use), which writes and launches repeatable runs under
# prototypes/transformadder/results/.
#
# Usage (from anywhere):
#   prototypes/transformadder/run.sh               # Interactive configuration wizard
#   prototypes/transformadder/run.sh --help        # Show CLI options
#   prototypes/transformadder/run.sh --dry-run ... # Scripted execution
# ==============================================================================

set -e

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found: install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

exec uv run python "$EXPERIMENT_DIR/scripts/configure_experiment.py" "$@"
