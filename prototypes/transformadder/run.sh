#!/usr/bin/env bash
# ==============================================================================
# TransformAdder prototype: Interactive Experiment Setup & Execution Shell Script
#
# Sets up the repo-level Python environment, verifies dependencies, and interactively
# configures, creates, and executes repeatable runs (written to prototypes/transformadder/results/).
#
# Usage (from anywhere):
#   prototypes/transformadder/run.sh               # Interactive configuration wizard
#   prototypes/transformadder/run.sh --help        # Show CLI options
#   prototypes/transformadder/run.sh --dry-run ... # Scripted execution
# ==============================================================================

set -e

# Repository root (this script lives in prototypes/transformadder/)
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
cd "$REPO_DIR"

# Ensure UV link mode avoids cross-filesystem hardlink failures (e.g. virtiofs / Docker mounts)
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# 1. Locate uv / Python
UV_BIN=""
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
fi

# 2. Virtual Environment Helper & Resolution
is_valid_venv() {
    local target="$1"
    local py=""
    if [ -x "$target/bin/python" ]; then
        py="$target/bin/python"
    elif [ -x "$target/bin/python3" ]; then
        py="$target/bin/python3"
    else
        return 1
    fi
    "$py" -c "import sys" >/dev/null 2>&1
}

OS_TYPE="$(uname -s | tr '[:upper:]' '[:lower:]')"

# Resolve VENV_DIR
if [ -z "$VENV_DIR" ]; then
    if is_valid_venv "$REPO_DIR/.venv"; then
        VENV_DIR="$REPO_DIR/.venv"
    elif is_valid_venv "$REPO_DIR/.venv-$OS_TYPE"; then
        VENV_DIR="$REPO_DIR/.venv-$OS_TYPE"
    elif [ -d "$REPO_DIR/.venv" ]; then
        # .venv exists but is not executable on this OS (e.g. Linux container vs macOS host in shared mount)
        echo "[*] Notice: Existing .venv interpreter is not executable on $(uname -s) (likely built in container/VM)."
        echo "[*] Using platform-isolated virtual environment: .venv-$OS_TYPE"
        VENV_DIR="$REPO_DIR/.venv-$OS_TYPE"
    else
        VENV_DIR="$REPO_DIR/.venv"
    fi
fi

# 3. Check / Setup Python Virtual Environment
if ! is_valid_venv "$VENV_DIR"; then
    if [ -d "$VENV_DIR" ]; then
        echo "[!] Existing virtual environment at $VENV_DIR is corrupted or invalid. Recreating..."
        rm -rf "$VENV_DIR"
    fi
    echo "[*] Setting up Python virtual environment at $VENV_DIR..."
    if [ -n "$UV_BIN" ]; then
        "$UV_BIN" venv "$VENV_DIR" --python python3.12 2>/dev/null || "$UV_BIN" venv "$VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
    fi
    echo "[✓] Virtual environment created."
fi

# 4. Activate Virtual Environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 5. Verify & Install Python Dependencies
if ! python -c "import torch.nn.functional, pytest" >/dev/null 2>&1; then
    echo "[*] Required Python packages missing or incomplete. Installing dependencies (torch, pytest)..."
    if [ -n "$UV_BIN" ]; then
        "$UV_BIN" pip install --reinstall --python "$VENV_DIR/bin/python" torch pytest
    else
        python -m pip install --upgrade pip
        python -m pip install --force-reinstall torch pytest
    fi
    echo "[✓] Dependencies installed."
fi

# 6. Set PYTHONPATH to repository root
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# 7. Delegate to the experiment configurator / runner
exec python "$EXPERIMENT_DIR/scripts/configure_experiment.py" "$@"
