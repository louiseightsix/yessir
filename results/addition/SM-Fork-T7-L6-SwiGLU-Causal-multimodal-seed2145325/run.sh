#!/usr/bin/env bash
# Standalone execution shell script for: SM-Fork-T7-L6-SwiGLU-Causal-multimodal-seed2145325
# Random seed: 2145325
set -e

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../.." && pwd)"

# Resolve active or platform-appropriate virtual environment
VENV_DIR=""
OS_TYPE="$(uname -s | tr '[:upper:]' '[:lower:]')"
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ] && "$VIRTUAL_ENV/bin/python" -c "import sys" >/dev/null 2>&1; then
    VENV_DIR="$VIRTUAL_ENV"
elif [ -x "$REPO_DIR/.venv/bin/python" ] && "$REPO_DIR/.venv/bin/python" -c "import sys" >/dev/null 2>&1; then
    VENV_DIR="$REPO_DIR/.venv"
elif [ -x "$REPO_DIR/.venv-$OS_TYPE/bin/python" ] && "$REPO_DIR/.venv-$OS_TYPE/bin/python" -c "import sys" >/dev/null 2>&1; then
    VENV_DIR="$REPO_DIR/.venv-$OS_TYPE"
fi

if [ -n "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

exec python "$EXP_DIR/run.py" "$@"
