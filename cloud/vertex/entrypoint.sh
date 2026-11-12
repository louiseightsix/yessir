#!/usr/bin/env bash
# Entrypoint for the yessir Vertex AI runner container.
#
# Required environment variables:
#   YESSIR_GIT_SHA   Full or short commit SHA (or ref) to fetch and check out.
# Optional environment variables:
#   YESSIR_REPO_URL  Git repository URL (default: https://github.com/louiseightsix/yessir.git).
#
# All command-line arguments ("$@") are forwarded to `python -m experiments`.

set -euo pipefail

REPO_URL="${YESSIR_REPO_URL:-https://github.com/louiseightsix/yessir.git}"
GIT_SHA="${YESSIR_GIT_SHA:-}"

if [ -z "$GIT_SHA" ]; then
    echo "error: YESSIR_GIT_SHA environment variable is required" >&2
    exit 1
fi

echo "=== yessir Vertex runner ==="
echo "repo:   $REPO_URL"
echo "commit: $GIT_SHA"
echo "args:   $*"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

REPO_DIR="/workspace/yessir"
rm -rf "$REPO_DIR"
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

git init --quiet
git remote add origin "$REPO_URL"
git fetch --depth 1 origin "$GIT_SHA"
git checkout --quiet FETCH_HEAD

echo "checked out $(git rev-parse HEAD) ($(git show -s --format=%cs HEAD))"

# Sync any newly added dependencies against the pre-populated /opt/venv, then run.
uv sync --frozen --no-install-project --no-dev
exec uv run --no-sync --no-dev python -m experiments "$@"
