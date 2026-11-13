#!/usr/bin/env bash
# Entrypoint for the yessir Vertex AI runner container.
#
# Required environment variables:
#   YESSIR_GIT_SHA       Full or short commit SHA (or ref) to fetch and check out.
# Optional environment variables:
#   YESSIR_REPO_URL      Git repository URL (default: https://github.com/louiseightsix/yessir.git).
#   YESSIR_LOCAL_OUT_DIR Local directory where the experiment writes artefacts during the run.
#   YESSIR_GCS_OUT_DIR   Mounted `/gcs/<bucket>/...` directory where artefacts are copied on exit.
#
# All command-line arguments ("$@") are forwarded to `python -m experiments`.

set -euo pipefail

REPO_URL="${YESSIR_REPO_URL:-https://github.com/louiseightsix/yessir.git}"
GIT_SHA="${YESSIR_GIT_SHA:-}"
LOCAL_OUT_DIR="${YESSIR_LOCAL_OUT_DIR:-}"
GCS_OUT_DIR="${YESSIR_GCS_OUT_DIR:-}"

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

if [ -n "$LOCAL_OUT_DIR" ]; then
    rm -rf "$LOCAL_OUT_DIR"
    mkdir -p "$LOCAL_OUT_DIR"
fi

sync_outputs_to_gcs() {
    if [ -n "$LOCAL_OUT_DIR" ] && [ -n "$GCS_OUT_DIR" ] && [ -d "$LOCAL_OUT_DIR" ]; then
        if [ -n "$(ls -A "$LOCAL_OUT_DIR" 2>/dev/null)" ]; then
            echo "Syncing artefacts from $LOCAL_OUT_DIR to $GCS_OUT_DIR ..."
            mkdir -p "$GCS_OUT_DIR"
            cp -f "$LOCAL_OUT_DIR"/* "$GCS_OUT_DIR"/
        fi
    fi
}
trap sync_outputs_to_gcs EXIT

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
uv run --no-sync --no-dev python -m experiments "$@"
