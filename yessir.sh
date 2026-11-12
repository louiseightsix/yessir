#!/usr/bin/env bash
# yessir runner.
#
# Usage:
#   ./yessir.sh run [experiment] [args]  # run a live experiment (or list experiments)
#   ./yessir.sh browser [astro args]     # open the results browser & config builder (astro dev)
#   ./yessir.sh test [pytest args]       # run the live experiments' tests
#   ./yessir.sh check                    # lint and typecheck live code (ruff check + ty check)
#   ./yessir.sh fix                      # typecheck, autofix lint issues, and format
#   ./yessir.sh format [ruff args]       # format live code (ruff format)
#   ./yessir.sh uv <uv args>             # any uv command against this repo's environment,
#                                        #   e.g. ./yessir.sh uv add numpy
#
# Environment: one venv per OS (.venv-darwin, .venv-linux), so a checkout shared
# between a macOS host and a Linux VM keeps a working venv on each side. Set
# UV_PROJECT_ENVIRONMENT to use a different one. uv creates/syncs it on first use.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found: install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-$OS}"
# In a Linux VM the checkout is usually a host mount, where uv can't hardlink from its cache.
[ "$OS" = linux ] && export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

usage() {
    sed -n '4,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

cmd="${1:-}"
[ $# -gt 0 ] && shift

case "$cmd" in
    run)     exec uv run python -m experiments "$@" ;;
    browser)
        if ! command -v pnpm >/dev/null 2>&1; then
            echo "pnpm not found: install it from https://pnpm.io/installation" >&2
            exit 1
        fi
        exec pnpm --dir web dev "$@"
        ;;
    test)    exec uv run pytest "$@" ;;
    check)
        uv run ruff check "$@"
        exec uv run ty check experiments yessir "$@"
        ;;
    fix)
        uv run ty check experiments yessir
        uv run ruff check --fix "$@"
        exec uv run ruff format "$@"
        ;;
    format) exec uv run ruff format "$@" ;;
    uv)     exec uv "$@" ;;
    ""|-h|--help|help) usage ;;
    *) echo "unknown command: $cmd" >&2; usage >&2; exit 1 ;;
esac
