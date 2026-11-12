# Reusable Vertex AI runner image for yessir.
#
# Pre-installs Python 3.12 and the locked dependencies from `pyproject.toml` / `uv.lock`
# (including CUDA-enabled PyTorch wheels) into `/opt/venv` so that jobs do not
# re-download multi-GB wheels at startup. At runtime, `entrypoint.sh` fetches the
# requested git commit from GitHub and runs `python -m experiments` against `/opt/venv`.

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

# Pre-create the Python 3.12 virtual environment and cache/install locked runtime
# dependencies (`torch`, `numpy`, `petname`) at image build time.
WORKDIR /opt/yessir-deps
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 && \
    uv sync --frozen --no-install-project --no-dev --python 3.12

COPY cloud/vertex/entrypoint.sh /usr/local/bin/yessir-entrypoint
RUN chmod +x /usr/local/bin/yessir-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/yessir-entrypoint"]
