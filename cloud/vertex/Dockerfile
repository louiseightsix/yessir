# Reusable Vertex AI runner image for yessir.
#
# Pre-installs Python 3.12 and the locked dependencies from `pyproject.toml` / `uv.lock`
# (including CUDA-enabled PyTorch wheels) into `/opt/venv` so that jobs do not
# re-download multi-GB wheels at startup. At runtime, `entrypoint.sh` fetches the
# requested git commit from GitHub and runs `python -m experiments` against `/opt/venv`.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# The plain Debian base doesn't set these, but the `nvidia/cuda` images do. They
# tell the NVIDIA container runtime to expose the GPUs and driver, and they make the
# host's mounted driver (`libcuda.so`, `nvidia-smi`) findable on GKE-style hosts.
# Without them, torch could silently fall back to CPU.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/usr/local/nvidia/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Pre-create the Python 3.12 virtual environment and install locked runtime
# dependencies (`torch`, `numpy`, `petname`) at image build time without
# keeping a duplicate wheel/archive cache in `/root/.cache/uv`.
WORKDIR /opt/yessir-deps
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 && \
    uv sync --frozen --no-install-project --no-dev --no-cache --python 3.12

COPY cloud/vertex/entrypoint.sh /usr/local/bin/yessir-entrypoint
RUN chmod +x /usr/local/bin/yessir-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/yessir-entrypoint"]
