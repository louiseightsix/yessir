#!/usr/bin/env bash
# Standalone Google Cloud Vertex AI Custom Job submission for: SM-MH-T1-L6-GELU-Causal-bimodal-seed45364
# Random seed: 45364
# Created at: 2026-09-18T20:00:47.281333
set -e

PROJECT_ID="${GCP_PROJECT:-}"
REGION="${GCP_REGION:-us-east4}"
BUCKET="${GCP_BUCKET:-}"
ACCELERATOR="${GCP_ACCELERATOR:-NVIDIA_L4}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-g2-standard-4}"

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: GCP project ID is required." >&2
    echo "Provide it via environment variable: export GCP_PROJECT=your-project-id" >&2
    exit 1
fi

if [ -z "$BUCKET" ]; then
    echo "ERROR: Cloud Storage staging bucket is required (e.g. gs://my-bucket)." >&2
    echo "Provide it via environment variable: export GCP_BUCKET=gs://my-bucket" >&2
    exit 1
fi

# Clean up bucket URI prefix if trailing slash exists
BUCKET="${BUCKET%/}"

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../.." && pwd)"

CLEAN_EXP_NAME="$(echo "SM-MH-T1-L6-GELU-Causal-bimodal-seed45364" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr '.' '-')"
JOB_NAME="ta-${CLEAN_EXP_NAME:0:45}-$(date +%s)"

echo "Building Python package from $REPO_DIR..."
cd "$REPO_DIR"
PYTHON_BIN=""
OS_TYPE="$(uname -s | tr '[:upper:]' '[:lower:]')"
for candidate in "$REPO_DIR/.venv/bin/python" "$REPO_DIR/.venv-$OS_TYPE/bin/python"; do
    if [ -x "$candidate" ] && "$candidate" -c "import sys" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
"$PYTHON_BIN" "$REPO_DIR/setup.py" sdist --formats=gztar --dist-dir="$REPO_DIR/dist" > /dev/null

PACKAGE_FILE="$(ls -t "$REPO_DIR"/dist/transformadder-*.tar.gz | head -n 1)"
TIMESTAMP="$(date +%s)"
PACKAGE_GCS_URI="$BUCKET/packages/transformadder-${TIMESTAMP}.tar.gz"

echo "Verifying staging bucket $BUCKET..."
if command -v gcloud >/dev/null 2>&1; then
    if ! gcloud storage buckets describe "$BUCKET" >/dev/null 2>&1; then
        echo "[*] Bucket $BUCKET does not exist. Creating bucket in $REGION ($PROJECT_ID)..."
        gcloud storage buckets create "$BUCKET" --project="$PROJECT_ID" --location="$REGION"
    fi
elif command -v gsutil >/dev/null 2>&1; then
    if ! gsutil ls -b "$BUCKET" >/dev/null 2>&1; then
        echo "[*] Bucket $BUCKET does not exist. Creating bucket in $REGION ($PROJECT_ID)..."
        gsutil mb -p "$PROJECT_ID" -l "$REGION" "$BUCKET"
    fi
fi

echo "Uploading package to $PACKAGE_GCS_URI..."
if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "$PACKAGE_FILE" "$PACKAGE_GCS_URI"
elif command -v gsutil >/dev/null 2>&1; then
    gsutil cp "$PACKAGE_FILE" "$PACKAGE_GCS_URI"
else
    echo "ERROR: Neither 'gcloud storage' nor 'gsutil' found." >&2
    exit 1
fi

if [ "$ACCELERATOR" = "none" ] || [ -z "$ACCELERATOR" ]; then
    IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-4.py310:latest"
    WORKER_POOL_SPEC="machine-type=$MACHINE_TYPE,executor-image-uri=$IMAGE_URI,python-module=transformadder.cli"
else
    IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
    WORKER_POOL_SPEC="machine-type=$MACHINE_TYPE,accelerator-type=$ACCELERATOR,accelerator-count=1,executor-image-uri=$IMAGE_URI,python-module=transformadder.cli"
fi

echo "========================================================================"
echo "SUBMITTING VERTEX AI CUSTOM JOB"
echo "========================================================================"
echo "  Job Name:     $JOB_NAME"
echo "  Project:      $PROJECT_ID"
echo "  Region:       $REGION"
echo "  Package:      $PACKAGE_GCS_URI"
echo "  Accelerator:  $ACCELERATOR (Machine: $MACHINE_TYPE)"
echo "  Image:        $IMAGE_URI"
echo "========================================================================"

JOB_RESOURCE_NAME=$(gcloud ai custom-jobs create \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --display-name="$JOB_NAME" \
    --python-package-uris="$PACKAGE_GCS_URI" \
    --worker-pool-spec="$WORKER_POOL_SPEC" \
    --args="--paradigm=sm_mh,--tier=1,--n-layers=6,--ffn-type=gelu,--attn-type=causal,--regime=bimodal,--seed=45364,--train-steps=100000,--eval-interval=5000,--batch-size=256,--eval-problems=100,--final-eval-problems=10000,--train-axiom=random,--eval-axioms=both,--log-dir=$BUCKET/experiments/SM-MH-T1-L6-GELU-Causal-bimodal-seed45364" \
    --format="value(name)")

echo ""
echo "[✓] Custom Job submitted successfully!"
echo "  Resource: $JOB_RESOURCE_NAME"
echo ""
echo "Cloud Console URL:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "To stream training logs live in your terminal, run:"
echo "  gcloud ai custom-jobs stream-logs $JOB_RESOURCE_NAME --region=$REGION --project=$PROJECT_ID"
