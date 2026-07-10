#!/usr/bin/env bash
set -euo pipefail

FLOW_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE_DIR="$(dirname "$FLOW_REPO_DIR")"
OPENARM_REPO_DIR="${OPENARM_REPO_DIR:-$WORKSPACE_DIR/Openarm-GR00T}"

BASE_IMAGE="${BASE_IMAGE:-$WORKSPACE_DIR/containers/gr00t-cu128-py310.sqsh}"
IMAGE_NAME="${IMAGE_NAME:-flow-matching-cu128-py312.sqsh}"
CONTAINER_DIR="${CONTAINER_DIR:-$WORKSPACE_DIR/containers}"

if ! command -v srun >/dev/null 2>&1; then
  echo "srun not found. Run this script from the Slurm login/outer environment, not inside the coding container." >&2
  exit 1
fi

if [ ! -f "$OPENARM_REPO_DIR/train_on_dgx/init_image.sh" ]; then
  echo "Missing $OPENARM_REPO_DIR/train_on_dgx/init_image.sh" >&2
  exit 1
fi

if [ ! -f "$BASE_IMAGE" ]; then
  echo "Missing base image: $BASE_IMAGE" >&2
  exit 1
fi

mkdir -p "$CONTAINER_DIR"

cd "$OPENARM_REPO_DIR"

./train_on_dgx/init_image.sh \
  --base-image "$BASE_IMAGE" \
  --container-dir "$CONTAINER_DIR" \
  --image-name "$IMAGE_NAME" \
  --mount-parent "$WORKSPACE_DIR" \
  --workdir /workspace/flow_matching \
  -- bash /workspace/flow_matching/scripts/install_py312_runtime.sh

echo "Saved image: $CONTAINER_DIR/$IMAGE_NAME"

