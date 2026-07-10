#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

JOB_NAME=${JOB_NAME:-flow-pusht}
GPUS=${GPUS:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-24}
TIME_LIMIT=${TIME_LIMIT:-infinite}
EXCLUDE=${EXCLUDE:-KDA00}
NODELIST=${NODELIST:-}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"$WORKSPACE_ROOT/containers/flow-matching-cu128-py312.sqsh"}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"$WORKSPACE_ROOT:/workspace,/mnt/syno127/volume1:/mnt/syno127/volume1"}

if [[ ! -f "$CONTAINER_IMAGE" ]]; then
  echo "Container image not found: $CONTAINER_IMAGE" >&2
  exit 1
fi

sbatch_node_args=()
if [[ -n "$NODELIST" ]]; then
  sbatch_node_args+=(--nodelist="$NODELIST")
fi
if [[ -n "$EXCLUDE" ]]; then
  sbatch_node_args+=(--exclude="$EXCLUDE")
fi

job_id=$(
  sbatch --parsable \
    --job-name="$JOB_NAME" \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$TIME_LIMIT" \
    "${sbatch_node_args[@]}" \
    --output="$SCRIPT_DIR/slurm-%x-%j.out" \
    --error="$SCRIPT_DIR/slurm-%x-%j.err" \
    --export=ALL \
    <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

srun --nodes 1 --ntasks 1 --gpus "$GPUS" --container-remap-root --no-container-mount-home --container-workdir=/workspace/flow_matching --container-mounts="$CONTAINER_MOUNTS" --container-image="$CONTAINER_IMAGE" bash -lc '
    set -euo pipefail
    if [ -f /workspace/.gitconfig ]; then
      mkdir -p "\${HOME:-/root}"
      ln -sf /workspace/.gitconfig "\${HOME:-/root}/.gitconfig"
    fi
    if [ -f /workspace/Openarm-GR00T/train_on_dgx/secrets/secrets.env ]; then
      set -a
      source /workspace/Openarm-GR00T/train_on_dgx/secrets/secrets.env
      set +a
    fi
    cd /workspace/flow_matching
    echo "Started at \$(date)"
    echo "Node: \$(hostname)"
    echo "Workdir: \$(pwd)"
    echo "Python: \$(venv_fm/bin/python --version 2>&1)"
    echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
    exec bash run.sh
  '
SBATCH
)

echo "Submitted batch job $job_id"
echo "  log out: $SCRIPT_DIR/slurm-$JOB_NAME-$job_id.out"
echo "  log err: $SCRIPT_DIR/slurm-$JOB_NAME-$job_id.err"
