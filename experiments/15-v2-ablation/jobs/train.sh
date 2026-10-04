#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/lib/pin-srun-cpus.sh"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/experiments/15-v2-ablation/jobs/common.bash"

: "${ARM:?ARM is required}"
: "${TRAIN_CONFIG:?TRAIN_CONFIG is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"

if [[ -f "$MODEL_DIR/COMPLETED" ]] && has_adapter "$MODEL_DIR"; then
  echo "Arm $ARM already completed"
  exit 0
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1
normalize_single_cuda_device

sync_uv_project "$SLURM_SUBMIT_DIR/finetune"
cd finetune
resume=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume+=(--resume)
fi
uv run python finetune.py "$TRAIN_CONFIG" "${resume[@]}"

cd "$SLURM_SUBMIT_DIR"
has_adapter "$MODEL_DIR" || {
  echo "Training did not produce a complete adapter for $ARM"
  exit 1
}
touch "$MODEL_DIR/COMPLETED"
