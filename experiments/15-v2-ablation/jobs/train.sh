#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/lib/pin-srun-cpus.sh"

: "${ARM:?ARM is required}"
: "${TRAIN_CONFIG:?TRAIN_CONFIG is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

if [[ -f "$MODEL_DIR/COMPLETED" ]] && has_adapter "$MODEL_DIR"; then
  echo "Arm $ARM already completed"
  exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

cd finetune
exec 9>"$SLURM_SUBMIT_DIR/finetune/.uv-sync.lock"
flock 9
uv sync
flock -u 9
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
