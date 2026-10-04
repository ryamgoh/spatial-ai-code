#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/lib/pin-srun-cpus.sh"

: "${ARM:?ARM is required}"
: "${EVAL_CONFIG:?EVAL_CONFIG is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${RESULT_DIR:?RESULT_DIR is required}"

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

[[ -f "$MODEL_DIR/COMPLETED" ]] && has_adapter "$MODEL_DIR" || {
  echo "Arm $ARM has no verified adapter"
  exit 1
}
if [[ -f "$RESULT_DIR/COMPLETED" && -f "$RESULT_DIR/results.json" ]]; then
  echo "Evaluation for $ARM already completed"
  exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

mkdir -p "$RESULT_DIR"
cd eval
exec 9>"$SLURM_SUBMIT_DIR/eval/.uv-sync.lock"
flock 9
uv sync
flock -u 9
uv run python eval_new.py \
  --config "$EVAL_CONFIG" \
  --stages "${EVAL_STAGES:-1}" \
  --output-dir "$RESULT_DIR"

[[ -f "$RESULT_DIR/results.json" ]] || {
  echo "Evaluation for $ARM did not produce results.json"
  exit 1
}
touch "$RESULT_DIR/COMPLETED"
