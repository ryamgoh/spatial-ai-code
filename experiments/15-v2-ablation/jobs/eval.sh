#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/lib/pin-srun-cpus.sh"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/experiments/15-v2-ablation/jobs/common.bash"

: "${ARM:?ARM is required}"
: "${EVAL_CONFIG:?EVAL_CONFIG is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${RESULT_DIR:?RESULT_DIR is required}"

[[ -f "$MODEL_DIR/COMPLETED" ]] && has_adapter "$MODEL_DIR" || {
  echo "Arm $ARM has no verified adapter"
  exit 1
}
if [[ -f "$RESULT_DIR/COMPLETED" && -f "$RESULT_DIR/results.json" ]]; then
  echo "Evaluation for $ARM already completed"
  exit 0
fi

normalize_single_cuda_device

mkdir -p "$RESULT_DIR"
sync_uv_project "$SLURM_SUBMIT_DIR/eval"
cd eval
uv run python eval_new.py \
  --config "$EVAL_CONFIG" \
  --stages "${EVAL_STAGES:-1}" \
  --output-dir "$RESULT_DIR"

[[ -f "$RESULT_DIR/results.json" ]] || {
  echo "Evaluation for $ARM did not produce results.json"
  exit 1
}
touch "$RESULT_DIR/COMPLETED"
