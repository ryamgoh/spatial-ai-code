#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/lib/pin-srun-cpus.sh"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/experiments/15-v2-ablation/jobs/common.bash"

: "${EVAL_NAME:?EVAL_NAME is required}"
: "${EVAL_CONFIG:?EVAL_CONFIG is required}"
: "${RESULT_DIR:?RESULT_DIR is required}"

if [[ "${UNTUNED:-0}" != "1" ]]; then
  : "${MODEL_DIR:?MODEL_DIR is required for adapter evaluation}"
  [[ -f "$MODEL_DIR/COMPLETED" ]] && has_adapter "$MODEL_DIR" || {
    echo "Evaluation $EVAL_NAME has no verified adapter at $MODEL_DIR" >&2
    exit 1
  }
fi
if [[ -f "$RESULT_DIR/COMPLETED" && -f "$RESULT_DIR/results.json" ]]; then
  echo "Evaluation $EVAL_NAME already completed"
  exit 0
fi

prepared_dir=${EVAL_PREPARED_DIR:-$SLURM_SUBMIT_DIR/experiments/spatialentail-eval/prepared-final}
python3 - "$prepared_dir/jobs.json" "$prepared_dir" "$SLURM_SUBMIT_DIR" <<'CHECK'
import hashlib, json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
base = pathlib.Path(sys.argv[2])
root = pathlib.Path(sys.argv[3])
for name, expected in receipt["file_sha256"].items():
    if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Prepared evaluation input changed: " + name)
for name, expected in receipt["environment_sha256"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Evaluation environment changed: " + name)
CHECK

normalize_single_cuda_device
mkdir -p "$RESULT_DIR"
sync_uv_project "$SLURM_SUBMIT_DIR/eval"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd eval
uv run python eval_new.py \
  --config "$EVAL_CONFIG" \
  --stages 2 \
  --output-dir "$RESULT_DIR"

[[ -f "$RESULT_DIR/results.json" ]] || {
  echo "Evaluation $EVAL_NAME did not produce results.json" >&2
  exit 1
}
touch "$RESULT_DIR/COMPLETED"
