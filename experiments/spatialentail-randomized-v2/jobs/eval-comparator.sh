#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/lib/pin-srun-cpus.sh"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/experiments/15-v2-ablation/jobs/common.bash"

: "${EVAL_NAME:?EVAL_NAME is required}"
: "${EVAL_CONFIG:?EVAL_CONFIG is required}"
: "${EVAL_PREPARED_DIR:?EVAL_PREPARED_DIR is required}"
: "${RESULT_DIR:?RESULT_DIR is required}"

if [[ -f "$RESULT_DIR/COMPLETED" && -f "$RESULT_DIR/results.json" ]]; then
  echo "Comparator evaluation $EVAL_NAME already completed"
  exit 0
fi

python3 - "$EVAL_PREPARED_DIR/jobs.json" "$EVAL_PREPARED_DIR" "$SLURM_SUBMIT_DIR" <<'CHECK'
import hashlib
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
prepared = pathlib.Path(sys.argv[2])
root = pathlib.Path(sys.argv[3])
for name, expected in receipt["file_sha256"].items():
    if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Prepared comparator input changed: " + name)
for name, expected in receipt["environment_sha256"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Comparator environment changed: " + name)
CHECK

normalize_single_cuda_device
mkdir -p "$RESULT_DIR"
sync_uv_project "$SLURM_SUBMIT_DIR/eval"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd eval
uv run python eval_comparators.py \
  --config "$EVAL_CONFIG" \
  --output-dir "$RESULT_DIR"

[[ -f "$RESULT_DIR/results.json" ]] || {
  echo "Comparator evaluation $EVAL_NAME did not produce results.json" >&2
  exit 1
}
touch "$RESULT_DIR/COMPLETED"
