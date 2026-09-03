#!/bin/bash
# Exp 6 — Baseline Task Feasibility.
# Zero-shot Qwen3.5-4B-Base and Qwen3.5-4B + Non-shot-2 on TQA-Corr (1329).
# summarize.py slices Single (1038) vs Multi (291) from the same run.
#
#   1. eval instruct  -> results/feasibility/instruct/
#   2. eval base      -> results/feasibility/base/
#   3. summarize.py   -> results/feasibility/SUMMARY.md
#
# Optional SINGLE_EVAL=1 also runs the dedicated TQA-Corr-Single task.
# SKIP_INSTRUCT=1 / SKIP_BASE=1 skip a checkpoint.
#
# Submit:  sbatch run_batch_06_feasibility.sh
#SBATCH --job-name=spatial6-feas
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/06-baseline-task-feasibility
FEAS_OUT=$EXP/results/feasibility
A_FEAS_OUT=$SLURM_SUBMIT_DIR/$FEAS_OUT
mkdir -p "$A_FEAS_OUT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true

cd eval
srun uv sync

run_eval() {
  local tag=$1 cfg=$2
  if [[ -f "$A_FEAS_OUT/$tag/results.json" ]]; then
    echo "=== skip $tag (already at $FEAS_OUT/$tag/results.json) ==="
    return 0
  fi
  echo "=== eval $tag ($cfg) ==="
  local status=0
  srun --cpu-bind=cores uv run python eval_new.py --config "$cfg" || status=$?
  local latest
  latest=$(ls -dt "$SLURM_SUBMIT_DIR/$EXP"/results/2*/ 2>/dev/null | head -1 || true)
  mkdir -p "$A_FEAS_OUT/$tag"
  if [[ -n "${latest:-}" && -f "${latest}results.json" ]]; then
    mv "${latest}"* "$A_FEAS_OUT/$tag/"
    rmdir "${latest}" 2>/dev/null || true
    echo "  -> $A_FEAS_OUT/$tag/"
  fi
  if [[ $status -ne 0 ]]; then
    echo "  eval $tag FAILED (status $status)"
    return "$status"
  fi
  echo "  eval $tag OK"
}

if [[ "${SKIP_INSTRUCT:-0}" != "1" ]]; then
  run_eval instruct ../experiments/06-baseline-task-feasibility/eval-instruct.yaml
fi
if [[ "${SKIP_BASE:-0}" != "1" ]]; then
  run_eval base ../experiments/06-baseline-task-feasibility/eval-base.yaml
fi
if [[ "${SINGLE_EVAL:-0}" == "1" ]]; then
  if [[ "${SKIP_INSTRUCT:-0}" != "1" ]]; then
    run_eval instruct-single ../experiments/06-baseline-task-feasibility/eval-instruct-single.yaml
  fi
  if [[ "${SKIP_BASE:-0}" != "1" ]]; then
    run_eval base-single ../experiments/06-baseline-task-feasibility/eval-base-single.yaml
  fi
fi

srun uv run --no-project python ../experiments/06-baseline-task-feasibility/scripts/summarize.py \
  || echo "summarize failed — check $FEAS_OUT manually"

echo "=== Exp 6 feasibility done. Summary: $A_FEAS_OUT/SUMMARY.md ==="
