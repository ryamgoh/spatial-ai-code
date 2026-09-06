#!/bin/bash
# Exp 9 — 4B mix5k SFT eval on TQA-Corr 1329 (H200, gpu, 3h).
#
#   sbatch run_batch_09_eval_corr_h200.sh
#SBATCH --job-name=spatial9-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ADAPTER=$SLURM_SUBMIT_DIR/experiments/09-multi-sft/models/qwen3.5-4b-sft-mix5k
OUT=$SLURM_SUBMIT_DIR/experiments/09-multi-sft/results/corr/4b-mix5k

if [[ ! -f "$ADAPTER/adapter_config.json" ]] ||
   { [[ ! -f "$ADAPTER/adapter_model.safetensors" ]] && [[ ! -f "$ADAPTER/adapter_model.bin" ]]; }; then
  echo "Missing mix5k adapter at $ADAPTER — aborting."
  exit 1
fi
if [[ -f "$OUT/results.json" ]]; then
  echo "Already have $OUT/results.json — nothing to do."
  exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true

cd eval
srun uv sync
mkdir -p "$OUT"
srun --cpu-bind=cores uv run python eval_new.py \
  --config ../experiments/09-multi-sft/eval-sft-4b-mix5k-corr.yaml \
  --output-dir "$OUT" || {
  echo "Exp 9 Corr eval FAILED"
  exit 1
}
echo "=== Exp 9 4B-mix5k → Corr OK -> $OUT/"
echo "    cd eval && uv run --no-project python ../experiments/09-multi-sft/scripts/summarize.py"
