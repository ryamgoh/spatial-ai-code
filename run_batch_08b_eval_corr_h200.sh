#!/bin/bash
# Exp 8b — 4B-5k Single-SFT → TQA-Corr 1329 on one H200 (gpu, 3h).
#   sbatch run_batch_08b_eval_corr_h200.sh
#SBATCH --job-name=spatial8b-corr
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

EXP=experiments/08b-sft-transfer-corr
ADAPTER=$SLURM_SUBMIT_DIR/experiments/08-dual-scaling/models/qwen3.5-4b-sft-5000
OUT=$SLURM_SUBMIT_DIR/$EXP/results/transfer/4b-5k-corr

if [[ ! -f "$ADAPTER/adapter_config.json" ]] ||
   { [[ ! -f "$ADAPTER/adapter_model.safetensors" ]] && [[ ! -f "$ADAPTER/adapter_model.bin" ]]; }; then
  echo "Missing 4B-5k adapter at $ADAPTER — aborting."
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
  --config ../experiments/08b-sft-transfer-corr/eval-sft-4b-5000-corr.yaml \
  --output-dir "$OUT" || {
  echo "8b Corr eval FAILED (if walltime, use run_batch_08b_eval_corr_h100_96.sh)"
  exit 1
}
echo "=== 8b 4B-5k → Corr OK -> $OUT/"
echo "    cd eval && uv run --no-project python ../experiments/08b-sft-transfer-corr/scripts/summarize.py"
