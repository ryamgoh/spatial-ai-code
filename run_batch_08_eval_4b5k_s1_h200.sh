#!/bin/bash
# Exp 8 — 4B-5k SFT on Single, --stages 1 (letter from LoRA CoT, no constrained re-ask).
# Does not overwrite results/scaling/4b-5k/ (that is stages 2).
#
#   sbatch run_batch_08_eval_4b5k_s1_h200.sh
#SBATCH --job-name=spatial8-4b5k-s1
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

ADAPTER=$SLURM_SUBMIT_DIR/experiments/08-dual-scaling/models/qwen3.5-4b-sft-5000
OUT=$SLURM_SUBMIT_DIR/experiments/08-dual-scaling/results/stages1/4b-5k

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
  --config ../experiments/08-dual-scaling/eval-sft-4b-5000.yaml \
  --stages 1 \
  --output-dir "$OUT" || {
  echo "4B-5k stages-1 eval FAILED"
  exit 1
}
echo "=== 4B-5k stages 1 OK -> $OUT/"
echo "    Compare to results/scaling/4b-5k/ (stages 2, 98.5%)"
