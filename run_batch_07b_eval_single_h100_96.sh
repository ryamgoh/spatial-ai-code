#!/bin/bash
# Exp 7b — untuned 4B Instruct on TQA-Corr-Single, H100-96 fallback.
#   sbatch run_batch_07b_eval_single_h100_96.sh
#SBATCH --job-name=spatial7b-zs
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/07b-zero-shot-single
OUT=$SLURM_SUBMIT_DIR/$EXP/results/zero-shot-single/instruct

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
  --config ../experiments/07b-zero-shot-single/eval-instruct-single.yaml \
  --output-dir "$OUT" || {
  echo "7b zero-shot Single eval FAILED"
  exit 1
}
echo "=== 7b Instruct zero-shot Single OK -> $OUT/"
echo "    cd eval && uv run --no-project python ../experiments/07b-zero-shot-single/scripts/summarize.py"
