#!/bin/bash
# Exp 7b — untuned 4B Instruct on TQA-Corr-Single, H100-96 fallback.
#   sbatch experiments/07b-zero-shot-single/slurm/eval-h100-96.sh
#SBATCH --job-name=spatial7b-zs
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=experiments/07b-zero-shot-single/logs/%x-%j.out
#SBATCH --error=experiments/07b-zero-shot-single/logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# #SBATCH --cpus-per-task and SLURM_CPUS_PER_TASK must never differ (Slurm 23+).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

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
