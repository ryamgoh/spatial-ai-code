#!/bin/bash
# Exp 8 — 0.8B-20k eval-only on one H200 141GB (xgpk*, partition gpu, 3h).
# Adapter must already exist at models/qwen3.5-0.8b-sft-20000/.
# One card, not :4 — eval_new.py is a single vLLM process.
#
#   sbatch run_batch_08_eval_0.8b_h200.sh
#SBATCH --job-name=spatial8-eval-08b
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

EXP=experiments/08-dual-scaling
ADAPTER=$SLURM_SUBMIT_DIR/$EXP/models/qwen3.5-0.8b-sft-20000
OUT=$SLURM_SUBMIT_DIR/$EXP/results/scaling/0.8b-20k

if [[ ! -f "$ADAPTER/adapter_config.json" ]] ||
   { [[ ! -f "$ADAPTER/adapter_model.safetensors" ]] && [[ ! -f "$ADAPTER/adapter_model.bin" ]]; }; then
  echo "Missing 0.8B adapter at $ADAPTER — aborting."
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
  --config ../experiments/08-dual-scaling/eval-sft-0.8b-20000-h200.yaml \
  --output-dir "$OUT" || {
  echo "0.8B eval FAILED"
  exit 1
}
echo "=== 0.8B eval OK -> $OUT/"
