#!/bin/bash
# One-H200 retention evaluation for a completed V14 GRPO probe.
# Designed to be submitted with --dependency=afterok:<GRPO_JOB_ID>.
#SBATCH --job-name=spatial14-retain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/14-v13-grpo-feasibility/logs/%x-%j.out
#SBATCH --error=experiments/14-v13-grpo-feasibility/logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/slurm/lib/pin-srun-cpus.sh"

EXP=experiments/14-v13-grpo-feasibility
MERGED=$EXP/models/qwen3.5-4b-delta-merged
GRPO_ADAPTER=$EXP/models/qwen3.5-4b-grpo-probe

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

if [[ ! -f "$MERGED/config.json" ]]; then
  echo "Missing merged delta-SFT policy base: $MERGED"
  exit 1
fi
if ! has_adapter "$GRPO_ADAPTER"; then
  echo "Missing completed GRPO adapter: $GRPO_ADAPTER"
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
mkdir -p "$EXP/logs" "$EXP/results"
nvidia-smi -L || true

cd eval
uv sync || exit 1
for row in \
  "grpo-v13-stage1 eval-grpo-v13.yaml" \
  "grpo-breakpoint-stage1 eval-grpo-breakpoint.yaml"; do
  set -- $row
  tag=$1 config=$2 out=../$EXP/results/$1
  if [[ "${FORCE_EVAL:-0}" != "1" && -f "$out/results.json" ]]; then
    echo "=== skip $tag (already complete) ==="
    continue
  fi
  echo "=== evaluate GRPO retention: $tag ==="
  uv run python eval_new.py \
    --config ../$EXP/"$config" \
    --stages 1 \
    --output-dir "$out" || exit 1
done
cd "$SLURM_SUBMIT_DIR"

uv run --no-project python "$EXP/scripts/summarize.py" || exit 1
echo "Retention complete: $EXP/results/SUMMARY.md"
