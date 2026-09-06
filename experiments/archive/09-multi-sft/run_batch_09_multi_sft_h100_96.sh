#!/bin/bash
# Exp 9 — generate 5k Corr-mix traces + QLoRA SFT Qwen3.5-4B Instruct.
#
#   sbatch run_batch_09_multi_sft_h100_96.sh
# Overrides: SKIP_TRAIN=1
#SBATCH --job-name=spatial9-sft
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/09-multi-sft
SFT_DATA=data/spatial_sft_corr_mix_5000_train.jsonl
A_SFT_DATA=$SLURM_SUBMIT_DIR/$SFT_DATA
ADAPTER=$EXP/models/qwen3.5-4b-sft-mix5k
A_ADAPTER=$SLURM_SUBMIT_DIR/$ADAPTER

has_adapter() {
  [[ -f "$1/adapter_config.json" ]] &&
    { [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]; }
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_NO_TELEMETRY=1

echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L || true

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "SKIP_TRAIN=1"
  exit 0
fi

cd finetune
export PYTHONPATH="$SLURM_SUBMIT_DIR/finetune${PYTHONPATH:+:$PYTHONPATH}"
srun uv sync

if [[ -s "$A_SFT_DATA" ]]; then
  echo "=== 5k mix already at $SFT_DATA ==="
else
  echo "=== Generate 5000 Corr-mix traces (seed 42) ==="
  srun uv run python generate_all.py \
    --out ../data/spatial_sft_corr_mix_5000.jsonl \
    --test-split 0 \
    --seed 42 \
    --num-type0-1-answer 1249 \
    --num-type0-2-answer 350 \
    --num-type0-4-answer 0 \
    --num-type1-1-answer 1136 \
    --num-type1-2-answer 0 \
    --num-type1-4-answer 745 \
    --num-type2 1520 || {
    echo "SFT data generation FAILED — aborting."
    exit 1
  }
fi
if [[ ! -s "$A_SFT_DATA" ]]; then
  echo "Missing $A_SFT_DATA after generate; aborting"
  exit 1
fi

if has_adapter "$A_ADAPTER"; then
  echo "=== skip SFT (adapter complete at $ADAPTER) ==="
else
  echo "=== SFT QLoRA 4B mix5k ==="
  srun --cpu-bind=cores uv run axolotl train \
    ../experiments/09-multi-sft/train-sft-4b-mix5k.yaml --launcher python || {
    echo "SFT FAILED — aborting."
    exit 1
  }
  if ! has_adapter "$A_ADAPTER"; then
    echo "SFT did not produce a complete adapter at $A_ADAPTER; aborting"
    exit 1
  fi
fi

echo "=== Exp 9 SFT done. Eval: sbatch run_batch_09_eval_corr_h200.sh ==="
