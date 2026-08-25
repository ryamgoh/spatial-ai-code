#!/bin/bash
# Two full H100 96GB on one node: cuda:0 vLLM, cuda:1 train.
# 3h partition cap. ~2000 prompts, checkpoints every 10 steps.
# KV is capped at max_model_len 3072 (do not raise; Qwen3 default is 131072).
#SBATCH --job-name=spatialgrpo-h100
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100-96:2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

cd finetune
srun uv sync --extra vllm

DATA_2K=../spatial_grpo_data_2k.jsonl
DATA_4K=../spatial_grpo_data.jsonl
if [[ ! -s "$DATA_2K" ]]; then
  if [[ -s "$DATA_4K" ]]; then
    echo "Writing 2000-row subset from $DATA_4K"
    head -n 2000 "$DATA_4K" > "$DATA_2K"
  else
    srun uv run python generate_grpo.py --n 2000 --out "$DATA_2K"
  fi
fi
srun uv run python generate_grpo.py --annotate --out "$DATA_2K"

SFT_ADAPTER=./outputs/deepseek-r1-qwen3-8b
MERGED=./outputs/deepseek-r1-qwen3-8b-merged
if [[ ! -f "$MERGED/config.json" ]]; then
  echo "Merging SFT QLoRA into bf16 at $MERGED (adapter left untouched)"
  CUDA_VISIBLE_DEVICES="" srun uv run python merge_sft.py \
    --adapter "$SFT_ADAPTER" \
    --out "$MERGED"
fi
if [[ ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
  echo "SFT adapter missing after merge; aborting"
  exit 1
fi

CFG=./config/qwen3-8b-spatial-grpo-vllm-h100.yaml
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L
n_mig=$(nvidia-smi -L | grep -c 'MIG-' || true)
n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${n_mig}" -lt 2 && "${n_gpu}" -lt 2 ]]; then
  echo "Need 2 GPUs, saw MIG=${n_mig} GPU=${n_gpu}"
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0,1 (vLLM cannot parse UUIDs)"
  export CUDA_VISIBLE_DEVICES=0,1
fi
echo "vLLM  CUDA_VISIBLE_DEVICES=0  (max_model_len=3072, gpu_memory_utilization=0.70)"
echo "train CUDA_VISIBLE_DEVICES=1"

CUDA_VISIBLE_DEVICES=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run axolotl vllm-serve "$CFG" &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

ok=0
for _ in $(seq 1 90); do
  if curl -sfL "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1 \
    || curl -sfL "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 10
done
if [[ "$ok" -ne 1 ]]; then
  echo "vLLM server did not become ready"
  exit 1
fi

RESUME=()
if compgen -G "./outputs/deepseek-r1-qwen3-8b-grpo-h100/checkpoint-*" > /dev/null; then
  echo "Resuming from existing GRPO checkpoints"
  RESUME=(--resume)
fi

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK || true
CUDA_VISIBLE_DEVICES=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  uv run python finetune.py qwen3-8b-spatial-grpo-vllm-h100 "${RESUME[@]}"
