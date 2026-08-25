#!/bin/bash
# Same-node 2-GPU GRPO: GPU0 vLLM serve, GPU1 train.
# Cluster: a100-80 is 1 GPU/node. Dual named cards are h100-47:4 (xgpi*)
# or idle gpu:nv:2 (unknown VRAM). Request 2x h100-47 on one node.
#SBATCH --job-name=spatialgrpo-2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100-47:2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

cd finetune
srun uv sync --extra vllm

if [[ -s ../spatial_grpo_data.jsonl ]]; then
  srun uv run python generate_grpo.py --annotate --out ../spatial_grpo_data.jsonl
else
  srun uv run python generate_grpo.py --n 4000 --out ../spatial_grpo_data.jsonl
fi

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

CFG=./config/qwen3-8b-spatial-grpo-vllm.yaml
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Numeric CUDA_VISIBLE_DEVICES=1 makes Torch dynamo index GPU 1 with a
# length-1 property list. Pin by UUID so each process only has cuda:0.
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ ${#GPU_UUIDS[@]} -lt 2 ]]; then
  echo "Need 2 GPUs in this job, nvidia-smi saw ${#GPU_UUIDS[@]}"
  nvidia-smi -L
  exit 1
fi
echo "vLLM  GPU ${GPU_UUIDS[0]}"
echo "train GPU ${GPU_UUIDS[1]}"
nvidia-smi -L

CUDA_VISIBLE_DEVICES="${GPU_UUIDS[0]}" \
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

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK || true
CUDA_VISIBLE_DEVICES="${GPU_UUIDS[1]}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  uv run python finetune.py qwen3-8b-spatial-grpo-vllm
