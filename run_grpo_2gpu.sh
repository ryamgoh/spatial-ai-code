#!/bin/bash
# Same-node 2-device GRPO: cuda:0 vLLM serve, cuda:1 train.
# h100-47 is MIG 3g.47gb: two slices often share one physical H100 NVL.
# SLURM already isolates those slices. vLLM ModelConfig does int() on
# CUDA_VISIBLE_DEVICES, so pin with 0/1, not MIG- UUIDs.
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

SFT_ADAPTER=../experiments/03-sft-vs-baseline/models/deepseek-r1-qwen3-8b
MERGED=../experiments/05-grpo/models/deepseek-r1-qwen3-8b-merged
if [[ ! -f "$MERGED/config.json" ]]; then
  echo "Merging SFT QLoRA into bf16 at $MERGED (adapter left untouched)"
  CUDA_VISIBLE_DEVICES="" srun uv run python merge_sft.py \
    --base deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --adapter "$SFT_ADAPTER" \
    --out "$MERGED"
fi
if [[ ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
  echo "SFT adapter missing after merge; aborting"
  exit 1
fi

CFG=../experiments/05-grpo/train-grpo-8b-vllm.yaml
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi -L
n_mig=$(nvidia-smi -L | grep -c 'MIG-' || true)
n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${n_mig}" -lt 2 && "${n_gpu}" -lt 2 ]]; then
  echo "Need 2 GPUs or MIG slices, saw MIG=${n_mig} GPU=${n_gpu}"
  exit 1
fi
# SLURM may export MIG-UUIDs; vLLM then does int(CUDA_VISIBLE_DEVICES).
# Cgroup already has the two slices. 0/1 are the CUDA indices inside the job.
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  echo "Remapping CUDA_VISIBLE_DEVICES to 0,1 (vLLM cannot parse MIG UUIDs)"
  export CUDA_VISIBLE_DEVICES=0,1
fi
echo "vLLM  CUDA_VISIBLE_DEVICES=0"
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

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK || true
CUDA_VISIBLE_DEVICES=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  uv run python finetune.py ../experiments/05-grpo/train-grpo-8b-vllm.yaml
