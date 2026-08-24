#!/bin/bash
#SBATCH --job-name=spatialgrpo
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
# vLLM colocate sleep/memory pool asserts if expandable_segments is on.
# https://github.com/pytorch/pytorch/issues/147851
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_ALLOC_CONF=expandable_segments:False
export AXOLOTL_NO_TELEMETRY=1
export AXOLOTL_DO_NOT_TRACK=1

cd finetune
srun uv sync --extra vllm
srun uv run python generate_grpo.py --n 4000 --out ../spatial_grpo_data.jsonl
srun --cpu-bind=cores uv run python finetune.py qwen3-8b-spatial-grpo
