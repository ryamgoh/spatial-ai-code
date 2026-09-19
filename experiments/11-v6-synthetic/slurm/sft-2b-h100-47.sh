#!/bin/bash
# Exp 11 — Qwen3.5-2B 1.5k then 6k then 20k on one H100-47 MIG.
# 1-GPU python launcher (not DDP). cpus-per-task=16 (same as Exp 10).
#
#   sbatch experiments/11-v6-synthetic/slurm/sft-2b-h100-47.sh
#   sbatch experiments/11-v6-synthetic/slurm/sft-4b-small-h100-47.sh
#   sbatch experiments/11-v6-synthetic/slurm/sft-4b-20k-h100-47.sh
#SBATCH --job-name=spatial11-2b
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=experiments/11-v6-synthetic/logs/%x-%j.out
#SBATCH --error=experiments/11-v6-synthetic/logs/%x-%j.err

ONLY="${ONLY:-2b-1.5k,2b-6k,2b-20k}"
export ONLY
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/experiments/11-v6-synthetic/slurm/sft-driver.sh"
