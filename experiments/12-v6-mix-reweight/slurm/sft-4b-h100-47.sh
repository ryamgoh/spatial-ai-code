#!/bin/bash
# Exp 12 — Qwen3.5-4B 1.5k (default) on one H100-47 MIG.
# 1-GPU python launcher / acc 8. cpus-per-task=16 (same as Exp 10/11).
#
#   sbatch experiments/12-v6-mix-reweight/slurm/sft-4b-h100-47.sh
#   ONLY=4b-6k sbatch experiments/12-v6-mix-reweight/slurm/sft-4b-h100-47.sh
#   sbatch experiments/12-v6-mix-reweight/slurm/sft-4b-18k-h100-47.sh
#SBATCH --job-name=spatial12-4b
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=experiments/12-v6-mix-reweight/logs/%x-%j.out
#SBATCH --error=experiments/12-v6-mix-reweight/logs/%x-%j.err

ONLY="${ONLY:-4b-1.5k}"
export ONLY
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/experiments/12-v6-mix-reweight/slurm/sft-driver.sh"
