#!/bin/bash
# Exp 11 — Qwen3.5-4B 20k on one H100-47 MIG.
# 1-GPU python launcher / acc 8. cpus-per-task=16 (same as Exp 10).
#
#   sbatch run_batch_11_sft_h100_47_2b.sh
#   sbatch run_batch_11_sft_h100_47_4b_small.sh
#   sbatch run_batch_11_sft_h100_47_4b_20k.sh
#SBATCH --job-name=spatial11-4b20
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-4b-20k}"
export ONLY
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_11_sft_h100_47.sh"
