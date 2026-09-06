#!/bin/bash
# Exp 10 — 4B 1.5k then 5k on one H100-47 MIG.
# Same 1-GPU python launcher / acc 8 as the 96 path (not DDP).
# 4B-20k stays on h100-96.
#
#   sbatch run_batch_10_sft_h100_47_4b_small.sh
#SBATCH --job-name=spatial10-47-4bs
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-4b-1.5k,4b-5k}"
export ONLY
# vLLM/axolotl do int(CUDA_VISIBLE_DEVICES); SLURM may pass a MIG UUID.
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
# Same body as the 96 4B jobs. Do not sbatch run_batch_10_sft_h100_47.sh (DDP).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_10_sft_h100.sh"
