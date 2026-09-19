#!/bin/bash
# Exp 10 — 4B 1.5k then 5k then 20k on one H100-47 MIG.
# Same 1-GPU python launcher / acc 8 as the 96 path (not DDP).
#
#   sbatch experiments/10-option-e-full/slurm/sft-4b-small-h100-47.sh
#   sbatch experiments/10-option-e-full/slurm/sft-0.8b-h100-47.sh
#   sbatch experiments/10-option-e-full/slurm/sft-2b-h100-47.sh
#SBATCH --job-name=spatial10-47-4bs
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=experiments/10-option-e-full/logs/%x-%j.out
#SBATCH --error=experiments/10-option-e-full/logs/%x-%j.err

ONLY="${ONLY:-4b-1.5k,4b-5k,4b-20k}"
export ONLY
# vLLM/axolotl do int(CUDA_VISIBLE_DEVICES); SLURM may pass a MIG UUID.
if [[ "${CUDA_VISIBLE_DEVICES-}" == *MIG-* || "${CUDA_VISIBLE_DEVICES-}" == *GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
# Same body as the 96 4B jobs. Do not sbatch experiments/10-option-e-full/slurm/sft-ddp-h100-47x2.sh (DDP).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/experiments/10-option-e-full/slurm/sft-driver.sh"
