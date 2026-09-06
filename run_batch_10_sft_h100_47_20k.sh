#!/bin/bash
# Exp 10 — experimental 2-MIG DDP: 20k on h100-47:2 (one NVL).
# Known-working 1-GPU path: run_batch_10_sft_h100_96_20k.sh
#
#   sbatch run_batch_10_sft_h100_47.sh
#   sbatch run_batch_10_sft_h100_47_20k.sh
#SBATCH --job-name=spatial10-47-20k
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-47:2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-4b-20k}"
export ONLY
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_10_sft_h100_47_ddp.sh"
