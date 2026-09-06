#!/bin/bash
# Exp 10 — 0.8B 1.5k then 5k on one H100-47 MIG (1-GPU python, not DDP).
#
#   sbatch run_batch_10_sft_h100_47_0.8b_small.sh
#SBATCH --job-name=spatial10-08b-s
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-0.8b-1.5k,0.8b-5k}"
export ONLY
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_10_sft_h100_47_param.sh"
