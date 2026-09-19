#!/bin/bash
# Exp 10 — Qwen3.5-0.8B 1.5k then 5k then 20k on one H100-47 MIG.
# 1-GPU python launcher (not DDP).
#
#   sbatch experiments/10-option-e-full/slurm/sft-0.8b-h100-47.sh
#   sbatch experiments/10-option-e-full/slurm/sft-2b-h100-47.sh
#   sbatch experiments/10-option-e-full/slurm/sft-4b-small-h100-47.sh
#SBATCH --job-name=spatial10-08b
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=experiments/10-option-e-full/logs/%x-%j.out
#SBATCH --error=experiments/10-option-e-full/logs/%x-%j.err

ONLY="${ONLY:-0.8b-1.5k,0.8b-5k,0.8b-20k}"
export ONLY
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/experiments/10-option-e-full/slurm/sft-param-driver.sh"
