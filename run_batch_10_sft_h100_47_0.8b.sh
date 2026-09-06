#!/bin/bash
# Exp 10 — Qwen3.5-0.8B Instruct SFT on 20k Full mix. 1× H100-47 MIG.
# Same 1-GPU python launcher as Exp 8 0.8B (not DDP).
#
#   sbatch run_batch_10_sft_h100_47_0.8b.sh
#   sbatch run_batch_10_sft_h100_47_2b.sh
#SBATCH --job-name=spatial10-08b
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-0.8b-20k}"
export ONLY
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_10_sft_h100_47_param.sh"
