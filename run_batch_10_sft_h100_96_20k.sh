#!/bin/bash
# Exp 10 — 4B Full-mix data-scaling arm: 20k on a second H100 96GB.
#
#   sbatch run_batch_10_sft_h100_96.sh
#   sbatch run_batch_10_sft_h100_96_20k.sh
#SBATCH --job-name=spatial10-20k
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ONLY="${ONLY:-4b-20k}"
export ONLY
# Body + cell table live in run_batch_10_sft_h100.sh
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/run_batch_10_sft_h100.sh"
