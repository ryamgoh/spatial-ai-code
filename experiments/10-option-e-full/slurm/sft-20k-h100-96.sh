#!/bin/bash
# Exp 10 — 4B Full-mix data-scaling arm: 20k on a second H100 96GB.
#
#   sbatch experiments/10-option-e-full/slurm/sft-h100-96.sh
#   sbatch experiments/10-option-e-full/slurm/sft-20k-h100-96.sh
#SBATCH --job-name=spatial10-20k
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --output=experiments/10-option-e-full/logs/%x-%j.out
#SBATCH --error=experiments/10-option-e-full/logs/%x-%j.err

ONLY="${ONLY:-4b-20k}"
export ONLY
# Body + cell table live in experiments/10-option-e-full/slurm/sft-driver.sh
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/experiments/10-option-e-full/slurm/sft-driver.sh"
