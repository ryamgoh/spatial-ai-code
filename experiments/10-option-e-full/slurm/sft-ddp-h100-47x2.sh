#!/bin/bash
# Exp 10 — experimental 2-MIG DDP: 1.5k then 5k on h100-47:2 (one NVL).
# Known-working 1-GPU path: experiments/10-option-e-full/slurm/sft-h100-96.sh
#
#   sbatch experiments/10-option-e-full/slurm/sft-ddp-h100-47x2.sh
#   sbatch experiments/10-option-e-full/slurm/sft-ddp-20k-h100-47x2.sh
#SBATCH --job-name=spatial10-47
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-47:2
#SBATCH --output=experiments/10-option-e-full/logs/%x-%j.out
#SBATCH --error=experiments/10-option-e-full/logs/%x-%j.err

ONLY="${ONLY:-4b-1.5k,4b-5k}"
export ONLY
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/experiments/10-option-e-full/slurm/sft-ddp-driver.sh"
