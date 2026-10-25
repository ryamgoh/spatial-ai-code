#!/bin/bash
#SBATCH --job-name=spatialentail-eval-untuned
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic/logs/%x-%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"
export EVAL_NAME=untuned
export EVAL_CONFIG="$SLURM_SUBMIT_DIR/experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic/configs/untuned.yaml"
export EVAL_PREPARED_DIR="$SLURM_SUBMIT_DIR/experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic"
export MODEL_DIR="$SLURM_SUBMIT_DIR"
export RESULT_DIR="$SLURM_SUBMIT_DIR/experiments/spatialentail-randomized-v2/prepared-transfer-diagnostic/results/untuned"
export UNTUNED=1
bash experiments/spatialentail-eval/jobs/eval.sh
