#!/bin/bash
#SBATCH --job-name=spatialentail-prepare-general-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/experiments/15-v2-ablation/jobs/common.bash"
sync_uv_project "$SLURM_SUBMIT_DIR/eval"
uv run python -m spatial.v2.study.comparator_evaluation \
  experiments/spatialentail-randomized-v2/prepared-benchmark-candidate/source-natural.jsonl \
  experiments/spatialentail-randomized-v2/prepared-general-eval
