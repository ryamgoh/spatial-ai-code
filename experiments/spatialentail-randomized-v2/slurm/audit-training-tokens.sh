#!/bin/bash
#SBATCH --job-name=spatialentail-random-v2-token-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output=experiments/spatialentail-randomized-v2/logs/%x-%j.out
#SBATCH --error=experiments/spatialentail-randomized-v2/logs/%x-%j.err
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

candidate=${TRAINING_CANDIDATE:-experiments/spatialentail-randomized-v2/prepared-training-candidate}
out=${TOKEN_AUDIT_OUT:-$candidate/token-audit.json}

[[ -f "$candidate/manifest.json" ]] || {
  echo "Missing training candidate manifest: $candidate/manifest.json" >&2
  exit 1
}

echo "Auditing randomized SpatialEntail v2 tokenizer lengths"
echo "candidate=$candidate"
echo "output=$out"
echo "started=$(date --iso-8601=seconds)"

export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m spatial.v2.study.training_token_audit \
  "$candidate" \
  --out "$out"

echo "completed=$(date --iso-8601=seconds)"
