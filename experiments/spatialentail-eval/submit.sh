#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' \
    'sbatch experiments/spatialentail-eval/prepared-final/slurm/untuned.sh' \
    'sbatch experiments/spatialentail-eval/prepared-final/slurm/answer-only-seed-42.sh' \
    'sbatch experiments/spatialentail-eval/prepared-final/slurm/natural-seed-42.sh' \
    'sbatch --dependency=afterany:<three-eval-job-ids> experiments/spatialentail-eval/slurm/summarize.sh'
  exit 0
fi
: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"

prepared=experiments/spatialentail-eval/prepared-final/slurm
[[ -f experiments/spatialentail-eval/prepared-final/jobs.json ]] || {
  echo "Missing prepared evaluation bundle; run spatial.v2.study.evaluation first" >&2
  exit 1
}
untuned=$(sbatch --parsable "$prepared/untuned.sh")
answer=$(sbatch --parsable "$prepared/answer-only-seed-42.sh")
natural=$(sbatch --parsable "$prepared/natural-seed-42.sh")
summary=$(sbatch --parsable \
  --dependency="afterany:$untuned:$answer:$natural" \
  experiments/spatialentail-eval/slurm/summarize.sh)

printf 'untuned=%s\nanswer-only-seed-42=%s\nnatural-seed-42=%s\nsummary=%s\n' \
  "$untuned" "$answer" "$natural" "$summary"
