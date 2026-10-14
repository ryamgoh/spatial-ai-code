#!/bin/bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

prepared=experiments/spatialentail-eval/prepared-final
[[ -f "$prepared/jobs.json" ]] || {
  echo "Missing prepared evaluation bundle; run spatial.v2.study.evaluation first" >&2
  exit 1
}

# The prepared receipt is the source of truth for which models to submit, so a
# regenerated bundle (new seeds) needs no edit here.
scripts=()
while IFS= read -r script; do
  scripts+=("$script")
done < <(
  python3 - "$prepared/jobs.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
for job in receipt["jobs"]:
    print(job["script"])
PY
)

if [[ "${1:-}" == "--dry-run" ]]; then
  for script in "${scripts[@]}"; do
    printf 'sbatch %s\n' "$script"
  done
  printf 'sbatch --dependency=afterany:<eval-job-ids> experiments/spatialentail-eval/slurm/summarize.sh\n'
  exit 0
fi

ids=()
for script in "${scripts[@]}"; do
  job_id=$(sbatch --parsable "$script")
  ids+=("$job_id")
  printf '%s=%s\n' "$(basename "$script" .sh)" "$job_id"
done

dependency=$(IFS=:; echo "${ids[*]}")
summary=$(sbatch --parsable \
  --dependency="afterany:$dependency" \
  experiments/spatialentail-eval/slurm/summarize.sh)
printf 'summary=%s\n' "$summary"
