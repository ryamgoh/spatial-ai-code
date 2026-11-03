#!/usr/bin/env bash
# Four H100 adapter jobs and three H200 shot-setting baseline jobs.
set -euo pipefail
EXP=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
ROOT=$(dirname "$(dirname "$EXP")")
CAMPAIGN=$(realpath "${1:?Usage: bash submit-evaluation.sh /absolute/campaign [full-training-commit]}")
TRAINING_REVISION=${2:-}
command -v sbatch >/dev/null
command -v srun >/dev/null
check=(--run "$CAMPAIGN")
[[ -z "$TRAINING_REVISION" ]] || check+=(--training-revision "$TRAINING_REVISION")
# Run the CPU-only checks on a compute node, not on the login node.
echo "Preflight: requesting 2 CPUs and 8 GB RAM on normal; waiting for allocation."
srun --job-name=spatialentail-eval-preflight --partition=normal \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --chdir="$ROOT" uv --native-tls run --python 3.12 --no-project \
    --script "$EXP/generate.py" verify-evaluation "${check[@]}"
SUBMISSION=$(mktemp -d "$CAMPAIGN/eval-submission-XXXXXXXX")
printf 'campaign=%s\n' "$CAMPAIGN" > "$SUBMISSION/jobs.txt"
trap 'echo "Submission stopped; existing jobs were not cancelled. See '"$SUBMISSION"'/jobs.txt" >&2' ERR
submit() {
    local key=$1 gpu=$2 run=$3 mode=$4 shots=$5 result job
    result=$(sbatch --parsable --chdir="$ROOT" --export=ALL \
        --gres="gpu:$gpu:1" \
        --output="$SUBMISSION/$key-%j.out" --error="$SUBMISSION/$key-%j.err" \
        "$EXP/slurm/evaluate.sh" "$run" "$mode" "$shots" "$TRAINING_REVISION")
    job=${result%%;*}
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $result" >&2; exit 1; }
    printf '%s=%s\n' "$key" "$job" >> "$SUBMISSION/jobs.txt"
}
for seed in 42 43; do
    for style in natural symbolic; do
        submit "selected-$style-$seed" h100-47 "$CAMPAIGN/$style-seed-$seed" selected 0
    done
done
for shots in 0 1 3; do
    submit "untuned-$shots-shot" h200-141 "$CAMPAIGN/natural-seed-42" untuned "$shots"
done
echo "Submitted four tuned and three untuned jobs (20 results). Record: $SUBMISSION/jobs.txt"
