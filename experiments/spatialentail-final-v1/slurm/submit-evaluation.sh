#!/usr/bin/env bash
# Eight selected-adapter evaluations plus four style-matched untuned evaluations.
set -euo pipefail
EXP=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
ROOT=$(dirname "$(dirname "$EXP")")
CAMPAIGN=$(realpath "${1:?Usage: bash submit-evaluation.sh /absolute/campaign}")
command -v sbatch >/dev/null
for style in natural symbolic; do
    for seed in 42 43; do
        uv run --project "$ROOT" --frozen --no-sync python "$EXP/sft.py" verify \
            --run "$CAMPAIGN/$style-seed-$seed" --selected
    done
done
SUBMISSION=$(mktemp -d "$CAMPAIGN/eval-submission-XXXXXXXX")
printf 'campaign=%s\n' "$CAMPAIGN" > "$SUBMISSION/jobs.txt"
trap 'echo "Submission stopped; existing jobs were not cancelled. See '"$SUBMISSION"'/jobs.txt" >&2' ERR
for mode in selected untuned; do
    for style in natural symbolic; do
        seeds=(42 43)
        [[ "$mode" != untuned ]] || seeds=(42)
        for seed in "${seeds[@]}"; do
            for benchmark in sft-bench external-test; do
                key="$mode-$style-$seed-$benchmark"
                result=$(sbatch --parsable --chdir="$ROOT" --export=ALL \
                    --output="$SUBMISSION/$key-%j.out" --error="$SUBMISSION/$key-%j.err" \
                    "$EXP/slurm/evaluate.sh" "$CAMPAIGN/$style-seed-$seed" "$benchmark" "$mode")
                job=${result%%;*}
                [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $result" >&2; exit 1; }
                printf '%s=%s\n' "$key" "$job" >> "$SUBMISSION/jobs.txt"
            done
        done
    done
done
echo "Submitted eight tuned and four untuned evaluations. Record: $SUBMISSION/jobs.txt"
