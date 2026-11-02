#!/usr/bin/env bash
# Prepare once, validate all four runs, then submit exactly four SFT jobs.
set -euo pipefail
EXP=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
ROOT=$(dirname "$(dirname "$EXP")")
ARTIFACTS=$(realpath "${1:?Usage: bash submit-training.sh /absolute/artifacts}")
command -v sbatch >/dev/null
command -v uv >/dev/null
mkdir -p "$EXP/runs"
SUBMISSION=$(mktemp -d "$EXP/runs/sft-XXXXXXXX")
CAMPAIGN="$SUBMISSION/campaign"
uv run --project "$ROOT" --frozen --no-sync python "$EXP/sft.py" prepare \
    --artifacts "$ARTIFACTS" --output "$CAMPAIGN"
for style in natural symbolic; do
    for seed in 42 43; do
        uv run --project "$ROOT" --frozen --no-sync python "$EXP/sft.py" verify \
            --run "$CAMPAIGN/$style-seed-$seed"
    done
done
printf 'artifacts=%s\ncampaign=%s\n' "$ARTIFACTS" "$CAMPAIGN" > "$SUBMISSION/jobs.txt"
trap 'echo "Submission stopped; existing jobs were not cancelled. See '"$SUBMISSION"'/jobs.txt" >&2' ERR
for style in natural symbolic; do
    for seed in 42 43; do
        key="$style-seed-$seed"
        result=$(sbatch --parsable --chdir="$ROOT" --export=ALL \
            --output="$SUBMISSION/$key-%j.out" --error="$SUBMISSION/$key-%j.err" \
            "$EXP/slurm/train.sh" "$CAMPAIGN/$key")
        job=${result%%;*}
        [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $result" >&2; exit 1; }
        printf '%s=%s\n' "$key" "$job" >> "$SUBMISSION/jobs.txt"
    done
done
echo "Four SFT jobs submitted. No final evaluation jobs submitted. Record: $SUBMISSION/jobs.txt"
