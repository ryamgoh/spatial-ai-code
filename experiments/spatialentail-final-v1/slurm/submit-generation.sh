#!/usr/bin/env bash
# Submit generation only. This script never submits SFT or evaluation jobs.
set -euo pipefail
EXP=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
ROOT=$(dirname "$(dirname "$EXP")")
MODE=${1:-full}
case "$MODE" in
    full|smoke) ;;
    *) echo "Usage: bash $0 [full|smoke]" >&2; exit 2 ;;
esac
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 2; }
command -v uv >/dev/null || { echo "Install uv on the server first." >&2; exit 2; }
for file in \
    "$ROOT/data/spatialeval_org.jsonl" \
    "$ROOT/experiments/spatialentail-randomized-v2/prepared-corr-eval/data/corr.jsonl" \
    "$ROOT/experiments/spatialentail-randomized-v2/prepared-corr-eval/correction-summary.json"; do
    [[ -s "$file" ]] || { echo "Missing tracked external source: $file" >&2; exit 2; }
done
mkdir -p "$EXP/runs"
SUBMISSION=$(mktemp -d "$EXP/runs/generation-${MODE}-XXXXXXXX")
export FINAL_RUN_DIR="$SUBMISSION/artifacts"
export FINAL_GENERATION_MODE="$MODE"
export FINAL_EXPERIMENT_DIR="$EXP"
export UV_CACHE_DIR="$SUBMISSION/cache/uv"
export HF_HOME="$SUBMISSION/cache/huggingface"
printf 'mode=%s\nrun_dir=%s\n' "$MODE" "$FINAL_RUN_DIR" > "$SUBMISSION/submission.txt"
if ! result=$(sbatch --parsable --chdir="$ROOT" --export=ALL \
    --output="$SUBMISSION/generation-%j.out" \
    --error="$SUBMISSION/generation-%j.err" \
    "$EXP/slurm/generate.sh"); then
    echo "Submission failed. Preserved record: $SUBMISSION/submission.txt" >&2
    exit 1
fi
job=${result%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || {
    printf 'unexpected_sbatch_output=%s\n' "$result" >> "$SUBMISSION/submission.txt"
    echo "Unexpected sbatch output: $result" >&2
    exit 1
}
printf 'generation_job=%s\n' "$job" >> "$SUBMISSION/submission.txt"
printf 'Generation job: %s\nArtifacts: %s\nLogs: %s\n' "$job" "$FINAL_RUN_DIR" "$SUBMISSION"
