#!/usr/bin/env bash
# Run with bash from the repository root; submits exactly three jobs.
set -euo pipefail
EXP=experiments/spatialentail-composition-v1
if [[ ! -f "$EXP/active-model.json" || ! -f spatial/v2/entailment_thinking_training.py ]]; then
    echo "Run this command from the spatial-ai-code repository root." >&2
    exit 2
fi
for file in generation-exclusions-v1.json entailment-config-frozen-v1.json \
    CONTROLLED-TRAINING-BUILD-v2.manifest.json unique-only-selection-v1/selection.json; do
    if [[ ! -s "$EXP/$file" ]]; then
        echo "Missing tracked generation input: $file. Pull the complete code update." >&2
        exit 2
    fi
done
command -v sbatch >/dev/null || { echo "sbatch is not available on this host." >&2; exit 2; }
mkdir -p "$EXP/logs" "$EXP/runs"
run=$(mktemp -d "$EXP/runs/thinking-XXXXXXXX")
export PREPARED="$run/prepared"
trap 'echo "Submission record: '"$run"'/jobs.txt (already submitted jobs are not cancelled)" >&2' ERR

submit_job() {
    local result
    result=$(sbatch --parsable --chdir="$PWD" --export=ALL "$@") || return $?
    result=${result%%;*}
    if [[ ! "$result" =~ ^[0-9]+$ ]]; then
        echo "Unexpected sbatch job ID: $result" >&2
        return 1
    fi
    printf '%s\n' "$result"
}

cpu=$(submit_job "$EXP/slurm/prepare.sh")
printf 'prepare=%s\nprepared=%s\n' "$cpu" "$PREPARED" > "$run/jobs.txt"
natural=$(submit_job --dependency="afterok:$cpu" --kill-on-invalid-dep=yes \
    "$EXP/slurm/train-natural.sh")
printf 'natural=%s\n' "$natural" >> "$run/jobs.txt"
symbolic=$(submit_job --dependency="afterok:$cpu" --kill-on-invalid-dep=yes \
    "$EXP/slurm/train-symbolic.sh")
printf 'symbolic=%s\n' "$symbolic" >> "$run/jobs.txt"
printf 'Submitted: CPU=%s Natural=%s Symbolic=%s\n' "$cpu" "$natural" "$symbolic"
printf 'Both GPU jobs wait for CPU success. Record: %s/jobs.txt\n' "$run"
