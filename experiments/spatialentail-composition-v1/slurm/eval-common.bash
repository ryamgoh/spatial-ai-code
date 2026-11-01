#!/usr/bin/env bash
set -euo pipefail
: "${STYLE:?Set STYLE=natural or STYLE=symbolic}"
EXP=experiments/spatialentail-composition-v1
mode=$1
args=("$mode" --style "$STYLE")
if [[ -n "${PREPARED:-}" ]]; then
    args+=(--prepared "$PREPARED")
fi
if [[ "$mode" == dev ]]; then
    args+=(--benchmark "${EVAL_DATA:-$EXP/evaluation-internal}")
else
    : "${SPLIT:?Set SPLIT=in-range-test, extrapolation-test or external-test}"
    if [[ "$SPLIT" == external-test ]]; then
        args+=(--benchmark "${EVAL_DATA:-$EXP/spatialmap-unique-v1}")
    else
        args+=(--benchmark "${EVAL_DATA:-$EXP/evaluation-internal}")
    fi
    args+=(--split "$SPLIT")
fi
export AXOLOTL_DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Training's CPU job already installed the shared locked environment.
exec srun uv run --frozen --no-sync python -m spatial.v2.entailment_evaluation "${args[@]}"
