#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${RUN_DIR:?RUN_DIR is required}"

uv run --no-project python \
  experiments/15-v2-ablation/scripts/summarize.py \
  "$RUN_DIR"
