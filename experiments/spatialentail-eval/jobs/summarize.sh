#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 experiments/spatialentail-eval/scripts/summarize.py \
  experiments/spatialentail-eval/prepared-final
