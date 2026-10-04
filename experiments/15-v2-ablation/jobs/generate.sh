#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${MATRIX:?MATRIX is required}"
: "${DATA_OUTPUT:?DATA_OUTPUT is required}"

args=()
if [[ "${REPLACE_DATA:-0}" == "1" ]]; then
  args+=(--replace)
fi

uv run --no-project --with typer --with pyyaml --with z3-solver \
  python spatial/generate_matrix_v2.py \
  "$MATRIX" \
  --out "$DATA_OUTPUT" \
  "${args[@]}"
