#!/bin/bash
# Exp 7b — untuned 4B Instruct and/or Base on TQA-Corr-Single (SFT prompt).
# H200, partition gpu, 3h. Two 4B evals will not both fit in 3h — submit
# them as two jobs on two cards:
#
#   TAGS=instruct sbatch run_batch_07b_eval_single_h200.sh
#   TAGS=base     sbatch run_batch_07b_eval_single_h200.sh
#
# Corr 1329 (Single + Multi slices), same SFT prompt:
#   sbatch --export=ALL,TAGS=instruct-corr run_batch_07b_eval_single_h200.sh
#   sbatch --export=ALL,TAGS=base-corr     run_batch_07b_eval_single_h200.sh
#
# TAGS=instruct,base (default) is Single only; two tags in one 3h job is tight.
#SBATCH --job-name=spatial7b-zs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXP=experiments/07b-zero-shot-single
A_EXP=$SLURM_SUBMIT_DIR/$EXP
TAGS="${TAGS:-instruct,base}"
STAGES="${STAGES:-2}"
if [[ "$STAGES" != "1" && "$STAGES" != "2" ]]; then
  echo "STAGES must be 1 or 2, got $STAGES"
  exit 1
fi
S1_SUFFIX=""
if [[ "$STAGES" == "1" ]]; then
  S1_SUFFIX="-s1"
fi

declare -A CFG=(
  [instruct]=eval-instruct-single.yaml
  [base]=eval-base-single.yaml
  [instruct-corr]=eval-instruct-corr.yaml
  [base-corr]=eval-base-corr.yaml
)
declare -A OUT_REL=(
  [instruct]=results/zero-shot-single/instruct
  [base]=results/zero-shot-single/base
  [instruct-corr]=results/zero-shot-corr/instruct
  [base-corr]=results/zero-shot-corr/base
)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
echo "TAGS=$TAGS STAGES=$STAGES"
nvidia-smi -L || true

cd eval
srun uv sync

status=0
IFS=,
for tag in $TAGS; do
  tag=${tag// /}
  cfg=${CFG[$tag]:-}
  if [[ -z "$cfg" ]]; then
    echo "Unknown tag $tag (want instruct, base, instruct-corr, base-corr)"
    status=1
    continue
  fi
  out=$A_EXP/${OUT_REL[$tag]}${S1_SUFFIX}
  if [[ -f "$out/results.json" ]]; then
    echo "=== skip $tag stages=$STAGES (already have results.json) ==="
    continue
  fi
  echo "=== eval 7b $tag stages=$STAGES ($cfg) ==="
  mkdir -p "$out"
  srun --cpu-bind=cores uv run python eval_new.py \
    --config "../experiments/07b-zero-shot-single/$cfg" \
    --stages "$STAGES" \
    --output-dir "$out" || {
    echo "7b $tag FAILED"
    status=1
    continue
  }
  echo "  7b $tag OK -> $out/"
done

echo "=== 7b done."
echo "    Single:  cd eval && uv run --no-project python ../experiments/07b-zero-shot-single/scripts/summarize.py"
echo "    Corr:    cd eval && uv run --no-project python ../experiments/07b-zero-shot-single/scripts/summarize_corr.py"
exit "$status"
