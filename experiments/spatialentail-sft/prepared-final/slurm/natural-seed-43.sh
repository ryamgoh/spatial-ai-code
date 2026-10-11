#!/bin/bash
#SBATCH --job-name=spatialentail-natural-seed-43
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd "$SLURM_SUBMIT_DIR"
export ARM=natural-seed-43
export TRAIN_CONFIG="$SLURM_SUBMIT_DIR/experiments/spatialentail-sft/prepared-final/configs/natural-seed-43.yaml"
export MODEL_DIR="$SLURM_SUBMIT_DIR/experiments/spatialentail-sft/prepared-final/models/natural-seed-43"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-spatialentail-qwen3.5-4b}"
export WANDB_JOB_TYPE=natural
export WANDB_RESUME=allow
python3 - "$SLURM_SUBMIT_DIR/experiments/spatialentail-sft/prepared-final/jobs.json" "$SLURM_SUBMIT_DIR/experiments/spatialentail-sft/prepared-final" "$SLURM_SUBMIT_DIR" <<'CHECK'
import hashlib, json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
base = pathlib.Path(sys.argv[2])
for name, expected in receipt["file_sha256"].items():
    path = base / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Prepared job input changed: " + name)
for name, expected in receipt["environment_sha256"].items():
    path = pathlib.Path(sys.argv[3]) / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Training environment input changed: " + name)
CHECK
bash experiments/15-v2-ablation/jobs/train.sh
