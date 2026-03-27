#!/bin/bash
#SBATCH --job-name=spatialeval_h100
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100-47:1 
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
 
# FINETUNING
# export OMP_NUM_THREADS=16
# export MKL_NUM_THREADS=16
# export PYTORCH_ALLOC_CONF=expandable_segments:True
#
# export AXOLOTL_NO_TELEMETRY=1
# export AXOLOTL_DO_NOT_TRACK=1
# cd finetune
# srun uv sync
# srun --cpu-bind=cores uv run python finetune.py qwen3-8b-spatial-reasoning
# cd ../eval

# EVAL
cd eval
srun uv sync
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/Cosmos-Reason2_8B_Baseline.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/Cosmos-Reason2_8B_NonFinetuned_nonshot_2.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/Falcon-H1R-7B_Baseline.yaml
srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/Falcon-H1R-7B_NonFinetuned_nonshot_2.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/RoboBrain2.5-8B-NV_Baseline.yaml
srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/RoboBrain2.5-8B-NV_NonFinetuned_nonshot_2.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/SpaceQwen3-VL-2B-Thinking_Baseline.yaml
srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/SpaceQwen3-VL-2B-Thinking_nonshot_2.yaml
# srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/SpaceR_Baseline.yaml
srun --cpu-bind=cores uv run python eval_two_stage.py --config ./config/experiments_other_models/SpaceR_NonFinetuned_nonshot_2.yaml
