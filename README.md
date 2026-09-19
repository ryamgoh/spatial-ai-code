# Spatial AI experiments

This repository contains the data generation, fine-tuning, and evaluation code
for the SpatialEval experiments. Training uses Axolotl; evaluation uses
LM Evaluation Harness with a custom one- or two-stage vLLM backend.

## Repository map

| Path | Purpose |
| --- | --- |
| data/ | Versioned source and evaluation datasets |
| finetune/ | SFT/GRPO generation and training code |
| eval/ | Evaluation backend, metrics, and tests |
| experiments/<id>/ | One experiment's configs, notes, scripts, logs, and results |
| experiments/<id>/slurm/ | Slurm launchers for that experiment |
| experiments/tasks/ | Shared LM Evaluation Harness task definitions |
| slurm/lib/ | Shared Slurm helpers; not submitted directly |
| slurm/archive/ | Historical multi-experiment launchers |
| tools/ | Fast local repository checks |

The detailed experiment index and path conventions are in
[experiments/README.md](experiments/README.md).

## Setup

Python 3.11 or 3.12 and uv are required. The lock file is committed so the
CUDA/Axolotl/vLLM combination remains reproducible.

    uv sync

## Run locally

Relative paths in configs assume the command is launched from eval/ or
finetune/ respectively.

    cd eval
    uv run python eval_new.py --config ../experiments/00-baseline-model/eval-smoke.yaml

    cd ../finetune
    uv run python finetune.py ../experiments/07-sft-starting-state/train-sft-base.yaml

## Submit to Slurm

This development machine does not run Slurm. Perform local checks here, then
submit training and GPU evaluation jobs from the Slurm server.

Each experiment has a tracked `logs/` directory, so Slurm can open the
job's output files before execution begins. Submit from the repository root
because the launchers use `SLURM_SUBMIT_DIR` to resolve project paths.

    sbatch experiments/07b-zero-shot-single/slurm/eval-h200.sh

    # Experiment environment variables can be exported explicitly:
    sbatch --export=ALL,TAGS=instruct experiments/07b-zero-shot-single/slurm/eval-h200.sh
    sbatch --export=ALL,TAGS=base experiments/07b-zero-shot-single/slurm/eval-h200.sh

Launcher names describe the workload and requested accelerator. Driver files
such as sft-driver.sh are sourced by hardware-specific launchers and must not
be submitted themselves.

Job output is kept with its experiment:

    tail -f experiments/07b-zero-shot-single/logs/spatial7b-zs-<job-id>.out

## Validate before submitting

    python3 tools/validate_repo.py
    cd eval
    uv run --python 3.12 --no-project --with pytest --with typer \
      pytest test_spatial_laws.py -q

The validator is GPU-free. It checks shell syntax, Slurm headers, time limits,
shared CPU pinning, and local shell-script references. It also parses every
YAML file using PyYAML from the project environment or Ruby's standard parser.

The checks above validate launchers structurally; they do not replace an
end-to-end job run on the Slurm server.
