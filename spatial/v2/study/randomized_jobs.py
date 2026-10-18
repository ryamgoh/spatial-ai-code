"""Prepare lean Axolotl data and six randomized-v2 SFT jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

_MODEL = "Qwen/Qwen3.5-4B"
_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(output_relative: Path, package: str, seed: int) -> dict:
    data_prefix = Path("..") / output_relative / "data"
    model_dir = Path("..") / output_relative / "models" / f"{package}-seed-{seed}"
    return {
        "chat_template": "tokenizer_default",
        "base_model": _MODEL,
        "load_in_4bit": True,
        "adapter": "qlora",
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj",
            "up_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ],
        "lora_qkv_kernel": False,
        "lora_o_kernel": False,
        "lora_mlp_kernel": False,
        "embeddings_skip_upcast": True,
        "attn_implementation": "sdpa",
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "shuffle_merged_datasets": True,
        "learning_rate": 5e-5,
        "sequence_len": 4096,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optimizer": "paged_adamw_8bit",
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 0.05,
        "num_epochs": 2,
        "logging_steps": 1,
        "output_dir": str(model_dir),
        "datasets": [{
            "path": str(data_prefix / f"{package}-train.jsonl"),
            "type": "chat_template",
            "field_messages": "messages",
        }],
        "test_datasets": [{
            "path": str(data_prefix / f"{package}-development.jsonl"),
            "type": "chat_template",
            "field_messages": "messages",
        }],
        "dataloader_prefetch_factor": 8,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "dataset_num_proc": 16,
        "val_set_size": 0,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "revision_of_model": _REVISION,
        "use_wandb": True,
        "wandb_project": "spatialentail-randomized-v2-sft",
        "wandb_mode": "online",
        "seed": seed,
        "wandb_name": f"randomized-v2-{package}-seed-{seed}",
        "wandb_run_id": f"spatialentail-randomized-v2-{package}-seed-{seed}",
    }


def prepare_jobs(candidate: Path, token_audit: Path, output: Path, repo_root: Path) -> dict:
    """Verify admitted source rows and prepare lean data, configs, and Slurm jobs."""
    if output.exists():
        raise FileExistsError(f"prepared jobs output already exists: {output}")
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(token_audit.read_text())
    if audit.get("status") != "passed" or audit.get("recommended_train_max_tokens") != 4096:
        raise ValueError("training candidate lacks successful 4096-token admission")
    if audit.get("manifest_sha256") != _hash(manifest_path):
        raise ValueError("token audit does not match the training manifest")
    packages = {
        "single-natural": "single/{split}-natural.jsonl",
        "single-answer-only": "single/{split}-answer-only.jsonl",
        "multi-natural": "multi/{split}-natural.jsonl",
    }
    output.mkdir(parents=True)
    for directory in ("data", "configs", "slurm", "logs", "models"):
        (output / directory).mkdir()
    for package, pattern in packages.items():
        for split in ("train", "development"):
            relative = pattern.format(split=split)
            source = candidate / relative
            receipt = manifest["files"][relative]
            if _hash(source) != receipt["sha256"]:
                raise ValueError(f"candidate source changed: {relative}")
            destination = output / "data" / f"{package}-{split}.jsonl"
            with source.open() as inputs, destination.open("w") as lean:
                for line in inputs:
                    row = json.loads(line)
                    lean.write(json.dumps({"id": row["id"], "messages": row["messages"]}, ensure_ascii=False) + "\n")
    output_relative = output.resolve().relative_to(repo_root.resolve())
    jobs = []
    for package in packages:
        for seed in (42, 43):
            name = f"{package}-seed-{seed}"
            config_relative = Path("configs") / f"{name}.yaml"
            script_relative = Path("slurm") / f"{name}.sh"
            (output / config_relative).write_text(
                yaml.safe_dump(_config(output_relative, package, seed), sort_keys=False)
            )
            script = f"""#!/bin/bash
#SBATCH --job-name=random-v2-{package}-{seed}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user=e0958116@u.nus.edu
#SBATCH --output={output_relative}/logs/%x-%j.out
#SBATCH --error={output_relative}/logs/%x-%j.err
set -euo pipefail
: "${{SLURM_SUBMIT_DIR:?Submit this script from the repository root}}"
cd "$SLURM_SUBMIT_DIR"
export ARM={name}
export TRAIN_CONFIG="$SLURM_SUBMIT_DIR/{output_relative}/{config_relative}"
export MODEL_DIR="$SLURM_SUBMIT_DIR/{output_relative}/models/{name}"
export WANDB_REQUIRED=1
export WANDB_RUN_GROUP="${{WANDB_RUN_GROUP:-spatialentail-randomized-v2}}"
export WANDB_JOB_TYPE={package}
export WANDB_RESUME=allow
bash experiments/15-v2-ablation/jobs/train.sh
"""
            (output / script_relative).write_text(script)
            (output / script_relative).chmod(0o755)
            jobs.append({"name": name, "package": package, "seed": seed, "config": str(output_relative / config_relative), "script": str(output_relative / script_relative)})
    prepared = [path for path in output.rglob("*") if path.is_file()]
    receipt = {
        "schema": "spatialentail-randomized-v2-sft-jobs-v1",
        "base_model": _MODEL,
        "model_revision": _REVISION,
        "sequence_len": 4096,
        "training_manifest_sha256": _hash(manifest_path),
        "token_audit_sha256": _hash(token_audit),
        "jobs": jobs,
        "file_sha256": {
            str(path.relative_to(output)): _hash(path) for path in sorted(prepared)
        },
        "submitted": False,
    }
    (output / "jobs.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("token_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(prepare_jobs(args.candidate, args.token_audit, args.output, args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
