"""Prepare frozen SpatialEntail-SINGLE benchmark evaluation jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from spatial.v2.study import read_role_text

BENCH_SOURCES = (
    ("bench-core", "core", "bench-core"),
    ("bench-composition", "composition", "bench-comp"),
    ("bench-shift", "heldout-composition", "bench-composition"),
    ("bench-shift", "depth", "bench-depth"),
)
MODEL_SPECS = {
    "untuned": None,
    "answer-only-seed-42": (
        "experiments/spatialentail-sft/prepared-final/models/answer-only-seed-42"
    ),
    "natural-seed-42": (
        "experiments/spatialentail-sft/prepared-final/models/natural-seed-42"
    ),
    "answer-only-seed-43": (
        "experiments/spatialentail-sft/prepared-final/models/answer-only-seed-43"
    ),
    "natural-seed-43": (
        "experiments/spatialentail-sft/prepared-final/models/natural-seed-43"
    ),
}
_SAFE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _SAFE.sub("_", value.lower()).strip("_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_answer_rows(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("metadata", {}).get("trace_format") == "natural"
        and row.get("metadata", {}).get("supervision_arm") == "answer-only"
    ]
    all_ids = {row.get("metadata", {}).get("base_id") for row in rows}
    selected_ids = [row.get("metadata", {}).get("base_id") for row in selected]
    if None in all_ids or None in selected_ids:
        raise ValueError(f"{role} contains a row without metadata.base_id")
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != all_ids:
        raise ValueError(
            f"{role} does not contain exactly one answer-only row per base"
        )
    return selected


def _task_yaml(task_name: str, data_path: str, max_new_tokens: int) -> str:
    return f"""task: {task_name}
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files:
    test: {json.dumps(data_path)}
test_split: test
output_type: generate_until
process_docs: !function spatialentail_utils.process_docs_v2_sft
process_results: !function spatialentail_utils.process_results_v2_answer
doc_to_text: "{{{{evaluation_prompt}}}}"
doc_to_target: "{{{{oracle_option}}}}"
generation_kwargs:
  max_gen_toks: {max_new_tokens}
  temperature: 0
  repetition_penalty: 1
  until: []
metric_list:
  - metric: strict_acc
    aggregation: mean
    higher_is_better: true
  - metric: loose_acc
    aggregation: mean
    higher_is_better: true
  - metric: stage1_acc
    aggregation: mean
    higher_is_better: true
  - metric: stage1_format_valid
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
  note: "Frozen SpatialEntail answer evaluation; certificates remain offline"
"""


def _eval_config(
    *,
    tasks: list[str],
    include_path: str,
    model_revision: str,
    max_model_len: int,
    max_new_tokens: int,
    lora_path: str | None,
    choices: list[str] | None = None,
    allow_multiple_answers: bool = False,
) -> str:
    config = {
        "model": "vllm_staged_pass",
        "model_args": {
            "pretrained": "Qwen/Qwen3.5-4B",
            "revision": model_revision,
            "tokenizer_revision": model_revision,
            "lora_path": lora_path,
            "max_lora_rank": 64,
            "max_thinking_tokens": max_new_tokens,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.85,
            "max_model_len": max_model_len,
            "choices": choices or list("ABCDE"),
            "allow_multiple_answers": allow_multiple_answers,
            "record_stage1": True,
            "add_special_tokens": False,
        },
        "include_path": include_path,
        "tasks": tasks,
        "apply_chat_template": False,
        "bootstrap_iters": 0,
        "system_instruction": None,
        "batch_size": 4,
        "device": "cuda:0",
        "log_samples": True,
    }
    return yaml.safe_dump(config, sort_keys=False)


def _wrapper(
    *, output_relative: Path, model_name: str, config_relative: Path, lora: str | None
) -> str:
    model_dir = f"$SLURM_SUBMIT_DIR/{lora}" if lora else "$SLURM_SUBMIT_DIR"
    return f"""#!/bin/bash
#SBATCH --job-name=spatialentail-eval-{model_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200-141:1
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
export EVAL_NAME={model_name}
export EVAL_CONFIG="$SLURM_SUBMIT_DIR/{config_relative}"
export EVAL_PREPARED_DIR="$SLURM_SUBMIT_DIR/{output_relative}"
export MODEL_DIR="{model_dir}"
export RESULT_DIR="$SLURM_SUBMIT_DIR/{output_relative}/results/{model_name}"
export UNTUNED={1 if lora is None else 0}
bash experiments/spatialentail-eval/jobs/eval.sh
"""


def prepare_evaluation(
    release: Path, output: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    release = release.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    if not release.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError(
            "release and evaluation output must live inside the repository"
        )

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "spatialentail-study-release-v1"
        or manifest.get("status") != "sealed"
    ):
        raise ValueError("frozen evaluation requires a sealed study release")
    budget = manifest["context_budget"]
    revision = str(budget.get("tokenizer_revision") or "")
    if not revision:
        raise ValueError("sealed release must pin the model/tokenizer revision")

    files: dict[str, str] = {}
    task_records: list[dict[str, Any]] = []
    task_names: list[str] = []
    total_bases = 0
    files["tasks/spatialentail_utils.py"] = """\
import importlib.util
from pathlib import Path

source = Path(__file__).resolve().parents[4] / "experiments/tasks/utils.py"
spec = importlib.util.spec_from_file_location("spatialentail_task_utils", source)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load SpatialEntail task utilities from {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
process_docs_v2_sft = module.process_docs_v2_sft
process_results_v2_answer = module.process_results_v2_answer
"""
    for report_role, component, sealed_role in BENCH_SOURCES:
        source_text = read_role_text(release, sealed_role)
        artifact = manifest["artifacts"][sealed_role]
        if hashlib.sha256(source_text.encode()).hexdigest() != artifact["sha256"]:
            raise ValueError(
                f"{report_role} hash disagrees with the sealed manifest"
            )
        rows = [json.loads(line) for line in source_text.splitlines()]
        selected = _select_answer_rows(rows, report_role)
        if len(selected) != artifact["bases"]:
            raise ValueError(
                f"{report_role} base count disagrees with the sealed manifest"
            )
        total_bases += len(selected)
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            cell = str(row["metadata"].get("benchmark_cell") or "")
            if not cell:
                raise ValueError(f"{report_role} row lacks metadata.benchmark_cell")
            by_cell[cell].append(row)
        for cell, cell_rows in sorted(by_cell.items()):
            task_name = (
                f"spatialentail_{_slug(report_role)}_{_slug(component)}_{_slug(cell)}"
            )
            data_name = f"data/{report_role}-{component}--{cell}.jsonl"
            data_path = "../" + str((output / data_name).relative_to(root))
            files[data_name] = "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in cell_rows
            )
            files[f"tasks/{task_name}.yaml"] = _task_yaml(
                task_name, data_path, int(budget["max_new_tokens"])
            )
            metadata = cell_rows[0]["metadata"]
            task_records.append(
                {
                    "task": task_name,
                    "role": report_role,
                    "component": component,
                    "cell": cell,
                    "query_kind": metadata.get("query_kind"),
                    "count": len(cell_rows),
                }
            )
            task_names.append(task_name)

    include_path = "../" + str((output / "tasks").relative_to(root))
    jobs = []
    for model_name, lora in MODEL_SPECS.items():
        lora_from_eval = "../" + lora if lora else None
        config_relative = (output / "configs" / f"{model_name}.yaml").relative_to(root)
        files[f"configs/{model_name}.yaml"] = _eval_config(
            tasks=task_names,
            include_path=include_path,
            model_revision=revision,
            max_model_len=int(budget["eval_max_tokens"]),
            max_new_tokens=int(budget["max_new_tokens"]),
            lora_path=lora_from_eval,
        )
        script_relative = (output / "slurm" / f"{model_name}.sh").relative_to(root)
        files[f"slurm/{model_name}.sh"] = _wrapper(
            output_relative=output.relative_to(root),
            model_name=model_name,
            config_relative=config_relative,
            lora=lora,
        )
        jobs.append(
            {
                "name": model_name,
                "config": str(config_relative),
                "script": str(script_relative),
                "lora_path": lora,
            }
        )

    environment_files = (
        "pyproject.toml",
        "uv.lock",
        "eval/eval_new.py",
        "eval/generation_contract.py",
        "experiments/tasks/utils.py",
        "spatial/v2/study/evaluation.py",
        "experiments/spatialentail-eval/jobs/eval.sh",
        "experiments/spatialentail-eval/jobs/summarize.sh",
        "experiments/spatialentail-eval/scripts/summarize.py",
        "experiments/spatialentail-eval/slurm/summarize.sh",
        "experiments/spatialentail-eval/submit.sh",
    )
    receipt = {
        "schema": "spatialentail-frozen-eval-v1",
        "release_manifest_sha256": _sha256(manifest_path),
        "submitted": False,
        "total_bases": total_bases,
        "context_budget": budget,
        "tasks": task_records,
        "jobs": jobs,
        "environment_sha256": {
            name: _sha256(root / name)
            for name in environment_files
            if (root / name).is_file()
        },
        "file_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        },
    }
    output.mkdir(parents=True)
    for filename, content in files.items():
        path = output / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (output / "logs").mkdir()
    (output / "logs/.gitkeep").write_text("")
    (output / "results").mkdir()
    (output / "results/.gitkeep").write_text("")
    (output / "jobs.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (output / "README.md").write_text(
        """# Prepared SpatialEntail-SINGLE benchmark evaluation

This immutable bundle evaluates the untuned model plus answer-only and Natural
seed-42 adapters on exactly 4,300 sealed `SpatialEntail-SINGLE-Bench` bases. The
52 tasks retain the frozen cell boundaries used for CORE, COMP, and SHIFT
reporting. SHIFT keeps slice labels as cell metadata. Each data view contains
one answer-only row per base; answer certificates remain offline audit artifacts.

`jobs.json` records preparation only. Remote submission and completion are not
synchronized back into this directory. Transfer this directory together with
its ignored `data/` files, the sealed release, and trained adapter directories.
Submit from the repository root with:

```sh
bash experiments/spatialentail-eval/submit.sh
```
"""
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(prepare_evaluation(args.release, args.out), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
