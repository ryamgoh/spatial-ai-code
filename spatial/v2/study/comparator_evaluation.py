"""Prepare the general-instruction SpatialEntail comparator batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ComparatorSpec:
    name: str
    pretrained: str
    revision: str
    response_regime: str
    chat_template_kwargs: dict[str, Any]
    system_message_policy: str = "native"


GENERAL_INSTRUCTION_MODELS = (
    ComparatorSpec(
        name="qwen3-5-4b-thinking-off",
        pretrained="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        response_regime="thinking-off",
        chat_template_kwargs={"enable_thinking": False},
    ),
    ComparatorSpec(
        name="qwen3-5-9b-thinking-off",
        pretrained="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        response_regime="thinking-off",
        chat_template_kwargs={"enable_thinking": False},
    ),
    ComparatorSpec(
        name="gemma-4-e4b-it",
        pretrained="google/gemma-4-E4B-it",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        response_regime="instruction",
        chat_template_kwargs={},
    ),
    ComparatorSpec(
        name="gemma-3-4b-it",
        pretrained="google/gemma-3-4b-it",
        revision="093f9f388b31de276ce2de164bdc2081324b9767",
        response_regime="instruction",
        chat_template_kwargs={},
        system_message_policy="prepend_user",
    ),
    ComparatorSpec(
        name="ministral-3-8b-instruct",
        pretrained="mistralai/Ministral-3-8B-Instruct-2512",
        revision="5b26027e7b19eeb4b7352e1fed3926375dd2cb4d",
        response_regime="instruction",
        chat_template_kwargs={},
    ),
    ComparatorSpec(
        name="phi-4-mini-instruct",
        pretrained="microsoft/Phi-4-mini-instruct",
        revision="cfbefacb99257ffa30c83adab238a50856ac3083",
        response_regime="instruction",
        chat_template_kwargs={},
    ),
)

_SAFE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _SAFE.sub("_", value.lower()).strip("_")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_yaml(task_name: str, data_path: str) -> str:
    return f"""task: {task_name}
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files:
    test: {json.dumps(data_path)}
test_split: test
output_type: generate_until
process_docs: !function spatialentail_utils.process_docs_comparator
process_results: !function spatialentail_utils.process_results_v2_answer
doc_to_text: "{{{{model_input}}}}"
doc_to_target: "{{{{oracle_option}}}}"
generation_kwargs:
  max_gen_toks: 4096
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
  note: "Cross-family native-template evaluation; certificates remain offline"
"""


def _config(
    spec: ComparatorSpec,
    *,
    tasks: list[str],
    include_path: str,
    limit: int | None = None,
) -> str:
    config = {
        "model": "vllm_staged_chat_pass",
        "model_args": {
            "pretrained": spec.pretrained,
            "revision": spec.revision,
            "tokenizer_revision": spec.revision,
            "stages": 2,
            "max_thinking_tokens": 4096,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.85,
            "max_model_len": 8192,
            "choices": list("ABCDE"),
            "allow_multiple_answers": True,
            "record_stage1": True,
            "chat_template_kwargs": spec.chat_template_kwargs,
            "system_message_policy": spec.system_message_policy,
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
    if limit is not None:
        config["limit"] = limit
    return yaml.safe_dump(config, sort_keys=False)


def _wrapper(
    output_relative: Path,
    spec: ComparatorSpec,
    config_relative: Path,
    *,
    smoke: bool = False,
) -> str:
    run_name = f"smoke-{spec.name}" if smoke else spec.name
    result_relative = f"results/smoke/{spec.name}" if smoke else f"results/{spec.name}"
    return f"""#!/bin/bash
#SBATCH --job-name=spatialentail-{run_name}
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
export EVAL_NAME={run_name}
export EVAL_CONFIG="$SLURM_SUBMIT_DIR/{config_relative}"
export EVAL_PREPARED_DIR="$SLURM_SUBMIT_DIR/{output_relative}"
export RESULT_DIR="$SLURM_SUBMIT_DIR/{output_relative}/{result_relative}"
bash experiments/spatialentail-randomized-v2/jobs/eval-comparator.sh
"""


def _submit_script(
    output_relative: Path,
    specs: tuple[ComparatorSpec, ...],
    *,
    smoke: bool = False,
) -> str:
    directory = "slurm/smoke" if smoke else "slurm"
    scripts = "\n".join(
        f'  "$ROOT/{output_relative}/{directory}/{spec.name}.sh"' for spec in specs
    )
    return f"""#!/bin/bash
set -euo pipefail
ROOT=${{SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT"
SCRIPTS=(
{scripts}
)
dependency=()
for script in "${{SCRIPTS[@]}}"; do
  job_id=$(sbatch --parsable "${{dependency[@]}}" -p gpu -t 03:00:00 --gres=gpu:h200-141:1 "$script")
  printf '%s %s\n' "$job_id" "$script"
  dependency=(--dependency="afterok:$job_id")
done
"""


def prepare_general_instruction_evaluation(
    benchmark_jsonl: Path,
    output: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    root = repo_root.resolve()
    benchmark = benchmark_jsonl.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"comparator evaluation output already exists: {output}")
    if not benchmark.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("benchmark and output must live inside the repository")

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    with benchmark.open() as source:
        for line in source:
            row = json.loads(line)
            messages = row.get("messages") or []
            if [message.get("role") for message in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValueError("benchmark row does not have system/user/assistant messages")
            metadata = row.get("metadata") or {}
            key = (
                str(metadata["query_kind"]),
                int(metadata["gold_size"]),
                str(metadata["benchmark_partition"]),
            )
            prepared = dict(row)
            prepared["messages"] = messages[:2]
            groups.setdefault(key, []).append(prepared)
    if not groups:
        raise ValueError("benchmark contains no rows")

    files: dict[str, str] = {}
    files["tasks/spatialentail_utils.py"] = """\
import importlib.util
from pathlib import Path
from spatial.v2.evaluation_chat import encode_chat_input

source = Path(__file__).resolve().parents[4] / "experiments/tasks/utils.py"
spec = importlib.util.spec_from_file_location("spatialentail_task_utils", source)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load SpatialEntail task utilities from {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
process_results_v2_answer = module.process_results_v2_answer

def process_docs_comparator(dataset):
    def convert(doc):
        metadata = dict(doc.get("metadata") or {})
        letters = [str(letter).strip().upper() for letter in metadata.get("oracle_letters") or []]
        if not letters or letters != sorted(set(letters)) or any(letter not in "ABCDE" for letter in letters):
            raise ValueError("comparator row has invalid oracle letters")
        return {
            "text": next(str(message.get("content") or "") for message in doc["messages"] if message.get("role") == "user"),
            "model_input": encode_chat_input(doc["messages"]),
            "oracle_option": ",".join(letters),
            "trace_format": str(metadata.get("trace_format") or "natural"),
            "supervision_arm": "answer-only",
            "answer_mode": str(metadata.get("answer_mode") or ""),
            "query_kind": str(metadata.get("query_kind") or ""),
            "difficulty": dict(metadata.get("difficulty") or {}),
            "benchmark_partition": str(metadata.get("benchmark_partition") or ""),
            "gold_size": int(metadata.get("gold_size") or 0),
            "base_id": str(metadata.get("base_id") or ""),
        }
    return dataset.map(convert)
"""

    tasks = []
    task_names = []
    for (query, gold, partition), rows in sorted(groups.items()):
        task_name = (
            f"spatialentail_comparator_{_slug(query)}_g{gold}_{_slug(partition)}"
        )
        data_name = f"data/{query}-g{gold}-{partition}.jsonl"
        data_relative = (output / data_name).relative_to(root)
        files[data_name] = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        files[f"tasks/{task_name}.yaml"] = _task_yaml(
            task_name, "../" + str(data_relative)
        )
        tasks.append(
            {
                "task": task_name,
                "query_kind": query,
                "gold_size": gold,
                "partition": partition,
                "count": len(rows),
            }
        )
        task_names.append(task_name)

    include_path = "../" + str((output / "tasks").relative_to(root))
    jobs = []
    for model in GENERAL_INSTRUCTION_MODELS:
        config_relative = (output / "configs" / f"{model.name}.yaml").relative_to(root)
        script_relative = (output / "slurm" / f"{model.name}.sh").relative_to(root)
        smoke_config_relative = (
            output / "configs" / "smoke" / f"{model.name}.yaml"
        ).relative_to(root)
        smoke_script_relative = (
            output / "slurm" / "smoke" / f"{model.name}.sh"
        ).relative_to(root)
        files[f"configs/{model.name}.yaml"] = _config(
            model, tasks=task_names, include_path=include_path
        )
        files[f"configs/smoke/{model.name}.yaml"] = _config(
            model, tasks=task_names, include_path=include_path, limit=1
        )
        files[f"slurm/{model.name}.sh"] = _wrapper(
            output.relative_to(root), model, config_relative
        )
        files[f"slurm/smoke/{model.name}.sh"] = _wrapper(
            output.relative_to(root), model, smoke_config_relative, smoke=True
        )
        jobs.append(
            {
                **asdict(model),
                "config": str(config_relative),
                "script": str(script_relative),
                "smoke_config": str(smoke_config_relative),
                "smoke_script": str(smoke_script_relative),
            }
        )
    files["submit-h200-chain.sh"] = _submit_script(
        output.relative_to(root), GENERAL_INSTRUCTION_MODELS
    )
    files["submit-h200-smoke-chain.sh"] = _submit_script(
        output.relative_to(root), GENERAL_INSTRUCTION_MODELS, smoke=True
    )

    output.mkdir(parents=True)
    for relative, content in files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if relative.startswith("slurm/") or relative.startswith("submit-h200-"):
            path.chmod(0o755)
    (output / "logs").mkdir()
    (output / "results").mkdir()

    environment_files = (
        "pyproject.toml",
        "uv.lock",
        "eval/eval_new.py",
        "eval/eval_comparators.py",
        "eval/generation_contract.py",
        "spatial/v2/evaluation_chat.py",
        "experiments/tasks/utils.py",
        "experiments/spatialentail-randomized-v2/jobs/eval-comparator.sh",
        "spatial/v2/study/comparator_evaluation.py",
    )
    receipt = {
        "schema": "spatialentail-general-instruction-eval-v1",
        "benchmark_sha256": _hash(benchmark),
        "total_bases": sum(len(rows) for rows in groups.values()),
        "context_budget": {"max_model_len": 8192, "max_new_tokens": 4096},
        "tasks": tasks,
        "jobs": jobs,
        "file_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        },
        "environment_sha256": {
            name: _hash(root / name)
            for name in environment_files
            if (root / name).is_file()
        },
        "submitted": False,
    }
    (output / "jobs.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_general_instruction_evaluation(
                args.benchmark, args.output, repo_root=args.repo_root
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
