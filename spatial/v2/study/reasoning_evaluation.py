"""Prepare the text-only SpatialEntail reasoning-model batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spatial.v2.study.comparator_evaluation import (
    ComparatorSpec,
    _config,
    _hash,
    _slug,
    _submit_script,
    _wrapper,
)


@dataclass(frozen=True)
class ReasoningSpec:
    model: ComparatorSpec
    temperature: float
    top_p: float
    top_k: int | None = None
    seed: int = 42


REASONING_MODELS = (
    ReasoningSpec(
        model=ComparatorSpec(
            name="qwen3-5-4b-thinking-on",
            pretrained="Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            response_regime="thinking-on",
            chat_template_kwargs={"enable_thinking": True},
        ),
        temperature=1.0,
        top_p=0.95,
        top_k=20,
    ),
    ReasoningSpec(
        model=ComparatorSpec(
            name="qwen3-5-9b-thinking-on",
            pretrained="Qwen/Qwen3.5-9B",
            revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            response_regime="thinking-on",
            chat_template_kwargs={"enable_thinking": True},
        ),
        temperature=1.0,
        top_p=0.95,
        top_k=20,
    ),
    ReasoningSpec(
        model=ComparatorSpec(
            name="ministral-3-8b-reasoning",
            pretrained="mistralai/Ministral-3-8B-Reasoning-2512",
            revision="81eaece1948f3875421d9a45bc55487d10e2d894",
            response_regime="reasoning",
            chat_template_kwargs={},
        ),
        temperature=0.7,
        top_p=0.95,
    ),
    ReasoningSpec(
        model=ComparatorSpec(
            name="deepseek-r1-distill-qwen-7b",
            pretrained="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            revision="916b56a44061fd5cd7d6a8fb632557ed4f724f60",
            response_regime="reasoning",
            chat_template_kwargs={},
        ),
        temperature=0.6,
        top_p=0.95,
    ),
)


def _task_yaml(
    task_name: str,
    data_path: str,
    spec: ReasoningSpec,
) -> str:
    sampling = [
        "  max_gen_toks: 4096",
        f"  temperature: {spec.temperature}",
        f"  top_p: {spec.top_p}",
        "  repetition_penalty: 1",
        f"  seed: {spec.seed}",
    ]
    if spec.top_k is not None:
        sampling.append(f"  top_k: {spec.top_k}")
    sampling.append("  until: []")
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
{chr(10).join(sampling)}
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
  note: "Text-only native reasoning evaluation; certificates remain offline"
"""


def _verify_upstream(receipt: dict[str, Any], prepared: Path) -> None:
    if receipt.get("schema") != "spatialentail-general-instruction-eval-v1":
        raise ValueError("reasoning evaluation requires the prepared general package")
    for name, expected in receipt["file_sha256"].items():
        path = prepared / name
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"prepared general input changed: {name}")


def prepare_reasoning_evaluation(
    general_prepared: Path,
    output: Path,
    *,
    repo_root: Path,
    expected_total_bases: int = 3600,
) -> dict[str, Any]:
    root = repo_root.resolve()
    general = general_prepared.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"reasoning evaluation output already exists: {output}")
    if not general.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("prepared inputs and output must live inside the repository")

    receipt_path = general / "jobs.json"
    general_receipt = json.loads(receipt_path.read_text())
    _verify_upstream(general_receipt, general)
    task_records = general_receipt["tasks"]
    if (
        general_receipt.get("total_bases") != expected_total_bases
        or len(task_records) != 48
    ):
        raise ValueError(
            "reasoning batch requires the expected frozen-base/48-task package"
        )

    files: dict[str, str] = {
        "tasks/spatialentail_utils.py": (
            general / "tasks" / "spatialentail_utils.py"
        ).read_text()
    }
    jobs = []
    include_path = "../" + str((output / "tasks").relative_to(root))
    general_relative = general.relative_to(root)
    for reasoning in REASONING_MODELS:
        model = reasoning.model
        task_names = []
        for task in task_records:
            query = str(task["query_kind"])
            gold = int(task["gold_size"])
            partition = str(task["partition"])
            task_name = (
                f"spatialentail_reasoning_{_slug(model.name)}_"
                f"{_slug(query)}_g{gold}_{_slug(partition)}"
            )
            data_path = (
                f"../{general_relative}/data/{query}-g{gold}-{partition}.jsonl"
            )
            files[f"tasks/{task_name}.yaml"] = _task_yaml(
                task_name, data_path, reasoning
            )
            task_names.append(task_name)

        config_relative = (output / "configs" / f"{model.name}.yaml").relative_to(root)
        smoke_config_relative = (
            output / "configs" / "smoke" / f"{model.name}.yaml"
        ).relative_to(root)
        script_relative = (output / "slurm" / f"{model.name}.sh").relative_to(root)
        smoke_script_relative = (
            output / "slurm" / "smoke" / f"{model.name}.sh"
        ).relative_to(root)

        files[f"configs/{model.name}.yaml"] = _config(
            model, tasks=task_names, include_path=include_path
        )
        smoke_config = _config(model, tasks=task_names, include_path=include_path)
        smoke_config += "limit: 1\n"
        files[f"configs/smoke/{model.name}.yaml"] = smoke_config
        files[f"slurm/{model.name}.sh"] = _wrapper(
            output.relative_to(root), model, config_relative
        )
        files[f"slurm/smoke/{model.name}.sh"] = _wrapper(
            output.relative_to(root), model, smoke_config_relative, smoke=True
        )
        jobs.append(
            {
                **asdict(model),
                "sampling": {
                    "temperature": reasoning.temperature,
                    "top_p": reasoning.top_p,
                    "top_k": reasoning.top_k,
                    "seed": reasoning.seed,
                },
                "config": str(config_relative),
                "script": str(script_relative),
                "smoke_config": str(smoke_config_relative),
                "smoke_script": str(smoke_script_relative),
            }
        )

    models = tuple(spec.model for spec in REASONING_MODELS)
    files["submit-h200-chain.sh"] = _submit_script(
        output.relative_to(root), models
    )
    files["submit-h200-smoke-chain.sh"] = _submit_script(
        output.relative_to(root), models, smoke=True
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
        "spatial/v2/study/reasoning_evaluation.py",
    )
    result = {
        "schema": "spatialentail-reasoning-eval-v1",
        "upstream_general_receipt_sha256": _hash(receipt_path),
        "benchmark_sha256": general_receipt["benchmark_sha256"],
        "total_bases": general_receipt["total_bases"],
        "tasks_per_model": len(task_records),
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
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("general_prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_reasoning_evaluation(
                args.general_prepared, args.output, repo_root=args.repo_root
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
