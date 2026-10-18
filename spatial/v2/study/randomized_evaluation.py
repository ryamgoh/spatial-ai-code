"""Prepare seven-model evaluation jobs for the frozen randomized-v2 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from spatial.v2.context_budget import ContextBudget, admit_row_group, load_tokenizer
from spatial.v2.study.evaluation import _eval_config, _slug, _task_yaml, _wrapper
from spatial.v2.supervision_controls import apply_supervision_arm

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_MODELS = {
    "untuned": None,
    **{
        f"{package}-seed-{seed}": (
            f"experiments/spatialentail-randomized-v2/prepared-sft/models/"
            f"{package}-seed-{seed}"
        )
        for package in ("single-natural", "single-answer-only", "multi-natural")
        for seed in (42, 43)
    },
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_randomized_evaluation(
    benchmark_jsonl: Path,
    output: Path,
    *,
    repo_root: Path,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    budget = ContextBudget(
        "Qwen/Qwen3.5-4B",
        train_max_tokens=8192,
        eval_max_tokens=8192,
        max_new_tokens=4096,
        tokenizer_revision=_REVISION,
    )
    tokenizer = tokenizer or load_tokenizer(budget)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    with benchmark_jsonl.open() as source:
        for line in source:
            natural = json.loads(line)
            view = apply_supervision_arm(natural, "answer-only")
            admission = admit_row_group([view], tokenizer, budget)
            if not admission.accepted:
                raise RuntimeError(
                    f"benchmark prompt exceeds evaluation budget: "
                    f"{admission.row_rejections}"
                )
            row = admission.rows[0]
            metadata = row["metadata"]
            groups[
                (
                    str(metadata["query_kind"]),
                    int(metadata["gold_size"]),
                    str(metadata["benchmark_partition"]),
                )
            ].append(row)
    files = {}
    tasks = []
    task_names = []
    files["tasks/spatialentail_utils.py"] = """\
import importlib.util
from pathlib import Path
source = Path(__file__).resolve().parents[4] / "experiments/tasks/utils.py"
spec = importlib.util.spec_from_file_location("spatialentail_task_utils", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
process_docs_v2_sft = module.process_docs_v2_sft
process_results_v2_answer = module.process_results_v2_answer
"""
    for (query, gold, partition), rows in sorted(groups.items()):
        task = f"spatialentail_random_v2_{_slug(query)}_g{gold}_{_slug(partition)}"
        data_name = f"data/{query}-g{gold}-{partition}.jsonl"
        data_path = "../" + str((output / data_name).resolve().relative_to(repo_root.resolve()))
        files[data_name] = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        files[f"tasks/{task}.yaml"] = _task_yaml(task, data_path, 4096)
        tasks.append({"task": task, "query_kind": query, "gold_size": gold, "partition": partition, "count": len(rows)})
        task_names.append(task)
    include_path = "../" + str((output / "tasks").resolve().relative_to(repo_root.resolve()))
    jobs = []
    for name, lora in _MODELS.items():
        config_relative = (output / "configs" / f"{name}.yaml").resolve().relative_to(repo_root.resolve())
        files[f"configs/{name}.yaml"] = _eval_config(
            tasks=task_names,
            include_path=include_path,
            model_revision=_REVISION,
            max_model_len=8192,
            max_new_tokens=4096,
            lora_path=("../" + lora) if lora else None,
            choices=list("ABCDE"),
            allow_multiple_answers=True,
        )
        script_relative = (output / "slurm" / f"{name}.sh").resolve().relative_to(repo_root.resolve())
        files[f"slurm/{name}.sh"] = _wrapper(
            output_relative=output.resolve().relative_to(repo_root.resolve()),
            model_name=name,
            config_relative=config_relative,
            lora=lora,
        )
        jobs.append({"name": name, "config": str(config_relative), "script": str(script_relative), "lora_path": lora})
    output.mkdir(parents=True)
    for relative, content in files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if relative.startswith("slurm/"):
            path.chmod(0o755)
    (output / "logs").mkdir(); (output / "results").mkdir()
    receipt = {
        "schema": "spatialentail-randomized-v2-eval-jobs-v1",
        "benchmark_sha256": _hash(benchmark_jsonl),
        "total_bases": sum(len(rows) for rows in groups.values()),
        "context_budget": {
            "eval_max_tokens": 8192,
            "max_new_tokens": 4096,
            "tokenizer_name": budget.tokenizer_name,
            "tokenizer_revision": _REVISION,
        },
        "tasks": tasks,
        "jobs": jobs,
        "file_sha256": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()},
        "environment_sha256": {
            name: _hash(repo_root / name)
            for name in (
                "pyproject.toml",
                "uv.lock",
                "eval/eval_new.py",
                "eval/generation_contract.py",
                "experiments/tasks/utils.py",
                "experiments/spatialentail-eval/jobs/eval.sh",
                "spatial/v2/study/evaluation.py",
                "spatial/v2/study/randomized_evaluation.py",
            )
            if (repo_root / name).is_file()
        },
        "submitted": False,
    }
    (output / "jobs.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(prepare_randomized_evaluation(args.benchmark, args.output, repo_root=args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
