"""Prepare text-only preflight, smoke and full spatial-specialist jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from spatial.v2.study.comparator_evaluation import _hash


@dataclass(frozen=True)
class SpatialSpecialistSpec:
    name: str
    pretrained: str
    revision: str
    loader: str
    dtype: str
    gated: bool
    chat_template_kwargs: dict[str, Any]


SPATIAL_SPECIALISTS = (
    SpatialSpecialistSpec(
        name="spatialasm-qwen3-5-9b",
        pretrained="YuhongLiu/SpatialASM-Qwen3.5-9B",
        revision="56e3970b23f93bb128471c624718c556ef5d4cad",
        loader="standard",
        dtype="bfloat16",
        gated=False,
        chat_template_kwargs={"enable_thinking": True},
    ),
    SpatialSpecialistSpec(
        name="sensenova-si-1-2-internvl3-8b",
        pretrained="sensenova/SenseNova-SI-1.2-InternVL3-8B",
        revision="57bfba82ed67dc27cf6607626362ee48391ecaa4",
        loader="remote",
        dtype="bfloat16",
        gated=False,
        chat_template_kwargs={},
    ),
    SpatialSpecialistSpec(
        name="spaceqwen3-vl-2b-thinking",
        pretrained="remyxai/SpaceQwen3-VL-2B-Thinking",
        revision="405a1c95bd0cefa6bdbe7ed538d1bd854e15597e",
        loader="standard",
        dtype="float16",
        gated=False,
        chat_template_kwargs={},
    ),
    SpatialSpecialistSpec(
        name="robobrain2-5-8b-nv",
        pretrained="BAAI/RoboBrain2.5-8B-NV",
        revision="7ae187cc62e45cfa1c5904903a8f017f11b2037b",
        loader="standard",
        dtype="bfloat16",
        gated=True,
        chat_template_kwargs={},
    ),
    SpatialSpecialistSpec(
        name="cosmos-reason2-8b",
        pretrained="nvidia/Cosmos-Reason2-8B",
        revision="a9fae2cf89dc64db96b12860417f0eb403013bb9",
        loader="standard",
        dtype="bfloat16",
        gated=True,
        chat_template_kwargs={},
    ),
)


def _config(
    spec: SpatialSpecialistSpec,
    *,
    tasks: list[str],
    include_path: str,
    limit: int | None = None,
) -> str:
    config = {
        "model": (
            "vllm_staged_remote_chat_pass"
            if spec.loader == "remote"
            else "vllm_staged_chat_pass"
        ),
        "model_args": {
            "pretrained": spec.pretrained,
            "revision": spec.revision,
            "tokenizer_revision": spec.revision,
            "stages": 2,
            "max_thinking_tokens": 4096,
            "dtype": spec.dtype,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 8192,
            "choices": list("ABCDE"),
            "allow_multiple_answers": True,
            "record_stage1": True,
            "chat_template_kwargs": spec.chat_template_kwargs,
            "system_message_policy": "native",
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
    spec: SpatialSpecialistSpec,
    config_relative: Path,
    *,
    mode: str,
) -> str:
    if mode not in {"preflight", "smoke", "full"}:
        raise ValueError(f"unknown spatial run mode: {mode}")
    run_name = spec.name if mode == "full" else f"{mode}-{spec.name}"
    result_relative = (
        f"results/{spec.name}"
        if mode == "full"
        else f"results/{mode}/{spec.name}"
    )
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
bash experiments/spatialentail-randomized-v2/jobs/eval-spatial.sh
"""


def _submit_script(
    output_relative: Path,
    specs: tuple[SpatialSpecialistSpec, ...],
    *,
    mode: str,
) -> str:
    directory = "slurm" if mode == "full" else f"slurm/{mode}"
    if mode == "preflight":
        selection = "\n".join(
            f'SCRIPTS+=("$ROOT/{output_relative}/{directory}/{spec.name}.sh")'
            for spec in specs
        )
        dependency_type = "afterany"
    else:
        prerequisite = "preflight" if mode == "smoke" else "smoke"
        selection = "\n".join(
            f'''if [[ -f "$ROOT/{output_relative}/results/{prerequisite}/{spec.name}/COMPLETED" ]]; then
  SCRIPTS+=("$ROOT/{output_relative}/{directory}/{spec.name}.sh")
else
  echo "Skipping {spec.name}: {prerequisite} did not complete" >&2
fi'''
            for spec in specs
        )
        dependency_type = "afterok"
    return f"""#!/bin/bash
set -euo pipefail
ROOT=${{SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT"
SCRIPTS=()
{selection}
(( ${{#SCRIPTS[@]}} > 0 )) || {{ echo "No eligible {mode} jobs" >&2; exit 1; }}
dependency=()
for script in "${{SCRIPTS[@]}}"; do
  job_id=$(sbatch --parsable "${{dependency[@]}}" -p gpu -t 03:00:00 --gres=gpu:h200-141:1 "$script")
  printf '%s %s\n' "$job_id" "$script"
  dependency=(--dependency="{dependency_type}:$job_id")
done
"""


def _verify_general_package(
    receipt: dict[str, Any],
    prepared: Path,
    *,
    expected_total_bases: int,
) -> None:
    if receipt.get("schema") != "spatialentail-general-instruction-eval-v1":
        raise ValueError("spatial evaluation requires the prepared general package")
    if (
        receipt.get("total_bases") != expected_total_bases
        or len(receipt.get("tasks") or []) != 48
    ):
        raise ValueError("spatial evaluation requires the expected 48-task package")
    for name, expected in receipt["file_sha256"].items():
        path = prepared / name
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"prepared general input changed: {name}")


def prepare_spatial_specialist_evaluation(
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
        raise FileExistsError(f"spatial evaluation output already exists: {output}")
    if not general.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("prepared inputs and output must live inside the repository")

    general_receipt_path = general / "jobs.json"
    general_receipt = json.loads(general_receipt_path.read_text())
    _verify_general_package(
        general_receipt,
        general,
        expected_total_bases=expected_total_bases,
    )

    task_names = [str(task["task"]) for task in general_receipt["tasks"]]
    include_path = "../" + str((general / "tasks").relative_to(root))
    files: dict[str, str] = {}
    jobs = []
    for spec in SPATIAL_SPECIALISTS:
        config_relative = (output / "configs" / f"{spec.name}.yaml").relative_to(root)
        smoke_config_relative = (
            output / "configs" / "smoke" / f"{spec.name}.yaml"
        ).relative_to(root)
        preflight_config_relative = (
            output / "configs" / "preflight" / f"{spec.name}.yaml"
        ).relative_to(root)
        script_relative = (output / "slurm" / f"{spec.name}.sh").relative_to(root)
        smoke_script_relative = (
            output / "slurm" / "smoke" / f"{spec.name}.sh"
        ).relative_to(root)
        preflight_script_relative = (
            output / "slurm" / "preflight" / f"{spec.name}.sh"
        ).relative_to(root)

        files[f"configs/{spec.name}.yaml"] = _config(
            spec, tasks=task_names, include_path=include_path
        )
        files[f"configs/smoke/{spec.name}.yaml"] = _config(
            spec, tasks=task_names, include_path=include_path, limit=1
        )
        files[f"configs/preflight/{spec.name}.yaml"] = _config(
            spec, tasks=task_names[:1], include_path=include_path, limit=1
        )
        files[f"slurm/{spec.name}.sh"] = _wrapper(
            output.relative_to(root), spec, config_relative, mode="full"
        )
        files[f"slurm/smoke/{spec.name}.sh"] = _wrapper(
            output.relative_to(root), spec, smoke_config_relative, mode="smoke"
        )
        files[f"slurm/preflight/{spec.name}.sh"] = _wrapper(
            output.relative_to(root), spec, preflight_config_relative, mode="preflight"
        )
        jobs.append(
            {
                **asdict(spec),
                "input_modality": "text-only",
                "sampling_regime": "deterministic-stage1-and-stage2",
                "config": str(config_relative),
                "script": str(script_relative),
                "smoke_config": str(smoke_config_relative),
                "smoke_script": str(smoke_script_relative),
                "preflight_config": str(preflight_config_relative),
                "preflight_script": str(preflight_script_relative),
            }
        )

    for mode in ("preflight", "smoke", "full"):
        suffix = "chain" if mode == "full" else f"{mode}-chain"
        files[f"submit-h200-{suffix}.sh"] = _submit_script(
            output.relative_to(root), SPATIAL_SPECIALISTS, mode=mode
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

    upstream_hashes = {
        str((general / name).relative_to(root)): expected
        for name, expected in general_receipt["file_sha256"].items()
        if name.startswith("data/") or name.startswith("tasks/")
    }
    environment_files = (
        "pyproject.toml",
        "uv.lock",
        "eval/eval_new.py",
        "eval/eval_comparators.py",
        "eval/eval_spatial.py",
        "eval/generation_contract.py",
        "spatial/v2/evaluation_chat.py",
        "experiments/tasks/utils.py",
        "experiments/spatialentail-randomized-v2/jobs/eval-spatial.sh",
        "spatial/v2/study/spatial_specialist_evaluation.py",
    )
    receipt = {
        "schema": "spatialentail-spatial-specialist-eval-v1",
        "upstream_general_receipt_sha256": _hash(general_receipt_path),
        "upstream_file_sha256": upstream_hashes,
        "benchmark_sha256": general_receipt["benchmark_sha256"],
        "total_bases": general_receipt["total_bases"],
        "tasks": len(task_names),
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
    parser.add_argument("general_prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_spatial_specialist_evaluation(
                args.general_prepared, args.output, repo_root=args.repo_root
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
