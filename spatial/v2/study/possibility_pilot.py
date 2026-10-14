"""Prepare the SpatialEntail-ALL-POSSIBLE calibration pilot.

This is an engineering-only bundle, not a sealed benchmark. It generates a
small number of rows per draft cell, admits them through the shared context
budget, and writes lm-eval tasks, two model configs (untuned and the Natural
seed-42 adapter), and Slurm wrappers. A repository-level summarizer aggregates
the completed results. The point is to see whether the untuned base model
already solves the set-valued cells before any allocation is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from spatial.v2.benchmark_registry import (
    BenchmarkCell,
    CellReadiness,
    load_benchmark_registry,
)
from spatial.v2.context_budget import ContextBudget, admit_row_group
from spatial.v2.generation import SpatialGeneratorV2
from spatial.v2.study.evaluation import _eval_config, _slug, _task_yaml

# Qwen3.5-4B tokenizer pin, matched to the SINGLE release.
_TOKENIZER = "Qwen/Qwen3.5-4B"
_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# Menus are fixed at five options, matching SINGLE and the external benchmarks.
_CHOICES = list("ABCDE")
_MODELS = {
    "untuned": None,
    "natural-seed-42": (
        "experiments/spatialentail-sft/prepared-final/models/natural-seed-42"
    ),
}


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_TOKENIZER, revision=_REVISION)


def _resolve_worker_count(requested: int | None, *, cell_count: int) -> int:
    """Resolve usable generation workers without exceeding independent cells."""
    if cell_count <= 0:
        raise ValueError("cell_count must be positive")
    raw: int | str = (
        requested
        if requested is not None
        else os.environ.get("SLURM_CPUS_PER_TASK", "1")
    )
    try:
        workers = int(raw)
    except (TypeError, ValueError) as exc:
        source = "workers" if requested is not None else "SLURM_CPUS_PER_TASK"
        raise ValueError(f"{source} must be an integer, got {raw!r}") from exc
    if workers <= 0:
        raise ValueError("workers must be positive")
    return min(workers, cell_count)


def _generate_cell_rows(
    cell: BenchmarkCell,
    *,
    samples_per_cell: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate one independent cell inside a worker process."""
    generator = SpatialGeneratorV2(seed)
    policy = cell.generation_policy()
    return [
        generator.generate(policy, max_attempts=200).as_sft_row()
        for _ in range(samples_per_cell)
    ]


def _generate_ready_cells(
    ready: list[BenchmarkCell],
    *,
    samples_per_cell: int,
    seed: int,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    """Generate cells serially or in a bounded process pool."""
    started = time.perf_counter()
    generated: dict[str, list[dict[str, Any]]] = {}
    print(
        f"Generating {len(ready)} cells with {workers} worker process(es)",
        file=sys.stderr,
        flush=True,
    )

    if workers == 1:
        for completed, (index, cell) in enumerate(enumerate(ready), start=1):
            generated[cell.id] = _generate_cell_rows(
                cell,
                samples_per_cell=samples_per_cell,
                seed=seed + index,
            )
            print(
                f"[{completed}/{len(ready)}] generated {cell.id}",
                file=sys.stderr,
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _generate_cell_rows,
                    cell,
                    samples_per_cell=samples_per_cell,
                    seed=seed + index,
                ): cell
                for index, cell in enumerate(ready)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                cell = futures[future]
                try:
                    generated[cell.id] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"cell {cell.id}: generation failed") from exc
                print(
                    f"[{completed}/{len(ready)}] generated {cell.id}",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print(f"Generation completed in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return generated


def _pilot_wrapper(
    *, output_relative: Path, model_name: str, config_relative: Path, lora: str | None
) -> str:
    model_dir = f"$SLURM_SUBMIT_DIR/{lora}" if lora else "$SLURM_SUBMIT_DIR"
    untuned = 1 if lora is None else 0
    return f"""#!/bin/bash
#SBATCH --job-name=spatialentail-allpossible-{model_name}
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
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/lib/pin-srun-cpus.sh"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/experiments/15-v2-ablation/jobs/common.bash"

base="$SLURM_SUBMIT_DIR/{output_relative}"
result_dir="$base/results/{model_name}"
if [[ {untuned} != 1 ]]; then
  [[ -f "{model_dir}/COMPLETED" ]] && has_adapter "{model_dir}" || {{
    echo "pilot {model_name} has no verified adapter at {model_dir}" >&2
    exit 1
  }}
fi
if [[ -f "$result_dir/COMPLETED" && -f "$result_dir/results.json" ]]; then
  echo "pilot {model_name} already completed"
  exit 0
fi

python3 - "$base/jobs.json" "$base" "$SLURM_SUBMIT_DIR" <<'CHECK'
import hashlib, json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
base = pathlib.Path(sys.argv[2])
root = pathlib.Path(sys.argv[3])
for name, expected in receipt["file_sha256"].items():
    if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Prepared pilot input changed: " + name)
for name, expected in receipt["environment_sha256"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Pilot environment changed: " + name)
CHECK

normalize_single_cuda_device
mkdir -p "$result_dir"
sync_uv_project "$SLURM_SUBMIT_DIR/eval"
export PYTHONPATH="$SLURM_SUBMIT_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
cd eval
uv run python eval_new.py \\
  --config "$SLURM_SUBMIT_DIR/{config_relative}" \\
  --stages 2 \\
  --output-dir "$result_dir"

[[ -f "$result_dir/results.json" ]] || {{
  echo "pilot {model_name} did not produce results.json" >&2
  exit 1
}}
touch "$result_dir/COMPLETED"
"""


def prepare_pilot(
    registry_path: Path,
    output: Path,
    *,
    samples_per_cell: int,
    seed: int = 42,
    max_new_tokens: int = 12288,
    eval_max_tokens: int = 16384,
    repo_root: Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"pilot output already exists: {output}")
    if not output.is_relative_to(root):
        raise ValueError("pilot output must live inside the repository")
    if samples_per_cell <= 0:
        raise ValueError("samples_per_cell must be positive")

    registry = load_benchmark_registry(registry_path)
    budget = ContextBudget(
        tokenizer_name=_TOKENIZER,
        train_max_tokens=max_new_tokens,
        eval_max_tokens=eval_max_tokens,
        max_new_tokens=max_new_tokens,
        tokenizer_revision=_REVISION,
    )
    files: dict[str, str] = {}
    task_records: list[dict[str, Any]] = []
    task_names: list[str] = []
    total_rows = 0
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

    ready = [c for c in registry.cells if c.readiness is CellReadiness.READY]
    generation_workers = _resolve_worker_count(workers, cell_count=len(ready))
    generated = _generate_ready_cells(
        ready,
        samples_per_cell=samples_per_cell,
        seed=seed,
        workers=generation_workers,
    )
    tokenizer = _tokenizer()
    for cell in ready:
        rows: list[dict[str, Any]] = []
        for row in generated[cell.id]:
            admission = admit_row_group([row], tokenizer, budget)
            if not admission.accepted:
                raise RuntimeError(
                    f"cell {cell.id}: row exceeded the context budget: "
                    f"{admission.rejection_reasons}"
                )
            rows.append(admission.rows[0])
        total_rows += len(rows)
        task_name = f"spatialentail_allpossible_{_slug(cell.id)}"
        data_name = f"data/{cell.id}.jsonl"
        data_path = "../" + str((output / data_name).relative_to(root))
        files[data_name] = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        files[f"tasks/{task_name}.yaml"] = _task_yaml(task_name, data_path, max_new_tokens)
        task_records.append(
            {
                "task": task_name,
                "role": "bench-all-possible",
                "component": cell.query_kind.value,
                "cell": cell.id,
                "query_kind": cell.query_kind.value,
                "count": len(rows),
            }
        )
        task_names.append(task_name)

    include_path = "../" + str((output / "tasks").relative_to(root))
    jobs = []
    for model_name, lora in _MODELS.items():
        files[f"configs/{model_name}.yaml"] = _eval_config(
            tasks=task_names,
            include_path=include_path,
            model_revision=_REVISION,
            max_model_len=eval_max_tokens,
            max_new_tokens=max_new_tokens,
            lora_path=("../" + lora) if lora else None,
            choices=_CHOICES,
            allow_multiple_answers=True,
        )
        config_relative = (output / "configs" / f"{model_name}.yaml").relative_to(root)
        files[f"slurm/{model_name}.sh"] = _pilot_wrapper(
            output_relative=output.relative_to(root),
            model_name=model_name,
            config_relative=config_relative,
            lora=lora,
        )
        jobs.append(
            {
                "name": model_name,
                "config": str(config_relative),
                "script": str((output / "slurm" / f"{model_name}.sh").relative_to(root)),
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
        "spatial/v2/study/possibility_pilot.py",
        "spatial/v2/possibility.py",
        "experiments/spatialentail-eval/jobs/eval.sh",
        "experiments/spatialentail-allpossible-pilot/scripts/summarize.py",
    )
    receipt = {
        "schema": "spatialentail-all-possible-pilot-v1",
        "status": "engineering-only",
        "registry_fingerprint": registry.fingerprint,
        "samples_per_cell": samples_per_cell,
        "seed": seed,
        "generation_workers": generation_workers,
        "total_rows": total_rows,
        "choices": _CHOICES,
        "allow_multiple_answers": True,
        "tasks": task_records,
        "jobs": jobs,
        "environment_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
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
    (output / "jobs.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        """# SpatialEntail-ALL-POSSIBLE calibration pilot

Engineering-only bundle. It checks whether the untuned base model already
solves the draft set-valued cells before any allocation is frozen. Rows are
freshly generated and admitted through the shared context budget; the answer
contract is `ALL_POSSIBLE` on a fixed five-option menu, so stage-2 decoding is
constrained to a nonempty subset of A-E.

Transfer this directory with its ignored `data/` files and the Natural seed-42
adapter, then submit from the repository root:

```sh
sbatch experiments/spatialentail-allpossible-pilot/prepared/slurm/untuned.sh
sbatch experiments/spatialentail-allpossible-pilot/prepared/slurm/natural-seed-42.sh
```

After both jobs complete, aggregate the exact-set and diagnostic results:

```sh
python3 experiments/spatialentail-allpossible-pilot/scripts/summarize.py \
  experiments/spatialentail-allpossible-pilot/prepared
```
"""
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-cell", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        help="generation processes (default: SLURM_CPUS_PER_TASK, otherwise 1)",
    )
    args = parser.parse_args()
    receipt = prepare_pilot(
        args.registry,
        args.out,
        samples_per_cell=args.samples_per_cell,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
