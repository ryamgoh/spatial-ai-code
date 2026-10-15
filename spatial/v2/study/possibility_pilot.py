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
import copy
import hashlib
import json
import math
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
from spatial.v2.supervision_controls import apply_supervision_arm

# Qwen3.5-4B tokenizer pin, matched to the SINGLE release.
_TOKENIZER = "Qwen/Qwen3.5-4B"
_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# Menus are fixed at five options, matching SINGLE and the external benchmarks.
_CHOICES = list("ABCDE")
_LENGTH_THRESHOLDS = (12288, 16384, 20000, 24576, 32768)
_GENERATION_ENVIRONMENT_FILES = (
    "spatial/v2/generation.py",
    "spatial/v2/possibility.py",
    "spatial/v2/study/possibility_pilot.py",
    "spatial/v2/proofs.py",
    "spatial/v2/answer_certificates.py",
    "spatial/v2/which_certificates.py",
    "spatial/v2/solver.py",
    "spatial/v2/text.py",
)
_MODELS = {
    "untuned": None,
    "natural-seed-42": (
        "experiments/spatialentail-sft/prepared-final/models/natural-seed-42"
    ),
}


def _section(cell_id: str) -> str:
    parts = cell_id.split("-")
    if "COMP" in parts:
        return "comp"
    if "SHIFT" in parts:
        return "shift"
    return "matched"


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_TOKENIZER, revision=_REVISION)


def _resolve_worker_count(requested: int | None, *, work_item_count: int) -> int:
    """Resolve usable generation workers without exceeding independent samples."""
    if work_item_count <= 0:
        raise ValueError("work_item_count must be positive")
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
    return min(workers, work_item_count)


def _sample_seed(
    base_seed: int, cell_id: str, sample_index: int, uniqueness_retry: int = 0
) -> int:
    payload = f"{base_seed}:{cell_id}:{sample_index}:{uniqueness_retry}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _generate_one_row(
    cell: BenchmarkCell,
    *,
    seed: int,
    uniqueness_retry: int = 0,
) -> dict[str, Any]:
    """Generate one independently seeded sample inside a worker process."""
    generator = SpatialGeneratorV2(seed)
    policy = cell.generation_policy()
    sample = generator.generate(policy, max_attempts=200)
    row = sample.as_sft_row(include_audit=True)
    row.setdefault("metadata", {})["uniqueness_retry"] = uniqueness_retry
    return row


def _duplicate_slots(
    generated: dict[str, list[dict[str, Any] | None]], ready: list[BenchmarkCell]
) -> list[tuple[BenchmarkCell, int]]:
    seen = set()
    duplicates = []
    for cell in ready:
        for sample_index, row in enumerate(generated[cell.id]):
            if row is None:
                continue
            signature = row.get("metadata", {}).get("structure_signature")
            if not isinstance(signature, str) or not signature:
                raise ValueError(f"cell {cell.id}: generated row lacks structure signature")
            if signature in seen:
                duplicates.append((cell, sample_index))
            else:
                seen.add(signature)
    return duplicates


def _replace_duplicate_structures(
    generated: dict[str, list[dict[str, Any] | None]],
    ready: list[BenchmarkCell],
    *,
    seed: int,
    workers: int,
    max_rounds: int = 50,
) -> None:
    retries: dict[tuple[str, int], int] = {}
    for _ in range(max_rounds):
        duplicates = _duplicate_slots(generated, ready)
        if not duplicates:
            return
        print(
            f"Regenerating {len(duplicates)} duplicate canonical structure(s)",
            file=sys.stderr,
            flush=True,
        )
        work = []
        for cell, sample_index in duplicates:
            key = (cell.id, sample_index)
            retries[key] = retries.get(key, 0) + 1
            retry = retries[key]
            work.append(
                (
                    cell,
                    sample_index,
                    retry,
                    _sample_seed(seed, cell.id, sample_index, retry),
                )
            )
        if workers == 1:
            for cell, sample_index, retry, sample_seed in work:
                generated[cell.id][sample_index] = _generate_one_row(
                    cell, seed=sample_seed, uniqueness_retry=retry
                )
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(work))) as executor:
                futures = {
                    executor.submit(
                        _generate_one_row,
                        cell,
                        seed=sample_seed,
                        uniqueness_retry=retry,
                    ): (cell, sample_index)
                    for cell, sample_index, retry, sample_seed in work
                }
                for future in as_completed(futures):
                    cell, sample_index = futures[future]
                    generated[cell.id][sample_index] = future.result()
    remaining = _duplicate_slots(generated, ready)
    raise RuntimeError(
        f"could not obtain unique canonical structures after {max_rounds} rounds; "
        f"remaining duplicates: {len(remaining)}"
    )


def _evaluation_row(natural_row: dict[str, Any]) -> dict[str, Any]:
    row = apply_supervision_arm(natural_row, "answer-only")
    row.pop("audit", None)
    return row


def _measure_natural_row(row: dict[str, Any], tokenizer: Any) -> dict[str, int]:
    messages = row["messages"]
    assistant = messages[-1]["content"]
    training_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    evaluation_prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    return {
        "training_model_tokens": len(
            tokenizer.encode(training_text, add_special_tokens=False)
        ),
        "evaluation_prompt_model_tokens": len(
            tokenizer.encode(evaluation_prompt, add_special_tokens=False)
        ),
        "natural_target_model_tokens": len(
            tokenizer.encode(assistant, add_special_tokens=False)
        ),
    }


def _distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("length statistics require at least one value")

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "over_threshold": {
            str(threshold): sum(value > threshold for value in ordered)
            for threshold in _LENGTH_THRESHOLDS
        },
    }


def _length_statistics(measurements: dict[str, list[dict[str, int]]]) -> dict[str, Any]:
    fields = (
        "training_model_tokens",
        "evaluation_prompt_model_tokens",
        "natural_target_model_tokens",
    )
    flattened = [item for rows in measurements.values() for item in rows]
    return {
        "schema": "spatialentail-natural-length-statistics-v1",
        "thresholds": list(_LENGTH_THRESHOLDS),
        "overall": {
            field: _distribution([item[field] for item in flattened])
            for field in fields
        },
        "cells": {
            cell_id: {
                field: _distribution([item[field] for item in rows])
                for field in fields
            }
            for cell_id, rows in sorted(measurements.items())
        },
    }


def _generation_cache_path(output: Path) -> Path:
    return output.with_name(output.name + ".generation-cache")


def _cache_contract(
    root: Path,
    registry_fingerprint: str,
    *,
    samples_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "spatialentail-all-possible-generation-cache-v1",
        "registry_fingerprint": registry_fingerprint,
        "samples_per_cell": samples_per_cell,
        "seed": seed,
        "environment_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in _GENERATION_ENVIRONMENT_FILES
        },
    }


def _write_generation_cache(
    cache: Path,
    generated: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
) -> None:
    if cache.exists():
        raise FileExistsError(f"generation cache already exists: {cache}")
    source = cache / "source-natural"
    source.mkdir(parents=True)
    file_sha256 = {}
    for cell_id, rows in sorted(generated.items()):
        content = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        name = f"source-natural/{cell_id}.jsonl"
        (cache / name).write_text(content)
        file_sha256[name] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {**contract, "file_sha256": file_sha256}
    (cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _load_generation_cache(
    cache: Path,
    contract: dict[str, Any],
    cell_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"incomplete generation cache: {cache}")
    manifest = json.loads(manifest_path.read_text())
    actual_contract = {key: manifest.get(key) for key in contract}
    if actual_contract != contract:
        raise ValueError(f"generation cache contract changed: {cache}")
    generated = {}
    signatures = set()
    for cell_id in cell_ids:
        name = f"source-natural/{cell_id}.jsonl"
        path = cache / name
        content = path.read_text()
        if hashlib.sha256(content.encode()).hexdigest() != manifest[
            "file_sha256"
        ].get(name):
            raise ValueError(f"generation cache file changed: {name}")
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) != int(contract["samples_per_cell"]):
            raise ValueError(f"generation cache row count changed: {name}")
        for row in rows:
            signature = row.get("metadata", {}).get("structure_signature")
            if not isinstance(signature, str) or not signature:
                raise ValueError(f"generation cache row lacks structure: {name}")
            if signature in signatures:
                raise ValueError(f"generation cache contains duplicate structure: {name}")
            signatures.add(signature)
        generated[cell_id] = rows
    return generated


def _generate_ready_cells(
    ready: list[BenchmarkCell],
    *,
    samples_per_cell: int,
    seed: int,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    """Generate cells serially or in a bounded process pool."""
    started = time.perf_counter()
    generated: dict[str, list[dict[str, Any] | None]] = {
        cell.id: [None] * samples_per_cell for cell in ready
    }
    work_items = [
        (
            cell,
            sample_index,
            _sample_seed(seed, cell.id, sample_index),
        )
        for cell in ready
        for sample_index in range(samples_per_cell)
    ]
    print(
        f"Generating {len(work_items)} samples across {len(ready)} cells with "
        f"{workers} worker process(es)",
        file=sys.stderr,
        flush=True,
    )

    if workers == 1:
        for completed, (cell, sample_index, sample_seed) in enumerate(
            work_items, start=1
        ):
            generated[cell.id][sample_index] = _generate_one_row(
                cell,
                seed=sample_seed,
                uniqueness_retry=0,
            )
            print(
                f"[{completed}/{len(work_items)}] generated {cell.id} "
                f"sample {sample_index + 1}/{samples_per_cell}",
                file=sys.stderr,
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _generate_one_row,
                    cell,
                    seed=sample_seed,
                    uniqueness_retry=0,
                ): (cell, sample_index)
                for cell, sample_index, sample_seed in work_items
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                cell, sample_index = futures[future]
                try:
                    generated[cell.id][sample_index] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"cell {cell.id} sample {sample_index}: generation failed"
                    ) from exc
                print(
                    f"[{completed}/{len(work_items)}] generated {cell.id} "
                    f"sample {sample_index + 1}/{samples_per_cell}",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print(f"Generation completed in {elapsed:.1f}s", file=sys.stderr, flush=True)
    if any(row is None for rows in generated.values() for row in rows):
        raise RuntimeError("generation completed with missing sample slots")
    _replace_duplicate_structures(
        generated, ready, seed=seed, workers=workers
    )
    return {
        cell_id: [row for row in rows if row is not None]
        for cell_id, rows in generated.items()
    }


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
        train_max_tokens=eval_max_tokens,
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
    generation_workers = _resolve_worker_count(
        workers, work_item_count=len(ready) * samples_per_cell
    )
    cache = _generation_cache_path(output)
    cache_contract = _cache_contract(
        root,
        registry.fingerprint,
        samples_per_cell=samples_per_cell,
        seed=seed,
    )
    if cache.exists():
        print(f"Reusing generation cache at {cache}", file=sys.stderr, flush=True)
        generated = _load_generation_cache(
            cache, cache_contract, [cell.id for cell in ready]
        )
    else:
        generated = _generate_ready_cells(
            ready,
            samples_per_cell=samples_per_cell,
            seed=seed,
            workers=generation_workers,
        )
        _write_generation_cache(cache, generated, cache_contract)
        print(f"Saved generation cache at {cache}", file=sys.stderr, flush=True)
    tokenizer = _tokenizer()
    measurements: dict[str, list[dict[str, int]]] = {}
    for cell in ready:
        rows: list[dict[str, Any]] = []
        natural_rows = []
        cell_measurements = []
        for cached_row in generated[cell.id]:
            natural_row = copy.deepcopy(cached_row)
            measurement = _measure_natural_row(natural_row, tokenizer)
            natural_row.setdefault("metadata", {})["length_diagnostics"] = measurement
            natural_rows.append(natural_row)
            cell_measurements.append(measurement)
            evaluation_row = _evaluation_row(natural_row)
            admission = admit_row_group([evaluation_row], tokenizer, budget)
            if not admission.accepted:
                raise RuntimeError(
                    f"cell {cell.id}: evaluation row exceeded the context budget: "
                    f"{admission.rejection_reasons}"
                )
            rows.append(admission.rows[0])
        measurements[cell.id] = cell_measurements
        total_rows += len(rows)
        task_name = f"spatialentail_allpossible_{_slug(cell.id)}"
        data_name = f"data/{cell.id}.jsonl"
        source_name = f"source-natural/{cell.id}.jsonl"
        data_path = "../" + str((output / data_name).relative_to(root))
        files[data_name] = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        files[source_name] = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in natural_rows
        )
        files[f"tasks/{task_name}.yaml"] = _task_yaml(task_name, data_path, max_new_tokens)
        task_records.append(
            {
                "task": task_name,
                "role": "bench-all-possible",
                "component": cell.query_kind.value,
                "section": _section(cell.id),
                "cell": cell.id,
                "query_kind": cell.query_kind.value,
                "count": len(rows),
            }
        )
        task_names.append(task_name)

    length_statistics = _length_statistics(measurements)
    files["target-length-stats.json"] = (
        json.dumps(length_statistics, indent=2, sort_keys=True) + "\n"
    )

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
        "spatial/v2/supervision_controls.py",
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
        "generation_work_items": len(ready) * samples_per_cell,
        "generation_parallelism": "sample",
        "generation_duplicate_replacements": sum(
            int(row.get("metadata", {}).get("uniqueness_retry", 0) > 0)
            for rows in generated.values()
            for row in rows
        ),
        "total_rows": total_rows,
        "choices": _CHOICES,
        "allow_multiple_answers": True,
        "context_budget": {
            "tokenizer_name": budget.tokenizer_name,
            "tokenizer_revision": budget.tokenizer_revision,
            "train_max_tokens": budget.train_max_tokens,
            "eval_max_tokens": budget.eval_max_tokens,
            "max_new_tokens": budget.max_new_tokens,
        },
        "natural_length_statistics": length_statistics,
        "generation_cache": str(cache.relative_to(root)),
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
    output_relative = output.relative_to(root)
    (output / "README.md").write_text(
        f"""# SpatialEntail-ALL-POSSIBLE calibration pilot

Engineering-only bundle. It checks whether the untuned base model already
solves the draft set-valued cells before any allocation is frozen. Every sample
is generated with a replay-checked certificate. Full Natural source rows are
preserved under `source-natural/` and measured in `target-length-stats.json`;
they are not culled by a provisional training limit. Derived answer-only rows
under `data/` are admitted for evaluation, where the answer contract is
`ALL_POSSIBLE` on a fixed five-option menu and stage-2 decoding is constrained
to a nonempty subset of A-E. The resumable generation cache is recorded in
`jobs.json`.

Transfer this directory with its ignored `data/` files and the Natural seed-42
adapter, then submit from the repository root:

```sh
sbatch {output_relative}/slurm/untuned.sh
sbatch {output_relative}/slurm/natural-seed-42.sh
```

After both jobs complete, aggregate the exact-set and diagnostic results:

```sh
python3 experiments/spatialentail-allpossible-pilot/scripts/summarize.py \
  {output_relative}
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
    parser.add_argument("--max-new-tokens", type=int, default=12288)
    parser.add_argument("--eval-max-tokens", type=int, default=16384)
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
        max_new_tokens=args.max_new_tokens,
        eval_max_tokens=args.eval_max_tokens,
        workers=args.workers,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
