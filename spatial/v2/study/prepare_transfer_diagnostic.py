"""Prepare a fresh, paired answer-policy x premise-layout diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

from spatial.v2.legacy.generation import (
    GenerationMode,
    GenerationPolicy,
    QueryKind,
    SemanticShape,
    SpatialGeneratorV2,
    _system_prompt,
)
from spatial.v2.grading import AnswerMode, encode_menu_answer, resolve_answer
from spatial.v2.solver import (
    Direction,
    DirectionAnalysis,
    SpatialSolverV2,
    WhichAnalysis,
)
from spatial.v2.study.analyze_randomized import archive_digest, source_rows
from spatial.v2.study.evaluation import _eval_config, _task_yaml, _wrapper
from spatial.v2.study.summarize_randomized import digest
from spatial.v2.text import SpatialTextAdapter

MODES = ("single", "all-possible")
LAYOUTS = ("paragraph", "lines")
SCHEMA = "spatialentail-transfer-diagnostic-v1"


def possible_values(analysis):
    if isinstance(analysis, DirectionAnalysis):
        return [v.value for v in analysis.possible_directions]
    if isinstance(analysis, WhichAnalysis):
        return list(analysis.possible_entities)
    return [str(v) for v in analysis.possible_counts]


def paired_views(sample, rng):
    """Keep the semantic problem and all menu positions identical across views."""
    values = possible_values(sample.analysis)
    if not 1 <= len(values) <= 4:
        raise ValueError("Diagnostic requires one to four complete possible values")
    if isinstance(sample.analysis, DirectionAnalysis):
        universe = [v.value for v in Direction]
    elif isinstance(sample.analysis, WhichAnalysis):
        universe = list(sample.problem.query.candidates)
    else:
        universe = [str(v) for v in range(len(sample.problem.query.candidates) + 1)]
    distractors = [v for v in universe if v not in values]
    options = values + rng.sample(distractors, 4 - len(values))
    rng.shuffle(options)
    menu = dict(zip("ABCD", options, strict=True))
    menu["E"] = "Cannot be determined"
    parser = SpatialTextAdapter()
    solver = SpatialSolverV2("z3")
    rows = []
    for mode in MODES:
        resolution = resolve_answer(sample.analysis, mode)
        answer = encode_menu_answer(resolution, menu)
        if not answer.is_resolved:
            raise ValueError("Incomplete diagnostic menu")
        paragraph = SpatialGeneratorV2._prompt(sample.problem, menu, AnswerMode(mode))
        header, premises, question = paragraph.split("\n\n")
        for layout in LAYOUTS:
            rendered = (
                premises if layout == "paragraph" else premises.replace(". ", ".\n")
            )
            text = f"{header}\n\n{rendered}\n\n{question}"
            parsed = parser.parse(text)
            if parsed.problem != sample.problem or parsed.options != menu:
                raise ValueError("Layout changed problem or menu")
            replay = solver.analyze(parsed.problem, include_witnesses=False)
            scored = encode_menu_answer(resolve_answer(replay, mode), parsed.options)
            if replay.error or scored.letters != answer.letters:
                raise ValueError("Reparsed solver answer changed")
            prompt = (
                f"<|im_start|>system\n{_system_prompt(AnswerMode(mode))}<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
            )
            rows.append(
                {
                    "id": f"{sample.base_id}-{mode}-{layout}",
                    "base_id": sample.base_id,
                    "query_kind": sample.policy.query_kind.value,
                    "status": "unique" if len(values) == 1 else "ambiguous",
                    "answer_mode": mode,
                    "layout": layout,
                    "text": text,
                    "evaluation_prompt": prompt,
                    "oracle_option": ",".join(sorted(answer.letters)),
                    "trace_format": "natural",
                    "supervision_arm": "answer-only",
                }
            )
    return rows, {
        "base_id": sample.base_id,
        "structure_signature": sample.structure_signature,
        "generation_seed": sample.seed,
        "attempt": sample.attempt,
        "possible_values": values,
        "options": menu,
        "query_kind": sample.policy.query_kind.value,
        "status": rows[0]["status"],
        "solver": sample.analysis.engine,
        "source_prompt": sample.prompt,
    }


def exclusions(experiment, training_archive, benchmark_archive):
    """Bind exclusion sources to the already audited experiment provenance."""
    provenance = json.loads(
        (experiment / "research-analysis/provenance.json").read_text()
    )
    signatures = set()
    sources = {}
    with tarfile.open(training_archive) as archive:
        for name, info in provenance["files"].items():
            member = "prepared-training-candidate/" + name
            if archive_digest(archive, member) != info["sha256"]:
                raise ValueError(f"Training exclusion hash mismatch: {name}")
            count = 0
            for row in source_rows(archive, member):
                signatures.add(row["metadata"]["structure_signature"])
                count += 1
            if count != info["rows"]:
                raise ValueError("Training exclusion count mismatch")
            sources[member] = {"sha256": info["sha256"], "rows": count}
    with tarfile.open(benchmark_archive) as archive:
        member = "prepared-benchmark-candidate/source-natural.jsonl"
        expected = provenance["benchmark_sha256"]
        if archive_digest(archive, member) != expected:
            raise ValueError("Benchmark exclusion hash mismatch")
        count = 0
        for row in source_rows(archive, member):
            signatures.add(row["metadata"]["structure_signature"])
            count += 1
        if count != provenance["benchmark_bases"]:
            raise ValueError("Benchmark exclusion count mismatch")
        sources[member] = {"sha256": expected, "rows": count}
    return signatures, sources


def generate(per_cell, seed, forbidden):
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    rows, audits = [], []
    occupied = set(forbidden)
    rejected_overlap = 0
    for qindex, query in enumerate(("direction", "which", "count")):
        for sindex, status in enumerate(("unique", "ambiguous")):
            accepted = 0
            for attempt in range(per_cell * 100):
                candidate_seed = seed + qindex * 100_000 + sindex * 10_000 + attempt
                sample = SpatialGeneratorV2(
                    candidate_seed, solver=SpatialSolverV2("z3")
                ).generate(
                    GenerationPolicy(
                        query_kind=QueryKind(query),
                        semantic_shape=SemanticShape(status),
                        ambiguity_size=2 if status == "ambiguous" else None,
                        num_entities=8,
                        num_premises=9,
                        oracle_only=True,
                        measure_support=False,
                        generation_mode=GenerationMode.WORLD_FIRST,
                    ),
                    max_attempts=500,
                )
                if sample.structure_signature in occupied:
                    rejected_overlap += 1
                    continue
                views, audit = paired_views(sample, random.Random(candidate_seed))
                if audit["status"] != status:
                    raise ValueError("Generated wrong semantic status")
                occupied.add(sample.structure_signature)
                rows.extend(views)
                audits.append(audit)
                accepted += 1
                if accepted == per_cell:
                    break
            if accepted != per_cell:
                raise ValueError(f"Could not fill fresh cell {query}/{status}")
    return rows, audits, rejected_overlap


def prepare(
    experiment,
    output,
    training_archive,
    benchmark_archive,
    tokenizer_path,
    root,
    per_cell=10,
    seed=260026,
):
    import z3
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(f"Refusing to replace package: {output}")
    relative = output.resolve().relative_to(root.resolve())
    corr = experiment / "prepared-corr-eval"
    receipt = json.loads((corr / "jobs.json").read_text())
    if digest(tokenizer_path) != receipt["tokenizer_sha256"]:
        raise ValueError("Pinned tokenizer hash mismatch")
    for path, expected in receipt["environment_sha256"].items():
        if digest(root / path) != expected:
            raise ValueError(f"Frozen evaluator source mismatch: {path}")
    for path, expected in receipt["file_sha256"].items():
        if digest(corr / path) != expected:
            raise ValueError(f"Frozen Corr input mismatch: {path}")
    forbidden, excluded_sources = exclusions(
        experiment, training_archive, benchmark_archive
    )
    # Corr is never a generation input. Compare canonical structures only for exclusion.
    from spatial.v2.structure import canonical_structure_profile

    parser = SpatialTextAdapter()
    corr_count = 0
    for line in (corr / "data/corr.jsonl").read_text().splitlines():
        problem = parser.parse(json.loads(line)["text"]).problem
        signature = hashlib.sha256(
            json.dumps(
                canonical_structure_profile(problem),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        forbidden.add(signature)
        corr_count += 1
    rows, audits, overlap_rejections = generate(per_cell, seed, forbidden)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    max_tokens = max(
        len(tokenizer.encode(r["evaluation_prompt"], add_special_tokens=False).ids)
        for r in rows
    )
    budget = receipt["context_budget"]
    if max_tokens + budget["max_new_tokens"] + 32 > budget["eval_max_tokens"]:
        raise ValueError("Diagnostic exceeds frozen context budget")
    files = {
        "audit.jsonl": "".join(json.dumps(a, sort_keys=True) + "\n" for a in audits),
        "tasks/spatialentail_utils.py": "from experiments.tasks.utils import process_results_v2_answer\n",
    }
    tasks = []
    for mode in MODES:
        for layout in LAYOUTS:
            task = f"spatialentail_transfer_diag_{mode.replace('-', '_')}_{layout}"
            subset = [
                r for r in rows if r["answer_mode"] == mode and r["layout"] == layout
            ]
            filename = f"data/{mode}-{layout}.jsonl"
            files[filename] = "".join(
                json.dumps(r, sort_keys=True) + "\n" for r in subset
            )
            files[f"tasks/{task}.yaml"] = _task_yaml(
                task,
                "../" + str(relative / filename),
                budget["max_new_tokens"],
            ).replace(
                "process_docs: !function spatialentail_utils.process_docs_v2_sft\n", ""
            )
            tasks.append(
                {
                    "task": task,
                    "count": len(subset),
                    "data": filename,
                    "answer_mode": mode,
                    "layout": layout,
                }
            )
    jobs = []
    for job in receipt["jobs"]:
        name, lora = job["name"], job["lora_path"]
        config = relative / f"configs/{name}.yaml"
        script = relative / f"slurm/{name}.sh"
        files[f"configs/{name}.yaml"] = _eval_config(
            tasks=[t["task"] for t in tasks],
            include_path="../" + str(relative / "tasks"),
            model_revision=budget["tokenizer_revision"],
            max_model_len=budget["eval_max_tokens"],
            max_new_tokens=budget["max_new_tokens"],
            lora_path="../" + lora if lora else None,
            choices=list("ABCDE"),
            allow_multiple_answers=True,
        )
        files[f"slurm/{name}.sh"] = _wrapper(
            output_relative=relative,
            model_name=name,
            config_relative=config,
            lora=lora,
        )
        jobs.append(
            {
                "name": name,
                "config": str(config),
                "script": str(script),
                "lora_path": lora,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "per_cell": per_cell,
        "unique_bases": len(audits),
        "views_per_base": 4,
        "rows_per_model": len(rows),
        "tasks": tasks,
        "jobs": jobs,
        "submitted": False,
        "headline": False,
        "parent_corr_jobs_sha256": digest(corr / "jobs.json"),
        "context_budget": budget,
        "tokenizer_sha256": digest(tokenizer_path),
        "max_prompt_tokens": max_tokens,
        "preparation_z3_version": z3.get_version_string(),
        "excluded_sources": excluded_sources,
        "corr_exclusion_rows": corr_count,
        "corr_data_sha256": digest(corr / "data/corr.jsonl"),
        "excluded_unique_signatures": len(forbidden),
        "overlap_rejections": overlap_rejections,
        "accepted_overlap": 0,
        "cells": dict(Counter(f"{a['query_kind']}/{a['status']}" for a in audits)),
        "protocol": "Fixed set-capable Stage 2 across all views; not the frozen Corr SINGLE decoder.",
        "file_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in files.items()
        },
        "environment_sha256": dict(receipt["environment_sha256"]),
    }
    manifest["preparation_source_sha256"] = preparation_source_hashes(root)
    manifest["preparation_source_sha256"][
        str(Path(__file__).resolve().relative_to(root))
    ] = digest(Path(__file__))
    output.mkdir(parents=True)
    for name, text in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (output / "jobs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def preparation_source_hashes(root):
    """Include the relocated curriculum in new historical preparation receipts."""
    package = root / "spatial/v2"
    paths = {*package.glob("*.py"), *(package / "legacy").glob("*.py")}
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-archive", type=Path, required=True)
    parser.add_argument("--benchmark-archive", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=10)
    parser.add_argument("--seed", type=int, default=260026)
    args = parser.parse_args()
    result = prepare(
        args.experiment,
        args.output,
        args.training_archive,
        args.benchmark_archive,
        args.tokenizer,
        Path(__file__).resolve().parents[3],
        args.per_cell,
        args.seed,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("unique_bases", "rows_per_model", "cells", "submitted")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
