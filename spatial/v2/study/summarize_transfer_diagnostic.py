"""Validate paired diagnostic results without changing any headline scores."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean

import yaml

from spatial.v2.study.analyze_randomized import (
    WRAPPER,
    answer_set,
    archive_digest,
    source_rows,
)
from spatial.v2.study.summarize_randomized import METRICS, digest, metric, pct, table


def describe(rows):
    return {
        "n": len(rows),
        **{key: mean(r[key] for r in rows) for key in METRICS},
        "stage2_e_rate": mean(r["stage2_pred"] == "E" for r in rows),
        "stage2_multi_rate": mean("," in r["stage2_pred"] for r in rows),
    }


def paired_effects(rows):
    index = {(r["base_id"], r["answer_mode"], r["layout"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate paired view")
    bases = sorted({r["base_id"] for r in rows})
    if len(rows) != len(bases) * 4:
        raise ValueError("Incomplete four-view pairing")
    output = []
    comparisons = [
        ("policy", ("all-possible", layout), ("single", layout))
        for layout in ("paragraph", "lines")
    ] + [
        ("layout", (mode, "paragraph"), (mode, "lines"))
        for mode in ("single", "all-possible")
    ]
    for label, first, second in comparisons:
        for status in ("unique", "ambiguous"):
            for query in ("all", "direction", "which", "count"):
                pairs = [
                    (index[(base, *first)], index[(base, *second)]) for base in bases
                ]
                selected = [
                    (a, b)
                    for a, b in pairs
                    if a["status"] == status
                    and (query == "all" or a["query_kind"] == query)
                ]
                if not selected:
                    continue
                output.append(
                    {
                        "factor": label,
                        "first": "/".join(first),
                        "second": "/".join(second),
                        "status": status,
                        "query": query,
                        "n": len(selected),
                        "wins": sum(
                            not a["strict_acc"] and b["strict_acc"] for a, b in selected
                        ),
                        "losses": sum(
                            a["strict_acc"] and not b["strict_acc"] for a, b in selected
                        ),
                        "difference_pp": 100
                        * mean(b["strict_acc"] - a["strict_acc"] for a, b in selected),
                    }
                )
    return output


def summarize(archive_path, prepared, root):
    receipt = json.loads((prepared / "jobs.json").read_text())
    if receipt["schema"] != "spatialentail-transfer-diagnostic-v1":
        raise ValueError("Wrong diagnostic schema")
    for base, hashes in (
        (prepared, receipt["file_sha256"]),
        (root, receipt["environment_sha256"]),
    ):
        for path, expected in hashes.items():
            if digest(base / path) != expected:
                raise ValueError(f"Frozen hash mismatch: {path}")
    spec = importlib.util.spec_from_file_location(
        "diag_scorer", root / "experiments/tasks/utils.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    models, hashes = {}, {}
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Duplicate archive member")
        for m in members:
            path = PurePosixPath(m.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (m.isfile() or m.isdir())
            ):
                raise ValueError("Unsafe archive member")
        if archive_digest(archive, "jobs.json") != digest(prepared / "jobs.json"):
            raise ValueError("Archive receipt mismatch")
        present = {
            PurePosixPath(m.name).parts[1]
            for m in members
            if m.isfile() and m.name.startswith("results/")
        }
        if present != {j["name"] for j in receipt["jobs"]}:
            raise ValueError("Model coverage mismatch")
        for job in receipt["jobs"]:
            prefix = f"results/{job['name']}/"
            if not archive.getmember(prefix + "COMPLETED").isfile():
                raise ValueError("Missing completion")
            stream = archive.extractfile(prefix + "results.json")
            if stream is None:
                raise ValueError("Missing results")
            with stream:
                payload = json.load(stream)
            expected_config = yaml.safe_load((root / job["config"]).read_text())
            runtime = payload["config"]
            if (
                runtime.get("limit") is not None
                or runtime["model"] != expected_config["model"]
            ):
                raise ValueError("Limited run or wrong backend")
            for key, value in expected_config["model_args"].items():
                if runtime["model_args"].get(key) != value:
                    raise ValueError(f"Runtime argument mismatch: {key}")
            if set(payload["results"]) != {t["task"] for t in receipt["tasks"]}:
                raise ValueError("Task coverage mismatch")
            rows = []
            for task in receipt["tasks"]:
                sources = {
                    r["id"]: r
                    for r in map(
                        json.loads, (prepared / task["data"]).read_text().splitlines()
                    )
                }
                task_rows, seen = [], set()
                reported = payload["results"][task["task"]]
                generation = {
                    "max_gen_toks": receipt["context_budget"]["max_new_tokens"],
                    "temperature": 0,
                    "repetition_penalty": 1,
                    "until": [],
                }
                for sample in source_rows(
                    archive, prefix + f"responses_{task['task']}.jsonl"
                ):
                    identity = sample["doc"]["id"]
                    if identity in seen or identity not in sources:
                        raise ValueError("Duplicate/unknown sample")
                    seen.add(identity)
                    source = sources[identity]
                    if (
                        sample["doc"] != source
                        or sample["target"] != source["oracle_option"]
                    ):
                        raise ValueError("Document/target mismatch")
                    if sample["arguments"] != [
                        [source["evaluation_prompt"], generation]
                    ]:
                        raise ValueError("Prompt/generation mismatch")
                    (response,) = sample["filtered_resps"]
                    if (
                        sample["resps"] != [[response]]
                        or WRAPPER.fullmatch(response) is None
                    ):
                        raise ValueError(
                            "Raw/filtered response or stage wrapper mismatch"
                        )
                    pred = answer_set(response)
                    if not pred or not pred <= set("ABCDE"):
                        raise ValueError("Invalid stage-2 set")
                    replay = module.process_results_v2_answer(source, [response])
                    if any(sample[k] != v for k, v in replay.items()):
                        raise ValueError("Scorer replay mismatch")
                    task_rows.append(
                        {
                            **{
                                k: source[k]
                                for k in (
                                    "base_id",
                                    "query_kind",
                                    "status",
                                    "answer_mode",
                                    "layout",
                                )
                            },
                            **{k: replay[k] for k in METRICS},
                            "stage2_pred": ",".join(sorted(pred)),
                        }
                    )
                if (
                    seen != sources.keys()
                    or len(seen) != task["count"]
                    or reported["sample_len"] != len(seen)
                ):
                    raise ValueError("Task sample coverage mismatch")
                effective = (
                    payload.get("n-samples", {}).get(task["task"], {}).get("effective")
                )
                if effective not in (None, len(seen)):
                    raise ValueError("Effective sample coverage mismatch")
                for key in METRICS:
                    if not math.isclose(
                        mean(r[key] for r in task_rows),
                        metric(reported, key),
                        abs_tol=1e-10,
                    ):
                        raise ValueError("Aggregate score mismatch")
                rows.extend(task_rows)
            groups = defaultdict(list)
            for row in rows:
                groups[
                    f"{row['answer_mode']}/{row['layout']}/{row['query_kind']}/{row['status']}"
                ].append(row)
            models[job["name"]] = {
                "rows": rows,
                "slices": {k: describe(v) for k, v in sorted(groups.items())},
                "paired_effects": paired_effects(rows),
            }
        for member in members:
            if member.isfile():
                hashes[member.name] = archive_digest(archive, member.name)
    return {
        "schema": "spatialentail-transfer-diagnostic-summary-v1",
        "archive_sha256": digest(archive_path),
        "jobs_sha256": digest(prepared / "jobs.json"),
        "member_sha256": hashes,
        "models": models,
    }


def render(report):
    lines = [
        "# Matched Transfer Diagnostic",
        "",
        (
            "Diagnostic only. All views use the same set-capable decoder. Not comparable "
            "as a replacement for Corr headline scores. Differences are second minus first. "
            "Counts are descriptive; do not treat four views as independent problems."
        ),
        "",
    ]
    for name, model in report["models"].items():
        lines += [f"## {name}", ""]
        table(
            lines,
            ["Slice", "N", "Stage 1", "Stage 2", "E rate", "Multi-letter rate"],
            [
                [
                    key,
                    str(s["n"]),
                    *[
                        pct(s[k])
                        for k in (
                            "stage1_acc",
                            "strict_acc",
                            "stage2_e_rate",
                            "stage2_multi_rate",
                        )
                    ],
                ]
                for key, s in model["slices"].items()
            ],
        )
        table(
            lines,
            [
                "Factor",
                "First",
                "Second",
                "Status",
                "Query",
                "N",
                "Wins",
                "Losses",
                "Delta pp",
            ],
            [
                [
                    r["factor"],
                    r["first"],
                    r["second"],
                    r["status"],
                    r["query"],
                    str(r["n"]),
                    str(r["wins"]),
                    str(r["losses"]),
                    f"{r['difference_pp']:.2f}",
                ]
                for r in model["paired_effects"]
            ],
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace diagnostic results")
    result = summarize(args.archive, args.prepared, Path(__file__).resolve().parents[3])
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "SUMMARY.md").write_text(render(result))
    print(f"Validated {len(result['models'])} complete diagnostic runs")


if __name__ == "__main__":
    main()
