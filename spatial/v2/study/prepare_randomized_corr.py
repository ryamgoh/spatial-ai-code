"""Prepare seven hash-bound, SINGLE-contract Corr transfer jobs; do not submit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from spatial.v2.generation import _system_prompt
from spatial.v2.grading import AnswerMode
from spatial.v2.study.evaluation import _eval_config, _task_yaml, _wrapper
from spatial.v2.study.summarize_randomized import digest


def prepare(correction: Path, sft: Path, output: Path, tokenizer_path: Path, root: Path):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(f"Transfer package already exists: {output}")
    relative = output.resolve().relative_to(root.resolve())
    receipt = json.loads((sft / "jobs.json").read_text())
    summary = json.loads((correction / "spatialeval_audit_summary.json").read_text())
    if summary["solver"] != {"answer_mode": "single", "backend": "z3"}:
        raise ValueError("Corr requires the current SINGLE Z3 correction")
    if summary["counts"]["status"] != {"exact-match": 667, "underdetermined-oracle-possible": 833}:
        raise ValueError("Correction counts differ from the documented release")
    if digest(root / "data/spatialeval_org.jsonl") != summary["input"]["sha256"]:
        raise ValueError("Correction input differs from the local released source")
    for info in summary["outputs"].values():
        if digest(correction / info["file"]) != info["sha256"]:
            raise ValueError("Correction artifact hash mismatch")
    for path, expected in receipt["environment_sha256"].items():
        if digest(root / path) != expected:
            raise ValueError(f"Frozen internal evaluation environment changed: {path}")
    audits = {
        row["id"]: row
        for row in map(json.loads, (correction / "spatialeval_audit.jsonl").read_text().splitlines())
    }
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    budget = receipt["context_budget"]
    records, ids, max_tokens = [], set(), 0
    system = _system_prompt(AnswerMode.SINGLE)
    for line in (correction / "spatialmap_tqa_corr.jsonl").read_text().splitlines():
        source = json.loads(line)
        if source["id"] in ids:
            raise ValueError("Duplicate correction ID")
        ids.add(source["id"])
        if source["oracle_option"] not in tuple("ABCDE"):
            raise ValueError("Corr gold must be exactly one option")
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{source['text']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        max_tokens = max(max_tokens, tokens)
        # Reserve the answer-stage continuation as well as the reasoning budget.
        if tokens + budget["max_new_tokens"] + 32 > budget["eval_max_tokens"]:
            raise ValueError("Corr prompt exceeds the pinned context budget")
        audit = audits[source["id"]]
        records.append({
            **source, "evaluation_prompt": prompt, "trace_format": "natural",
            "supervision_arm": "answer-only", "answer_mode": "single",
            "query_kind": audit["query_kind"], "audit_status": audit["status"],
        })
    if len(ids) != 1500 or ids != set(audits):
        raise ValueError("Correction must cover exactly all 1500 audited rows")
    output.mkdir(parents=True)
    for name in ("data", "tasks", "configs", "slurm", "logs", "results"):
        (output / name).mkdir()
    (output / "data/corr.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    (output / "correction-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    task = "spatialentail_randomized_v2_corr_single"
    task_text = _task_yaml(task, "../" + str(relative / "data/corr.jsonl"), budget["max_new_tokens"])
    task_text = task_text.replace("process_docs: !function spatialentail_utils.process_docs_v2_sft\n", "")
    (output / f"tasks/{task}.yaml").write_text(task_text)
    (output / "tasks/spatialentail_utils.py").write_text(
        "from experiments.tasks.utils import process_results_v2_answer\n"
    )
    jobs = []
    for job in receipt["jobs"]:
        name, lora = job["name"], job["lora_path"]
        config = relative / "configs" / f"{name}.yaml"
        script = relative / "slurm" / f"{name}.sh"
        (root / config).write_text(_eval_config(
            tasks=[task], include_path="../" + str(relative / "tasks"),
            model_revision=budget["tokenizer_revision"],
            max_model_len=budget["eval_max_tokens"], max_new_tokens=budget["max_new_tokens"],
            lora_path="../" + lora if lora else None, choices=list("ABCDE"),
            allow_multiple_answers=False,
        ))
        (root / script).write_text(_wrapper(
            output_relative=relative, model_name=name, config_relative=config, lora=lora,
        ))
        jobs.append({"name": name, "lora_path": lora, "config": str(config), "script": str(script)})
    result = {
        "schema": "spatialentail-randomized-v2-corr-transfer-v1",
        "source_sft_evaluation_receipt_sha256": digest(sft / "jobs.json"),
        "corrected_sha256": summary["outputs"]["corrected"]["sha256"],
        "tokenizer_sha256": digest(tokenizer_path), "total_bases": 1500,
        "max_prompt_tokens": max_tokens, "context_budget": budget,
        "gold_counts": dict(Counter(r["oracle_option"] for r in records)),
        "protocol": "Current 1500-row SINGLE Corr; not legacy 1329-row all-possible Corr.",
        "tasks": [{"task": task, "count": 1500}], "jobs": jobs, "submitted": False,
        "file_sha256": {str(p.relative_to(output)): digest(p)
                       for p in sorted(output.rglob("*")) if p.is_file()},
        "environment_sha256": {
            **receipt["environment_sha256"],
            "spatial/v2/study/prepare_randomized_corr.py": digest(Path(__file__)),
        },
    }
    (output / "jobs.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--sft-prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.correction, args.sft_prepared, args.output, args.tokenizer,
                     Path(__file__).resolve().parents[3])
    print(json.dumps({k: result[k] for k in ("total_bases", "max_prompt_tokens", "gold_counts", "submitted")}, indent=2))


if __name__ == "__main__":
    main()
