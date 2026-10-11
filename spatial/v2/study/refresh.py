"""Refresh repeated Count witness prose in a new candidate, never in-place."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from spatial.v2.context_budget import ContextBudget, admit_row_group, load_tokenizer
from spatial.v2.natural_evidence import share_natural_evidence


def refresh_count_prose(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refresh output exists: {output}")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "candidate-awaiting-quality-review":
        raise ValueError("only an unsealed candidate may be refreshed")
    budget = ContextBudget(**manifest["context_budget"])
    tokenizer = load_tokenizer(budget)
    rendered = {}
    replacement_rows = {}
    changed = Counter()
    for role in manifest["quotas"]:
        path = source / (role + ".jsonl")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["artifacts"][role]["sha256"]
        ):
            raise ValueError("source artifact hash differs")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        groups = {}
        for row in rows:
            groups.setdefault(row["metadata"]["base_id"], []).append(row)
        result = []
        for group in groups.values():
            altered = False
            for row in group:
                if (
                    row["metadata"]["query_kind"] not in {"count", "which"}
                    or row["metadata"]["trace_format"] != "natural"
                    or row["metadata"]["supervision_arm"] != "checked-trace"
                ):
                    continue
                before = row["messages"][-1]["content"]
                trace, footer = before.removeprefix("<think>\n").split(
                    "\n</think>\n", 1
                )
                after = (
                    "<think>\n"
                    + share_natural_evidence(trace)
                    + "\n</think>\n"
                    + footer
                )
                row["messages"][-1]["content"] = after
                row["metadata"]["target_characters"] = len(after)
                row["metadata"]["target_whitespace_tokens"] = len(after.split())
                altered |= before != after
            if altered:
                admission = admit_row_group(group, tokenizer, budget)
                if not admission.accepted:
                    raise ValueError("refreshed paired group no longer admits")
                group = list(admission.rows)
                changed[role] += 1
            result.extend(group)
            replacement_rows.update({row["id"]: row for row in group})
        rendered[role] = result
    for role, record in manifest["artifacts"].items():
        if not role.startswith("train-"):
            continue
        path = source / (role + ".jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("nested source hash mismatch")
        rendered[role] = [
            replacement_rows[json.loads(line)["id"]]
            for line in path.read_text().splitlines()
        ]
    serialized = {
        role: "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        for role, rows in rendered.items()
    }
    for role, rows in rendered.items():
        record = manifest["artifacts"][role]
        record["sha256"] = hashlib.sha256(serialized[role].encode()).hexdigest()
        record["target_tokens"] = sum(
            row["metadata"]["target_model_tokens"] for row in rows
        )
        record["rows"] = len(rows)
        record["bases"] = len({row["metadata"]["base_id"] for row in rows})
        record["unique_structures"] = len(
            {row["metadata"]["structure_signature"] for row in rows}
        )
        record["signature_fallback_rows"] = sum(
            row["metadata"]["structure_profile"]["canonicalization"] != "exact"
            for row in rows
        )
        record["answer_position_rows"] = dict(
            Counter(
                letter for row in rows for letter in row["metadata"]["oracle_letters"]
            )
        )
        record["target_tokens_by_package"] = {
            package: sum(
                row["metadata"]["target_model_tokens"]
                for row in rows
                if row["metadata"]["trace_format"]
                + ":"
                + row["metadata"]["supervision_arm"]
                == package
            )
            for package in record["target_tokens_by_package"]
        }
    manifest["implementation_sources_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("spatial/v2").rglob("*.py"))
    }
    manifest["software_dirty_after_refresh"] = True
    manifest["render_refresh"] = {
        "schema": "shared-membership-witness-prose-v2",
        "source_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
        "changed_bases": dict(changed),
        "renderer_sha256": hashlib.sha256(
            Path("spatial/v2/natural_evidence.py").read_bytes()
        ).hexdigest(),
        "quality_receipts_reused": False,
    }
    output.mkdir(parents=True)
    for role, text in serialized.items():
        (output / (role + ".jsonl")).write_text(text)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest["render_refresh"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(refresh_count_prose(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
