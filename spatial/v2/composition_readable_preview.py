"""Export eleven readable example pairs for feedback, not a dataset release."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from spatial.v2.composition_contracts import encode, parse_prompt, render_prompt
from spatial.v2.composition_evidence import build_evidence
from spatial.v2.composition_feasibility import token_measurement
from spatial.v2.composition_readable import (
    VERSION,
    build_outline,
    check_pair,
    render_outline,
)
from spatial.v2.composition_training import REVISION, digest
from spatial.v2.serialization import tagged_dataclass_to_dict
from spatial.v2.solver import SpatialSolverV2
from spatial.v2.tests.test_pivot_evidence import BY_ID

CASES = (
    (
        "D1",
        ("West", "Northeast", "South", "Northwest"),
        "single",
        "Mixed-direction composition",
    ),
    ("D2", ("West", "East", "South", "Northeast"), "single", "Ambiguous Direction"),
    ("S1", ("U", "V", "W", "T"), "single", "Unique Selection"),
    ("S4", ("U", "V", "W", "T"), "single", "Contingent Selection"),
    ("C2", (0, 1, 2, 3), "single", "Invariant Count"),
    ("C3", (0, 1, 2, 3), "single", "Zero Count"),
    ("C1", (0, 1, 2, 3), "all-possible", "Joint-variable Count"),
    ("I1", ("North", "East", "South", "West"), "single", "Strict-cycle contradiction"),
    (
        "I3",
        ("North", "East", "South", "West"),
        "single",
        "Forced-coincidence contradiction",
    ),
    ("S6", ("U", "V", "W", "T"), "all-possible", "Unlisted possible answer"),
)


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        raise ValueError("refusing to overwrite a preview")
    if args.tokenizer.name != REVISION:
        raise ValueError("expected pinned Qwen tokenizer directory")
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "experiments/spatialentail-composition-v1/depth-feasibility-compact-v1/attempts.jsonl"
    )
    source_hash = digest(source)
    sources = {
        str(p.relative_to(root)): digest(p)
        for p in sorted((root / "spatial/v2").glob("*.py"))
    }
    sources["spatial/v2/tests/test_pivot_evidence.py"] = digest(
        root / "spatial/v2/tests/test_pivot_evidence.py"
    )
    saved = next(
        json.loads(line)
        for line in source.read_text().splitlines()
        if json.loads(line)["story_id"] == "depth-10-sample-4"
        and json.loads(line)["query_kind"] == "direction"
    )
    old_row = next(
        row
        for row in saved["data"]["rows"]
        if row["id"] == "depth-10-sample-4-direction-m0-all-possible"
    )
    problem, menu, mode = parse_prompt(old_row["prompt"])
    examples = [
        (
            old_row["id"],
            "The same depth-ten question",
            problem,
            menu,
            mode,
            old_row["answer"],
        ),
        *(
            (name, title, BY_ID[name].problem(), menu, mode, None)
            for name, menu, mode, title in CASES
        ),
    ]
    solver = SpatialSolverV2(backend="z3", timeout_ms=5000)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = []
    sections = [
        "# Readable Explanation Preview",
        (
            "Eleven actual renderer outputs for readability feedback. This is not "
            "a training set, benchmark, full pilot rerender, or release audit. "
            "Natural and Symbolic use the same selected explanation units. "
            "All meaningful proof-path links remain. Numerical witnesses are audit-only; "
            "possibilities are expressed as qualitative orders or spatial relations. "
            "Repeated possibilities share case labels. Citations refer to the numbered premise "
            "sentences. Symbols N/NE/E/SE/S/SW/W/NW denote the eight exact directions."
        ),
        (
            "Scope: exact-direction paths and possibility sets, ambiguity, "
            "membership cases, supported joint Count constraints, two conflict "
            "types and missing-menu witnesses. Every qualitative alternative has "
            "a checked whole-map completion. Historical artifacts remain unchanged."
        ),
        (
            "The text-integrity checker regenerates the expected templates from "
            "checked evidence. It is not an independent natural-language theorem "
            "prover. Tests inspect specific rendered comparisons, and actual examples "
            "still require readability and semantic review."
        ),
    ]
    for name, title, problem, menu, mode, prior_gold in examples:
        evidence = build_evidence(problem, solver=solver)
        outline = build_outline(evidence, menu, mode, solver=solver)
        pair = render_outline(outline)
        check_pair(evidence, menu, mode, pair, solver=solver)
        prompt = render_prompt(problem, menu, mode)
        answer = encode(evidence.semantics, menu, mode).letters
        if prior_gold is not None and answer != prior_gold:
            raise ValueError("depth-ten gold changed")
        row = {
            "id": name,
            "title": title,
            "prompt": prompt,
            "contract": str(getattr(mode, "value", mode)),
            "split": "readability-preview-only",
            "answer": answer,
            "references": pair,
            "outline": outline,
            "audit_evidence": tagged_dataclass_to_dict(evidence),
        }
        row["tokens"] = [token_measurement(tokenizer, row, style) for style in pair]
        rows.append(row)
        token_text = "; ".join(
            f"{t['style']}: {t['target_tokens']} target tokens "
            f"({t['supervised_tokens']} with ending); admitted={t['admitted']}"
            for t in row["tokens"]
        )
        # The raw JSON retains the complete interface; show the map/menu here.
        question = "Objects: " + prompt.split("Objects: ", 1)[1]
        sections.extend(
            [
                f"## {title}",
                f"ID: `{name}`. Contract: `{row['contract']}`. {token_text}.",
                "### Question",
                "```text\n" + question + "\n```",
                "### Natural",
                pair["natural"],
                "### Symbolic",
                "```text\n" + pair["symbolic"] + "\n```",
            ]
        )
    if digest(source) != source_hash or any(
        digest(root / p) != h for p, h in sources.items()
    ):
        raise ValueError("source changed during preview export")
    payload = {
        "status": "preview-awaiting-readability-feedback",
        "renderer": VERSION,
        "source_pilot_sha256": source_hash,
        "source_sha256": sources,
        "tokenizer_sha256": {
            p.name: digest(p) for p in sorted(args.tokenizer.iterdir()) if p.is_file()
        },
        "versions": {
            p: version(p) for p in ("z3-solver", "transformers", "tokenizers")
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    markdown.write_text("\n\n".join(sections) + "\n")
    print(
        json.dumps(
            {
                "pairs": len(rows),
                "markdown": str(markdown),
                "max_target_tokens": {
                    style: max(
                        t["target_tokens"]
                        for row in rows
                        for t in row["tokens"]
                        if t["style"] == style
                    )
                    for style in ("natural", "symbolic")
                },
                "all_admitted": all(
                    t["admitted"] for row in rows for t in row["tokens"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
