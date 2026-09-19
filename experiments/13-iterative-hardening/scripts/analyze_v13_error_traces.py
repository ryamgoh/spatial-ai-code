"""Compare V13 stage-1 traces on identical prompts across model cells.

Deterministic cohorts identify regressions. Trace labels are intentionally
heuristic and every exported example includes the raw prompt and completions
for manual verification.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("v13_summary_for_audit", SUMMARY_PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)

TAGS = ("baseline-stage1", "v12-4b-1.5k-stage1", "v12-4b-6k-stage1")
RELATION_RE = re.compile(
    r"The (.+?) is to the "
    r"(Northeast|Northwest|Southeast|Southwest|North|South|East|West) "
    r"of the (.+?)\.",
    re.I,
)
STEP_RE = re.compile(r"### Step \d+\s*(.*?)(?=### Step \d+|### Final Deduction|### Consistency Check|$)", re.S)
EXTRACTION_RE = {
    axis: re.compile(rf"\*\*{axis}-Extraction\*\*:\s*([^\n]+)", re.I)
    for axis in ("X", "Y")
}


def _text(sample: dict) -> str:
    return str((sample.get("doc") or {}).get("text") or "")


def _response(sample: dict) -> str:
    raw = sample.get("resps")
    while isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw or "")


def _difficulty(sample: dict) -> dict:
    return dict((sample.get("doc") or {}).get("difficulty") or {})


def _index(rows: list[dict]) -> dict[str, dict]:
    out = {_text(row): row for row in rows}
    if len(out) != len(rows):
        raise ValueError("response file contains duplicate prompt text")
    return out


def _step_extractions(response: str) -> list[dict[str, str]]:
    result = []
    for block in STEP_RE.findall(response):
        item = {}
        for axis, pattern in EXTRACTION_RE.items():
            match = pattern.search(block)
            if match:
                item[axis.lower()] = match.group(1).strip()
        result.append(item)
    return result


def cardinal_axis_errors(prompt: str, response: str) -> dict[str, int]:
    """Count template-shaped cardinal extraction mistakes.

    This only scores steps whose X/Y extraction lines can be aligned by index
    with prompt relations. Missing or free-form steps are counted separately.
    """
    relations = list(RELATION_RE.finditer(prompt.split("Question:", 1)[0]))
    steps = _step_extractions(response)
    counts = Counter()
    if len(steps) != len(relations):
        counts["unaligned_trace"] += 1
        return dict(counts)
    for relation, step in zip(relations, steps):
        subject, direction_raw, reference = relation.groups()
        direction = direction_raw.lower()
        active = "x" if direction in {"east", "west"} else "y" if direction in {"north", "south"} else None
        if active is None:
            continue
        inactive = "y" if active == "x" else "x"
        active_value = step.get(active, "")
        inactive_value = step.get(inactive, "")
        if not active_value or "none" in active_value.lower():
            counts["missing_active_axis_update"] += 1
        else:
            low, high = (
                (reference, subject)
                if direction in {"east", "north"}
                else (subject, reference)
            )
            expected = f"{low.strip()} < {high.strip()}"
            normalized = re.sub(r"\s+", " ", active_value).strip()
            if normalized != expected:
                counts["incorrect_active_axis_update"] += 1
        if inactive_value and "none" not in inactive_value.lower():
            counts["cardinal_cross_axis_update"] += 1
    return dict(counts)


def classify(sample: dict) -> list[str]:
    difficulty = _difficulty(sample)
    labels = []
    if not SUMMARY.prediction(sample):
        labels.append("missing_answer")
    if difficulty.get("world_consistency") == "inconsistent":
        labels.append("missed_global_inconsistency")
    subtype = SUMMARY._semantic_subtype(sample)
    family = SUMMARY._question_family(sample)
    if subtype == "dir-2":
        labels.append("multi_answer_policy")
    elif family == "which":
        labels.append("entity_set_enumeration")
    elif family == "count":
        labels.append("count_or_menu_policy")
    for label, count in cardinal_axis_errors(_text(sample), _response(sample)).items():
        if count:
            labels.append(label)
    return sorted(set(labels or ["unclassified"]))


def build_audit(results_dir: Path, limit: int) -> tuple[str, list[dict]]:
    indexed = {tag: _index(SUMMARY.load_samples(results_dir, tag)) for tag in TAGS}
    missing = [tag for tag, rows in indexed.items() if not rows]
    if missing:
        raise ValueError(f"missing response samples for: {', '.join(missing)}")
    prompt_sets = [set(rows) for rows in indexed.values()]
    if not all(prompts == prompt_sets[0] for prompts in prompt_sets[1:]):
        raise ValueError("model response files do not contain identical prompts")

    cohorts = {
        "base-correct_1.5k-wrong": [],
        "1.5k-correct_6k-wrong": [],
        "base-correct_both-adapters-wrong": [],
    }
    for prompt in sorted(prompt_sets[0]):
        rows = {tag: indexed[tag][prompt] for tag in TAGS}
        correct = {tag: SUMMARY.strict_correct(row) for tag, row in rows.items()}
        if correct[TAGS[0]] and not correct[TAGS[1]]:
            cohorts["base-correct_1.5k-wrong"].append(rows)
        if correct[TAGS[1]] and not correct[TAGS[2]]:
            cohorts["1.5k-correct_6k-wrong"].append(rows)
        if correct[TAGS[0]] and not correct[TAGS[1]] and not correct[TAGS[2]]:
            cohorts["base-correct_both-adapters-wrong"].append(rows)

    lines = [
        "# V13 paired error-trace audit",
        "",
        "All cohorts compare identical prompts. Cohort membership and answer correctness are deterministic. Trace labels are heuristics and must be verified against the included raw completions.",
        "",
        "| cohort | rows | cardinal/mixed | depth 3–5 |",
        "|---|---:|---:|---:|",
    ]
    exported = []
    for name, groups in cohorts.items():
        representative = [group[TAGS[1] if "1.5k-wrong" in name else TAGS[2]] for group in groups]
        cardinal_mixed = sum(_difficulty(row).get("relation_mix") in {"cardinal-only", "mixed"} for row in representative)
        deep = sum(max(_difficulty(row).get("x_depth") or 0, _difficulty(row).get("y_depth") or 0) >= 3 for row in representative)
        lines.append(f"| `{name}` | {len(groups)} | {cardinal_mixed} | {deep} |")
    lines += ["", "## Heuristic labels on regressed completions", ""]
    for name, groups in cohorts.items():
        wrong_tag = TAGS[1] if "1.5k-wrong" in name else TAGS[2]
        counts = Counter(label for group in groups for label in classify(group[wrong_tag]))
        axis_counts = Counter()
        affected_axis_traces = Counter()
        for group in groups:
            wrong = group[wrong_tag]
            errors = cardinal_axis_errors(_text(wrong), _response(wrong))
            for label, count in errors.items():
                axis_counts[label] += count
                if count:
                    affected_axis_traces[label] += 1
        rendered = ", ".join(f"{label}={count}" for label, count in counts.most_common()) or "none"
        lines.append(f"- `{name}`: {rendered}")
        if axis_counts:
            rendered_axis = ", ".join(
                f"{label}={axis_counts[label]} occurrences in "
                f"{affected_axis_traces[label]} traces"
                for label in sorted(axis_counts)
            )
            lines.append(f"  - axis-extraction evidence: {rendered_axis}")

        ranked = sorted(
            groups,
            key=lambda group: (
                max(_difficulty(group[wrong_tag]).get("x_depth") or 0, _difficulty(group[wrong_tag]).get("y_depth") or 0),
                _difficulty(group[wrong_tag]).get("num_relations") or 0,
            ),
            reverse=True,
        )[:limit]
        for group in ranked:
            wrong = group[wrong_tag]
            exported.append({
                "cohort": name,
                "labels": classify(wrong),
                "prompt": _text(wrong),
                "difficulty": _difficulty(wrong),
                "gold": sorted(SUMMARY.gold(wrong)),
                "predictions": {tag: sorted(SUMMARY.prediction(group[tag])) for tag in TAGS},
                "responses": {tag: _response(group[tag]) for tag in TAGS},
            })

    lines += [
        "",
        "Representative raw prompts and traces are written to `ERROR-TRACE-EXAMPLES.jsonl`.",
        "",
    ]
    return "\n".join(lines), exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    report, examples = build_audit(args.results_dir, args.limit)
    (args.results_dir / "ERROR-TRACE-AUDIT.md").write_text(report + "\n")
    with (args.results_dir / "ERROR-TRACE-EXAMPLES.jsonl").open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
