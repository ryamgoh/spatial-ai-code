"""Summarize Exp 11: v6 SFT on synthetic 20% test vs SpatialMap v6.

Reads results/<tag>/{results.json,responses_*.jsonl}.
Writes results/SUMMARY.md.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
TASKS = ("spatial_eval_v6_synth", "spatial_eval_v6_spatialmap")
ANSWER_RE = re.compile(r"Answer:\s*([A-E](?:\s*,\s*[A-E])*)", re.I)
TAGS = [
    "2b-1.5k", "2b-6k", "2b-20k",
    "4b-1.5k", "4b-6k", "4b-20k",
]


def qtype(text: str) -> str:
    if "In which direction is" in text:
        return "dir"
    if "Which object is in the" in text:
        return "which"
    if "How many objects are in the" in text:
        return "count"
    return "?"


def letters(raw: object) -> list[str]:
    s = str(raw or "").strip().upper()
    if not s:
        return []
    return [p for p in re.split(r"[,;| ]+", s) if p and p in "ABCDE"]


def gold_of(sample: dict) -> str:
    doc = sample.get("doc") or {}
    return str(doc.get("oracle_option") or sample.get("target") or "")


def acc_from_results(data: dict, task: str, metric: str) -> float | None:
    res = data.get("results") or {}
    for name, row in res.items():
        if name == task or str(name).endswith(task):
            val = row.get(f"{metric},extract-answer")
            if val is None:
                val = row.get(metric)
            if val is None:
                return None
            return round(float(val) * 100, 1)
    return None


def load_samples(run_dir: Path, task: str) -> list[dict]:
    p = run_dir / f"responses_{task}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def fmt(v: float | None) -> str:
    return f"{v}%" if v is not None else "—"


def main() -> None:
    lines = [
        "# Exp 11 summary",
        "",
        "SFT Qwen3.5 2B/4B × {1.5k, 6k, 20k} v6 mix.",
        "Eval: synthetic 20% test vs SpatialMap with v6 gold.",
        "",
        "| tag | synth strict | synth loose | smap strict | smap loose |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        d = OUT / tag
        if not (d / "results.json").is_file():
            lines.append(f"| `{tag}` | — | — | — | — |")
            continue
        data = json.loads((d / "results.json").read_text())
        lines.append(
            "| `{tag}` | {ss} | {sl} | {ms} | {ml} |".format(
                tag=tag,
                ss=fmt(acc_from_results(data, TASKS[0], "strict_acc")),
                sl=fmt(acc_from_results(data, TASKS[0], "loose_acc")),
                ms=fmt(acc_from_results(data, TASKS[1], "strict_acc")),
                ml=fmt(acc_from_results(data, TASKS[1], "loose_acc")),
            )
        )
    lines.append("")
    (OUT / "SUMMARY.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((OUT / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
