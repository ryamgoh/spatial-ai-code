"""Summarize Exp 12: reweighted Type-0 SFT vs the v6 synth test + SpatialMap.

Headline is dir-2 / dir-incomplete / dir-cycle macro on the synth test, plus
SpatialMap v6. Writes results/SUMMARY.md.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinds import HARD, kind, user_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
OUT = ROOT / "results"
SYNTH_TEST = REPO / "data" / "spatial_sft_v12_2000_test.jsonl"
SMAP = "spatial_eval_v6_spatialmap"
SYNTH = "spatial_eval_v12_synth"
TAGS = ["4b-1.5k", "4b-6k", "4b-18k"]
BUCKETS = [
    "dir-1",
    "dir-2",
    "dir-undetermined",
    "dir-cycle",
    "dir-incomplete",
    "dir-omit",
    "which-1",
    "which-2",
    "which-3",
    "which-4",
    "which-0",
    "count-1",
    "count-omit",
]


def letters(raw: object) -> list[str]:
    s = str(raw or "").strip().upper()
    if not s:
        return []
    return [p for p in re.split(r"[,;| ]+", s) if p and p in "ABCDE"]


def gold_of(sample: dict) -> str:
    doc = sample.get("doc") or {}
    return str(doc.get("oracle_option") or sample.get("target") or "")


def pred_of(sample: dict) -> str:
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        inner = fr[0]
        if isinstance(inner, list):
            return str(inner[0] if inner else "")
        return str(inner or "")
    return str(sample.get("resps") or "")


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


def synth_kinds() -> dict[str, str]:
    if not SYNTH_TEST.is_file():
        return {}
    out = {}
    for line in SYNTH_TEST.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[user_text(row)] = kind(row)
    return out


def bucket_acc(samples: list[dict], index: dict[str, str]) -> dict[str, tuple[int, int]]:
    tallies: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for sample in samples:
        doc = sample.get("doc") or {}
        text = str(doc.get("text") or "")
        tag = index.get(text, "?")
        g = set(letters(gold_of(sample)))
        p = set(letters(pred_of(sample)))
        tallies[tag][1] += 1
        if g and p == g:
            tallies[tag][0] += 1
    return {k: (v[0], v[1]) for k, v in tallies.items()}


def pct(ok: int, n: int) -> float | None:
    if n <= 0:
        return None
    return round(100.0 * ok / n, 1)


def hard_macro(buckets: dict[str, tuple[int, int]]) -> float | None:
    vals = []
    for tag in HARD:
        ok, n = buckets.get(tag, (0, 0))
        if n:
            vals.append(ok / n)
    if not vals:
        return None
    return round(100.0 * (sum(vals) / len(vals)), 1)


def main() -> None:
    index = synth_kinds()
    lines = [
        "# Exp 12 summary",
        "",
        "4B SFT on hierarchical-uniform mix + three letter rules. Eval is the matched 2k holdout + SpatialMap v6.",
        "Headline: macro of dir-2 / dir-incomplete / dir-cycle (strict letter-set).",
        "",
        "| tag | synth strict | hard macro | smap strict | smap loose |",
        "|---|---:|---:|---:|---:|",
    ]
    per_tag_buckets: dict[str, dict[str, tuple[int, int]]] = {}
    for tag in TAGS:
        d = OUT / tag
        if not (d / "results.json").is_file():
            lines.append(f"| `{tag}` | — | — | — | — |")
            continue
        data = json.loads((d / "results.json").read_text())
        samples = load_samples(d, SYNTH)
        buckets = bucket_acc(samples, index) if samples else {}
        per_tag_buckets[tag] = buckets
        lines.append(
            "| `{tag}` | {ss} | {hm} | {ms} | {ml} |".format(
                tag=tag,
                ss=fmt(acc_from_results(data, SYNTH, "strict_acc")),
                hm=fmt(hard_macro(buckets)),
                ms=fmt(acc_from_results(data, SMAP, "strict_acc")),
                ml=fmt(acc_from_results(data, SMAP, "loose_acc")),
            )
        )
    lines += ["", "## 13 buckets (synth test, strict)", ""]
    header = "| bucket |" + "".join(f" {t} |" for t in TAGS)
    split = "|---|" + "---:|" * len(TAGS)
    lines += [header, split]
    for bucket in BUCKETS + ["?"]:
        cells = []
        any_n = False
        for tag in TAGS:
            ok, n = per_tag_buckets.get(tag, {}).get(bucket, (0, 0))
            if n:
                any_n = True
                cells.append(f"{pct(ok, n)}% ({ok}/{n})")
            else:
                cells.append("—")
        if bucket == "?" and not any_n:
            continue
        lines.append("| `{bucket}` | {c} |".format(bucket=bucket, c=" | ".join(cells)))
    lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((OUT / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
