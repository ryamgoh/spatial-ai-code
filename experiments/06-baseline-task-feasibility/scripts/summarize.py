"""Summarize Exp 6 baseline-feasibility evals.

Reads experiments/06-baseline-task-feasibility/results/feasibility/<tag>/
for tags base / instruct (and optional *-single). Writes SUMMARY.md.

Accuracy is sliced from the 1329-row TQA-Corr run:
  all     — SpatialMap-TQA-Corr (nonempty oracle)
  single  — gold is exactly one A–D letter (TQA-Corr-Single)
  multi   — gold has 2+ letters

Generation anomalies are scored on the logged thinking+answer string:
  format_fail   — no extractable Answer: A-D line
  missing_entry — empty predicted letter set
  token_loop    — repeated 20–80 char chunk
  over_select   — more predicted letters than gold
  all_four      — predicted A,B,C,D
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "feasibility"
TASK_CORR = "spatial_eval_gen_cleaned_1329"
TASK_SINGLE = "spatial_eval_gen_cleaned_single"
ANSWER_RE = re.compile(r"Answer:\s*([A-D](?:\s*,\s*[A-D])*)", re.I)
LOOP_RE = re.compile(r"(.{20,80}?)\1{4,}", re.DOTALL)


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
    return [p for p in re.split(r"[,;| ]+", s) if p and p in "ABCD"]


def pred_from_sample(sample: dict) -> str:
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        inner = fr[0]
        if isinstance(inner, list):
            return str(inner[0] if inner else "")
        return str(inner or "")
    return ""


def resp_text(sample: dict) -> str:
    resps = sample.get("resps")
    if isinstance(resps, list) and resps:
        inner = resps[0]
        if isinstance(inner, list):
            return str(inner[0] if inner else "")
        return str(inner or "")
    return ""


def gold_of(sample: dict) -> str:
    doc = sample.get("doc") or {}
    return str(doc.get("oracle_option") or sample.get("target") or "")


def n_gold(sample: dict) -> int:
    return len(letters(gold_of(sample)))


def slice_name(sample: dict) -> str:
    n = n_gold(sample)
    if n == 1:
        return "single"
    if n >= 2:
        return "multi"
    return "empty"


def anomalies(sample: dict) -> set[str]:
    text = resp_text(sample)
    gold = letters(gold_of(sample))
    pred = letters(pred_from_sample(sample) or "")
    if not pred:
        m = list(ANSWER_RE.finditer(text))
        pred = letters(m[-1].group(1) if m else "")
    flags: set[str] = set()
    if not ANSWER_RE.search(text):
        flags.add("format_fail")
    if not pred:
        flags.add("missing_entry")
    thinking = ANSWER_RE.split(text)[0] if text else ""
    if thinking and LOOP_RE.search(thinking):
        flags.add("token_loop")
    if gold and len(pred) > len(gold):
        flags.add("over_select")
    if set(pred) == set("ABCD") and len(pred) == 4:
        flags.add("all_four")
    return flags


def load_samples(run_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for name in (TASK_CORR, TASK_SINGLE):
        p = run_dir / f"responses_{name}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                samples.append(json.loads(line))
    return samples


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


def mean_acc(samples: list[dict], metric: str) -> float | None:
    if not samples:
        return None
    vals = [float(s.get(metric, 0) or 0) for s in samples]
    return round(100.0 * sum(vals) / len(vals), 1)


def pct(n: int, d: int) -> str:
    if d == 0:
        return "—"
    return f"{100.0 * n / d:.1f}%"


def fmt_acc(v: float | None) -> str:
    return f"{v}%" if v is not None else "—"


def summarize_tag(tag: str, run_dir: Path) -> dict:
    rj = run_dir / "results.json"
    results = json.loads(rj.read_text()) if rj.exists() else {}
    samples = load_samples(run_dir)
    # Prefer the 1329-row file for slicing; if this tag is a dedicated
    # Single run, every sample is already single-letter.
    by_slice: dict[str, list[dict]] = {"all": [], "single": [], "multi": []}
    by_type: Counter[tuple[str, str]] = Counter()
    acc_type: dict[tuple[str, str], list[float]] = {}
    flag_counts: dict[str, Counter[str]] = {
        "all": Counter(),
        "single": Counter(),
        "multi": Counter(),
    }
    any_anom: Counter[str] = Counter()
    for s in samples:
        sl = slice_name(s)
        if sl == "empty":
            continue
        by_slice["all"].append(s)
        by_slice[sl].append(s)
        t = qtype((s.get("doc") or {}).get("text") or "")
        by_type[(t, sl)] += 1
        acc_type.setdefault((t, sl), []).append(float(s.get("strict_acc", 0) or 0))
        flags = anomalies(s)
        for f in flags:
            flag_counts["all"][f] += 1
            flag_counts[sl][f] += 1
        if flags:
            any_anom["all"] += 1
            any_anom[sl] += 1
    return {
        "tag": tag,
        "n": {k: len(v) for k, v in by_slice.items()},
        "strict": {
            "all": acc_from_results(results, TASK_CORR, "strict_acc")
            or mean_acc(by_slice["all"], "strict_acc"),
            "single": mean_acc(by_slice["single"], "strict_acc"),
            "multi": mean_acc(by_slice["multi"], "strict_acc"),
        },
        "loose": {
            "all": acc_from_results(results, TASK_CORR, "loose_acc")
            or mean_acc(by_slice["all"], "loose_acc"),
            "single": mean_acc(by_slice["single"], "loose_acc"),
            "multi": mean_acc(by_slice["multi"], "loose_acc"),
        },
        "single_task_strict": acc_from_results(results, TASK_SINGLE, "strict_acc"),
        "by_type": by_type,
        "acc_type": acc_type,
        "flags": flag_counts,
        "any_anom": any_anom,
    }


def main() -> None:
    if not OUT.is_dir():
        raise SystemExit(f"no results yet at {OUT}")
    rows = []
    for d in sorted(OUT.iterdir()):
        if d.is_dir() and (d / "results.json").exists():
            rows.append(summarize_tag(d.name, d))
    if not rows:
        raise SystemExit(f"no completed evals under {OUT}")

    flag_names = [
        "format_fail",
        "missing_entry",
        "token_loop",
        "over_select",
        "all_four",
    ]
    lines = [
        "# Exp 6 — Baseline Task Feasibility",
        "",
        "Zero-shot Qwen3.5-4B-Base vs Qwen3.5-4B on SpatialMap-TQA-Corr,",
        "sliced into Single (one gold letter) vs Multi (2+ gold letters).",
        "",
        "## Accuracy",
        "",
        "| config | n (all/single/multi) | strict all | strict single | strict multi | loose all |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["n"]
        lines.append(
            f"| {r['tag']} | {n['all']}/{n['single']}/{n['multi']} | "
            f"{fmt_acc(r['strict']['all'])} | {fmt_acc(r['strict']['single'])} | "
            f"{fmt_acc(r['strict']['multi'])} | {fmt_acc(r['loose']['all'])} |"
        )

    lines += ["", "## Generation anomalies (rate of flagged samples)", ""]
    header = "| config | slice | n | any | " + " | ".join(flag_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (4 + len(flag_names)))
    for r in rows:
        for sl in ("all", "single", "multi"):
            n = r["n"][sl]
            flags = r["flags"][sl]
            cells = [
                r["tag"],
                sl,
                str(n),
                pct(r["any_anom"][sl], n),
            ]
            cells.extend(pct(flags[f], n) for f in flag_names)
            lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Strict acc by question type × answer cardinality", ""]
    lines.append("| config | type | single n / acc | multi n / acc |")
    lines.append("|---|---|---|---|")
    for r in rows:
        types = sorted({t for t, _ in r["by_type"]})
        for t in types:
            def cell(sl: str) -> str:
                n = r["by_type"][(t, sl)]
                vals = r["acc_type"].get((t, sl)) or []
                if not vals:
                    return "—"
                acc = round(100.0 * sum(vals) / len(vals), 1)
                return f"{n} / {acc}%"

            lines.append(f"| {r['tag']} | {t} | {cell('single')} | {cell('multi')} |")

    out = "\n".join(lines) + "\n"
    (OUT / "SUMMARY.md").write_text(out)
    print(out)
    print(f"wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
