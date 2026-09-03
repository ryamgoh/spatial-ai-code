"""Summarize Exp 7 SFT starting-state runs.

Reads:
  results/starting-state/<tag>/{results.json,responses_*.jsonl}
  models/qwen3.5-4b-{base,instruct}-sft/trainer_state.json

Writes results/starting-state/SUMMARY.md.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "starting-state"
TASK = "spatial_eval_gen_cleaned_single"
ANSWER_RE = re.compile(r"Answer:\s*([A-D](?:\s*,\s*[A-D])*)", re.I)
LOOP_RE = re.compile(r"(.{20,80}?)\1{4,}", re.DOTALL)
FLAG_NAMES = [
    "format_fail",
    "missing_entry",
    "token_loop",
    "over_select",
    "all_four",
]
TAG_TO_MODEL = {
    "base-sft": ROOT / "models" / "qwen3.5-4b-base-sft",
    "instruct-sft": ROOT / "models" / "qwen3.5-4b-instruct-sft",
}


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


def acc_from_results(data: dict, metric: str) -> float | None:
    res = data.get("results") or {}
    for name, row in res.items():
        if name == TASK or str(name).endswith(TASK):
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


def load_samples(run_dir: Path) -> list[dict]:
    p = run_dir / f"responses_{TASK}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def trainer_stats(model_dir: Path) -> dict:
    state_path = model_dir / "trainer_state.json"
    if not state_path.exists():
        ckpts = sorted(
            model_dir.glob("checkpoint-*/trainer_state.json"),
            key=lambda p: int(p.parent.name.split("-")[-1]),
        )
        state_path = ckpts[-1] if ckpts else None
    if state_path is None or not state_path.exists():
        return {}
    data = json.loads(state_path.read_text())
    hist = data.get("log_history") or []
    losses = [(h.get("step"), h["loss"]) for h in hist if "loss" in h]
    evals = [(h.get("step"), h["eval_loss"]) for h in hist if "eval_loss" in h]
    best_step = data.get("best_global_step")
    best_metric = data.get("best_metric")
    return {
        "global_step": data.get("global_step"),
        "last_loss": losses[-1][1] if losses else None,
        "last_eval_loss": evals[-1][1] if evals else None,
        "first_eval_loss": evals[0][1] if evals else None,
        "n_log": len(losses),
        "n_eval": len(evals),
        "best_step": best_step,
        "best_eval_loss": best_metric,
    }


def fmt_loss(v: object) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def main() -> None:
    if not OUT.is_dir():
        raise SystemExit(f"no results yet at {OUT}")
    lines = [
        "# Exp 7 — Architectural Starting State for SFT",
        "",
        "SFT on 5,000 single-answer traces; eval on SpatialMap-TQA-Corr-Single (1,038).",
        "",
        "## Accuracy (TQA-Corr-Single)",
        "",
        "| config | n | strict | loose |",
        "|---|---|---|---|",
    ]
    eval_rows = []
    for d in sorted(OUT.iterdir()):
        if not (d.is_dir() and (d / "results.json").exists()):
            continue
        results = json.loads((d / "results.json").read_text())
        samples = load_samples(d)
        n = len(samples)
        strict = acc_from_results(results, "strict_acc") or mean_acc(
            samples, "strict_acc"
        )
        loose = acc_from_results(results, "loose_acc") or mean_acc(
            samples, "loose_acc"
        )
        flags: Counter[str] = Counter()
        any_anom = 0
        for s in samples:
            f = anomalies(s)
            flags.update(f)
            if f:
                any_anom += 1
        eval_rows.append((d.name, n, strict, loose, flags, any_anom))
        lines.append(
            f"| {d.name} | {n} | {fmt_acc(strict)} | {fmt_acc(loose)} |"
        )

    if len(eval_rows) >= 2:
        by_tag = {t: s for t, _, s, _, _, _ in eval_rows}
        b, i = by_tag.get("base-sft"), by_tag.get("instruct-sft")
        if b is not None and i is not None:
            lines.append("")
            lines.append(f"base-sft − instruct-sft (strict): {b - i:+.1f} pp")

    lines += [
        "",
        "## Formatting adherence",
        "",
        "| config | n | any | " + " | ".join(FLAG_NAMES) + " |",
        "|" + "---|" * (3 + len(FLAG_NAMES)),
    ]
    for tag, n, _, _, flags, any_anom in eval_rows:
        cells = [tag, str(n), pct(any_anom, n)]
        cells.extend(pct(flags[f], n) for f in FLAG_NAMES)
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Loss convergence",
        "",
        "| config | steps | last train loss | first eval_loss | last eval_loss | best eval_loss (step) |",
        "|---|---|---|---|---|---|",
    ]
    for tag, model_dir in TAG_TO_MODEL.items():
        st = trainer_stats(model_dir)
        if not st:
            lines.append(f"| {tag} | — | — | — | — | — |")
            continue
        best = st.get("best_eval_loss")
        best_step = st.get("best_step")
        best_s = (
            f"{fmt_loss(best)} ({best_step})"
            if best is not None
            else "—"
        )
        lines.append(
            f"| {tag} | {st.get('global_step') or '—'} | "
            f"{fmt_loss(st.get('last_loss'))} | "
            f"{fmt_loss(st.get('first_eval_loss'))} | "
            f"{fmt_loss(st.get('last_eval_loss'))} | {best_s} |"
        )

    out = "\n".join(lines) + "\n"
    (OUT / "SUMMARY.md").write_text(out)
    print(out)
    print(f"wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
