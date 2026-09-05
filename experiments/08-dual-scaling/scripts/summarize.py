"""Summarize Exp 8 dual-scaling runs.

Reads:
  results/scaling/<tag>/{results.json,responses_*.jsonl}
  models/qwen3.5-*-sft-*/trainer_state.json

Writes results/scaling/SUMMARY.md (2.1 data curve, 2.2 param curve,
format flags, loss).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scaling"
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
# tag -> (n_train, n_params_b, model_dir)
CELLS = {
    "4b-1.5k": (1500, 4.0, ROOT / "models" / "qwen3.5-4b-sft-1500"),
    "4b-5k": (5000, 4.0, ROOT / "models" / "qwen3.5-4b-sft-5000"),
    "4b-20k": (20000, 4.0, ROOT / "models" / "qwen3.5-4b-sft-20000"),
    "2b-20k": (20000, 2.0, ROOT / "models" / "qwen3.5-2b-sft-20000"),
    "2b-20k-96b": (20000, 2.0, ROOT / "models" / "qwen3.5-2b-sft-20000-96b"),
    "0.8b-20k": (20000, 0.8, ROOT / "models" / "qwen3.5-0.8b-sft-20000"),
}
DATA_TAGS = ("4b-1.5k", "4b-5k", "4b-20k")
PARAM_TAGS = ("0.8b-20k", "2b-20k", "4b-20k")


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


def fmt_err(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{100.0 - v:.1f}%"


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
    return {
        "global_step": data.get("global_step"),
        "last_loss": losses[-1][1] if losses else None,
        "last_eval_loss": evals[-1][1] if evals else None,
        "first_eval_loss": evals[0][1] if evals else None,
        "best_step": data.get("best_global_step"),
        "best_eval_loss": data.get("best_metric"),
    }


def fmt_loss(v: object) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def loglinear_note(rows: dict) -> list[str]:
    """Error vs ln(n) for the 4B data-scaling cells that have strict acc."""
    pts = []
    for tag in DATA_TAGS:
        n, _, _ = CELLS[tag]
        strict = rows.get(tag, {}).get("strict")
        if strict is None:
            continue
        err = (100.0 - strict) / 100.0
        pts.append((n, err, tag, strict))
    if len(pts) < 2:
        return ["Need at least two 4B cells to check log-linear error vs n."]
    lines = [
        "Error (1 − strict) vs ln(n) on 4B. Hypothesis: roughly constant "
        "slope; 20k lowest error.",
        "",
        "| tag | n | ln n | strict | error | Δerror / Δln n vs prev |",
        "|---|---|---|---|---|---|",
    ]
    prev = None
    for n, err, tag, strict in pts:
        ln_n = math.log(n)
        slope = "—"
        if prev is not None:
            dn = ln_n - prev[0]
            slope = f"{(err - prev[1]) / dn:.4f}" if dn else "—"
        lines.append(
            f"| {tag} | {n} | {ln_n:.3f} | {strict}% | {err * 100:.1f}% | {slope} |"
        )
        prev = (ln_n, err)
    return lines


def summarize_eval(tag: str, run_dir: Path) -> dict:
    results = json.loads((run_dir / "results.json").read_text())
    samples = load_samples(run_dir)
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
    return {
        "tag": tag,
        "n_eval": n,
        "strict": strict,
        "loose": loose,
        "flags": flags,
        "any_anom": any_anom,
    }


def main() -> None:
    if not OUT.is_dir():
        raise SystemExit(f"no results yet at {OUT}")

    eval_rows: dict[str, dict] = {}
    for d in sorted(OUT.iterdir()):
        if d.is_dir() and (d / "results.json").exists():
            eval_rows[d.name] = summarize_eval(d.name, d)

    lines = [
        "# Exp 8 — Dual Scaling Laws",
        "",
        "SFT on nested single-answer traces; eval on SpatialMap-TQA-Corr-Single (1,038).",
        "Instruct Qwen3.5 only. 4B-20k is shared across 2.1 and 2.2.",
        "",
        "## 2.1 Data scaling (Qwen3.5-4B Instruct)",
        "",
        "| config | n train | ln n | n eval | strict | loose | error (1−strict) |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag in DATA_TAGS:
        n_train, _, _ = CELLS[tag]
        r = eval_rows.get(tag)
        if r is None:
            lines.append(
                f"| {tag} | {n_train} | {math.log(n_train):.3f} | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {tag} | {n_train} | {math.log(n_train):.3f} | {r['n_eval']} | "
            f"{fmt_acc(r['strict'])} | {fmt_acc(r['loose'])} | {fmt_err(r['strict'])} |"
        )

    present_21 = [t for t in DATA_TAGS if t in eval_rows and eval_rows[t]["strict"] is not None]
    if len(present_21) >= 2:
        a, b = present_21[0], present_21[-1]
        da = eval_rows[b]["strict"] - eval_rows[a]["strict"]
        lines += [
            "",
            f"{b} − {a} (strict): {da:+.1f} pp",
        ]
        best = max(present_21, key=lambda t: eval_rows[t]["strict"])
        lines.append(
            f"lowest error among completed 2.1 cells: {best} "
            f"(strict {eval_rows[best]['strict']}%)"
        )

    lines += ["", *loglinear_note(eval_rows)]

    lines += [
        "",
        "## 2.2 Parameter scaling (20,000 samples)",
        "",
        "| config | params | n eval | strict | loose | error (1−strict) |",
        "|---|---|---|---|---|---|",
    ]
    param_tags = []
    for tag in PARAM_TAGS:
        if tag == "2b-20k" and "2b-20k-96b" in eval_rows:
            param_tags.append("2b-20k-96b")
        else:
            param_tags.append(tag)

    for tag in param_tags:
        _, params, _ = CELLS[tag]
        r = eval_rows.get(tag)
        if r is None:
            lines.append(f"| {tag} | {params:g}B | — | — | — | — |")
            continue
        lines.append(
            f"| {tag} | {params:g}B | {r['n_eval']} | "
            f"{fmt_acc(r['strict'])} | {fmt_acc(r['loose'])} | {fmt_err(r['strict'])} |"
        )

    present_22 = [t for t in param_tags if t in eval_rows and eval_rows[t]["strict"] is not None]
    if len(present_22) >= 2:
        small, big = present_22[0], present_22[-1]
        d = eval_rows[big]["strict"] - eval_rows[small]["strict"]
        lines += [
            "",
            f"{big} − {small} (strict): {d:+.1f} pp",
        ]

    lines += [
        "",
        "## Formatting adherence",
        "",
        "| config | n | any | " + " | ".join(FLAG_NAMES) + " |",
        "|" + "---|" * (3 + len(FLAG_NAMES)),
    ]
    flag_tags = list(DATA_TAGS) + [
        t for t in param_tags if t not in DATA_TAGS
    ]
    if "2b-20k-96b" in eval_rows and "2b-20k-96b" not in flag_tags:
        flag_tags.append("2b-20k-96b")
    for tag in flag_tags:
        r = eval_rows.get(tag)
        if r is None:
            continue
        n = r["n_eval"]
        cells = [tag, str(n), pct(r["any_anom"], n)]
        cells.extend(pct(r["flags"][f], n) for f in FLAG_NAMES)
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Loss convergence",
        "",
        "| config | steps | last train loss | first eval_loss | last eval_loss | best eval_loss (step) |",
        "|---|---|---|---|---|---|",
    ]
    for tag, (_, _, model_dir) in CELLS.items():
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text(out)
    print(out)
    print(f"wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
