"""Build a 16-row stratified SpatialMap-TQA-CORR smoke benchmark.

Covers every SFT/GRPO question-type and answer-count case that exists in
the cleaned 1329 TQA-CORR set, so a 16-sample run exercises the full
pipeline:

    dir   1-ans x4, 2-ans x1          (direction, single + multi-letter)
    which 1-ans x3, 4-ans x2          (which-entity, single + multi-letter)
    count 1-ans x6                    (count, single-letter)

Cases intentionally not in this set (they do not exist in TQA-CORR, whose
empty-oracle rows were dropped during cleaning):
  - direction 4-answer  (undetermined on both axes)
  - which-entity 2-answer
  - count / direction rows with empty oracle
The direction 4-answer case is still exercised by the SFT training data
(300 samples) and the GRPO prompt mix; it simply cannot appear on this
benchmark.

Deterministic (seeded). Writes data/spatialeval_4a_smoke16.jsonl.
Run from the repo root:
    uv run --no-project python experiments/04a-smoke/scripts/make_smoke16.py
"""
import json
import random
from pathlib import Path

SRC = Path("data/spatialeval_cleaned.jsonl")
OUT = Path("data/spatialeval_4a_smoke16.jsonl")
SEED = 42
WANT = 16


def qtype(text: str) -> str:
    if "In which direction is" in text:
        return "dir"
    if "Which object is in the" in text:
        return "which"
    if "How many objects are in the" in text:
        return "count"
    return "?"


def n_answers(oracle: str) -> int:
    oracle = (oracle or "").strip()
    return len(oracle.split(",")) if oracle else 0


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    # Group by (question type, answer-letter count) over nonempty-oracle rows.
    buckets: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        o = (r.get("oracle_option") or "").strip()
        if not o:
            continue
        key = (qtype(r["text"]), n_answers(o))
        buckets.setdefault(key, []).append(r)

    # (type, #answer-letters, how many) — sums to WANT.
    plan = [
        ("dir", 1, 4),
        ("dir", 2, 1),
        ("which", 1, 3),
        ("which", 4, 2),
        ("count", 1, 6),
    ]
    assert sum(k for _, _, k in plan) == WANT

    rng = random.Random(SEED)
    picked: list[dict] = []
    for t, n, k in plan:
        pool = list(buckets.get((t, n), []))
        if len(pool) < k:
            raise SystemExit(f"bucket {t}/{n}-ans has {len(pool)}, need {k}")
        picked.extend(rng.sample(pool, k))

    seen = set()
    out = []
    for r in picked:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    assert len(out) == WANT, f"picked {len(out)}, want {WANT}"
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(f"wrote {len(out)} rows -> {OUT}")
    for r in out:
        print(f"  {r['id']}  {qtype(r['text']):<6} {r['oracle_option']}")


if __name__ == "__main__":
    main()
