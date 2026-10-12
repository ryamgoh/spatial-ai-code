# SpatialEntail frozen benchmark evaluation

This evaluates the untuned Qwen3.5-4B baseline and the answer-only/Natural
seed-42 adapters on all 4,300 sealed benchmark bases. It selects one answer-only
row per base, so replayable certificates remain offline audit artifacts and no
base is evaluated more than once.

The benchmark has three formal reporting groups:

- CORE: 1,300 bases;
- COMP: 1,200 matched-composition bases; and
- SHIFT: 1,800 bases, with held-out composition (900) and depth 3–8 (900)
  retained as diagnostic submetrics.

Prepare the immutable bundle from the repository root. The recommended workflow
is to run this directly on the cluster after updating the checkout, so no
generated benchmark shards need to be transferred:

```sh
uv run python -m spatial.v2.study.evaluation \
  results/spatialentail-study/release-final \
  --out experiments/spatialentail-eval/prepared-final
```

The command verifies the sealed release, selects exactly one answer-only row per
base, creates 52 cell tasks, and writes three hash-bound evaluation configs. Then
submit from the same repository root:

```sh
bash experiments/spatialentail-eval/submit.sh
```

Each model runs independently on one H200. The summary job waits for all three
with `afterany`, records incomplete jobs explicitly, and writes
`prepared-final/SUMMARY.md` plus cell/depth details in `summary.json`.
