# SpatialEntail frozen benchmark evaluation

The current provisional measurements and open analysis items are tracked in
[`RESULTS.md`](RESULTS.md).

This evaluates the untuned Qwen3.5-4B baseline and the answer-only/Natural
seed-42 adapters on all 4,300 sealed benchmark bases. It selects one answer-only
row per base, so replayable certificates remain offline audit artifacts and no
base is evaluated more than once.

Evaluation uses two passes from the same evaluated checkpoint: free generation,
then constrained answer selection. The current `SINGLE` benchmark permits
exactly one of A–E. The evaluator also supports a nonempty, canonical subset of
the configured choices for future set-valued answer contracts. Output-format
compliance is diagnostic and does not determine semantic accuracy. Each run
records stage-1 native accuracy and format validity alongside the primary
stage-2 constrained accuracy.

The benchmark has three formal reporting groups:

- CORE: 1,300 bases;
- COMP: 1,200 matched-composition bases; and
- SHIFT: 1,800 bases, with slice labels retained as cell metadata.

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
`prepared-final/SUMMARY.md` plus `summary.json`. Both retain the cumulative
CORE/COMP/SHIFT scores and the complete cell breakdown for every section;
query-family and depth diagnostics are retained alongside them.
