# All-Controlled Prototype

Separate feasibility experiment, not a replacement training dataset.

- Fixed attempts: 260; statuses: {'accepted': 260}.
- All ten construction spans; two matched scene sizes per span.
- L denotes verified positive certificate depth only. Ambiguous and inconsistent E rows have no positive L label.
- Count sole values cycle through 0-4. This is not full depth-by-value coverage or the final 1,200/600/200 allocation.
- Existing Natural/Symbolic renderers are reused without prose changes.
- Every completed case is independently replayed, including counterexamples and positive/conflict certificates.
- Maximum target tokens: {'natural': 2780, 'symbolic': 3000}.
- Original artifact fingerprints unchanged: True.

## Interpretation

All 260 fixed slots passed; no retry or refill was needed. This establishes
that controlled construction can supply the required answer cases at matched
scene sizes while preserving verified positive depths. The size-only probe
has 50% balanced accuracy for each query type.

This does not establish production readiness. Structural-count probes still
reach 70.0% Direction, 64.2% Which and 71.2% Count balanced accuracy. With only
ten seed groups and a small recipe set, these are exploratory results rather
than reliable generalization estimates or proof of absence of other shortcuts.
The prototype covers 15-31 premises only; it does not yet establish balanced
coverage for the smaller 3-9-premise scenes seen in the existing random pool.
Supporting all feasible smaller scene sizes remains a production-design task.

## Scene Sizes

| Construction span | Size variant | Objects / premises | Cases matched |
|---|---|---|---|
| 1 | 0 | [(13, 15)] | True |
| 1 | 3 | [(16, 18)] | True |
| 2 | 0 | [(15, 20)] | True |
| 2 | 3 | [(18, 23)] | True |
| 3 | 0 | [(16, 19)] | True |
| 3 | 3 | [(19, 22)] | True |
| 4 | 0 | [(17, 21)] | True |
| 4 | 3 | [(20, 24)] | True |
| 5 | 0 | [(18, 23)] | True |
| 5 | 3 | [(21, 26)] | True |
| 6 | 0 | [(19, 22)] | True |
| 6 | 3 | [(22, 25)] | True |
| 7 | 0 | [(20, 24)] | True |
| 7 | 3 | [(23, 27)] | True |
| 8 | 0 | [(21, 26)] | True |
| 8 | 3 | [(24, 29)] | True |
| 9 | 0 | [(22, 26)] | True |
| 9 | 3 | [(25, 29)] | True |
| 10 | 0 | [(23, 28)] | True |
| 10 | 3 | [(26, 31)] | True |

## Shortcut Checks

Five grouped folds, keeping all variants of each seed/span together. Ten construction families only; no SFT model or evaluation examples. The structural tree uses min_samples_leaf=5 because this prototype is much smaller than the previous training-set diagnostic (25).

| Query | Probe | Accuracy | Balanced accuracy |
|---|---|---|---|
| direction | size_only | 50.0% | 50.0% |
| direction | structural_counts | 70.0% | 70.0% |
| which | size_only | 45.0% | 50.0% |
| which | structural_counts | 63.0% | 64.2% |
| count | size_only | 50.0% | 50.0% |
| count | structural_counts | 71.2% | 71.2% |

## Limits

- Exact object/premise-count matching does not establish absence of graph, wording, semantic-recipe, or explanation shortcuts.
- The current ambiguity recipe uses an unordered pair above an anchor; the inconsistency recipe uses a three-edge strict cycle. These are narrow recognizable mechanisms, not broad E coverage.
- Context is constructed on an excluded side of the reference. Every context object participates in Which/Count verification, but this is a restricted family of scenes.
- No canonical deduplication, final family-floor check, held-out overlap screening, or production allocation was performed. All prototype rows are development-only and must not enter training.
- Prototype case proportions differ from the final study; these diagnostic percentages are not directly comparable to the old 70%-non-E training distribution.
- No full replacement, source-quota migration, evaluation changes, sealing, training, or automatic extra sampling.

Manifest: `/Users/bytedance/Code/spatial-ai-code/experiments/spatialentail-composition-v1/ALL-CONTROLLED-PROTOTYPE-v1.manifest.json`

Full evidence and references: `/var/folders/g4/ccx77z2d60sgsjrt9ck_ppwr0000gn/T/spatial-all-controlled-prototype-es4fmrhv/records.jsonl`
