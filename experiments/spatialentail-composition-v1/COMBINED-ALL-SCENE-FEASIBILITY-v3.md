# Fresh Revised All-Scene Feasibility

## Authorization

The user clarified that "continue" meant a fresh revised run, not completion
from saved results, and requested that it follow all the latest updates.
The v2 continuation was stopped. This run uses `--fresh`, which rejects every
source/controlled saved-result option. Historical checkpoint directories and
both completed and partial prior metadata files are not read.

## Current Rules

The authoritative specification is `ENTAILMENT-ONLY-v1.md`. The run uses:

- `plain-all-scene-entailment-v1` and all nonreference scene objects for
  Selection/Count, independent of the menu.
- `all-scene-forest-v3` fixed-slot repairs, including the depth-one context
  diversity repair and preserved negative forests.
- Ordinary place names, one-paragraph spatial premises, no visible candidate
  legends or technical headers, and explanation-only self-contained P citations.
- All three entailment-only contracts and matched Natural/Symbolic references.
- Unique Selection decision-depth support in all four splits, with unchanged
  minima of 1,400/280/560/400. A multiple-entailment E needs its cheapest pair
  at the requested depth; Listed/Exhaustive substantive depth is unchanged.
- All existing numeric base/source/Count quotas, depth-family floors,
  direction presence, coverage safeguards, and seeded omission companions.
- One diagnostic overall target of 900 multi-entailment Selection bases out
  of 3,600, including E/F cases in the denominator. No per-split/depth
  multi-answer quota or final G2:G3:G4 allocation is approved.

The historical frozen config supplies only the fixed source inventory and
unchanged numeric requirements. The runner applies the revised diagnostic
policy before any new admission. It does not reuse old labels or acceptance.

## Execution

Reconstruct the fixed schedule, freshly check every noncontrolled job under
all-scene semantics, map the repaired controlled scenes, quarantine overlaps,
freshly admit and independently replay controlled jobs, deduplicate canonical
questions, then test combined retention if supply permits.

All seeds and input slots remain unchanged. Workload counts are calculated
by preflight, not newly frozen dataset sizes. Unknown/unsupported/error results
are not answer labels. A retention timeout is inconclusive, not infeasible.

The completed receipt is `COMBINED-ALL-SCENE-FEASIBILITY-v3.json`.
The run is not a replacement dataset export, a seal, or training authorization.

## Startup Checks

454 updated generation, protocol, naming, policy, multi-Selection, and retention
tests passed. The final 22-case combined diagnostic suite includes four tests
that reject each saved-result option under `--fresh`. Ruff and
`git diff --check` passed.

The fresh run was launched with `--fresh`, four workers, the pinned local
tokenizer, and no source/controlled metadata arguments. The old continuation
is stopped.

## Completed Result

Runtime: 2,948.1 seconds, approximately 49 minutes 8 seconds.
Status: `combined-diagnostic-retention-infeasible`.
Fresh execution is confirmed: no source reuse and zero controlled reused slots.

- All 7,632 noncontrolled jobs checked: 3,928 accepted, 3,540 rejected because
  declared objects were absent from premises, and 164 unsupported. Every raw
  noncontrolled source quota has sufficient supply.
- All 21,758 controlled jobs accepted and independently replayed.
- 1,261 overlapping story slots quarantined during mapping.
- 25,344 distinct verified candidate questions remain after deduplication and
  final scene isolation.
- Exactly one supply shortage: `dev/count/2/4` requires 7 distinct bases,
  but only 6 are available. This means depth-2 dev Count questions whose
  entailed total is 4, not questions selecting four answer letters.
- No other individual quota or coverage/family supply shortage was reported.
- Maximum controlled target tokens: Natural 3,037; Symbolic 3,419.

The retention MILP was not run because the one-base shortage already prevents
the exact requirements from being met. Passing the other individual supply
checks does not prove their joint compatibility or establish that the exact
900 multi-Selection target can be retained. There is no selected 10,800-base
dataset, no seal, and no training.

No additional seeds, altered quotas, or corrective generation were introduced
after this result. The shortage requires a separate next-step decision.
