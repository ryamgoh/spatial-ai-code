# Revised Combined Feasibility

## Approved Change

The user approved extending Unique Selection decision-depth coverage to all
four splits. The numeric primary-depth minima remain 1,400 train, 280 dev,
560 in-range test, and 400 extrapolation.

Singleton answers retain substantive justification. Multiple-entailment E
answers qualify only when the cheapest two-member proof reaches the requested
depth. Long unrelated chains cannot make a shallow Unique decision deep.
Listed and Exhaustive substantive-depth requirements remain unchanged.

The old frozen configuration and scoped artifacts remain untouched. The
revision is applied to a nonfrozen diagnostic policy.

## Check

Test the same fixed reserve against all existing numeric base/source/Count
quotas, semantic coverage, direction presence, depth-family floors, and one
overall multi-Selection target of 900 out of 3,600 bases (25%). There are no
per-split or per-depth multi-answer quotas, and no approved G2:G3:G4 allocation.

Reuse the completed 7,632-slot source audit after checking its exact inventory,
SHA-256, token flags, and independent replay markers. Admission semantics for
those noncontrolled sources are unchanged. Reconstruct and preflight the fixed
schedule, map one recipe to each controlled slot, explicitly reject overlaps,
fully admit/replay controlled queries, and deduplicate before retention.

Any supply shortage is reported with its exact requirement and measured bases
and families. A retention solver timeout is reported as inconclusive, not
infeasible. Any feasible subset must pass independent retention validation.

## Progress

The revised run was interrupted. Its attempted prefix continuation was also
stopped after the user clarified that they wanted a fresh revised run with no
saved-result reuse. This attempt has no completed retention receipt.
The fresh run is documented in `COMBINED-ALL-SCENE-FEASIBILITY-v3.md`.
The revised policy and diagnostic tests passed 36 targeted cases.

The complete regression run passed 822 tests, with another four integration
cases confirming deep Unique E coverage in each split. A focused 166-case
admission run preserved rejection of shallow Unique failures, and 77 shared
execution/retention compatibility tests passed.

Mapped-scene preflight rejected 1,261 overlapping story slots. The remaining
controlled inventory has 21,758 query jobs. The user stopped the run and then
authorized continuing. The stopped run saved 16,404 complete, accepted records
in its temporary controlled metadata file.

The stopped continuation was configured to validate that saved prefix against the regenerated fixed
job inventory, canonical query keys, current recipe/protocol, token admissions,
and independent replay markers before reusing it. Its pinned SHA-256 is
`e5841e84004eb8db336d2ce955f65d4fdff27fafac29bf811c04d95d61129571`.
This is explicitly trusted metadata from the interrupted current run, not a
general historical-checkpoint migration or an automatic software-identity
certificate. Old scoped checkpoint directories are not read.

The guarded resume and diagnostic regression suite passed 18 tests; Ruff and
`git diff --check` passed. There are 5,354 controlled jobs remaining after
prefix validation. Those saved results are not reused by the fresh v3 run.
Admission and combined retention for this attempt remain incomplete.

No new seeds, silent numeric quota relaxations, replacement dataset export,
sealing, or training.
