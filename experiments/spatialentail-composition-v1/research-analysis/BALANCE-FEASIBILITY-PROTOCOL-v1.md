# Balance Feasibility Protocol

## Status

Bounded generator-development experiment only. No production quotas, dataset,
benchmark, seal or training launch. The earlier suggested 60/20/20 allocation
is an untested proposal, not a requirement for this run.

## Fixed Schedule

Seed: 20261030. Solver timeout: 5000 ms per solver call.
Exactly 80 premise sets, each queried as Direction, Selection and Count:
240 base attempts, with no retries, seed search, replacement or adaptive top-up.

| Source | Stories | Details |
|---|---:|---|
| Controlled depth | 30 | Three existing generator variants at each depth 1-10 |
| Random world | 24 | Existing random coordinates-to-atomic-graph generator |
| Random atoms | 16 | Independently sampled exact atomic relations |
| Targeted controls | 10 | Existing nine recipes, including both cycle variants |

Reuse existing generators to measure their current ability rather than claim
new structural richness. All records have role `balance-feasibility-only`.
These are development inputs, not train/dev/test assignments or held-out data.
Depth 9/10 here cannot become final extrapolation test families.

## Menu Candidate Inventory

For a checked answer domain of size D with P possible values, a four-option
menu can contain k possible values exactly when
`max(0, 4-(D-P)) <= k <= min(4,P)`.
Select one seeded menu per feasible k, using actual domain values, with all
three contracts and both styles attached. At most five menus per base.
This is a bounded candidate inventory, not the final exposure distribution.

Distinguish interface-permitted combinations from spatially witnessed
combinations and from renderer-admitted combinations. An unobserved cell is
not a proof of impossibility. E/F are not substantive size-one gold.
Literal strings reachable by slot permutations are labeled potential, not
counted as new independent examples.

## Checks and Rejections

Analyze full semantics, then build and verify certificates. Record semantic
inventory even if certificate construction later fails. Preserve stage and
reason for failures, unknowns and timeouts. For admitted candidates:

- Use the current matched qualitative renderer with audit-only numerical maps.
- Run template integrity and independent printed-text evidence checks.
- Measure target tokens using the pinned Qwen3.5-4B tokenizer and unchanged
  4096 response / 8192 context limits.
- Check prompt round trips, gold encoding and rename/order invariance.
- Check controlled distractor removability and requested depth.
- Admit complete bases only if every generated menu/contract/style passes.

No certificate support is broadened in this sampling experiment. The known
indirect-coincidence limitation remains an explicit rejection category.

## Diversity and Weighting

Report attempted/admitted stories, bases, menu bundles and views separately.
Measure conservative canonical story signatures, query-aware signatures with
normalized premise inversions, and label-free topology refinement signatures.
Topology signatures also exclude disconnected distractor-only components;
they are coarse collision-prone descriptors, not proof of independent reasoning
structures. Report source recipes separately. Renaming is not new structure.

Depth belongs to the base support descriptor. A view only qualifies for the
substantive depth cohort if it is not E/F and its selected outline contains a
checked path attaining that depth. This is a nominated-support check, not a
proof of minimal explanation complexity.

As an availability diagnostic, select at most one menu bundle per base and
one base per coarse query-core topology, then equalize the number of bundles
across the three question types. Prefer substantive full-coverage bundles,
using seeded tie-breaking. All three contracts remain together. Report the
resulting semantic/depth/E/F mix; do not call this an approved training mixture.
Do not let view permutations stand in for distinct bases or families.

## Output and Checkpoint

Save fixed schedule and source/tokenizer hashes, complete attempt records,
cell matrix, rejection inventory, token and diversity summaries, and the
diagnostic balanced subset IDs (not training files). Report which observed
cells have only one base or topology. Preserve old artifacts.

Stop for user review. Final quotas require feasible coverage, structural
variety, explicit family splitting and production-size decisions.
