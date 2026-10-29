# Coverage-Protected Selection Results

Existing 240-base pool only. No new sampling, explanations, training files, split assignment or seal. Original artifacts are unchanged.

Solver result: **optimal**.

## Verdict

Coverage loss is fixed, but the sample remains inadequate for production balance.
The revised subset retains every observed protected category and restores
substantive Unique Selection at depth 2. Total E/F falls from 190/630 (30.2%)
to 175/630 (27.8%). Unique Selection remains 50/70 E/F (71.4%).

Only 22 substantive Unique Selection bases exist in the whole pool; even without
other restrictions, selecting 70 distinct bases forces at least 48 E/F (68.6%).
The current policy retains 20 substantive bases. Its optimum minimizes total
escapes across all contracts, not Unique Selection escapes independently.

This experiment demonstrates selection capability, not sufficient quotas.
Fine-grained coverage improves, but cells with only one selected base increase
from 45 to 60. The remaining 117 of the pool's 240 observed fine cells are
not retained. None of the 159 interface-permitted but unobserved cells can be
created by selection.

## Same-Size Comparison

| Metric | Previous | Revised |
| --- | ---: | ---: |
| Views per style | 630 | 630 |
| Fine-grained cells | 111 | 123 |
| Task/topology combinations | 210 | 202 |
| Source families | 53 | 53 |
| Cells with only one base | 45 | 60 |
| Missing protected categories | 7 | 0 |

## Escape Rates

| Task / contract | Previous E/F | Revised E/F | Revised substantive |
| --- | ---: | ---: | ---: |
| direction/single | 25/70 (35.7%) | 25/70 (35.7%) | 45 |
| direction/all-possible | 22/70 (31.4%) | 22/70 (31.4%) | 48 |
| direction/visible-possible | 5/70 (7.1%) | 4/70 (5.7%) | 66 |
| which/single | 50/70 (71.4%) | 50/70 (71.4%) | 20 |
| which/all-possible | 24/70 (34.3%) | 20/70 (28.6%) | 50 |
| which/visible-possible | 15/70 (21.4%) | 11/70 (15.7%) | 59 |
| count/single | 33/70 (47.1%) | 33/70 (47.1%) | 37 |
| count/all-possible | 11/70 (15.7%) | 8/70 (11.4%) | 62 |
| count/visible-possible | 5/70 (7.1%) | 2/70 (2.9%) | 68 |

## Substantive Unique Depths

| Task | Depth | Pool bases | Previous | Revised |
| --- | ---: | ---: | ---: | ---: |
| direction | 1 | 21 | 17 | 18 |
| direction | 2 | 6 | 4 | 4 |
| direction | 3 | 3 | 3 | 3 |
| direction | 4 | 4 | 4 | 4 |
| direction | 5 | 3 | 3 | 3 |
| direction | 6 | 3 | 3 | 2 |
| direction | 7 | 3 | 3 | 3 |
| direction | 8 | 3 | 2 | 2 |
| direction | 9 | 3 | 3 | 3 |
| direction | 10 | 3 | 3 | 3 |
| which | 1 | 4 | 4 | 4 |
| which | 2 | 2 | 0 | 1 |
| which | 3 | 2 | 2 | 2 |
| which | 4 | 2 | 2 | 2 |
| which | 5 | 2 | 2 | 1 |
| which | 6 | 2 | 2 | 2 |
| which | 7 | 2 | 2 | 2 |
| which | 8 | 2 | 2 | 2 |
| which | 9 | 2 | 2 | 2 |
| which | 10 | 2 | 2 | 2 |
| count | 1 | 8 | 5 | 5 |
| count | 2 | 9 | 7 | 7 |
| count | 3 | 5 | 5 | 5 |
| count | 4 | 3 | 3 | 3 |
| count | 5 | 3 | 3 | 3 |
| count | 6 | 3 | 3 | 3 |
| count | 7 | 3 | 3 | 3 |
| count | 8 | 3 | 2 | 2 |
| count | 9 | 3 | 3 | 3 |
| count | 10 | 3 | 3 | 3 |

## Pool Limits

| Task | Available substantive Unique bases | Minimum E/F at 70 bases |
| --- | ---: | ---: |
| direction | 52 | 18/70 |
| which | 22 | 48/70 |
| count | 43 | 27/70 |

These are availability bounds before protected coverage and topology caps, not promises that every available substantive base can coexist in the selected set.

## Interpretation

Hard minima preserve each observed semantic case, returned size, escape reason, Count-zero category and substantive depth by task/contract. They do not require every combination of these features or every literal answer permutation. The JSON lists missing fine-grained cells and all selected bundle IDs.

The optimizer first minimizes total E/F under those hard constraints, then maximizes fine-cell coverage, then topology diversity, then a fixed seeded rank preference. Optimality applies only to this fixed pool and stated policy.

Repeating a topology across different semantic/depth strata is allowed. Thus topology diversity can decrease relative to the previous blanket filter. Topology and source-family counts are not proofs of independent reasoning structures. No quota percentages, split budgets or independence thresholds have been approved for production.

One example per protected category is a feasibility minimum, not adequate training or benchmark coverage. Selection cannot create unseen Exhaustive Direction size-four cases, invariant Count with varying identities, or additional generator families. The inherited certificate limitation is unchanged.

All constraints are independently rechecked in Python after solving. Both styles and the three contracts are retained from each original bundle. No reference text, gold, depth or prompt is modified.

## Computation and Verification

The first Z3 Optimize attempt reached its fixed 120-second limit and returned
`unknown: sat.canceled`. It is preserved in `COVERAGE-SELECTION-RESULTS-v1.json`
and was not treated as a valid selection.

The unchanged policy was then solved using SciPy 1.18.1 / HiGHS binary MILP,
as declared in `COVERAGE-SELECTION-PROTOCOL-v2.md`. Each stage fixes the preceding
integer optimum. All stages reported zero relative gap:

| Objective | Optimum | Solver dual bound |
| --- | ---: | ---: |
| Minimize E/F | 175 | 175 |
| Maximize fine cells | 123 | 123 |
| Maximize task/topology combinations | 202 | 202 |
| Minimize seeded rank sum | 44041 | 44041 |

Optimality is solver-reported with floating-point tolerances, not an independent
formal optimality proof. Python independently verified every hard constraint
and each integer objective value. All original artifact hashes and the selector
source hash matched after execution.

Final regression suite: **709 passed, one skipped**. Selector lint passed.
The two solver backends agree on small feasible and infeasible test cases.

## Next Work

1. Add structurally varied generators for exactly one guaranteed Selection
   match, including multiple independent constructions per depth. The present
   pool has only two such bases at most depths.
2. Set minimum distinct-base and family counts within supported coverage cells,
   then solve bundle quotas jointly. One example per category is not enough.
3. Investigate the unobserved Exhaustive Direction size-four and
   invariant-count/varying-membership categories before treating them as required.
4. Resolve or explicitly scope the known indirect-coincidence certificate
   limitation before a production proposal.

No additional generation is authorized by this report. Current development
depths 9/10 remain excluded from final held-out benchmark families.
