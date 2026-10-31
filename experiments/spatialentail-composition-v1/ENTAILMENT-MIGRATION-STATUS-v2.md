# Entailment Production Integration Checkpoint

## Verified Progress

The shared CPU coordinator now routes `entailment-production-v1` configurations
to the entailment-only production implementation. Historical configurations
retain their original semantics. New production admission, source/depth/Count
quota accounting, coverage checks, checkpoint replay, early-stop auditing and
input-only exports are connected.

Tests exercise actual one- and two-worker admissions, exact serial/parallel
record equality, checkpoint replay and rejection of mismatched identities.
An end-to-end coordinator test exports the new contract views and reports
expected shortfalls on a deliberately small development workload. Test-only
approval markers and sizes are not production approval or a candidate release.

The implementation passed:

- 142 focused semantic, evidence and production tests.
- 139 production, coverage, admission and historical-executor regression tests.

These selections overlap and must not be summed. They are not a full
repository test run or proof of complete supported-source coverage.

## Deterministic Counterexamples

The serial/parallel equality test exposed different valid Z3 model choices
after different solver histories. Answers and tested reference prose agreed,
but full audit records differed.

A full-coordinate lexicographic optimizer fixed equality but slowed the
six-query development workload to 24.92 seconds versus 19.97 seconds for the
historical path. This was not retained as the atomic-study witness algorithm.

The current atomic path constructs deterministic ranks from equality groups
and strict-order graphs. When a different counterexample is needed, incremental
unbounded SAT checks fix only query relations. Count fixes a jointly feasible
alternative count and candidate relations; context pairs are not individually
canonicalized. Full premise, distinctness and false-claim checks still apply.

The graph construction helper is shared with the small reference backend.
Counterexample validation independently evaluates the full premises and query;
reference-backend agreement must not be described as wholly independent of
that shared construction code.

The latest receipt, `ENTAILMENT-DEVELOPMENT-v5.json`, records six accepted
existing development queries, 33 views per style, and:

| Path | Local elapsed time |
|---|---:|
| Historical possibility-based admission | 19.90 seconds |
| Current entailment-only graph-witness admission | 3.36 seconds |

Maximum reference lengths were 644 Natural tokens and 622 Symbolic tokens.
This small local comparison answers different tasks and checks agreement of
the underlying entailed sets, not identical historical/new labels or prose.
It is not a production-wide or Slurm runtime estimate.

## Configuration and Gates

`entailment-config-draft-v1.json` preserves the approved retained sizes,
source mix, depth weights, Count weights, family floors and omission probability.
It records the numerical source configuration and exclusion inventory hashes.
Its review status is pending and it cannot be executed as production.

The derived matrix now also carries the approved nonhomogeneous-depth minima
and feasible literal-gold presence requirements. Listed/Exhaustive difference
requirements for Direction and Count remain explicitly inapplicable. Negative
exposure minima are proposed, not empirically optimized or silently approved.

## Outstanding

1. Verify source and explanation supply for the revised coverage matrix,
   including sole-entailment-plus-contingent Selection cases and both negative
   categories. Audit the declared proof-support limits before freezing.
2. Review and freeze the new executable configuration. Do not resume the old
   possibility-based allocation under the new interface.
3. Complete and verify the separately versioned external SpatialMap views.
4. Generate the full 6,000 / 1,200 / 2,400 / 1,200 unsealed candidate and review
   distributions, family isolation, rejections, and matched explanations.
5. Obtain separate approval before sealing or model training.

No production dataset, Slurm submission, seal or model training was performed
at this checkpoint. The active goal remains unfinished.
