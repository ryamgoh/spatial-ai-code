# Depth and Gold Sampling Specification

## Status

Design amendment with an implemented first feasibility pilot, not a production freeze.
Subordinate to [the authoritative pivot](pivot_20261026.md).
The user requires a report before production generation and final sealing.
Candidate sizes, source proportions, depth weights, family floors and paired-menu
probability were approved on 2026-10-30 and are recorded in the
[pivot's approved settings](pivot_20261026.md#approved-candidate-settings).
The [compiled allocation](PRODUCTION-ALLOCATION-v1.md) now supplies source batches,
semantic/gold checks and development exclusions. Structural preflight passed
with 31,251 query attempts; the concrete settings still await execution
approval/freeze. No seal or training is authorized.
The [first feasibility verdict](research-analysis/DEPTH-FEASIBILITY-VERDICT.md)
reports passing depth mechanics but a Direction token blocker and unfinished
coverage/diversity requirements. Gate 1 is not yet passed.

The [distribution audit](research-analysis/DISTRIBUTION-AND-DEPTH-AUDIT.md)
motivates replacing the prior sampling policy: Unique gold is 84.8% E/F,
and only one recorded four-link Direction question exists, in training.
The old packages remain unchanged. Their correctness/integrity receipts
do not demonstrate conformance to this specification.

## Split Roles

| Role | Depth-Controlled Cohort | Other Semantic Cases | Selection Use |
|---|---|---|---|
| Train | 1-8 | Required supported semantic matrix | Training only |
| Dev | 1-8 | Same declared semantic coverage | Stage-1 checkpoint selection |
| In-range test | 1-8 | Same declared semantic coverage | Final evaluation only |
| Extrapolation test | 9-10 | Only explicitly applicable depth cases | Final evaluation only |

Together the two test roles form the synthetic benchmark, but results remain
separate. External SpatialMap remains a separate evaluation source.
Null-depth cases are not evidence of extrapolation.

Assign source families to roles before query construction and admission.
All query/contract/menu/style variants stay with the source. Renamings,
reorderings, matched consistent/inconsistent cases and deliberately related
depth variants cannot cross roles. Do not create depth 9 by extending a
training story and then call it an independent test family.

Pilot examples at depths 9/10 are generator-development fixtures, not the final
held-out benchmark; exclude their families and copies from final benchmark
admission. No model inference on these examples is part of feasibility.

## Depth Measurement

Keep the existing supporting-path descriptor. Add a versioned
`shortest-atomic-axis-support-v1` descriptor for the controlled cohort.
Its initial implementation and tests are in `spatial/v2/composition_depth.py`
and `spatial/v2/tests/test_composition_depth.py`; production coverage remains pending.

### Direction

For consistent, entailed exact direction queries, compute the shortest valid
x-path and y-path over the complete atomic premise graph. A strict comparison
requires at least one strict edge; equality uses equality edges only.
Record both lengths, premise IDs and `depth = max(x_length, y_length)`.
Use the existing `derive_axis_proof` as the starting implementation, not a new
solver. Test the measurer against independently enumerated small graphs.

A graph path is a sound support. Absence of such a path is not a solver
impossibility verdict, since distinct-point axioms couple the axes.
Unsupported proof forms receive a reason, not an invented path depth.
No claim is made about globally smallest proof, optimal reasoning algorithm,
minimum human steps or minimum token count.

### Selection

Start with fixed memberships: every eligible candidate is either guaranteed
to qualify or impossible to qualify.

- For a guaranteed match, record shortest x/y inclusion paths and their maximum.
- For an impossible match, record a shortest supported axis comparison that
  contradicts at least one required component of the exact queried direction.
  If several such comparisons suffice, choose the shortest deterministically.
- Set the query's membership depth to the maximum over all eligible candidates,
  retaining each candidate's positive/negative evidence and individual lengths.
- If any necessary classification lacks this supported proof form, the query
  cannot fill a controlled-depth quota. It can remain a separately supported
  semantic case if all ordinary evidence requirements pass.

Contingent memberships are not assigned this fixed-membership depth.
Unique, no-match and multiple-certain cases must remain distinct in metadata.

### Count

Begin controlled-depth feasibility with invariant counts from fully fixed
memberships. Reuse the complete membership profile above, and independently
verify the exact count with the solver. The maximum membership depth is a
descriptor of that evidence, not automatically of the shortest count proof.

For both Selection and Count, identify a candidate attaining the reported
maximum. Perform single-premise ablations on its nominated support and require
at least one conclusive change in the complete semantic query answer.
For Count this must change possible counts, not merely a candidate's identity.
Keep the ablation results. This is a dependence check, not a proof of global
minimality. If it fails, do not use the case to establish required count depth.

Invariant counts with varying member identities and variable joint counts
remain separately labeled semantic cases. Preserve their joint witnesses and
exclusion checks; do not remove them to simplify the depth metric.

### Controls and Limits

Recompute depth after every accepted context/distractor addition. Reject
requested-depth cases with shorter graph support. Recheck full semantics,
distinctness, coordinate witnesses, and distractor removability.

Depth 10 may use intermediate objects beyond the eligible candidate set.
Keep candidate domains explicit and independent of the answer menu. Begin
with 4-8 eligible candidates where four real menu values are feasible;
do not raise the current eight-candidate Count evidence bound automatically.
Freeze total object and premise bounds only after token/solver feasibility.

Apply the depth ceiling to all admitted depth-applicable training/dev questions,
not just one selected question about each story. Do not introduce incidental
depth-9/10 questions through another query view of a training story.

Record total objects/premises, candidates, branches, distractors and token
lengths by depth. Match their distributions where feasible; disclose inevitable
increases such as the objects needed for a longer simple path. A depth result
alone does not isolate these effects.

## Gold Coverage

Keep three separate coverage tables:

1. Semantic cases: unique, underdetermined, inconsistent, and Selection
   no-match/multiple-certain; retain zero/nonzero and joint Count distinctions.
2. Literal gold strings: six for Unique; 17 for Listed and Exhaustive.
3. Underlying possible-answer cardinality, displayed-answer cardinality,
   full/partial coverage, E/F reason and substantive slot positions.

Aim for all feasible literal strings per contract in train/dev/in-range test,
with minimum distinct-base counts to be approved after feasibility.
Do not demand all 17 strings at every query/depth cell or force them into a
determinate-only extrapolation cohort. Record unsupported intersections.
Gold inspection must cover each present contract/string and escape reason,
with overlap allowed, in addition to the pivot's existing audit minimum.

Uniform labels are not the objective. Set semantic proportions and substantive
minima first; choose menu weights jointly with them. All three contracts
remain present for each retained menu, with Natural/Symbolic paired exactly.
An A-D permutation must be shared by the matched views.

Make full coverage the control when feasible; sample a declared bounded subset
for paired omission variants rather than always doubling it. Preserve cases
with more than four possible values as necessarily incomplete. Never truncate
the semantic candidate domain, add fake options, or alter a gold answer.

## Generation Procedure

1. Freeze a seeded pilot attempt schedule, supported evidence forms and
   per-stratum caps before running the pilot. Report failures, not just successes.
2. Generate varied atomic graphs conditioned on a requested depth/semantic
   category, plus broader randomized semantic cases. Randomize graph structure,
   relation directions, premise orientations, names and order.
3. Check consistency and complete semantics. Build and verify both reference
   renderings; measure actual depth and reject mismatches.
4. Generate menus using the declared coverage policy and map true semantic
   answers to letters. Preserve complete paired groups.
5. Admit within fixed role/stratum caps using deterministic selection. No
   infinite retry-to-quota, seed search, moving rejected items across splits,
   or unreported relaxation of requirements.
6. Produce pre-release distributions with both base/family and rendered-view
   denominators. Count repeated views separately from independent examples.

No requirement exists to generate exactly three query types from every story
if that would defeat depth or semantic coverage. Any queries that are generated
still share the story's role. Query coverage is enforced across the dataset.

## Numeric Decisions

Approved values are distinguished from remaining choices below. Do not silently
inherit pending settings from the 200-story release.

| Setting | Required Decision |
|---|---|
| Pilot budget | Fixed attempts per supported depth/query and semantic category |
| Production size | Approved base counts: 6,000 / 1,200 / 2,400 / 1,200; actual story/family counts still audited |
| Depth quotas | Approved weights and family minima: 50 / 10 / 20 / 50 per applicable query/depth cell |
| Source mix | Approved 70% controlled, 15% random-world, 10% random-atoms, 5% targeted in-range; extrapolation all controlled |
| Semantic mix | Within-source semantic proportions and substantive minima remain to be completed |
| Menu policy | Approved 20% seeded omission chance among eligible bases, always retaining the primary control |
| Literal gold coverage | Minimum feasible examples per contract/string |
| Controls | Candidate/object/premise/distractor bounds and matching policy |
| Execution budget | Seeds, solver timeouts, attempt caps and acceptance ordering |
| Token admission | Measured feasibility under matched 4,096-token Stage-1 cap |

Do not invent an E/F threshold before examining feasible paired-contract
mixtures. Report proposed thresholds/ranges for approval. Compute the fixed
contract-majority baseline from training labels only; it is not a trained arm.

## Gates and User Checkpoints

**Gate 1: Feasibility, before production approval**

- Tests demonstrate measured depths 1-10 for each proposed supported query type,
  deliberate-shortcut rejection, membership exclusions, Count dependence,
  family isolation, prompt round trips and error/timeout quarantine.
- Matched references pass evidence checks and pinned-tokenizer admission.
  Unsupported cases and long-trace failures are visible.
- Present a depth-by-role prototype table, gold/reason counts, escape rates,
  token distributions, rejection rates, sampled explanations, and proposed
  numeric production settings. Pilot roles are provisional, not final test data.
- Stop and report. Do not generate production data until the user approves
  the concrete settings and supported matrix.

**Gate 2: Production audit, before final sealing**

- Freeze the approved policy before generating final test families. Inspect
  dataset validity/coverage but do not tune repeatedly against held-out scores
  or label frequencies; a failed gate requires a documented revision.
- All approved depth/gold/semantic quotas pass in the relevant roles.
  Extra views cannot compensate for missing underlying examples.
- All existing solver/evidence, audit, split, token, and integrity gates pass,
  including verified explanations for every synthetic benchmark example.
- Report actual counts, distributions, full limitations and training-selected
  premise-blind baseline. Keep substantive depth results separate from escapes.
- Stop for review before writing the final seal. A seal is not approval merely
  because the source and file checksums match.

**After sealing:** GPU preflight and training remain separate authorization
steps. Keep two matched adapters, one shared initial training seed, the
Stage-1-only in-range dev selection rule, and both evaluation stages.
