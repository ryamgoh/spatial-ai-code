# Production Integration v1

Status: candidate numeric settings approved on 2026-10-30; retained-source
quotas implemented; source allocation compiled and structurally checked.
The [allocation receipt](PRODUCTION-ALLOCATION-v1.md) records 31,251 scheduled
query attempts, completed development exclusions and passing structural supply
checks. The concrete executable schedule still awaits approval/freeze.
No production dataset, seal, GPU preflight or training run has been created.
The authoritative specification remains [the pivot](pivot_20261026.md).

## Implemented

`spatial.v2.composition_production` connects the existing random-world,
random-atom, targeted, controlled-depth, unique-Selection, sampled-forest and
sampled-targeted constructors to
the current readable admission pipeline. Historical development entry points
and their saved results are unchanged.

- Explicit source batches assign story roles before query admission.
- Canonical story and source-family collisions across roles block the run.
  Related short/long controlled constructions share one lineage, even though
  the development generator used different depth-band family labels.
- Development family IDs and canonical story keys are excluded from dev and
  both benchmark splits using a required, hash-pinned inventory.
- Requested depths must match measured full-scene depth. Every primary controlled
  contract must print substantive evidence attaining that depth. Unlisted
  deep rivals are included in substantive Selection menus when needed.
- Incidental measured depth above eight is rejected from train/dev/in-range
  test, including questions in the non-controlled semantic cohort.
- Full semantics, evidence, distractor removal, prompt round trips, menu
  permutations, matched readable explanations and actual token admission are
  checked through the existing components. Unsupported cases stay rejected.
- One primary menu and optionally one matched omission menu are retained per
  selected base, with all three contracts per menu and both styles. A companion
  replaces one possible value with an undisplayed impossible value; every other
  slot remains fixed. Prompt round trips, permutations and contract-specific
  omission effects are checked. No companion is fabricated for an inconsistent,
  empty, overfull, or replacement-infeasible domain. Omission views retain base
  depth but cannot substitute for the primary control's printed-depth support.
  Canonical
  duplicate questions do not increase the base count. Fixed seeded order and
  per-role/query/cohort/depth caps determine admission; no retry-to-quota loop.
- Full attempted evidence is streamed to disk. Summary statistics use compact
  records, and model inputs are exported separately from paired gold/references.
- Missing quotas, distinct-family floors, mandatory semantic/gold coverage and
  additional approved minima are reported. Missing any of the eight directions
  in a split/query type also fails the presence check. These produce a nonzero
  CLI exit, not an automatic refill. Counts ignore menu/contract/style copies.
- Retained source quotas enforce the approved mixture separately within each
  split/query type. A verified base beyond its source cap is recorded as
  `source-quota-excess`; it does not consume another source's allowance or fill
  the overall cohort quota. Rejections and duplicate queries never fill quotas.
- Controlled Count caps use verified invariant values, with separate per-depth
  counts and shortfalls. Surplus values are `count-quota-excess`, never silently
  substituted for missing values. Sampled Count batches require explicit matches.
- Same-menu contract differences are independently recomputed from gold letters
  at summary time. Reports count distinct bases per query and split, not views.
  Approved safeguards also require semantic-case depth-independent minima and
  nonhomogeneous proof support in controlled depth cells above one.
- Manifests record the policy, schedule, source/tokenizer/output hashes and
  versions. Every result remains an unsealed candidate requiring manual audit.

Canonical grouping reuses the existing bounded structural canonicalizer.
It is not a proof of equivalence for every rotated or semantically equivalent
graph. Explicit lineage grouping is still necessary.

## Approved Candidate Configuration

[production-config-v1.json](production-config-v1.json) is `numeric-approved`.
The user approved the candidate sizes, query/source allocation, depth weights,
family floors, paired menus and direction presence on 2026-10-30. This is
separate from `frozen`, which authorizes execution only after remaining source,
exclusion and coverage gates have been completed.

| Setting | Candidate Value |
|---|---|
| Underlying base targets | 6,000 train; 1,200 dev; 2,400 in-range test; 1,200 extrapolation |
| Query allocation | Equal Direction / Selection / Count base counts |
| In-range source allocation | Approved 70% controlled; 15% random-world; 10% random-atoms; 5% targeted semantic |
| Training depth weights | 8, 9, 10, 11, 13, 14, 16, 19 for depths 1-8 |
| Evaluation depth weights | Equal within 1-8 or 9-10 respectively |
| Primary menus | Maximum possible-set coverage; controlled bases require substantive printed depth under all three primary contracts |
| Matched omission menus | Approved 20% seeded chance, at most one additional menu per eligible base in either cohort |
| Depth family floors | Approved train 50; dev 10; in-range test 20; extrapolation 50 per query/depth cell |
| Direction presence | All eight per split/query type; no equal-frequency or per-depth quota |
| Outer scene ceilings | Existing implementation ceilings: 32 objects, 32 premises, 8 eligible candidates |
| Solver timeout | Existing draft: 5 seconds per solver invocation, not per entire base |
| Fixed base-attempt budget | Proposed 32,400, to be assigned to explicit source batches |

The semantic cohort may contain measurable-depth questions, but these do not
fill controlled quotas. The new `paired_omission_percent` setting replaces the
earlier draft `semantic_partial_menu_percent`, which selected a partial menu
instead of retaining its control. The approved 20% chance is not an exact
realized share. Feasibility, admission and the
seeded draws affect the number of companions. Each adds three views per style,
never another base. Necessarily incomplete primary menus remain in the dataset.
The outer scene ceilings do not make the existing random constructor sample
32-object scenes; it still samples 5-8 objects. Controlled scenes can be larger.

For example, the approved training plan allocates 1,400 controlled and 600 semantic
bases to each query type. Each query's depth-1 target is 112 bases and depth-8
target is 266. The 6,000 training bases produce 18,000 primary contract views per
style, plus three per admitted omission companion. Planning reports primary
counts and an upper bound, not a falsely exact view total.

| Split | Controlled | Random World | Random Atoms | Targeted |
|---|---:|---:|---:|---:|
| Train | 4,200 | 900 | 600 | 300 |
| Dev | 840 | 180 | 120 | 60 |
| In-range test | 1,680 | 360 | 240 | 120 |
| Extrapolation | 1,200 | 0 | 0 | 0 |

Each source count is divided equally across the three query types. Source
proportions describe accepted underlying bases, not attempt proportions.
The 78 explicit family floors cover all supported split/query/depth cells.

## Allocation Checkpoint

The following list records the requirements that were pending before the
allocation work. Items 1-3 are now implemented in
`production-allocation-v1/config.json` and its hash-pinned companion inventory.
Structural preflight passed; review the concrete settings in
[Production Allocation v1](PRODUCTION-ALLOCATION-v1.md) before execution.
The separate external adapter in item 4 remains unfinished and does not
prevent structural preparation of the synthetic candidate.

1. **Frozen held-out source allocation.** The new sampled-forest constructor
   supplies graph variation beyond the earlier bounded development lineages.
   Assign its scaffold families to roles before admission and freeze the actual
   source batches. A new seed alone does not establish independence; equal
   canonical scaffold keys are one family, and related depth/direction/context
   variants cannot cross roles. Canonical scene collisions also block preflight.
2. **Frozen source batches and remaining coverage minima.** The numeric mixture
   and depth-family floors are approved and recorded. Populate `source_batches`
   and `coverage_minima`; finish the unapproved attempt schedule and remaining
   semantic/gold settings before execution. Baseline gates
   require all 33 query/contract/semantic cells in each in-range role, every
   contract-level literal gold string, and substantive invariant Count-zero.
   `depth_family_minima` must specify a positive floor for every populated
   controlled split/query/depth stratum, never exceeding that stratum's base
   quota. Historical depth-family aliases count as one lineage. A missing map
   blocks generation; sparse actual family coverage fails candidate checks.
   Additional rules match row metadata and count distinct underlying bases
   and/or families.
   Unsupported targets require a documented policy revision, not silent waivers.
3. **Development exclusion inventory.** Supply a reviewed JSON file containing
   `family_ids` and `story_keys`, with its path and SHA-256 in the config.
   Depth-family aliases are normalized across historical depth bands.
4. **External evaluation adapter.** The separate three-contract
   SpatialMap-TQA-Corr adaptation is not implemented by this synthetic runner.
   Its candidate-domain policy still needs to be frozen explicitly.

## Commands and Source Batches

Planning does not sample scenes, invoke the solver or load a tokenizer:

```sh
python -m spatial.v2.composition_production \
  --config experiments/spatialentail-composition-v1/production-config-v1.json \
  --plan
```

Execution additionally requires `--output` and the pinned `--tokenizer` path,
a frozen configuration and passing schedule preflight. It refuses existing
output directories. This is not authorization to execute production now.

Each source batch declares `generator`, `seed`, `count`, `split` and
`query_kinds` (`direction`, `which`, `count`). Controlled batches also declare
`depth` and `sample`; unique-Selection batches add `pattern` and `direction`.
Targeted batches declare `per_recipe`. Their `count` counts constructor calls,
each producing multiple stories, including matched cycle variants. Random and
controlled batches' `count` counts stories. The resulting number of requested
base queries must exactly match `attempt_budget`.

`sampled-forest` batches declare `depth` but do not use the old `sample` index.
Their `count` draws scaffolds at successive seeds starting with `seed`.
Optional `direction` fixes any one of the eight exact directions; otherwise
the seed selects it. Optional `variant` changes presentation and context
without changing the family. `matches` fixes the number of qualifying
candidates (valid from 0 through the sampled candidate count) and is mandatory
for batches requesting Count. Other queries retain the default of one match.
`proof_pattern` selects `homogeneous` (default), `mixed-chain` (diagonals only,
depth at least two) or `axis-split` (all directions, depth at least two).
An incompatible request fails instead of falling back to a homogeneous proof.
Count variants also share the original scaffold family. These draws remain subject to duplicate, family,
development-exclusion, verification and quota checks; no retries are added.

Coverage rules use `{"match": {...}, "minimum_bases": N,
"minimum_families": F}` with at least one minimum. A match can include split,
query kind, contract, semantic case, answer, depth, escape reason, `menu_role`
(`primary` or `omission`), or `substantive_depth_support`. For a substantive
depth rule, explicitly match the primary role and printed support, rather than
counting an omission's unchanged base-depth annotation as a depth-supported view.
The mandatory `depth_family_minima` map uses keys such as `train/which/2`;
it counts only accepted controlled bases, which have passed primary depth
verification. These are post-admission gates, not instructions to alter labels
or resample. Depth-family values are now approved; additional semantic/gold
minima remain to be completed.

Direction summaries use base denominators and normalized family counts,
including explicit zero cells, cardinal/diagonal groups and direction-by-depth
tables. Presence means a verified singleton Direction answer or a queried
Selection/Count direction on a consistent scene. It does not mean an option
merely occurs in a menu. Ambiguous Direction sets and inconsistent scenes do
not fill the eight-direction check. Every direction must appear per split/query
type, but no uniform frequency or eight-by-every-depth grid is enforced.

## Sampled Forest Scope

`spatial/v2/composition_scene_sampler.py` implements `sampled-forest-v1`:

- Sample 4-8 eligible candidates, nonmatching path lengths and shared-parent
  choices, with verified support reaching the nominated endpoint at depth 1-10.
  For positive match counts that endpoint qualifies; for Count-zero it is
  excluded. Additional guaranteed matches use direct relations.
- Add up to four consistent relations between nonmatching candidates. These
  can create cycles; they never connect back into the positive chain to create
  a shortcut. Internal ranks choose a compatible order, not unit-step distances
  or extra constraints in the problem.
- Add 1-3 connected context leaves and 0-2 disconnected relation pairs. Mark
  these context premises for individual and joint removability verification.
  Other premises can also be irrelevant; marked context is not an exhaustive
  classification of relevance.
- Keep at most 24 core objects (23 for separate-axis supports), 31 total objects
  and 32 total premises. The separate-axis core reserves room for its extra edge.
  Shuffle names, premise order and relation orientation through the existing
  presentation helper.
- Build family IDs from the canonical core graph at a fixed depth-10 North
  homogeneous anchor. Actual depth, queried direction, proof pattern and added context do not change that
  identity. The anchor is an identity convention, not a claim that cardinal
  constraints can be rotated by 45 degrees; actual diagonal scenes are built
  with diagonal relations.

All these controlled scenes are consistent and have fixed candidate memberships.
Default one-match scenes support all three query types under substantive
contracts. The optional fixed-count variants cover no-match and multiple-match
Selection and non-unit Count. No-match/multiple-match Selection has an E answer
under Unique and cannot fill an all-contract substantive depth quota; use its
Count/Direction views for controlled quotas. The sampler does not replace the
ambiguous, inconsistent or variable-count semantic sources. Production
proportions across those sources are approved above; their concrete family
allocation is not yet frozen.

### Proof Variation

The homogeneous mode preserves the old constructor and family anchor. The two
additional modes change the supporting relations, not just surrounding context:

- `mixed-chain`: alternate East and North to derive Northeast, with quarter-turn
  variants for the other diagonals. Equality propagation and strict comparisons
  must both be composed along the path.
- `axis-split`: combine a two-link horizontal-support path with a depth-length
  vertical-support path, rotated as appropriate. For North, the short path uses
  North then South to establish equal x without fixing endpoint y order. The long
  path uses Northeast links then Northwest to establish increasing y without
  fixing x order. Neither alone determines North. The diagonal construction
  instead combines Southeast/Northeast horizontal support with
  Northwest/Northeast vertical support.

For diagonal zero-count variants, the short path keeps the query's horizontal
sign and the long path establishes the conflicting vertical sign. Opposing
both signs would let the short path exclude the candidate immediately, losing
the requested Count depth. This is checked across all eight query directions.

Depth remains the maximum measured required axis-support length, not the number
of paths or total premises. These fixed-membership modes still yield identical
contract golds on each menu. Different proofs are not contract-discriminating
examples, nor are they separate families when they share a scaffold.

A depth-2 North example has `A Northeast O`, `C Northwest A`, `B North O`
and `C South B`. The first path puts C above O but leaves their horizontal
order open; the second fixes equal x but leaves their vertical order open.
Together they entail North. A rendered software fixture was inspected: Natural
states the equal horizontal positions followed by the bottom-to-top ordering;
Symbolic uses the corresponding equality chain and strict y chain. Neither
style prints the audit witness coordinates.

### Approved Diversity Settings

The original source mixture remains approved and unchanged. The additional
thresholds below were initially draft and received approval on 2026-10-30 when
the user authorized proceeding with the safeguards and production-setting work.
`diversity_checks.status` now records that approval:

- Equal controlled Count weights for 0-4 per split/depth cell. Largest-remainder
  rounding preserves each cell's existing base quota. Semantic Count is not
  restricted to these five values.
- At least 30% nonhomogeneous bases in each controlled query/depth cell above
  depth 1. This counts mixed-chain or axis-split primary views after depth
  admission, once per base; it does not require equal directions at each depth.
- Each semantic case requires at least 5% of the query's semantic quota:
  30 train, 6 dev and 12 in-range bases. Fixed unique cases can also be supplied
  by controlled sources; the percentage sets a floor, not a source restriction.
- For each query and each contract pair, require same-menu gold differences in
  at least 20% of the semantic quota: 120 train, 24 dev and 48 in-range bases.
  There are three separately checked pairs, so a single comparison cannot
  stand in for all contracts. The same base may satisfy multiple pair checks.
- Both semantic and contract-pair minima additionally require 10 train,
  3 dev and 5 in-range families. Extrapolation remains definite depth practice,
  not an ambiguity or contract-discrimination benchmark.

Flags `unique_listed_differ`, `unique_exhaustive_differ`,
`listed_exhaustive_differ` and `contracts_differ` are attached to all three rows
of each identical-menu bundle. A full-menu A followed by omission F is **not**
itself a contract difference. A base with a genuine difference on either menu
counts once for that pair. Summary recomputation prevents stale flags from
passing coverage checks. Full/omission companions, styles and contracts do not
inflate minima. These safeguards are not evidence that the ungenerated candidate
will satisfy them. The CLI still blocks unless the executable policy is frozen.

## Targeted Source Allocation Check

The existing `construct_stories` supports nine recipes with five sample IDs
each: **45 source families**, producing 50 stories because cycle variants share
a family. Every one appears in the historical development candidate:
`study-candidate-v2/pilot.json`, SHA-256
`03bcd8cd880d1c697ab7f8b61f94d084b8b8dfa742eafa4283ce0356904bfb39`.
The source family IDs are recoverable from its `split_assignments` story keys,
even though the old union-based split assignment also stored hashed group IDs.

Changing the constructor seed retains the same 45 family IDs. Once the
development exclusions are applied, none can supply the approved 60 dev or
120 in-range-test targeted bases. The 300 training targets may reuse development
lineages under the existing training policy, but that does not solve held-out
coverage. The new sampled-targeted source described below extends this
constructor without changing historical families. Source allocation must use
fresh, unexcluded scaffold families rather than count renamed stories as new
families. No new pilot or production scene generation was used for this check.

The fixed software fixtures use seeds 0-31: 32 canonical scaffold families and
32 distinct depth-5 core topologies, excluding added context from that comparison.
These are graph families sampled by one construction method, not 32 independent
construction methods or evidence of representative coverage. Their family IDs
must join the development exclusion inventory before benchmark generation.
Shallow projections of distinct scaffolds can coincide; the full-scene
canonical split check remains required in addition to scaffold identity.

## Sampled Targeted Scope

`spatial/v2/composition_targeted_sampler.py` adds `sampled-targeted-v1`.
It reuses sampled forest cores, including varying candidate counts, shared
negative paths and candidate-to-candidate edges. This is structural variation
within construction methods, not an assertion of novel reasoning methods.

| Pattern | Intended Semantic Situation |
|---|---|
| `singleton` | One guaranteed matching candidate |
| `empty` | No matching candidates; invariant Count zero |
| `multiple` | Two, three or four guaranteed matches |
| `contingent` | One possible but not guaranteed match; Count possibilities 0 and 1 |
| `consistent-cycle` | A redundant closure consistent with the positive chain |
| `inconsistent-cycle` | The same closure reversed, contradicting the positive chain |
| `coincidence` | Two named candidates forced to share both coordinates |

All patterns accept all eight queried directions. Contingency uses two
independently positioned endpoints above a common diagonal anchor; the queried
target can have all eight directions relative to the reference. Diagonal
relations are constructed directly, with only quarter-turn orientation changes.
No coordinate witnesses are added to supervised premises.

The `span` parameter is bounded at 1-3, with cycles requiring at least 2.
It controls construction length, **not** a requested or measured proof depth.
One connected context leaf and optionally one disconnected pair remain subject
to full-semantic removal checks. Construction is bounded at 32 objects and
32 premises, with 4-8 eligible candidates. The existing evidence, text and token
admission checks still reject unsupported or oversized results without retries.

Example source-batch schema (software fixture, not a production allocation):

```json
{
  "generator": "sampled-targeted",
  "seed": 3,
  "count": 2,
  "pattern": "contingent",
  "span": 2,
  "split": "dev",
  "query_kinds": ["direction", "which", "count"]
}
```

The batch emits one scene per seed (`seed + i`), unlike the legacy constructor
that expands every recipe per draw. Optional `direction`, `variant`, and
`matches` (only 2-4 for `multiple`) configure related variants.
Requested `depth` is forbidden, including for extrapolation. These bases count
against the targeted semantic source quota, never the controlled quota.

**Cross-generator family isolation:** targeted samples retain the same
`sampled-forest-v1/<canonical anchor>` family identity as the controlled forest
with the same scaffold. Pattern, span, direction, count, names, order and context
do not create independent families. Preflight rejects related targeted/controlled
variants in different splits. A fresh seed can still collide with an existing
canonical family; the exclusion inventory and preflight remain mandatory.

Tests use seeds 0-31, already designated development forest lineages. All their
targeted variants must be excluded from final held-out splits as well. A
fixture comparison removes added context and checks core topology variation and
noncollision with legacy targeted cores; it does not prove that every possible
draw is distinct from every previous development scene.

This addition does **not** implement arbitrary joint-count correlations or all
possible-set cardinalities. Those remain the responsibility of the broader
random sources and explicit coverage gates. The final source schedule,
development inventory and post-admission distribution still need completion.

## Verification Boundary

Larger scenes exposed repeated canonical-map construction in checks that only
compare answer sets. The composition analysis wrapper now exposes the solver's
existing `include_witnesses=False` mode for depth, ablation, transformation and
independent label comparisons. It still rejects inconclusive results and checks
the full answer domain. Evidence construction keeps the default complete witness
maps, and evidence verification still independently checks their coverage and
coordinates. Regression tests forbid map construction in label-only checks and
confirm that removing required evidence maps still fails admission.

Regression tests include actual pinned-tokenizer admission for Direction,
Selection and Count at depths 2 and 9, and an end-to-end temporary CLI run
that verifies hashes, unsealed status, paired exports and deliberate shortfalls.
Small test fixtures are software tests, not another dataset feasibility pilot.
They do not establish production family diversity or adequate class proportions.

Before adding the sampled-forest constructor, 33 integration tests passed; the full composition and
pivot-evidence regression command passed 1,120 tests in 176.53 seconds. Ruff
and `git diff --check` also passed. `--plan` returned the expected quotas and
four explicit pre-generation blockers without sampling scenes.

After adding the sampler and label-only verification optimization, the same full
composition/pivot-evidence command passed **1,260 tests in 742.39 seconds**
(12 minutes 22 seconds), including 140 sampler tests. These cover all eight
directions at depths 1-10, full-scene semantics and context removal, actual
pinned-tokenizer depth-10 admission for all three query types, fixed Count 0-4,
family isolation, and mandatory witness-map checks. Lint and whitespace checks
passed. Larger full-evidence scenes remain substantially slower than the old
small fixtures; production throughput has not been measured or promised.
No production dataset, seal or training job was created.

## Family and Menu Integration Checks

The updated production regression file passes **77 tests** (24.47 seconds).
It verifies explicit per-depth family floors, normalized family identities,
matched omission effects, unchanged shared option positions, unchanged base
depth, paired exports, and primary-only qualification for controlled depth
quotas. Actual pinned-tokenizer checks cover both primary-only and paired
admission for all three query types at depths 2 and 9. Ambiguous Selection and
Count pairs also pass the full readable admission path. Companion token failure
rejects the whole base instead of silently discarding the companion.

Direction tests require all eight per split/query while deliberately omitting
other depths and using unequal frequencies. Tests also show that an ambiguous
Direction menu does not establish direction presence. Family, direction,
base-quota and semantic coverage failures each independently fail the candidate
success check. Ruff and whitespace checks passed. At that historical checkpoint,
draft planning reported five blockers, including unapproved family floors.
The broader suite subsequently passed 1,299 tests in 594.94 seconds; five
additional production tests were included in the separate 77-test run.

After recording numeric approval and adding retained-source caps, all **93
production tests passed in 26.40 seconds**. These include exact approved totals,
all 78 family floors, small-total rounding, invalid mixture rejection, source
cap isolation, menu-copy accounting and exclusion of legacy targeted families
under different seeds. Ruff and `git diff --check` passed. `--plan` verifies
the approved allocations without generating scenes and reports the four
remaining freeze/source/exclusion/coverage blockers. The earlier 1,299-test
result predates this source-quota change; the full suite was not rerun for it.

After adding sampled-targeted construction, **193 tests passed in 43.84
seconds**: 100 new targeted-sampler tests and the existing 93 production tests.
Coverage includes seven patterns across all eight directions, 21 actual pinned
tokenizer/readable admissions (each pattern and query type), cross-generator
family isolation, fixed scheduled draws, invalid input rejection, context
removability and core-graph variation. The first run found a test-helper error
that discarded an unconstrained candidate while stripping legacy context;
retaining the query domain fixed the helper without changing sampler semantics.
Ruff and `git diff --check` passed. The full historical suite was not rerun.
No final candidate scenes or exports were generated.

## Proof and Contract Diversity Checks

After adding mixed-chain and separate-axis proofs, verified Count caps and
same-menu contract comparisons, **157 diversity tests passed in 228.01 seconds**.
These include all supported proof/direction combinations at depths 2, 5, 8, 9
and 10, actual pinned-tokenizer readable admission at depth 10, fixed Count
0, 2 and 4, unchanged scaffold family identities, scene bounds, explicit Count
batch settings, quota isolation and recomputed contract-difference flags.
All eight zero-count query directions are checked at depths 2, 8 and 10.

The checks caught two construction issues. Separate-axis support adds an extra
edge, so its core now reserves one object slot to preserve the existing premise
ceiling. For diagonal Count-zero, opposing both query axes allowed a short
horizontal exclusion instead of the requested long proof. The repaired
construction keeps the horizontal sign compatible and makes the long vertical
path establish exclusion. The depth gate correctly rejected the original
construction; it was not relaxed. The focused Count regression passed 30 tests
before the complete 157-test rerun.

Compatibility checks also passed: **194 production/targeted tests** in 50.12
seconds, **217 depth/readable/shared-path tests** in 12.89 seconds, and **393
contract/evidence/balance/depth tests** in 14.49 seconds. The latter two
commands overlap on depth tests and must not be summed as a unique total.
These are selected regression runs, not a rerun of the entire historical suite.
The original sampled-forest regression also passed **140 tests in 573.76
seconds**, preserving its default homogeneous construction checks. Final Ruff
and `git diff --check` checks passed.

At that verification checkpoint, the approved 70/15/10/5 mixture was unchanged
and the new Count weights and diversity floors remained draft. The subsequent
approval is recorded above; it does not retroactively authorize generation.
No production dataset, seal or training run was created.
