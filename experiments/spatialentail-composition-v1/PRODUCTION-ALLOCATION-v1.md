# Production Allocation v1

This is an executable-settings review, not a labeled dataset, production seal or
training authorization. The approved target sizes, source mixture, depth weights,
family floors, omission probability and diversity safeguards are unchanged.

## Completed Allocation

The fixed compilation passed structural preflight on 2026-10-30. It removed
430 scene slots, representing 1,149 query attempts, because their families
participated in development-scene or cross-split scene collisions. No slots
were replaced and no accepted-base target was reduced.

| Role | Remaining Query Attempts | Unchanged Retained Target |
|---|---:|---:|
| Train | 17,394 | 6,000 |
| Dev | 3,414 | 1,200 |
| In-range test | 6,843 | 2,400 |
| Extrapolation test | 3,600 | 1,200 |
| Total | 31,251 | 10,800 |

There are 18,290 remaining scheduled scene slots. All structural supply checks
pass: overall and source quotas, depth-family floors, requested Count-value
quotas and nonhomogeneous-proof floors have sufficient scheduled supply.
Every scene is within the declared size bounds. These checks do not deduplicate
canonical queries or predict subsequent solver/evidence/token admission.

The 4,096-seed scaffold pool contained 4,000 canonical families; two were
development-used. The remaining 3,998 families were partitioned into pools of
1,863 train, 245 dev, 490 in-range and 1,400 extrapolation families before scene
construction. Pool counts are not final accepted family counts; some families
are unused by the fixed schedule and some are subsequently quarantined.

The development inventory covers 30 saved files, 368 recorded/declared family
IDs and 3,218 canonical scene keys, including 2,462 reconstructed sampler fixture
scenes before canonical deduplication. All four manifest-listed output hashes passed independent
verification. The compiled configuration's planning check reports just one
remaining execution gate: approval/freeze of this concrete schedule.

## Fixed Attempt Schedule

The compiler schedules three attempts per requested retained base, including
three per requested controlled Count value. This implements the earlier proposed
32,400-attempt ceiling; it is not a forecast of how many examples will pass.

| Role | Retained Base Target | Raw Query Attempt Slots |
|---|---:|---:|
| Train | 6,000 | 18,000 |
| Dev | 1,200 | 3,600 |
| In-range test | 2,400 | 7,200 |
| Extrapolation test | 1,200 | 3,600 |
| Total | 10,800 | 32,400 |

Direction and Selection share scheduled one-match controlled scenes. Count has
explicit separate allocations for values 0-4. Semantic scenes supply all three
query kinds. This gives 18,720 raw scheduled scenes, not 32,400 independent
stories. Related scene variants also share their original family identities.

At depth 1, attempts cycle through the eight homogeneous directions. At depths
2-10, a fixed 20-slot cycle contains eight homogeneous, eight separate-axis and
four diagonal mixed-chain constructions. Thus approximately 60% of those
attempts are nonhomogeneous, providing supply for the approved retained minimum
of 30%; actual retained coverage still needs verification.

The targeted attempt weights are empty 3, multiple 3, contingent 2, singleton 1,
consistent-cycle 1, inconsistent-cycle 1 and coincidence 1. They apply only
inside the fixed targeted attempt budget, with largest-remainder rounding.
They are concrete execution settings for review, not new retained semantic
quotas or changes to the approved 5% targeted share. Random-world and random-atom
attempts retain their existing unconstrained sampling procedures.

## Family Assignment

A bounded pool contains 4,096 scaffold seeds, starting at 2026110200. Canonical
anchor family IDs group colliding seeds; the lowest seed represents each group.
Development-used families are removed before allocation. Remaining families are
ordered by a fixed hash of the existing production seed and family identity.
Role pool sizes are proportional to the maximum required controlled attempt
count per depth cell, rounded by largest remainder.

Every family is assigned once before constructing the requested scene variants
or computing any answer. Short/long, direction, proof and Count variants reuse
that assignment. Sampled-targeted scenes use those same role pools, preserving
cross-generator family isolation. The allocator does not search for seeds with
desired gold answers or retry when a source pool is insufficient.

Canonical full-scene checks follow construction. A development-story collision
or a scene shared across split roles quarantines every participating family,
including its other variants. Quarantine propagates through same-split copies
as well. Nothing is reassigned or replenished. Excluded slots and reasons are
saved; the compiled executable budget is the remaining number of query
attempts. All retained quotas and coverage requirements stay unchanged.

## Development Exclusions

The inventory reads original pilot data, saved attempts, manifests, smoke packs
and explanation previews under this experiment. It includes rejected attempts,
not just released examples. It collects recorded family/story identities and
recomputes canonical keys from saved atomic premises, independently of labels.
Source files are hash-pinned.

The inventory additionally includes the bounded software fixtures: sampled
scaffold seeds 0-31, legacy recipe samples 0-4, legacy depth samples 0-5, and
the two small random-source reproducibility fixtures. Unsaved sampler scenes
are reconstructed from a conservative bounded superset of the existing test
parameters, including context variants, and their canonical scene hashes are
also excluded. This covers shallow scene collisions under a different family
anchor, not just reuse of a known family ID. No new families are introduced by
the reconstruction. Tests and their versions are recorded. These exclusions
cover the declared development artifacts and lineages; the canonicalizer is not
a proof of equivalence for arbitrary graphs, rotations or semantically
equivalent premises.

## Coverage Policy

- Literal gold strings: at least one distinct base per feasible string per
  contract in train/dev/in-range test, using the existing mandatory rules.
  This is coverage, not equal label frequencies.
- Substantive primary answers: at least 1,400 train, 280 dev, 560 in-range and
  400 extrapolation bases per query and contract. These floors are already
  implied by the approved controlled quotas, not additional dataset size.
- Semantic cases, all three pairwise contract differences, Count values,
  nonhomogeneous proofs, depth-family counts and direction presence retain
  their approved checks.
- No new E/F proportion, per-depth direction balance, distractor quota or
  rejection-rate target is introduced.

The structural supply report checks available attempts and families, requested
Count values, nonhomogeneous constructions and scene bounds. These are necessary
conditions only. Actual semantics, printed depth, evidence support, readability,
tokens, literal gold coverage and retained distributions require the subsequent
solver-verified candidate run.

## Artifacts and Commands

The numeric policy remains [production-config-v1.json](production-config-v1.json).
The compiled executable settings and receipts live separately:

- [Compiled configuration](production-allocation-v1/config.json)
- [Development exclusions](production-allocation-v1/development-exclusions.json)
- [Family allocation](production-allocation-v1/family-allocation.json)
- [Structural supply report](production-allocation-v1/allocation-report.json)
- [Integrity manifest](production-allocation-v1/manifest.json)

Compile into a fresh directory, without solver admission:

```sh
python -m spatial.v2.composition_allocation \
  --config experiments/spatialentail-composition-v1/production-config-v1.json \
  --output experiments/spatialentail-composition-v1/production-allocation-v1
```

Check the compiled schedule without authorizing generation:

```sh
python -m spatial.v2.composition_production \
  --config experiments/spatialentail-composition-v1/production-allocation-v1/config.json \
  --preflight
```

The original compiled configuration remains `numeric-approved`, with
`production_freeze=false` and `training_authorized=false`, preserving the
allocation receipt. The user subsequently approved executing the generation
plan. A separate `execution-config-v1.json` changes only status and approval
metadata, freezing the identical 31,251-attempt settings for one unsealed run.
Distribution review and manual explanation audit still precede any final seal.
The separate three-contract SpatialMap-TQA-Corr adapter remains unfinished.

## Execution Launch

The approved command was launched as background job
`job-f3439dbd4b3343a2a54fca482b827e44`, targeting
`../production-candidate-v1/`. The launch was recorded at
`2026-10-29T18:42:38Z` (UTC; the local approval date is 2026-10-30).
The execution configuration SHA-256 is
`727c28c68fc237c2088cfb057c6e6150beaa7a0cf6d86e7c90a0e15024fc5471`.
All original allocation source and artifact hashes matched before launch, and
the execution planning check reported no blockers. No sampler, solver or
admission code was changed for this launch.

The process begins by reconstructing and checking the fixed scene plan before
creating the candidate output directory. It then loads the pinned local
Qwen tokenizer and performs Z3 admission, streaming each attempted outcome.
Results are pending. A nonzero exit or completed exports do not establish
candidate acceptance; final coverage, integrity and manual audits must still
be checked. No automatic refill, final sealing or training is authorized.

## Verification

The final allocation/production regression passed **125 tests in 25.22
seconds**, including 31 allocation tests. These cover fixed seed pools,
cross-split collision quarantine and propagation, exact unsaved fixture
reconstruction, preserved schedule IDs, Count and family supply accounting,
input hashes, and preflight without solver admission or execution approval.

A separate focused run passed **118 tests in 21.77 seconds**, covering the
targeted sampler and the affected schedule, Count-cap and diversity checks.
The complete historical test suite was not rerun. Ruff and whitespace checks
also passed.

The first structural compilation was stopped before it wrote artifacts to
include unsaved fixture scene hashes as well as family IDs. The replacement
uses the same fixed seed pool and raw attempt slots. Neither run evaluates
solver golds or tunes the schedule against held-out answer frequencies.
