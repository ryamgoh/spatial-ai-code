# SpatialEntail Composition v1

New experiment for atomic spatial reasoning, verified evidence, and matched
Unique Answer, Listed Possibilities, and Exhaustive Possibilities contracts.

## Working Documents

- [Main pivot](pivot_20261026.md): source of truth for research decisions.
- [Evidence examples](research-analysis/PIVOT-EVIDENCE.md): worked examples and
  validation requirements supporting the pivot.
- [Literature note](research-analysis/PIVOT-LITERATURE.md): primary-source
  findings and dated recommendations, not a second experiment specification.
- [Generation freeze](GENERATION-FREEZE.md): fixed random-plus-targeted budget,
  admission rules and split policy for the prior 200-story release.
- [Depth and gold specification](DEPTH-AND-GOLD-SPEC.md): amended 1-8 training
  and 9-10 benchmark scope, proposed measurement rules, and pre-generation gates.
- [Production integration](PRODUCTION-INTEGRATION-v1.md): candidate runner,
  approved candidate settings, verification boundaries and remaining source-family work.
- [Production allocation](PRODUCTION-ALLOCATION-v1.md): fixed source schedule,
  development exclusions, family assignment and executable-settings review.
- [CPU execution](CPU-EXECUTION-v1.md): question-local solver reuse, ordered
  multiprocessing, checkpoints, development timing and the Slurm normal launcher.
- [Depth feasibility protocol](research-analysis/DEPTH-FEASIBILITY-PROTOCOL.md):
  fixed 94-story / 282-query pilot, recorded before execution.
- [Depth feasibility verdict](research-analysis/DEPTH-FEASIBILITY-VERDICT.md):
  completed measurements, token blocker, gold gaps and required next work.
- [Distribution audit](research-analysis/DISTRIBUTION-AND-DEPTH-AUDIT.md):
  measured answer imbalance and depth gaps in the prior release.
- [Training runbook](TRAINING-RUNBOOK.md): prior preparation counts and commands,
  on hold pending the amended dataset and user approval.

## Current Status

The pivot now requires a depth-controlled replacement: train/dev/in-range test
at depths 1-8, plus a separate extrapolation benchmark at depths 9-10.
The depth measurer, readable paired explanations and all-direction controlled
Selection checks are implemented. The first pilot's 35 Direction token failures
are historical; subsequent compact/readable replays addressed that blocker.
See the [all-direction results](research-analysis/DIRECTION-COVERAGE-RESULTS-v2.md)
and [question-led prose results](research-analysis/PROSE-ORGANIZATION-RESULTS-v5.md).

The new [`composition_production.py`](../../spatial/v2/composition_production.py)
assembles existing constructors, verified admission, quota accounting and paired
candidate exports. Its [configuration](production-config-v1.json) has approved
numeric candidate settings, but is not yet an executable production freeze. The new
[`sampled-forest constructor`](../../spatial/v2/composition_scene_sampler.py)
varies core graph structure and preserves related variants under one scaffold
family ID. The new allocation compiler assembles explicit source batches,
development exclusions and structural supply checks without solver admission.
Its compiled configuration is separate from the numeric policy. Structural
preflight passed with 31,251 query attempts after collision exclusions, with
enough scheduled supply for every required quota and no scene-bound failures.
The executable settings were subsequently approved and the serial candidate
was launched, then stopped by the user after 554 completed questions because
throughput was inadequate. Its partial output is preserved; this is not a
completed solver-verified candidate dataset. The new CPU executor supports
bounded solver-result reuse and 1-128 worker processes with ordered quota
selection and hash-pinned checkpoints. No replacement production run or Slurm
job has been launched. See the CPU execution note before restarting.
The forest source now also supports mixed-relation chains and separately
necessary axis paths, with measured depth and shared lineage identities.
Controlled Count batches must explicitly request their match count; retained
quotas use solver-verified values. Same-menu contract differences are reported
separately from full/omission answer changes. The Count weights and stronger
diversity floors received subsequent approval on 2026-10-30, recorded separately
from the earlier numeric approval.
The runner now enforces explicit depth-family floors, retains bounded matched
full/omission menu pairs, and checks all eight directions per split/query type
without equal-frequency or per-depth direction quotas. The source mixture,
50/10/20/50 family floors and 20% omission chance are approved. Retained source
quotas enforce the mixture rather than allowing excess from another source to
fill it. The new
[`sampled-targeted constructor`](../../spatial/v2/composition_targeted_sampler.py)
adds ambiguity, contradiction and fixed-membership patterns on varying forest
cores. It shares family identities with the controlled generator, so related
versions cannot cross splits under different generator names. All 45 legacy
targeted family IDs remain development-used. The allocation compiler partitions
a bounded fresh scaffold pool and quarantines structural collisions without
reassignment or refill.
The separate three-contract SpatialMap-TQA-Corr adapter is also still pending.
Report feasibility and proposed production settings before generating a new
training/benchmark set, then report the production audit before final sealing.
Do not launch training from an earlier package as though it meets this design.

## Prior Preparation

`study-candidate-v2/` contains 200 premise sets, 600 questions and 2,652
contract/menu examples per style, with all 33 semantic cells in every split.
`prepared-training-v2/` contains matched QLoRA datasets/configs, pinned-token
measurements and an 88-pair agent audit. Its `seal.json` binds that prior
release only; it does not certify the amended depth/coverage requirements.
No training or GPU inference has been launched.

Preparation v2 simplifies evaluation scoring without changing scores, examples,
tokenization or training settings. The original v1 package and archive remain
unchanged; use their archived source to reproduce v1, and v2 with current source.
The original paired audit is reused only after verifying identical data hashes.

Prior train/dev/test contain 1,596/522/534 examples per style.
The earlier smoke packs and pilot below are historical feasibility milestones,
not alternative training releases. `study-candidate-v1/` failed split coverage
and remains unsealed. External SpatialMap contract views are not included in
this synthetic preparation, and legacy Corr SINGLE remains unchanged.

## First Implemented Path

The separately versioned
[`composition_contracts.py`](../../spatial/v2/composition_contracts.py) provides
full-domain solver analysis, independent coordinate-witness checks, A-F mapping,
all nine canonical question templates, full-prompt parsing, and strict answer
serialization. Its mapping is tested against all fourteen pivot fixtures and
their nineteen menu views. It does not change historical grading or Corr SINGLE.

[`composition_direction.py`](../../spatial/v2/composition_direction.py) reuses the
existing axis proof builder and independent proof checker to construct matched
Natural/Symbolic references for consistent, provable unique Direction cases.
Both styles expose exactly the same proof steps, including any orientation
rewrites. Solver results, witnesses, premise provenance, and proofs remain in
audit metadata. This is a first feasibility renderer, not the final compact
supervision format or a token-savings result.

The [Direction smoke pack](direction-smoke-v1.json) contains one D1 story,
six contract/menu views, and both reference styles per view. It is not six
independent examples, a train/dev/test split, or a passed dataset release.
Regenerate it from the repository root:

```bash
uv --system-certs run --python 3.12 --no-project \
  --with 'z3-solver>=4.15,<5' \
  python -m spatial.v2.composition_direction \
  --output experiments/spatialentail-composition-v1/direction-smoke-v1.json
```

Current serialization is ascending, undelimited uppercase letters (`AB`), or
`E`/`F` alone. Unique permits only one letter. The Stage-1 parser requires
exactly one terminal `Answer: <letters>` line; the Stage-2 parser accepts only
the answer, with surrounding whitespace ignored. The current
[`composition_evaluation.py`](../../spatial/v2/composition_evaluation.py)
connects it to two-stage evaluation and structured decoding, with local helper
tests passed and GPU validation still required.

## Expanded Evidence Path

Use [`composition_evidence.py`](../../spatial/v2/composition_evidence.py) for
the expanded path. It reuses shared Direction and Selection certificate
builders/checkers and provides matched renderings for all fourteen fixtures:
direction ambiguity, unique/empty/multiple/contingent Selection, invariant and
joint Count, strict-cycle conflicts, and forced coincidence.

The [expanded evidence pack](evidence-smoke-v1.json) contains 57 contract/menu
views with 114 reference renderings, over 12 distinct stories and 11 families.
Count/Selection views of identical premises share story IDs, and I1/I2 share a
family ID. Premises remain the hand-authored fixture inputs; evidence and
explanations are generated. No random story generator or split is implied.

```bash
uv --system-certs run --python 3.12 --no-project \
  --with pytest --with 'z3-solver>=4.15,<5' \
  python -m spatial.v2.tests.build_composition_pack \
  --output experiments/spatialentail-composition-v1/evidence-smoke-v1.json
```

The new [`composition_conflicts.py`](../../spatial/v2/composition_conflicts.py)
constructs replayable axis paths with premise provenance. Both coordinate
equalities are required to certify forced coincidence. Z3 separately rechecks
the cited conflicting subset. Neither path construction nor the subset check
claims minimum-size conflicts; shared direction-sign definitions are reused.

Count evidence currently supports at most eight eligible objects. It proves
fixed memberships and pairwise incompatibilities, then checks every assignment
for each excluded count against those constraints. Possible counts retain
independently checked whole-map witnesses. This supports C1 without summing
independent possibilities, but is not complete for higher-order correlations.
If an excluded count has any assignment not ruled out by the supported
evidence, the example is rejected explicitly.

Evidence selection is matched per contract: Unique ambiguity uses differing
valid maps; Listed covers displayed values; ordinary Exhaustive covers the
full domain; Exhaustive F for missing coverage uses one unlisted witness.
No-match/multiple-certain Selection uses its actual membership justification,
not fabricated ambiguity. Audit records retain full certificates and semantic
sets even when supervised evidence is shorter.

At this feasibility milestone, the pack did not establish production readiness.
The later study candidate adds a completed category/cell audit, compact Unique
Direction selection, pinned-token measurements and matched training preparation.
The fixture pack itself remains unchanged and is not a full release receipt or
verification of arbitrary model-written explanations.

## Generated Story Pilot

[`composition_pilot.py`](../../spatial/v2/composition_pilot.py) constructs a
bounded pilot without importing the fourteen fixtures. Nine parameterized
recipes vary chains, forks, memberships, candidate counts, orientation,
premise order, disconnected distractors, and consistency controls. Recipes
are construction labels, not a frozen evidence-category taxonomy.

```bash
uv --system-certs run --python 3.12 --no-project \
  --with 'z3-solver>=4.15,<5' \
  python -m spatial.v2.composition_pilot \
  --output experiments/spatialentail-composition-v1/generator-pilot-v1
```

The fixed default attempts 50 stories, with Direction, Selection, and Count
queries for each. Families are assigned before queries are accepted or menu
views rendered. Query-independent canonical grouping merges renamed,
reordered, and inverse-oriented duplicate stories; matched consistency
variants stay together. Coarse canonical fallback may conservatively merge
different stories. Rotated structures are not claimed to be deduplicated.

Each accepted query passes full checked-evidence construction, prompt
round trips, rename/order semantics, A-D permutation checks, and individual
plus joint removal checks for the designated distractor set. Direction
supporting x/y path lengths are recorded when available, not claimed as
minimum reasoning difficulty; other queries have no assigned chain length.
Reference-backend comparisons cover Direction problems with at most six
objects and share domain/sign definitions with Z3. Selection/Count enumeration
was too slow for this bounded check; the report states the actual coverage.

Outputs in `generator-pilot-v1/`:

- `pilot.json`: bases, full evidence, paired views, split assignments, and all
  attempts, including quarantined unsupported or failed problems.
- `model-inputs.jsonl`: prompt-only messages and IDs, with no answers,
  references, or evidence. This is not an Axolotl training export.
- `report.md`: coverage, split gaps, answer frequencies, and rejections.
- `audit-samples.md`: pending paired-review samples.
- `receipt.json`: source/output hashes, solver version, seed, and audit status.

The pilot is **not a release**. Its historical token and audit gaps are addressed
by the later study candidate, not by relabeling the pilot. Do not infer a
balanced training distribution from menu pairing, or scale it automatically. See the
[pilot inspection note](research-analysis/GENERATOR-PILOT-INSPECTION.md)
for the scope of the initial spot-check.

## Execution commitment

The research direction is settled. Implement the pivot rather than reopening
project ideation. Changes to research scope require explicit user approval;
correctness failures must still be surfaced and fixed, never hidden to preserve
the schedule. This section is an execution guardrail, not a second specification.

- Deliver the verified atomic spatial-composition dataset first, covering
  Direction, Selection, and Count across all three answer contracts.
- Use the six-option interface and matched Natural/Symbolic explanations from
  shared verified evidence. Freeze remaining formats and sampling settings
  after feasibility checks and before training.
- Keep story/family splits intact. SpatialMap is external evaluation only;
  preserve Corr SINGLE and version new contract views separately.
- Train two matched mixed-contract adapters with one shared initial seed.
  Evaluate both and the style-matched untuned baselines. Select checkpoints
  using synthetic Stage-1 development accuracy; report both evaluation stages.
- Do not add Boolean-proof baselines, an answer-only arm, broad generalization
  campaigns, extra model sweeps, or automatic data/seed expansion.
- Reuse shared infrastructure and make targeted, regression-checked changes.
  A repository-wide rewrite or solver replacement is not a study deliverable.
- Follow the pivot's three-week boundary and protect the writing week.
  Negative or mixed training results do not invalidate the dataset contribution.

The next work is resolving the measured depth/gold feasibility blockers,
not production generation or GPU training. Follow the specification's user
checkpoints before production and final sealing. GPU preflight comes after an
approved replacement release; the earlier fixtures and seals do not establish
readiness for the amended design.

## Boundaries

Keep historical results and receipts in
[`spatialentail-randomized-v2`](../spatialentail-randomized-v2/).
Reuse shared solver, tests, and training infrastructure in their existing
locations. The evidence fixtures remain in
[`test_pivot_evidence.py`](../../spatial/v2/tests/test_pivot_evidence.py).

Train, develop, and test on our own story-grouped synthetic splits. SpatialMap
contract variants are additional external evaluation only, never training or
checkpoint-selection data.
