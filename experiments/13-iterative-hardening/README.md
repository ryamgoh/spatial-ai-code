# Experiment 13 — iterative SFT task hardening (rough plan)

> **Status:** the v13 cardinal/diagonal foundation is implemented. No full v13
> dataset, training configuration, or result exists yet. This document remains
> intentionally provisional: make one structural change, measure it, and pause
> before choosing the next one.

The authoritative v13 meaning of worlds, questions, and special options is
[`docs/v13-semantic-contract.md`](../../docs/v13-semantic-contract.md). V6/v12
are frozen comparison suites, not semantic dependencies of v13.

## Implemented foundation

The first narrow implementation increment is complete:

- `spatial/spatial_solver_v13.py` independently implements the explicit v13
  semantic contract, parses cardinal/diagonal relations, and measures shortest
  X/Y proof paths. It does not subclass or import the v6 solver.
- `spatial/spatial_generation_v13.py` owns typed generation specs, scenes,
  subtype policies, constraint checking, and dataset construction.
- `spatial/generate_all_v13.py` is the CLI/compatibility adapter supporting
  `diagonal`, `cardinal`, and `mixed` relation modes across Type 0/1/2.
  Generated rows include `oracle_option`, solver-measured `difficulty`, and
  independent generator/schema version stamps.
- Every generated row is reparsed from its rendered user prompt; the
  generator's internal graph is not accepted as final gold.
- `experiments/tasks/utils.py::process_docs_v13_sft` is a versioned lm-eval
  adapter that preserves structural metadata. The v6 adapter is unchanged.
- `experiments/tasks/spatial_eval_v13_foundation.yaml` provides the first
  synthetic v13 evaluation task definition.
- Law, generator round-trip, loader, and existing v6 regression tests pass.

The next accepted increment is also implemented:

- all 12 consistent-world base semantic subtypes are available in every
  relation mode;
- subtype labels are inferred from the v13 solver's rendered-prompt verdict,
  not trusted from generator intent;
- batch generation crosses 3 relation modes × 12 base semantic subtypes;
- splitting is stratified within each generation cell; and
- an additional `mixed-dir-1-independent` cell requires the shortest X and Y
  proofs to share no supporting statement.

At the default 100 rows per cell, this produces 3,600 ordinary rows plus 100
independent-axis rows. This is a balanced construction interface, not yet the
final 1.5k/6k/18k experiment split recipe.

The current implementation supports solver-verified X/Y proof-depth ranges,
structured distractors, and global world-consistency checks. Any X/Y cycle
anywhere in the rendered premises invalidates the world before Type 0, Type 1,
or Type 2 query evaluation.

Foundation smoke command:

```bash
cd spatial
uv run --no-project --with typer python generate_all_v13.py \
  --out ../data/spatial_sft_v13_foundation.jsonl \
  --relation-modes diagonal,cardinal,mixed \
  --samples-per-cell 100 \
  --independent-mixed-dir1 100 \
  --test-split 0.2 \
  --seed 13
```

### Generation matrix

The generator exposes the two foundational dimensions independently:

| dimension | supported values |
|---|---|
| relation mode | `diagonal`, `cardinal`, `mixed` |
| base semantic subtype | `dir-1`, `dir-2`, `dir-undetermined`, `dir-incomplete`, `dir-omit`, `which-1`, `which-2`, `which-3`, `which-4`, `which-0`, `count-1`, `count-omit` |
| optional structural cell | `mixed-dir-1-independent` |
| proof depth | exact or ranged X/Y depths for independent `dir-1` in cardinal/mixed mode |
| distractor policy | exact-count `disconnected` or `query-branch` relations on depth-controlled `dir-1` |
| world consistency | globally consistent, or an X/Y/both-axis direct/indirect cycle with query-connected/disconnected placement |

Python callers should request any mixture as explicit `GenerationCell` values
passed to `generate_dataset`. The compatibility `batch_generate` adapter and
CLI expose the common cross-product selection through `--relation-modes` and
`--subtypes`. For example, generate only cardinal and mixed
`dir-2`/`count-omit` cells:

```bash
cd spatial
uv run --no-project --with typer python generate_all_v13.py \
  --out ../data/spatial_sft_v13_subset.jsonl \
  --relation-modes cardinal,mixed \
  --subtypes dir-2,count-omit \
  --samples-per-cell 100 \
  --independent-mixed-dir1 0 \
  --test-split 0.2 \
  --seed 13
```

Generation fails explicitly for an unknown dimension. Every emitted row is
accepted only after the solver reclassifies its rendered prompt as the
requested semantic subtype and relation mode. This means future experiments
can choose their own cell mixture without adding another generator entrypoint.

### Generation architecture

The extensible interface lives in `spatial/spatial_generation_v13.py`:

```text
GenerationSpec + seeded RNG
            │
            ▼
SpatialGenerator.generate
            │
            ├─ construct typed Scene / Relation values
            ├─ construct subtype-specific query and options
            ├─ render prompt and SFT trace
            └─ SpatialSolverV13.solve_and_analyze
                         │
                         ▼
              verified GeneratedExample

list[GenerationCell] ──► generate_dataset ──► stratified train/test JSONL
```

- `GenerationSpec` is the single description of one requested example.
- `StructuralConstraints` is where future proof-depth, distractor, and conflict
  requirements belong.
- `GenerationCell` gives an experiment cell a name, spec, and row count.
- `SpatialGenerator.generate` owns rejection sampling and raises
  `GenerationError` with per-reason rejection counts if the spec cannot be
  satisfied.
- `GeneratedExample.to_row` is the only conversion to JSON-compatible data and
  stamps `difficulty_schema_version`.
- `generate_dataset` owns cell-stratified splitting and JSONL output.
- `spatial/generate_all_v13.py` is intentionally only a CLI plus compatibility
  adapter for early v13 callers.
- `spatial/spatial_graph.py` holds stable graph primitives; v13 no longer
  imports implementation details from the v6 generator.
- `SpatialSolverV13.solve_and_analyze` returns one `SolvedProblem`, keeping the
  accepted grade and structural profile together.

New complexity should normally extend `GenerationSpec`/`StructuralConstraints`
and the solver profile rather than add more positional flags to
`generate_sample`.

### Trace contract

The rendered user prompt is the sole source of truth for v13 supervision.
`SolvedProblem` carries the entities and relations reparsed from that prompt,
and the assistant trace is rendered from those parsed facts rather than the
generator's internal scene. Therefore `Entities Detected` cannot contain an
entity that is absent from the question.

Every trace keeps the incremental Chain-of-State section:

```text
Sentence → X/Y extraction → updated X-State/Y-State
```

Its final deduction is query-focused and solver-backed:

- Type 0 shows target/reference, one shortest X proof, one shortest Y proof,
  each axis conclusion, and their compound-direction composition. Unknown or
  contradictory axes are stated explicitly instead of receiving invented
  paths.
- Type 1 shows the reference/direction, the required shortest axis paths for
  every proven entity, and the complete proven-entity set before evaluating
  the options.
- Type 2 shows the same proven-entity evidence and then the explicit derived
  count before evaluating the options.

All displayed paths use the state convention `lower < higher` (West→East or
South→North), including when the queried target lies West or South of the
reference.

### Proof-depth cells

Typed callers specify exact or ranged proof depths through `DepthRange`:

```python
GenerationSpec(
    semantic_subtype=SemanticSubtype.DIR_1,
    relation_mode=RelationMode.MIXED,
    constraints=StructuralConstraints(
        require_independent_axes=True,
        x_depth=DepthRange(3, 4),
        y_depth=DepthRange.exact(5),
    ),
    num_entities=11,
    num_relations=12,
)
```

The CLI accepts extra cells as `MODE:XxY:COUNT`; each axis may be an exact
integer or a `MIN-MAX` range:

```bash
cd spatial
uv run --no-project --with typer python generate_all_v13.py \
  --out ../data/spatial_sft_v13_depth.jsonl \
  --subtypes '' \
  --samples-per-cell 0 \
  --independent-mixed-dir1 0 \
  --depth-cells cardinal:1x1:100,mixed:2-3x3-4:100,mixed:6x8:100 \
  --test-split 0.2 \
  --seed 13
```

Depth cells currently require `dir-1`, independent axes, and cardinal or mixed
relations. The generator constructs disjoint cardinal proof skeletons and
places the mandatory mixed-mode diagonal relation away from the query paths.
The final rendered prompt is still measured by `SpatialSolverV13`; a candidate
is rejected if an accidental shortcut moves either shortest proof outside the
requested range.

### Structured-distractor cells

Typed callers add an exact distractor policy and count to the same structural
constraints:

```python
GenerationSpec(
    semantic_subtype=SemanticSubtype.DIR_1,
    relation_mode=RelationMode.MIXED,
    constraints=StructuralConstraints(
        require_independent_axes=True,
        x_depth=DepthRange.exact(3),
        y_depth=DepthRange.exact(4),
        distractors=DistractorSpec(
            policy=DistractorPolicy.QUERY_BRANCH,
            count=3,
        ),
    ),
    num_entities=10,
    num_relations=10,
)
```

The CLI accepts `MODE:XxY:POLICY:DISTRACTORS:COUNT`:

```bash
cd spatial
uv run --no-project --with typer python generate_all_v13.py \
  --out ../data/spatial_sft_v13_distractors.jsonl \
  --subtypes '' \
  --samples-per-cell 0 \
  --independent-mixed-dir1 0 \
  --distractor-cells \
    cardinal:3x4:disconnected:3:100,mixed:3x4:query-branch:3:100 \
  --test-split 0.2 \
  --seed 13
```

For these cells, a **relevant** relation is one used by the solver-selected
shortest X or Y proof. Every other relation is a distractor. The solver then
classifies each distractor from the rendered prompt's undirected relation
graph:

- `disconnected`: the relation is outside the component containing the query
  proof;
- `query-branch`: it is connected to that component but absent from both
  selected shortest proofs.

The requested distractor count is exact. Generation rejects a row if the
solver finds the wrong count/topology or if any distractor creates a shorter X
or Y proof. Distractor cells currently use exact rather than ranged proof
depths so their total relation count is explicit and auditable. New rows use
`generator_version=v13.3-structured-distractors` and
`difficulty_schema_version=2`.

### Global-consistency cells

V13 treats every premise as an objectively true statement about one world. A
cycle on either axis therefore makes the complete world unrealizable and
overrides every question family with `Cannot be determined`:

```text
parse all premises
    → detect a cycle anywhere on X or Y
        → inconsistent world: Cannot be determined
        → consistent world: evaluate the direction/which/count query
```

`CycleSpec` is the only v13 mechanism for requesting a cycle and is orthogonal
to the base semantic subtype:

```python
GenerationSpec(
    semantic_subtype=SemanticSubtype.WHICH_2,
    relation_mode=RelationMode.MIXED,
    cycle=CycleSpec(
        axes=CycleAxes.Y,
        topology=CycleTopology.INDIRECT,
        placement=CyclePlacement.DISCONNECTED,
        length=3,
    ),
    num_entities=11,
    num_relations=13,
)
```

The final solver subtype becomes `dir-cycle`, `which-cycle`, or `count-cycle`.
Top-level `base_semantic_subtype` preserves the consistent question constructed
before cycle injection. Solver metadata records `world_consistency`, cycle
axes, direct/indirect topology, query-connected/disconnected placement, cycle
lengths, statement indices, and one closed witness per cyclic axis.

The CLI format is
`BASE_SUBTYPE:MODE:AXES:TOPOLOGY:PLACEMENT:LENGTH:COUNT`:

```bash
cd spatial
uv run --no-project --with typer python generate_all_v13.py \
  --out ../data/spatial_sft_v13_cycles.jsonl \
  --subtypes '' \
  --samples-per-cell 0 \
  --independent-mixed-dir1 0 \
  --cycle-cells \
    dir-1:mixed:x:direct:query-connected:2:100,which-2:mixed:y:indirect:disconnected:3:100,count-1:mixed:both:indirect:query-connected:4:100 \
  --cycle-controls \
  --test-split 0.2 \
  --seed 13
```

Matched controls are enabled by default. Each control replaces the closed cycle
with an open chain while preserving the base question family, requested axis,
placement intent, relation count, and total entity budget. Invalid-world traces
show the closed witness under `### Consistency Check` and stop before local
query deduction. New rows use `generator_version=v13.4-global-consistency` and
`difficulty_schema_version=3`.

## Motivation

The current spatial task is close to saturated for Qwen3.5-4B after SFT. The
1.5k model is around 97% on the matched 2k test and the 6k model is around
96%. At that level, adding more examples from the same generator does not give
us a useful SFT scaling problem. It also leaves too little residual error to
study later training methods cleanly.

Experiment 13 is **not a GRPO experiment**. Its goal is to build a harder,
well-controlled SFT task where:

1. difficulty comes from spatial reasoning structure rather than label noise,
   hidden task rules, or arbitrary prompt length;
2. 1.5k and 6k SFT form a meaningful learning curve;
3. individual sources of difficulty can be measured separately; and
4. the existing v12 task and SpatialMap remain frozen regression checks.

The desired outcome is not a predetermined low aggregate score. A useful task
has an interpretable slope from easy to hard cases, improves with additional
SFT data, and preserves previously learned behavior.

## Main hypothesis

The current generator is easy because most ordinary relation sentences are
diagonal (for example, Northeast), so one sentence supplies an X-axis fact and
a Y-axis fact together. Queries are often direct or have short proofs, and the
special contradiction cases have recognizable construction patterns.

The primary v13 hypothesis is:

> SFT becomes meaningfully harder when the two axes must be reconstructed from
> separate facts over controlled multi-hop paths, especially in the presence
> of plausible but irrelevant graph structure.

We will test that hypothesis incrementally rather than adding every proposed
feature at once.

## Experimental controls

Unless a later decision explicitly changes them, keep these fixed across v13
iterations:

- Model: `Qwen/Qwen3.5-4B`.
- Training method: the same QLoRA/Axolotl recipe as Experiments 11 and 12.
- Epochs, LoRA rank, optimizer family, and effective batch size.
- Nested train sizes: **1.5k ⊂ 6k**. Run 18k only when the 1.5k to 6k
  comparison is informative.
- Strict exact letter-set accuracy as the primary answer metric.
- A separate validation split for checkpoint selection. Never select on test.
- Experiment 12's 2k test and SpatialMap v6 as frozen regression/transfer
  suites.
- Five-option question contract and the three Type-0 option rules from v12.

Keeping the training recipe fixed makes changes attributable mainly to the data
and reasoning structure. The option rules remain explicit because hiding an
unusual rule would create task ambiguity, not better spatial reasoning.

## General iteration protocol

For each iteration:

1. Add **one major complexity lever**.
2. Generate disjoint train, validation, and test splits.
3. Validate every gold answer with the symbolic solver.
4. Verify that generated examples actually satisfy their requested structural
   properties; reject accidental shortcuts.
5. Run the untuned 4B baseline, 1.5k SFT, and 6k SFT.
6. Report accuracy by structural bucket and macro-average the buckets.
7. Run all earlier frozen regression slices.
8. Stop, inspect errors, and decide whether to deepen, revise, or abandon the
   lever before implementing the next iteration.

Do not let a naturally frequent easy bucket dominate the headline. Overall
accuracy may still be included, but the primary result is the stratified table.

## Iteration 0 — measure the current task

**Change to task:** none.

Before modifying the generator, annotate or derive structural metadata for the
current v12 examples:

- question family and subtype;
- number of entities and relation statements;
- shortest X-axis proof depth for the queried conclusion;
- shortest Y-axis proof depth;
- whether either component is stated directly;
- whether the X and Y proofs use the same supporting statements/entities;
- number of statements relevant to at least one shortest proof;
- number of irrelevant statements;
- gold letter-set cardinality;
- contradiction type and affected axis; and
- whether the queried pair is the directly injected conflict pair.

### Questions

- Are most v12 questions direct or one/two-hop?
- Does entity count still predict errors after controlling for proof depth?
- Are there existing deep examples that can seed v13?
- Is the high overall result concealing a remaining structural pocket?

### Deliverable

A v12 structural census and accuracy report. No new training run is required
unless an existing response file is unavailable.

### Decision gate

- If current deep examples already form a useful difficulty curve, first build
  a balanced dataset around those structures.
- If all existing structural buckets are saturated, proceed to Iteration 1.

## Iteration 1 — cardinal one-axis relations

**New lever:** allow relation sentences that update only one axis.

**Implementation status:** accepted foundation complete. Diagonal-only,
cardinal-only, and mixed controls are available. The extra mixed hard cell
verifies that the queried `dir-1` answer actually uses independent X/Y
evidence; merely containing a cardinal sentence is not sufficient.

Examples:

```text
The Library is east of the Bank.     # X only
The School is north of the Museum.   # Y only
The Hospital is southwest of the Zoo. # X and Y
```

Keep map size, statement count, question families, option semantics, and
contradiction behavior otherwise close to v12. The purpose is to validate the
new representation boundary before introducing long proofs.

### Test buckets

- diagonal-only maps (compatibility control);
- cardinal-only maps;
- mixed cardinal/diagonal maps;
- direct one-axis queries; and
- compound directions assembled from separate cardinal facts.

### Decision gate

- If cardinal extraction itself fails badly, repair the representation or
  traces before increasing complexity.
- If all buckets remain saturated, proceed to independent-axis composition.
- If 6k clearly improves on 1.5k, retain the lever and examine the error types
  before choosing the next step.

## Iteration 2 — independent X/Y composition

**New lever:** force the two components of a compound direction to use
different evidence.

A qualifying example should usually satisfy all of the following:

- no relation directly states the queried compound direction;
- no single diagonal fact supplies both required components;
- the X proof and Y proof use different intermediate entities or edges; and
- the query cannot be answered from only one proof path.

Illustrative structure:

```text
X proof: Reference < A < B < Target
Y proof: Target < C < D < Reference
Conclusion: Target is Southeast of Reference
```

### Test buckets

- X and Y share their evidence;
- X and Y partially share evidence;
- X and Y use independent paths;
- one axis proven and one genuinely unknown; and
- both axes independently proven.

### Decision gate

There should be an interpretable gap between shared- and independent-evidence
cases. If both 1.5k and 6k fail uniformly, inspect trace correctness and SFT
coverage rather than immediately adding another lever.

## Iteration 3 — controlled minimum proof depth

**New lever:** construct queries with a verified minimum shortest-path depth.

**Implementation status:** exact and ranged X/Y depth cells are implemented
for independent-axis `dir-1` in cardinal and mixed modes. A feasibility stress
check generated 50 examples for each of cardinal/mixed × `1x1`, `2x3`, `4x5`,
and `6x8` with no failures or depth drift.

Initial bands:

| tier | shortest proof depth | role |
|---|---:|---|
| easy | 1 | direct-fact control |
| short | 2–3 | ordinary composition |
| medium | 4–5 | multi-hop SFT target |
| hard | 6–8 | long matched reasoning |

Depth must be measured independently for X and Y. The generator must reject an
example if an unintended direct edge or shorter alternate path exists. Merely
increasing the number of entities does not qualify as increasing proof depth.

### Primary report

| X depth | Y depth | baseline | SFT 1.5k | SFT 6k |
|---:|---:|---:|---:|---:|
| 1 | 1 | | | |
| 2–3 | 2–3 | | | |
| 4–5 | 4–5 | | | |
| 6–8 | 6–8 | | | |
| short | long | | | |
| long | short | | | |

### Decision gate

- A gradual decline with proof depth is desirable.
- If every band remains near ceiling, extend depth cautiously.
- If performance collapses abruptly, add intermediate coverage or inspect for
  accidental ambiguity/trace truncation.

## Iteration 4 — structured distractors

**New lever:** add graph structure that is irrelevant to the answer but
plausible enough to compete with the correct proof. Keep proof-depth buckets
fixed while measuring this lever.

**Implementation status:** `disconnected` and `query-branch` policies are
implemented for exact-depth independent-axis `dir-1` cells in cardinal and
mixed modes. The solver reports relevant/distractor statement indices and
counts for both topologies. Random filler remains outside this controlled
cell type.

Candidate distractors, introduced in small groups:

- disconnected irrelevant components;
- branches touching the target but never reaching the reference;
- branches touching the reference but ending at the wrong entity;
- a valid-looking X path paired with an irrelevant Y path;
- redundant facts that restate an implied relation; and
- nearly complete alternative paths missing one required edge.

Track total distractor statements, connected distractors, target/reference
contact, and the relevant-to-irrelevant ratio. Avoid arbitrary prose filler: it
mainly tests context length rather than graph reasoning.

### Decision gate

Compare equal-depth examples with and without distractors. Connected plausible
distractors should be more difficult than disconnected noise; otherwise the
construction is not testing the intended capability.

## Iteration 5 — contradiction relevance and scope

**New lever:** replace the recognizable direct-reversal pattern with varied
conflict structures.

**Decision:** all premises describe one objectively true world, so any cycle
on either axis invalidates the complete problem. Placement remains an
evaluation dimension measuring detection difficulty; it never changes the
gold. This applies equally to direction, which-object, and count questions.

**Implementation status:** global X/Y cycle detection, closed witnesses,
direct/indirect topology, query-connected/disconnected placement, all three
question families, and matched acyclic open-chain controls are implemented.

Possible ordered additions:

1. direct relevant reversal;
2. indirect relevant cycle such as `A < B < C < A`;
3. contradiction on one axis while the other axis remains valid;
4. irrelevant cycle in a disconnected component;
5. irrelevant cycle connected to one query entity; and
6. multiple cycles where only one can affect the query.

The intended rule is explicitly "any contradiction anywhere means Cannot be
determined." Matched open-chain controls prevent shortcutting on cycle length,
placement, prompt size, or recurring entities.

### Test buckets

- relevant versus irrelevant conflict;
- direct versus indirect conflict;
- X-only, Y-only, and both-axis conflict; and
- conflict depth/distance from the queried pair.

## Iteration 6 — structural generalization holdouts

**New lever:** hold out combinations or depths, rather than adding another
training feature.

Maintain three distinct evaluation regimes:

1. **Matched:** unseen maps with the same structures and depth distribution as
   training.
2. **Depth extrapolation:** for example, train through depth 5 and test on
   depths 6–8.
3. **Compositional holdout:** show individual phenomena during training but
   reserve their combination for test.

Example compositional holdout:

- training contains long paths;
- training contains independent-axis paths;
- training contains irrelevant cycles;
- training never combines all three; and
- test contains long independent-axis paths plus an irrelevant cycle.

Define and freeze these test strata before running the model. Do not build a
test set by mining the current model's failures.

## Iteration 7 — controlled language variation (optional)

**New lever:** controlled paraphrase families for atomic relations. This comes
after structural difficulty is calibrated so parsing failures are not confused
with reasoning failures.

```text
A is east of B.
A lies to B's east.
Relative to B, A is positioned eastward.
B has A on its eastern side.
```

Use some paraphrase families in training and reserve others for test. Always
retain a canonical-language slice so structural reasoning and linguistic
generalization can be reported separately.

## Cumulative evaluation suite

Each accepted iteration adds a frozen slice rather than replacing earlier
tests:

```text
v12 matched test              existing semantics and easy-task retention
SpatialMap v6                 external-distribution transfer
v13 cardinal                 one-axis relation extraction
v13 independent axes         separate X/Y composition
v13 depth                    controlled shortest proofs
v13 distractors              relevance filtering
v13 contradictions           inconsistency scope
v13 structural holdouts      depth and compositional generalization
```

## Result format

Every iteration should record at least:

| cell | matched macro | hardest new bucket | v12 regression | SpatialMap |
|---|---:|---:|---:|---:|
| untuned 4B | | | | |
| 4B SFT 1.5k | | | | |
| 4B SFT 6k | | | | |
| 4B SFT 18k (optional) | | | | |

Also include the complete per-bucket table and paired 1.5k-versus-6k errors. A
one-point aggregate difference is not meaningful unless we know which
structures changed and whether the two models fail on the same rows.

## Interpreting an iteration

| observation | likely interpretation | next action |
|---|---|---|
| high accuracy at all levels | task still saturated | deepen the same lever before adding another |
| low accuracy at all levels; 1.5k ≈ 6k | representation, specification, or coverage problem | inspect examples/traces and simplify |
| 6k clearly improves over 1.5k | useful SFT scaling regime | retain lever; analyze residuals |
| new slice improves but old slices regress | forgetting or mix imbalance | repair the mix before proceeding |
| only prompt length predicts errors | accidental attention benchmark | redesign structure, not just size |
| gradual degradation by depth/structure | intended controlled difficulty | freeze the slice and consider next lever |

Rough calibration targets, not acceptance requirements:

- old/easy regressions remain at roughly 95% or better;
- medium matched buckets land around 75–90%;
- hard matched buckets land around 50–75%;
- extrapolation may reasonably land around 30–60%; and
- 6k shows a repeatable gain over 1.5k on at least some hard buckets.

## Out of scope for v13 initially

- GRPO, PPO, or any reward-model training.
- Lowering model size merely to reduce accuracy.
- Label noise or deliberately ambiguous gold.
- Hiding the task's option semantics.
- Increasing entity count without controlling proof structure.
- Dynamic movement/state-update tasks. Those may be a later experiment after
  the static controlled-hardness ladder is understood.
- Implementing all iterations in one generator revision.

## Immediate next step

Generate one compact diagnostic suite spanning the implemented dimensions
before adding another mechanism: proof depth, distractor topology, and global
cycles with matched open-chain controls across all three question families.
Evaluate the untuned model and current v12 SFT adapter first; use their
bucket-level failure curves to choose the final 1.5k/6k training mixture and
structural holdouts.
