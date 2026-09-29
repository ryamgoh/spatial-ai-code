# Spatial solver V2 semantic contract

V2 is an experimental exact-direction solver. It does not replace or change
the V13 dataset contract.

## Module seams

V2 separates spatial reasoning from dataset concerns:

```text
dataset row or synthetic spec
    -> adapter
    -> SpatialProblem
    -> SpatialSolverV2
    -> QueryAnalysis
    -> grading, structured explanation, audit reporting, or map rendering
```

- `spatial_solver_v2.py` owns the spatial theory, model search, and coordinate
  witnesses. It accepts structured problems only.
- `spatial_explanations_v2.py` classifies individual claims, extracts readable
  axis proofs where possible, and records typed evidence without parsing text.
- `spatial_explanation_renderers_v2.py` renders that evidence as either a
  natural-language trace or a symbolic axis-chain trace, with an independent
  state-update schedule.
- `spatial_audit_rendering_v2.py` is the only renderer that formats coordinate
  witnesses.
- `spatial_text_v2.py` adapts the current prompt grammar into a
  `SpatialProblem` and returns options separately.
- `spatial_grading_v2.py` resolves answer semantics, encodes menu selections,
  and scores predicted letter sets against an existing query analysis.
- `spatialeval_adapter_v2.py` converts original SpatialEval rows, preserves
  their oracle metadata, and classifies the oracle against the model set.

The core solver has no knowledge of JSONL rows, chat messages, prompt wording,
option letters, oracle labels, or dataset-specific answer semantics. A future
synthetic generator and the implemented SpatialEval adapter both produce the same
`SpatialProblem` interface.

`SpatialProblem.query` selects `DirectionQuery`, `WhichQuery`, or `CountQuery`.
Each query carries its own candidates. A direction query defaults to all eight
exact directions, while Which/Count carry a set of exact directions describing
the requested spatial region. This is structured query input, not a dataset
mode inside the solver.

The implemented SpatialEval adapter reports `exact-match`,
`underdetermined-oracle-possible`, `oracle-underinclusive`,
`oracle-overinclusive`, `oracle-contradicted`, `partial-overlap`,
`inconsistent`, or `error`. Its
`SpatialEvalAudit.analysis.witnesses` provides the maps needed to substantiate
each possible answer. It never changes the original oracle or silently selects
the witness that happens to match it.

SpatialEval preparation is ordinal-only. The adapter accepts exact Northeast,
Northwest, Southeast, and Southwest premises, rejects cardinal/coarse/negated
premises, and sets those four directions as candidates for DirectionQuery.
SpatialEval Which/Count questions may use all eight compass labels. Cardinal
Which/Count labels are exact cardinal predicates; the solver remains compass-8
and permits axis equality in their witness maps.

Conceptually:

```python
SpatialProblem(
    objects=objects,
    premise=And(parsed_constraints),
    query=DirectionQuery(
        target=target,
        reference=reference,
        candidate_directions=frozenset({NE, NW, SE, SW}),
    ),
)
```

A future generic synthetic preparer can omit a DirectionQuery candidate set and
receive the all-eight default.

### File-level use cases

- Use `spatial_solver_v2.py` when the caller already has structured objects,
  formulas, and a query.
- Use `spatial_explanations_v2.py` after constructing a `SpatialProblem` when a
  generator or audit needs machine-readable evidence.
- Use `spatial_explanation_renderers_v2.py` to choose a training-trace ablation
  without exposing coordinates.
- Use `spatial_audit_rendering_v2.py` only for coordinate-bearing diagnostics.
- Use `spatial_text_v2.py` only to adapt the current rendered prompt grammar.
- Use `spatial_grading_v2.py` after semantic analysis to resolve answer
  semantics, encode an option menu, or score a model response.
- Use `spatialeval_adapter_v2.py` to preserve original SpatialEval oracle
  metadata while comparing it with the solver-derived model set.
- Use `test_spatial_solver_v2.py` as the executable semantic contract before
  changing the solver or building proof rendering.
- Use `test_spatial_explanations_v2.py` as the executable claim-evidence and
  explanation contract.

## Motivation: text must determine its own answer

The failure mode this model is intended to prevent is using a hidden grid as
the source of truth for a text-only question. A generator can construct a
complete grid, observe that `A` is northeast of `B`, and then render premises
that reveal only that `A` is northward of `B`. The hidden grid has one answer,
but the visible text permits three:

```text
hidden grid:       A is Northeast of B
visible evidence:  A is Northward of B
text possibilities: Northwest, North, Northeast
```

An image can expose the missing X relationship. A text-only example cannot use
that hidden information as its gold label. V2 therefore treats the rendered
premises, rather than generator coordinates, as authoritative.

## World model

For every pair of distinct objects, the X relationship is exactly one of
`West`, `Equal`, or `East`, and the Y relationship is exactly one of `South`,
`Equal`, or `North`. Two distinct objects may share one coordinate but may not
share both. The eight remaining products are the exact compass directions:

| Direction | X | Y |
|---|---|---|
| North | Equal | North |
| Northeast | East | North |
| East | East | Equal |
| Southeast | East | South |
| South | Equal | South |
| Southwest | West | South |
| West | West | Equal |
| Northwest | West | North |

Axis equality is transitive. Strict comparisons are transitive after equality
classes are merged. A strict cycle, a strict comparison inside an equality
class, or equality on both axes makes the premise set inconsistent.

## Premises

`RelationConstraint` is the atomic spatial proposition. `Not`, `And`, `Or`,
`Implies`, and `Iff` form a propositionally complete structured formula AST.
The text adapter maps the eight compass words to atomic constraints and combines
all rendered statements with `And`. It additionally accepts `Northward`,
`Eastward`, `Southward`, and `Westward` as coarse atomic domains. For example:

```text
East      = {East}
Eastward  = {Northeast, East, Southeast}
not East  = every exact direction except East
not NE    = every exact direction except Northeast
```

Negation is set complement over the eight exact directions. Missing evidence is
not negation. Both `A is not Northeast of B` and the generator-style
`A is not to the Northeast of B` are accepted.

Structured synthetic callers may build formulas directly:

```python
premise = And((
    Or((
        RelationConstraint("A", "B", {NE}),
        RelationConstraint("A", "B", {NW}),
    )),
    Not(RelationConstraint("A", "B", {NE})),
))
```

`Implies(P, Q)` and `Iff(P, Q)` are encoded with their ordinary Boolean
semantics. The original SpatialEval adapter deliberately uses only `And` of
positive singleton ordinal atoms; it does not expose the richer syntax in the
original dataset.

## Inference

The solver asks whether each exact query direction occurs in at least one model
that satisfies all premises. It first applies sound path-consistency propagation
over pairwise direction sets. The production backend then encodes the remaining
domains as integer X/Y constraints and performs one incremental Z3 check per
candidate query direction. Thus it preserves correlations such as
`not Northeast` rather than reducing them to independent X and Y facts.

`SpatialSolverV2(backend="auto")` uses Z3 when `z3-solver` is installed and
otherwise falls back to the dependency-free exhaustive reference backend.
Callers generating large datasets should request `backend="z3"` explicitly so
a missing dependency fails closed instead of silently selecting slower search.
The reference backend exists for small cases and differential correctness tests.

The core V2 interface supports pairwise direction queries only. Invalid
structured problems and Z3 timeouts fail closed. Unsupported question families,
unknown query objects, and unparsed premises are rejected by the text adapter.
The reference engine remains exponential in unresolved negated or coarse
premises; the Z3 backend is the supported path for larger workloads.

### What a unique answer means

Let `P` denote all rendered premises and let `D` be one of the eight exact
directions. V2 includes `D` in the query result exactly when this formula has a
model:

```text
P AND D(target, reference)
```

The answer is uniquely `D` when:

```text
P AND D                            is satisfiable
P AND every other direction D'    is unsatisfiable
```

Equivalently, `possible_directions` contains exactly one element. This means
that every coordinate map satisfying the text agrees on the queried direction.
It does not mean that the complete coordinate map is unique.

For example, all of these witnesses give the same Northeast answer:

```text
B=(0, 0),   A=(1, 1)
B=(4, 2),   A=(9, 7)
B=(-5, 3),  A=(20, 100)
```

The coordinate values vary, but `xA > xB` and `yA > yB` are invariant.

Missing evidence must remain distinct from explicit negation. If the only fact
is `A is Northward of B`, valid witnesses may place A northwest, north, or
northeast of B. Rendering one arbitrarily selected witness would reveal a
direction that the text does not entail.

## Text-first map construction

A text-and-image generator should use this order:

```text
choose the intended answer and structural difficulty
    -> construct a SpatialProblem
    -> solve the structured problem
    -> render textual constraints
    -> parse the rendered text back through its adapter
    -> solve the reparsed SpatialProblem
    -> require one possible direction, equal to the intended answer
    -> obtain any satisfying coordinate witness
    -> normalize its coordinates to compact integer ranks
    -> render the map
```

The acceptance condition is:

```python
analysis.consistent and analysis.possible_directions == (desired_direction,)
```

Checking only membership is insufficient:

```python
desired_direction in analysis.possible_directions  # ambiguous examples pass
```

Once the singleton condition holds, arbitrary choices for unconstrained parts
of the witness cannot change the query answer. They may change irrelevant map
details, which is acceptable for a single-question example.

### Constructing canonical coordinates

For exact positive constraints, merge equality classes on each axis, build the
strict-order DAG between those classes, and assign integer ranks in a
deterministic topological order. For example:

```text
X order: A < B < C
X ranks: A=0, B=1, C=2
```

Objects in the same equality class receive the same rank. The X and Y axes are
ranked independently, and no two distinct objects may receive both the same X
and the same Y value.

Negated and coarse premises contain disjunctions, so they do not always produce
one order DAG directly. In those cases, use a satisfying Z3 model to choose a
valid realization, then replace its arbitrary integer values with sorted compact
ranks while preserving `<`, `=`, and `>`.

The public `DirectionAnalysis` result reports consistency, possible query directions,
the selected engine, one convenience witness, and one witness for every
possible answer. A generator can use the convenience witness directly after
checking that the answer is unique:

```python
parsed = SpatialTextAdapter().parse(prompt)
analysis = SpatialSolverV2(backend="z3").analyze(parsed.problem)
if analysis.consistent and analysis.possible_directions == (desired_direction,):
    render(analysis.coordinates)
```

`analysis.coordinates` maps each object name to an `(x, y)` integer pair. Each
axis uses consecutive ranks beginning at zero, while required equality classes
retain the same rank. It is the first witness in direction order.

`analysis.witnesses` maps every possible direction to a coordinate assignment
that produces that direction while satisfying the same premises:

```python
for direction, coordinates in analysis.witnesses.items():
    render_counterexample(direction, coordinates)
```

This supports constructive ambiguity diagnostics. If the benchmark oracle is
Northeast but `witnesses` contains both Northeast and Northwest, the two maps
demonstrate that the visible text does not entail the oracle. Inconsistent and
invalid problems return no convenience witness and an empty witness mapping.

### Samples that may be rendered

For a paired text-and-image dataset:

- A singleton direction is safe to render.
- Multiple possible directions must be rejected or given additional premises
  for ordinary training data. For benchmark audits, retain them and render the
  per-direction witnesses as constructive counterexamples.
- An inconsistent premise set has no witness and must be rejected.
- A `Cannot be determined` text question should not be paired with a concrete
  image if both modalities are expected to have the same answer; the image will
  necessarily resolve some of the textual ambiguity.
- If several questions share one image, uniqueness must be checked separately
  for every query.

The resulting map is therefore arbitrary, but its answer is not.

## Structured explanations

`SpatialSolverV2.assess(problem, claim)` is the semantic interface used by the
explanation layer. It checks both `premise AND claim` and `premise AND NOT
claim` and returns:

- a satisfying coordinate witness when the claim is possible;
- a counterexample coordinate witness when the claim is not entailed;
- whether the premise is consistent; and
- whether the claim is possible and entailed.

The two checks distinguish the four relevant states without consulting an
answer menu or oracle:

| Claim | Negation | Status |
|---|---|---|
| satisfiable | unsatisfiable | entailed |
| satisfiable | satisfiable | contingent |
| unsatisfiable | satisfiable | impossible |
| unsatisfiable | unsatisfiable | inconsistent premise |

`SpatialExplainerV2.explain(problem, analysis)` converts these semantic results
into typed Direction, Which, or Count evidence. For positive conjunctions of
exact relations, it also decomposes directions onto the X and Y axes, finds
transitive paths, and recombines the two derived comparisons. Arbitrary
propositional formulas remain explainable through satisfying witnesses and the
unsatisfiability of a negated claim even when no simple axis path exists.

The structured explanation is authoritative. Renderers accept an optional
mapping from opaque object IDs to display labels and have no dataset schema or
prompt parser.

Training and audit output are deliberately separate:

- `TraceFormat.NATURAL` emits premise decomposition, axis reasoning, candidate
  classifications, and correlated count cases in natural language.
- `TraceFormat.SYMBOLIC` emits the same proof as
  compact `X[...]`, `Y[...]`, `X-State`, `Y-State`, direction-domain, and member-
  set symbols. This preserves the original Chain-of-Symbols ablation style.
- `StateMode.FINAL_ONLY` prints premise extraction followed by one final state.
- `StateMode.DELTA` prints only newly entailed direct or transitive facts after
  each premise, followed by one final state.
- `StateMode.FULL` prints the complete entailed state after every premise.
- `render_audit_explanation(...)` may include normalized coordinate witnesses.
  It is for diagnostics and must not be used as an SFT reasoning target.

Incremental axis states apply to positive conjunctions of exact relations—the
original SpatialEval statement grammar. A general Boolean premise is rendered
as one formula because `OR`, `IF`, and `IFF` may require branched proof states
rather than one monotonically growing pair of axis graphs.

Both training styles are coordinate-free by contract. Coordinate witnesses
remain inside the structured explanation so an auditor can establish that an
alternative world exists without teaching the model to invent a hidden map.

A synthetic generator should therefore construct once and render later:

```python
problem = build_structured_problem(seed)
analysis = solver.analyze(problem)
explanation = explainer.explain(problem, analysis)
resolution = resolve_answer(analysis, answer_mode)
menu_answer = encode_menu_answer(resolution, options)

row = {
    "prompt": prompt_renderer.render(problem),
    "answer": menu_answer.raw,
    "proof": explanation_to_dict(explanation),
    "explanation": render_training_trace(
        problem,
        explanation,
        trace_format=trace_format,
        state_mode=state_mode,
        labels=labels,
    ),
    "metadata": {
        "answer_mode": answer_mode.value,
        "trace_format": trace_format.value,
        "state_mode": state_mode.value,
        "audit_witnesses": analysis.witnesses,
        "seed": seed,
    },
}
```

For exact-answer generation, the generation policy rejects inconsistent or
ambiguous analyses before rendering. For ambiguity datasets and benchmark
audits, it retains the alternative witnesses as constructive counterexamples.
Prompt text, menus, answer mode, and generator metadata remain outside both the
solver and explanation modules.

## Completeness boundary

Within its supported language, V2 does not depend on a hand-written collection
of forward-chaining rules. Exact directions, coarse half-planes, their
negations, arbitrary finite Boolean combinations, equality, transitive
consequences, disjunctive elimination, inverse relations, and contradictions
are represented directly as Boolean integer constraints. Z3 checks the complete
set of models for each possible query direction.

This does not cover unrestricted spatial language. V2 currently excludes
quantifiers, distance, adjacency, betweenness, nearest-object questions, and
three-object orientation. It supports Direction, Which, and Count projections
over finite named entities. Its explanation layer provides axis-path proofs for
exact positive conjunctions and solver-backed possibility/entailment evidence
for arbitrary supported formulas. It does not yet expose a low-level Z3
unsatisfiable core for arbitrary propositional proofs.

## Query families

- `DirectionQuery` returns every satisfiable exact direction and one witness per
  direction.
- `WhichQuery` returns possible entities, entailed entities, and one witness per
  possible entity. Exact-set grading returns `Cannot be determined` when any
  candidate is contingent.
- `CountQuery` uses one correlated cardinality expression across all candidates.
  It returns every satisfiable count and one witness per count; it does not
  naively count individually possible entities.

Which/Count direction domains may contain any subset of the eight exact compass
directions. This lets adapters express either exact regions (`{North}`) or
coarse regions (`{Northwest, North, Northeast}`) without dataset logic in the
solver.

## Dataset scale

There is no existing repository-wide maximum of 30 entities. The default V13
entity pool contains 20 names, the challenge pool contains 48, and Experiment
14 includes 32-entity worlds. A new generator may choose a cap of 30, but it
should be an explicit generation-policy limit rather than a solver axiom.

## Option grading

`SpatialSolverV2.analyze()` is grading-policy neutral: it always returns the
complete model set through its query-specific analysis. Answering is split into
two explicit operations:

```python
parsed = SpatialTextAdapter().parse(prompt)
analysis = solver.analyze(parsed.problem)

resolution = resolve_answer(analysis, AnswerMode.SINGLE)
menu_answer = encode_menu_answer(resolution, parsed.options)
```

`AnswerMode` defines the complete question contract:

- `SINGLE` requires exactly one invariant answer. Multiple possible Direction
  values, Count values, or Which objects resolve to `Cannot be determined`.
- `ALL_POSSIBLE` returns every possible value and requires the menu to
  represent the complete set. If no possible value is visible, it can select an
  explicit `None of the Options`; if only some possible values are visible, it
  selects `Cannot be determined`.
- `VISIBLE_POSSIBLE` returns every possible value present in the menu and
  deliberately ignores possibilities that were not offered.

Original SpatialEval uses `SINGLE` for Direction, singular Which, and Count.
`Cannot be determined` is a semantic result of ambiguous `SINGLE` resolution and
an explicit menu-coverage result for partially represented `ALL_POSSIBLE`
answers; it is not used for `VISIBLE_POSSIBLE`.

The two result types preserve where a failure occurred:

| Status | Meaning |
|---|---|
| Resolution `exact` | One invariant answer exists under `SINGLE` |
| Resolution `possibilities` | All possible values were requested |
| Resolution `ambiguous` | `SINGLE` found no unique invariant answer |
| Menu `incomplete-menu` | A required value has no menu representation |
| Menu `undetermined` | An ambiguous single answer mapped to an explicit option |
| Menu `none-of-options` | No visible ordinary option is selected |
| `inconsistent` | No coordinate model satisfies the premises; generation fails |

`score_response` requires equality of the complete predicted and expected
letter sets. It also reports precision, recall, F1, and Jaccard for diagnostics;
those partial metrics do not redefine correctness.

With original SpatialEval configured as `SINGLE`, the current 1,500-row audit
has no adapter failures:

| Query | Exact oracle | Oracle possible but underdetermined |
|---|---:|---:|
| Direction | 332 | 168 |
| Which | 140 | 360 |
| Count | 195 | 305 |

The Which exact count requires the oracle object to be entailed and every other
menu object to be impossible. A singleton `possible_entities` result is not an
exact answer when that entity is absent in another valid world.
