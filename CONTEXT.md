# SpatialEntail

## Current research authority

The active study is
[`spatialentail-composition-v1`](experiments/spatialentail-composition-v1/README.md).
Read its [pivot](experiments/spatialentail-composition-v1/pivot_20261026.md)
before planning or changing research code. It is the source of truth for the
new study; the vocabulary below records earlier designs and does not override it.

The research direction is settled: a solver-verified atomic spatial-composition
dataset is the primary contribution, with matched Natural/Symbolic supervision
as a supporting comparison. Do not reopen the direction or expand the study
without explicit user approval. Surface correctness or feasibility blockers
with evidence; do not silently weaken release requirements to meet the schedule.

Reuse shared solver and training infrastructure. Preserve historical results,
identifiers, and frozen Corr SINGLE behavior. New contracts and workflows need
explicit versioning, not retroactive changes to old experiments.

## Historical vocabulary

The following terms describe earlier SpatialEntail problems, evidence, and
experimental dimensions. Old benchmark sizes, capability lenses, and optional
Symbolic supervision are not requirements for the composition study. Use the
pivot's descriptors and matched explanation requirements for new work.

## Language

**Spatial problem**:
A finite set of named objects, one visible premise formula, and one declared
query interpreted under the SpatialEntail semantic contract.
_Avoid_: Hidden map as the problem

**Spatial world**:
One complete assignment of X/Y relations that satisfies the visible premises
and the non-coincidence constraint.
_Avoid_: Gold answer source

**Query family**:
The requested projection of the shared entailment semantics: Direction
Identification (`DIR`), Entity Selection (`SEL`), or Cardinality Determination
(`CNT`). SpatialMap's source label `Which` adapts to `SEL`.
_Avoid_: Capability, difficulty tier

**Candidate status**:
The all-model classification of one query claim as entailed, contingent, or
impossible under consistent visible premises.
_Avoid_: Answer label, confidence

**Answer contract**:
The declared rule for turning candidate statuses into an output, such as one
invariant answer or all possible values.
_Avoid_: Output formatting alone

**Capability lens**:
A grouping used to present and evaluate related reasoning structures. It is not
a chronological training stage, universal hardness tier, or independent latent
cognitive module.
_Avoid_: Capability tier, difficulty level

**Relational Reasoning (RR)**:
Reasoning through relation access, inversion, decomposition, transitivity, and
axis recomposition.

**Logical Reasoning (LR)**:
Reasoning governed by Boolean operators, licensed rule applications, and branch
scope.

**Model-Theoretic Reasoning (MTR)**:
Reasoning about which query claims hold in every, some, or no satisfying
spatial model.
_Avoid_: Model-Based Reasoning, MR

**Reasoning-structure tag**:
An evidence-supported description of a relation or logical structure used in
an accepted derivation, such as `TRC`, `AXC`, `MP`, or `CSR`.
_Avoid_: Operator presence as verified use

**Semantic-status tag**:
A query-specific resolution condition such as unique/invariant, alternative
answers, multiple certain selections, or no match.
_Avoid_: Proof rule

**Evidence obligation**:
The claim that an explanation must establish, such as necessity,
impossibility, possibility, consistency, or joint Count coverage.
_Avoid_: Generic rationale, solver search log

**Answer certificate**:
A machine-checkable audit artifact establishing that a proposed answer satisfies
the declared semantics. It belongs to generation, verification, and evaluation;
it is not a default model response or SFT target.
_Avoid_: Reasoning trace, chain of thought, Symbolic supervision

**Natural solution trace**:
A readable, query-directed explanation rendered from verified reasoning for use
as an ordinary worked-solution target in SFT.
_Avoid_: Natural-language certificate dump, audit witness

**Compact symbolic trace**:
An optional CoS-like intermediate representation that abbreviates the
query-relevant spatial reasoning to reduce target and generation tokens. It is
not an answer certificate and is not required to be a replayable proof object.
_Avoid_: Certificate, formal proof, checked Symbolic

**Benchmark cell**:
A predeclared combination of query contract, support or rule structure,
semantic status, and evidence burden. Difficulty measurements annotate the
cell rather than define its task identity.
_Avoid_: Flat task bucket, depth tier

**Core benchmark**:
The 1,300-base coverage suite over the standard reasoning cells.

**Composition benchmark**:
The 1,200-base suite of new instances using composition families represented in
training.
_Avoid_: Comp benchmark, matched composition tier

**Shift benchmark**:
The 1,800-base structural-generalisation suite combining 900 held-out
composition instances with 900 depth-ladder instances. Its two components are
reported separately as diagnostics.
_Avoid_: Held-out composition tier, depth tier

**Structural difficulty**:
Measured properties such as spatial support depth, formula nesting, live branch
count, candidate-domain size, independent-axis support, distractors, and target
length.
_Avoid_: Trace length alone, depth as task identity

**External capability alignment**:
The relationship between an external task's native semantics and a declared
capability. Alignment may be direct, restricted, or absent.
_Avoid_: Shared spatial vocabulary as validation

**Native benchmark contract**:
An external dataset's stated input, assumptions, answer meanings, and scoring
rules, distinct from any adapted or newly audited view.
_Avoid_: Universal spatial answer contract
