# SpatialEntail

This context defines the canonical language for SpatialEntail problems,
questions, evidence, and experimental dimensions.

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

**Benchmark cell**:
A predeclared combination of query contract, support or rule structure,
semantic status, and evidence burden. Difficulty measurements annotate the
cell rather than define its task identity.
_Avoid_: Flat task bucket, depth tier

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
