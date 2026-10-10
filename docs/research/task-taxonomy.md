# SpatialEntail task taxonomy

Status: canonical design vocabulary, 2026-10-10. The taxonomy is fixed; the
cell allocation and quotas are not. It does not claim that the final corpus has
been generated or that the capability groups form a universal difficulty scale.

## One semantics, three query families

SpatialEntail applies one all-model semantic contract to three query families:

| Code | Query family | Candidate domain | Current `SINGLE` contract |
|---|---|---|---|
| `DIR` | Direction Identification | Declared exact compass directions | Return the direction only when exactly one is possible. |
| `SEL` | Entity Selection | Declared candidate entities | Return an entity only when exactly one is entailed and no other entity is possible. |
| `CNT` | Cardinality Determination | Integers from zero through the candidate count | Return the count only when exactly one count is possible. |

`SEL` is the generic name for SpatialMap's `Which` question. Dataset adapters
may preserve the source label `Which`, but research tables use `SEL` to expose
the singular selection contract.

For premises `P` and candidate claim `c`:

- `c` is possible when `P AND c` is satisfiable;
- `c` is entailed when `P AND NOT c` is unsatisfiable, after separately checking
  that `P` is consistent; and
- `c` is impossible when `P AND c` is unsatisfiable.

Entailment is therefore the shared semantics, not a fourth query family.
Direction classifies directional claims, Selection classifies candidate
memberships, and Count classifies exact-count claims while preserving joint
membership dependencies.

## Answer contracts

`SINGLE` is the headline contract. `ALL_POSSIBLE` is the strict diagnostic that
returns every possible direction, entity, or count in the declared candidate
domain. `VISIBLE_POSSIBLE` remains a menu-relative diagnostic.

For Selection, `ALL_POSSIBLE` is the union of individually possible entities;
it is not a domain of complete membership sets. Two entities may both be
entailed, in which case the current singular contract cannot select one even
though neither membership is uncertain. Record these cases separately:

- unique selectable entity;
- multiple entailed matches;
- contingent membership without a unique selection;
- no possible match.

A future `ALL_ENTAILED` or set-valued Selection contract would be an explicit
protocol extension, not a fourth query family. It is not part of the frozen
main study.

## Three capability lenses

| Code | Capability | Organizing question |
|---|---|---|
| `RR` | Relational Reasoning | Which spatial consequences follow by relation access, inversion, decomposition, transitivity, and axis recomposition? |
| `LR` | Logical Reasoning | How do negation, conjunction, disjunction, implication, equivalence, and branch scope control inference? |
| `MTR` | Model-Theoretic Reasoning | Which query claims are necessary, possible, or impossible across satisfying spatial models? |

These are complementary presentation headings, not independent latent skills,
chronological training stages, or guaranteed hardness tiers. Every accepted
problem has model-theoretic semantics; controlled uncertainty, exclusion, and
joint-dependency cells place particular stress on MTR. `MTR` avoids collision
with SODA's `MR` abbreviation for Mental Rotation.

## Experimental source of truth

A benchmark cell is identified by four dimensions:

```text
query contract × support/rule structure × semantic status × evidence burden
```

Difficulty and presentation measurements remain separate:

- spatial support depth;
- propositional depth and formula nesting;
- live branch count;
- candidate and possibility-domain sizes;
- independent-axis support;
- distractor count and deletion support;
- prompt and target token lengths; and
- generation rejection rate.

Depth is not a task identity. A longer chain is not automatically a different
capability or a harder problem.

## Compact structure and obligation codes

The following codes label dimensions inside a cell. They are not a flat list of
peer task types and do not imply equal allocation.

### Relational structures

| Code | Structure | Required distinction |
|---|---|---|
| `ARC` | Atomic Relation Comprehension | Direct access versus target/reference inversion is retained as a tag. |
| `TRC` | Transitive Relation Composition | The queried relation is supported through intermediate entities rather than a direct query fact. |
| `AXC` | Axis Composition | X and Y conclusions are separately supported and recombined; a diagonal label alone does not qualify. |

### Logical structures

| Code | Structure | Required distinction |
|---|---|---|
| `MP` | Modus Ponens | An established antecedent licenses the consequent; converse inference is invalid. |
| `IFF` | Equivalence Elimination | The required direction of an equivalence is used. |
| `NEG` | Negation | Exact-relation complement semantics and double-negation elimination are reported as separate subtypes. |
| `DJE` | Disjunctive Elimination | A disjunct is removed using checked negative evidence. |
| `CSR` | Case-Split Reasoning | A conclusion is established in every live case; nesting and branch count remain fields. |

Mixed logical-spatial composition is a generalisation regime, not a
miscellaneous task code. Enumerate its rule compounds explicitly, expose every
primitive during training, and reserve selected compounds or formula trees for
the premise-first holdout.

### Semantic status and evidence obligations

| Code | Status or obligation | Meaning |
|---|---|---|
| `UNI` | Unique/invariant answer | The query contract resolves to one answer. |
| `ALT` | Alternative answers | Different premise-satisfying worlds yield different complete query answers. |
| `MUL` | Multiple certain selections | Several entities are entailed under singular `SEL`; this is not missing information. |
| `NOM` | No match | No declared Selection candidate is possible. |
| `EXC` | Candidate exclusion | A declared candidate is impossible and carries checked negative evidence. |
| `JCR` | Joint Count obligation | Count reasoning preserves correlated memberships rather than combining marginals independently. |

`EXC` is an evidence burden and `ALT`, `MUL`, and `NOM` are query-resolution
conditions. They may define controlled evaluation strata, but they are not peer
inference rules. `JCR` identifies a distinctive Count dependency obligation.

Example cell identifiers remain compositional, such as `DIR-TRC-UNI`,
`SEL-MP-MUL`, or `CNT-CSR-JCR-UNI`. The final registry must list supported
combinations rather than instantiate an artificial full cross-product.

## Admission claims

Report three increasingly strong, measurable properties separately:

1. **Syntax exposure:** the relevant operator or relation occurs.
2. **Verified evidence use:** replay-valid dependencies use the declared rule or
   obligation.
3. **Controlled semantic dependence:** a predeclared intervention changes the
   answer or relevant status.

None alone proves that a rule is globally unavoidable across every derivation.
Do not turn dataset admission into an unrestricted shortest-proof problem.

## Training, evaluation, and external alignment

All capability cells are mixed during SFT; the groups are not a staged
curriculum. A separate depth ladder measures support-depth extrapolation, and a
premise-first holdout measures generalisation to selected formula/rule
compositions.

External benchmarks retain their native contracts:

- corrected StepGame and Text2Space primarily test relational Direction
  transfer under semantics that are not identical to SpatialEntail;
- SpatialMap-TQA-Corr evaluates the three same-source query projections and
  repaired underdetermination, but is not an independent new source;
- SpartQA-Human supplies partial human-language and Yes/No/DK transfer with
  coarse axes and a broader relation ontology;
- ReSQ is a native binary realistic-language stress test with no unknown label;
  and
- ProofWriter OWA is an optional, separately scored nonspatial LR/MTR diagnostic.

No cross-dataset aggregate is reported. Native scores, compatible audited
subsets, and unsupported portions remain separate.

## Allocation status

The former four-tier, 17-bucket allocation is withdrawn. The maximum nested
pool sizes `4K ⊂ 8K ⊂ 17K` remain candidate calibration budgets, not generated
corpora or finalized cell quotas. Before generation, freeze:

1. the supported cell registry;
2. requested counts and minimums across query, structure, and semantic strata;
3. matched-test macro-averaging cells;
4. quality-review stratification; and
5. group-sensitive size-calibration safeguards.

Primary supporting analysis:
[capability and framework synthesis](capability-validation-and-framework.md),
[SODA taxonomy check](soda-task-taxonomy.md),
[framework literature review](capability-framework-literature-review.md),
[relational external review](external-relational-capability-review.md), and
[human external review](external-human-capability-review.md).
