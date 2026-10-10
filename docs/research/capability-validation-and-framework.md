# SpatialEntail capabilities, external validation, and a teachable procedure

Status: primary-source research synthesis, 2026-10-10. This is a proposal, not a
new implementation, revised frozen allocation, or executed transfer result.
The supporting reviews cover SODA, adjacent logic/proof work, StepGame,
Text2Space, SpatialMap-TQA, SpartQA-Human, and ReSQ.

## 1. Working conclusion and scope

RR (Relational Reasoning), LR (Logical Reasoning), and MTR (Model-Theoretic
Reasoning) are defensible organizing groups for the declared scope: finite
named objects, qualitative two-dimensional directions, ground Boolean
formulas, and Direction/Which/Count questions. They are not a literature-
standard, exhaustive taxonomy of spatial intelligence and are not disjoint
abilities or a calibrated hierarchy.

Their organizing questions differ:

- RR: how do stated spatial relations constrain further relations?
- LR: how does Boolean structure determine which claims follow?
- MTR: which query claims hold in every, some, or no satisfying world?

A fourth capability group is not currently justified solely to enlarge the
benchmark. Broader spatial reasoning would require explicit additions such as
reference-frame transformations, topology/containment, metric geometry, time
and state updates, navigation, or quantified claims. None is implied by the
current DIR(X,Y) grammar. Natural-language parsing and synonym resolution are
an interface/transfer challenge, not proof that the formal spatial ontology
already covers those concepts.

The grouping is partly orthogonal: all logical consequences have an all-model
meaning, exclusion can accompany any query, and aggregation is an output-level
operation. RR/LR/MTR should therefore be described as complementary evaluation
lenses, not three independent latent skills. EXC versus NEG and QAG versus JCR
need especially careful cell definitions. Prefer merged families or secondary
tags over duplicate task identities if a controlled contrast cannot isolate
the proposed distinction.

The canonical [task taxonomy](task-taxonomy.md) retains compact structure and
obligation codes without treating them as one flat list of peer task types.
MTR means reasoning across satisfying models, not neural-model parameters.
SODA already uses MR for Mental Rotation, so Model-Based Reasoning/MR is rejected
as the public abbreviation. Spell out Model-Theoretic Reasoning at first use;
the expanded label is more important than the acronym.

## 2. Taxonomy versus procedure versus assurance

Three distinct claims must not be conflated:

1. **Taxonomy:** RR/LR/MTR groups the capabilities an example is intended to test.
2. **Procedure:** an explicit policy chooses and constructs relevant reasoning
   and evidence from visible premises before producing an answer.
3. **Assurance pipeline:** the generator, semantic oracle, proof replay, witness
   validation, and text round trip check the resulting supervision artifact.

A neutral method description is **premise-first, proof-and-model-guided spatial
reasoning**. Here premise-first describes the evidence authority at inference:
start from the visible premises, not a privileged answer-bearing world. It must
not overwrite generation provenance: current proof-template-first, premise-first,
and world-first proposal routes remain distinct. No mnemonic has been approved,
and a mnemonic is not a contribution by itself.

### Rejected working mnemonic: PRISM

An initial working mnemonic was **PRISM**:

1. **P — Parse** the visible premises, objects, query, candidates, and answer
   contract.
2. **R — Reason or refute** with scoped, query-relevant dependencies.
3. **I — Inspect witnesses** for consistency, possibility, or contrasting
   answers when the obligation requires them.
4. **S — Synthesize** the query-relevant possible/necessary result, preserving
   correlations rather than treating candidates independently.
5. **M — Map** that result through the declared Direction, Which, Count, and
   menu contract.

Do not use PRISM as the public framework name. A 2026 search found direct
collisions with *PRISM: Planning and Reasoning with Intent in Simulated
Embodied Environments* (arXiv:2605.11534) and an SLM robot-planning distillation
framework (arXiv:2506.17486). The stage wording is still useful, but the acronym
is not distinctive. Until a later naming review, call the method the
**premise-grounded proof-and-model evidence policy**. A mnemonic is not a
contribution by itself. Proof and witness construction may interleave
operationally.

The implemented certificate stack supplies much of the assurance pipeline.
It does not establish that an LLM has learned an inference policy, that every
semantic consequence has an automatically constructible proof, or that the
external Natural trace reveals internal cognition. Sources:
[certificate contract](../spatialentail/certificates.md), especially supported
language, automatic construction fragment, and assurance limits;
[trace formats](../spatialentail/trace-formats.md).

## 3. Candidate teachable procedure

The current supervision constructor receives solver-classified outcomes and
models before assembling evidence (`generation.py` and
`answer_certificates.py`, `build_direction_answer_set`). Its checked gold output
is not an autonomous learned search policy. For the proposed policy, specify
candidate order, evidence-attempt budgets, when branch or witness construction
is attempted, query-specific completion conditions, and failure handling. If
required coverage is incomplete, report a process failure/abstention under an
explicit evaluation rule; do not label search exhaustion as semantic ambiguity
or impossibility. A witness may be any satisfying completion, not exact match
to one canonical gold map.

Working description: **Interpret → Derive/Refute → Witness → Decide**. This is
not a branded acronym or a demand to print four boilerplate
paragraphs for every question. Proof and witness construction can alternate
inside scoped branches; it is an evidence policy rather than an exhaustive
chronological log of SMT internals.

### Interpret

Identify named objects, atom meanings, explicit Boolean scopes, the query's
candidate domain, and its answer contract. Decompose exact directions into X/Y
comparisons while preserving Boolean scope. Do not assert both sides of an OR
as facts or activate an implication without its antecedent. Do not treat an
external dataset's No label as a formal negation without examining its native
contract.

### Derive or refute

Construct a query-relevant dependency DAG using supported spatial and logical
rules. For candidate exclusion, open a declared candidate assumption and close
it by contradiction. For case analysis, reach the claimed conclusion in every
live case; close inconsistent cases within their own scopes. An absent proof
is not evidence of a negative claim and a failed bounded proof search is not
evidence that a candidate is impossible.

### Supply witnesses when required

Check that the premises have a satisfying world so positive necessity claims
are non-vacuous. A witness establishes possibility; contrasting complete
answers establish non-uniqueness. Two different maps with the same query answer
do not establish answer ambiguity. Two different answer witnesses establish
non-uniqueness but do not exhaust the complete possible-answer domain; an
ALL_POSSIBLE certificate needs additional coverage/exclusion obligations.
Canonical rank witnesses are a selected representation of satisfying worlds,
not a hidden source of the gold answer.
The local-proof comparison concerns whether explicit global ranks belong in
the learning target; removing them must not silently weaken truth checking.
That arm is planned, not an exposed current supervision variant. Unique
Direction's existing grammar requires a consistency witness. A local-proof
protocol must define whether eligible examples alone form the training domain,
or all base IDs remain paired with ranks omitted only in eligible cells; it
must also define independent premise-consistency assurance, process acceptance,
and comparable metrics. Stripping required witness fields from existing records
is not currently a valid implementation of this arm.

### Decide under the query contract

For each relevant claim c and premises P, the reference semantics checks
possibility via SAT(P AND c), and necessity via UNSAT(P AND NOT c), with
premise consistency checked separately. These semantic definitions are the
oracle/checker's contract, not instructions that the untethered model has Z3
available during native test inference.

The answer projection is query-specific, not universally a set of complete
world answers:

| Query | Current SINGLE criterion | Current ALL_POSSIBLE output |
|---|---|---|
| Direction | Exactly one possible declared direction. | All possible declared directions. |
| Which | Exactly one entailed entity and no other possible entity. | Union of individually possible entities, not possible complete membership sets. |
| Count | Exactly one possible count, preserving joint membership constraints. | All possible counts. |

Which is a **singular entity-selection contract**. With East(A,R) and
East(B,R), both memberships are certain and the membership set is invariant
{A,B}, but current SINGLE cannot select one entity and returns Cannot be
determined. This is multiple certain matches, not unknown spatial membership.
With no possible entity, the separate no-match/menu policy applies. The current
implementation reports several distinct failure-to-select situations through
its AMBIGUOUS resolution status; a capability analysis should separate them.

A truly set-valued Which task would be a protocol change. Do not present the
current ALL_POSSIBLE union as a domain of complete membership sets or as
correlation-aware Which enumeration. Count is already correlation-aware.
Source: `spatial/v2/grading.py`, `_possible_values`, `_single_value`, and
`resolve_answer`; `spatial/v2/which_certificates.py`, `is_exact_single`. A
2026-10-10 runtime check confirmed both entities entailed/possible, SINGLE
status `ambiguous`, and ALL_POSSIBLE status `possibilities` for this example.

Do not conflate inconsistent premises, a refuted candidate, and an uncertain
answer. The current generated training pool rejects inconsistent premises;
adding an inconsistency task would require a new explicit output contract.
Sources: [semantics](../spatialentail/semantics.md),
[answer/menu controls](../spatialentail/generation-and-knobs.md).

### A checked mixed logical-spatial derivation

Premises: IF East(A,B) THEN North(C,D); East(A,B); North(D,E).
Query: Direction(C,E). The existing `build_direction_proof` produced a locally
replay-checked derivation using modus ponens, direction decomposition, axis
transitivity/inversion, and direction recomposition on 2026-10-10. Deleting
any one of its three premises changed the answer domain from {North} to all
eight directions under the Z3 backend. This establishes one constructive
example and semantic dependence on each premise; it does not prove globally
minimal rule use, full MLC policy admission, or learning by an LLM.

### A useful JCR example

(East(A,R) AND West(B,R)) OR (West(A,R) AND East(B,R)).
Each object's east membership is contingent, but the count of {A,B} east of R
is always one. The existing Z3 backend confirmed count domain {1} and each
object's direction domain {East,West} on 2026-10-10. This is a checked semantic
example, not a released training certificate. It shows why MTR/JCR cannot be
replaced by an 'uncertain premise means Cannot be determined' heuristic.

## 4. Primary-source findings

### SODA and procedural supervision

The [framework literature review](capability-framework-literature-review.md)
checks the official papers. SODA's OODA is not just its three-tier taxonomy:
its training-free workflow orchestrates stage-specific prompts (§3.1), its
SPOD-143k SFT supplies OODA traces and GRPO uses answer/format rewards (§3.3),
and robotic control re-encodes executed state (Appendix D). Trace construction
is answer-conditioned and sampled-reviewed, not formal step replay. OODA
benefits do not establish hidden certified cognition or isolate every effect
of the stage policy. Primary:
[SODA ACL paper](https://aclanthology.org/2026.acl-long.1382.pdf), §§3.1–3.3,
Tables 1–3, 5. The [exact naming check](soda-task-taxonomy.md) gives its actual
three tiers and 13 task abbreviations.

ProofWriter is a useful precedent for learning explicit proof DAGs and an
iterative procedure, not merely longer prose (§§3.1, 3.6, 4). Its OWA native
True/False/Unknown distinction uses supported rule closure and explicit
negative evidence, not unrestricted classical Boolean entailment. CWA and OWA
must not be pooled. Primary:
[ProofWriter](https://aclanthology.org/2021.findings-acl.317.pdf).

### Native spatial-task alignment

The [relational external review](external-relational-capability-review.md)
checks primary papers and official source code. Alignment below is task-contract
alignment, not proof of necessity in every row or an executed adapter result.

| Dataset/view | RR alignment | LR alignment | Model-based alignment | Use and boundary |
|---|---|---|---|---|
| StepGame, corrected 2024 | Native extraction/converse and multi-hop composition; AXC only partial. | No verified explicit Boolean-premise task. | No native unknown or correlated counts; overlap is coincidence, not ambiguity. | Native DIR transfer under unit-grid offsets; nine labels; score overlap separately. |
| Text2Space | Stated/inferred direction questions; independently controlled axes not established. | No verified Boolean-premise evaluation. | Nonunique layouts are not unknown answers; queried pairs are selected as uniquely inferable. | Description-only full-direction native transfer; projected-axis tasks separate; no gold ASCII. |
| SpatialMap-TQA-Corr | Direction/Which/Count under our audited contract; necessary capability tagging still needs checks. | Does not validate the full explicit Boolean family suite. | Our repaired unknown-answer view tests calibrated response to nonunique answers, with singular Which caveats. | Same-source audited transfer, not an independent new external dataset or JCR proof. |
| SpartQA-Human | Directional relation reading/chaining and CO selection analogues are partial because axes are coarse and the ontology includes distance, topology, and containment. | Negated and quantified questions occur, but this does not establish visible IF/IFF/disjunctive-premise coverage in the supported fragment. | Native YN distinguishes Yes/No/DK; DK motivates partial-information transfer but is not automatically a certificate of differing complete SpatialEntail answers. | Report YN, FR, CO, and FB under their native contracts; any directional projection is separately audited and scored. |
| ReSQ | Natural-language relational transfer is useful, but front/behind, topology, distance, commonsense, quantification, and reference resolution exceed the current ontology. | No compatible LR family coverage is established by incidental words such as `or` or `not`. | No native unknown class; No must not be silently equated with formally refuted under our ontology. | Preserve native binary scoring; a compatible formal subset requires predeclared filtering and separate reporting. |

For StepGame, official generator/reasoner offsets are fixed (0,±1), (±1,0), and
(±1,±1). This extra prior can yield a unique answer where qualitative premises
alone cannot. Its story k is not guaranteed shortest query support. Sources:
[original paper](https://arxiv.org/html/2204.08292v1), §3.2;
[official generator](https://github.com/ShiZhengyan/StepGame/blob/main/Code/parameterized_step_game_8relation.py);
[correction](https://arxiv.org/html/2401.03991v1) and
[official correction repository](https://github.com/Fangjun-Li/SpatialLM-StepGame).

Text2Space's full queries differ from vertical/horizontal projections. It
selects uniquely inferable query pairs even when whole layouts are nonunique.
Its gold-ASCII condition is an oracle upper bound, not text-only inference.
The review did not establish fixed-unit edge semantics, a public release path,
or scene/isomorphism-disjoint splits. Sources:
[Text2Space v1](https://arxiv.org/html/2604.14641v1), §3.1, §4.4, Appendix A,
Appendix C Tables 7–8, Appendix F Algorithm 1, Appendices H/J.3.

Fresh StepGame/Text2Space per-row capability audits and exact retained corrected
split sizes remain unverified. Shared compass vocabulary alone does not make
their native contracts identical to SpatialEntail.

SpartQA-Human's FR output is a set of simultaneously true labels such as left,
below, and near, not a set of alternative exact compass answers. Its CO `both`
and `neither` outputs are selection outcomes, not numeric Count labels. FB is
primarily containment. ReSQ's volunteers wrote binary Yes/No questions over
image descriptions and can require commonsense; it has no DK output. Therefore
neither source validates JCR, and only a separately audited projection can be
called compatible with the point-compass fragment. Primary sources, pinned
release hashes, concrete records, and proposed exclusion accounting are in the
[human external review](external-human-capability-review.md).

### Optional auxiliary logical validation

Recommend at most one addition: **ProofWriter OWA**, separately scored with
native inference semantics and labels. It tests some LR rule chaining/negative
evidence and a related unknown-recognition behavior outside spatial wording.
It does not validate all IMP/EQV/NEG/DJE/CSR/MLC families, geometry, or JCR.
FOLIO is a quantified-FOL language stress test beyond the ground spatial
fragment; PrOntoQA's native binary repeated-modus-ponens tasks are not a complete
uncertainty benchmark. Sources:
[ProofWriter](https://aclanthology.org/2021.findings-acl.317.pdf), §§3.2, 3.6;
[FOLIO](https://aclanthology.org/2024.emnlp-main.1229.pdf), §4, Appendix E;
[PrOntoQA final author version](https://arxiv.org/pdf/2210.01240v4), §§3–4.
No new external benchmark or training arm is frozen by this recommendation.

## 5. Proposed evaluation architecture

Keep separate scorecards rather than one cross-dataset macro score:

1. Internal capability tests and premise-first held-out compositions under the
   complete SpatialEntail semantics, with checked dependency/evidence outcomes.
2. External native task accuracy, preserving each source's ontology and labels.
3. Predeclared compatible/tagged external subsets, with retained/excluded counts
   and alignment uncertainty reported; native and adapted scores remain separate.
4. Optional non-spatial logic transfer as an auxiliary diagnostic, not evidence
   of compass-direction transfer or spatial Count competence.

External task presence does not establish necessity of a named capability.
Where possible, validate dependency tags against an audited formalization or
manual review. Freeze subset rules using source semantics before final model
comparison, never selecting rows because a model fails them. Retain the
original external release unchanged.

A concrete ontology-mismatch diagnostic is Northeast(A,B) and Southeast(B,C).
A unit-step interpretation adds (1,1) and (1,-1), obtaining East(A,C). Under
our qualitative exact-direction ontology, only A's X order above C is fixed;
its Y order may be above, equal, or below, so the complete domain is
{Northeast,East,Southeast}. The existing solver verified this domain on
2026-10-10. This is a general semantic warning, not a claim that a specific
external row has been audited. A native offset-based dataset needs its native
interpretation/scoring preserved; it cannot be relabeled qualitative merely
because it uses the same compass words.

Native external prompts must not include their gold maps, answers, proof labels,
or the solver's computed intermediate evidence. A semantic oracle can audit or
score outputs without supplying hints. A tool-assisted inference condition, if
added, needs its own untuned tool-access baseline and separate result label.

## 6. Operational teachability checks by capability

| Group | Supervised behavior | What a held-out test should check | What success would not establish |
|---|---|---|---|
| RR | Decompose, invert, chain and recombine supported comparisons; aggregate query memberships. | New entity graphs and independent-axis supports, with replay-valid dependencies and correct native answers. | Metric distance, geometry, rotations or arbitrary language understanding. |
| LR | Preserve Boolean scopes, activate only licensed consequents, eliminate alternatives, and close all cases. | Predeclared new rule compounds/formula trees with familiar primitives; count illegal converse/branch leakage failures separately. | Quantified FOL, missing typed rewrites, or all logically equivalent presentations. |
| MTR | Establish candidate possibility/exclusion and query-level invariants; preserve correlated memberships. | Uniquely answered versus genuinely ambiguous pairs, witness/candidate coverage, and invariant/variable joint counts. | Calibrated confidence, probability estimates, or exhaustive reasoning inside a model. |

For Symbolic output, parse success alone is not success. Evaluate reasoning
replay, decision validity, domain agreement, and full answer correctness
separately. Witness-only and proof-only failures can reveal which part of the
explicit evidence policy transfers. Generated Natural output remains an
answer/readability evaluation unless an independent prose formalizer is added;
gold Natural correctness does not confer automatic process scoring on model
prose.

If external native formats cannot represent our typed certificates, native
answer accuracy remains the primary transfer result. A separately requested
structured-process condition needs a validated adapter and its own baseline;
process validity cannot silently be reported for all external answers.

## 7. Remaining work before a framework claim

- Make the source of truth **query contract × support/rule structure × semantic
  status × evidence burden**, with RR/LR/MTR as readable presentation headings.
  QAG is an aggregation/control dimension, EXC an exclusion obligation, AMB a
  query-specific semantic/selection condition, and MLC a composition regime;
  they are not automatically primitive task peers of transitivity or modus
  ponens. Retain a named family only where a controlled contrast makes its score
  and exposure interpretable. Overlapping tags do not support independent causal
  estimates of RR versus LR versus MTR.
- Enumerate MLC rule compounds and train/holdout relationships rather than
  defining a miscellaneous hard class.
- Operationalize measurable admission claims separately: syntax exposure,
  verified evidence use, and controlled semantic dependence. A rule used in one
  proof or deletion of one premise does not prove the rule is unavoidable in all
  derivations. Do not turn this into an unrestricted minimal-proof project.
- Implement and smoke-test proposed compositions within current constraints:
  non-atomic Boolean templates cannot currently combine with controlled axis
  depth/independence/distractor controls outside premise-first mode; typed
  contraposition and De Morgan conversions are absent from the constructor.
- Validate group/query coverage and context yield before setting revised quotas.
- Adapt, source-pin, and freeze external task interfaces and capability subsets.
- Teach the same evidence policy through both Natural and Symbolic gold targets,
  while preserving current distinctions in output validation and token cost.
- Measure learning with answer-only, checked Natural, checked Symbolic and the
  validity/rank-witness mechanisms, without multiplying training runs merely to
  introduce a mnemonic. A separate procedure-prompt ablation is optional if the
  causal claim specifically concerns stage instructions.
- Test primitive retention, unseen logical-spatial compounds, ambiguity and
  correlated-count calibration, evidence validity, and native external transfer.

An operational teachability test should inspect more than a correct final
answer. For each primitive family, verify legal dependency use, branch scope,
candidate coverage, and whole-answer consistency. Then test the same primitive
in held-out mixtures and premise-first structures. That is evidence of reuse of
an explicit procedure; it still does not establish the model's hidden causal
computation. Training targets must not encode an instruction to consult an
oracle the model lacks at test time. Witness generation and proof search are
learned tasks whose context and computational costs need measurement.

A change in task taxonomy also changes matched-test macro averaging, per-task
review stratification, and possibly the development size-calibration rule.
Keep the frozen report's existing numbers until that revision is agreed as a
whole. The high-level groups alone do not establish empirical hardness,
learnability, coverage of all spatial reasoning, or novelty.
