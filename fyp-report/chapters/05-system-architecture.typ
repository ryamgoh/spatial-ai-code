#pagebreak(weak: true)
#import "../figures/solver-lifecycle.typ": system-lifecycle, solver-lifecycle

= System Architecture and Question Lifecycle

== Separation of Responsibilities

The system is organised around one stable boundary: `SpatialProblem`. Everything
before that boundary depends on where a question came from. Everything inside
the reasoning core depends only on locations, spatial formulas, and a query.
Everything after the core decides how semantic results should be presented,
graded, explained, or stored.

This distinction prevents dataset conventions from changing spatial truth. A
reader adapter may recognise a particular sentence template or option format,
but it cannot decide which direction is correct. Likewise, the solver has no
knowledge of JSONL fields, option letters, SpatialEval or SpatialEntail names,
or training-file formats.

The reader also does not construct a map. It extracts constraints. Concrete
coordinates first appear later, when the solving backend finds one assignment
that satisfies those constraints. Such an assignment is a witness, not a
reconstruction of a hidden source map.

SpatialEntail uses a controlled textual language. Its atomic form is
`X is to the DIR of Y`; explicit parentheses and `NOT`, `AND`, `OR`,
`IF ... THEN`, and `IFF` compose atoms into formulas. The adapter renders and
parses this language deterministically. Unsupported free-form paraphrases are
rejected instead of being assigned a guessed formal meaning.

== Module Responsibilities

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 7pt, y: 5pt),
    stroke: .5pt + luma(120),
    fill: (_, y) => if y == 0 { luma(235) } else { none },
    table.header([*Logical module*], [*Responsibility*]),
    [Reader adapter], [Parse supported text and dataset fields into the shared problem contract.],
    [Problem contract], [Represent locations, propositional premises, and one typed query.],
    [Reasoning core], [Compute consistency, possibility, entailment, counts, and witnesses.],
    [Answer policy], [Turn semantic results into single-answer or set-valued decisions.],
    [Proof layer], [Build replayable derivations and render matched Natural or Symbolic traces.],
    [Model-certificate layer], [Validate constructive witnesses, countermodels, and contingency evidence.],
    [Answer-certificate layer], [Require exhaustive typed evidence for every declared candidate.],
    [Audit layer], [Retain checked evidence and source provenance.],
    [Pipeline coordinator], [Run the stages in order and write validated artifacts.],
  ),
  caption: [Logical module boundaries in the V2 architecture.],
) <solver-module-responsibilities>

The implementation maps these responsibilities to `spatial/v2/text.py` and
`spatial/v2/spatialeval_adapter.py` for reading; `spatial/v2/solver.py` for the shared
contract and reasoning core; `spatial/v2/grading.py` for answer policy and menu
encoding; `spatial/v2/proofs.py` and its renderer for checked derivations;
`spatial/v2/model_certificates.py` for constructive evidence;
`spatial/v2/answer_certificates.py` for complete Direction candidate sets;
`spatial/v2/which_certificates.py` for complete three-valued membership sets;
`spatial/v2/count_certificates.py` for correlation-preserving Count domains;
`spatial/v2/certificate_generation.py` for oracle-assisted certificate
construction;
`spatial/v2/difficulty.py` for structural measurements;
`spatial/v2/audit_spatialeval.py` or `spatial/v2/generation.py` for the two pipelines.

== End-to-End Lifecycle

@solver-system-lifecycle shows the shared processing path and the two supported
entry points. A released dataset row is parsed through a dataset adapter. A
synthetic workload instead begins from an explicit generation policy and builds
the same structured problem directly. Both routes then use the same solver and
produce the same typed semantic analysis.

#system-lifecycle <solver-system-lifecycle>

The output path deliberately branches only after semantic analysis. Answer
policy can map the result into a single answer or a set-valued answer. The
proof layer exposes separate renderers for compact training traces and
exhaustive audit certificates. Training traces omit numeric coordinates but may
show qualitative X/Y rank orders when possibility requires a constructive
model. Audit certificates retain the full per-candidate evidence, witness
coordinates, and provenance. This separation prevents arbitrary coordinate
values from being presented as deductive reasoning.

== Inside the Reasoning Core

The solver receives exactly three things: a finite list of locations, a premise
formula, and a Direction, Which, or Count query. @solver-internal-lifecycle
shows what happens next.

#solver-lifecycle <solver-internal-lifecycle>

The coordinates are initially variables with no assigned values. The solver
adds the translated spatial statements and the rule that two different
locations cannot occupy the same point. It may first narrow obvious pairwise
possibilities, but the complete premise formula remains present during every
exact check.

The first check asks whether any coordinate assignment satisfies all premises.
If none exists, the problem is inconsistent. Otherwise, the solver tests the
values relevant to the query. A successful check produces one normalised
coordinate assignment as a witness. A failed check means that no map satisfying
the premises can realise that candidate. Entailment is established by searching
for a counterexample and finding that none exists.

The solver does not create one map and reuse it as the answer key. Each candidate
is checked against the entire set of valid maps, and different successful
checks may produce different witnesses. This is what allows the system to
distinguish a possible answer from a uniquely determined answer.

== From Semantic Results to Answers

The solver returns a typed analysis rather than an option letter. Depending on
the query, this contains possible directions, possible and entailed entities,
or possible counts, together with consistency status and witnesses.

The answer-policy module then applies the requested contract:

+ `SINGLE` returns one answer only when the semantics determine exactly one;
  otherwise it requests _Cannot be determined_.
+ `ALL_POSSIBLE` requires the complete possibility set.
+ `VISIBLE_POSSIBLE` returns possible values visible in the supplied menu and is
  retained as a menu-relative diagnostic.

Only after this step does the menu encoder translate semantic values into option
letters. The grader compares the complete selected-letter set with the expected
set. This keeps menu layout and grading policy outside the spatial reasoning
core.

== Two Uses of the Architecture

=== SpatialEval audit

The audit pipeline reads an untouched SpatialEval row, parses it through the
SpatialEval adapter, solves the resulting `SpatialProblem`, and compares the
published oracle with the solver's possible and entailed answers. Witnesses are
revalidated against every parsed premise before the row-level audit is written.
SpatialMap-TQA-Corr is then produced as a separate artifact; the released source
row is not overwritten.

=== SpatialEntail generation

The generation pipeline starts from an explicit policy describing query type,
answer mode, semantic shape, difficulty, and rendering choices. It constructs a
candidate `SpatialProblem`, solves it, and rejects it if its semantics or
difficulty do not match the policy. It then builds the answer menu and prompt,
parses that rendered prompt back into a second `SpatialProblem`, and solves it
again. A row is emitted only when the round trip preserves both the structured
problem and its answer.

Boolean curriculum cells construct a proof schema before semantic validation.
The implemented schemas cover modus ponens, equivalence elimination, double
negation, disjunctive syllogism, case split, and bounded nested case split over
ordinary directional atoms. Thus the target conclusion is fixed by the proof
obligation rather than selected from a previously sampled coordinate world.
Rows identify this path as `proof-template-first`. The stricter
`certificate-first` label would require the typed certificate before the problem.
The distinct `premise-first` route samples visible relation premises and a query
without a coordinate map, solves and classifies the problem, then constructs
checked evidence. World-first proposals remain labelled separately and their
answers are also recomputed from visible premises.

Canonical signatures describe source formulas and queries under object renaming
and premise or commutative ordering. Exact canonicalization has a permutation
budget; its conservative fallback may merge distinct structures. Structural
clustering and three-way overlap validation keep related variants in one split.
They do not establish a withheld composition: explicit holdout cells must be
reserved before development selection.

Natural narrates checked dependencies and scopes, while Symbolic serializes
typed records from the same accepted evidence. Corpus-wide information and cost
matching remains to be audited. Only Symbolic model output
has a strict parser: it reconstructs typed evidence, replays proof and model
checks, validates the menu decision, and checks the final answer footer.
Natural process validity is unavailable. Neither gold certificate checking nor
model-output replay demonstrates the model's internal reasoning process.

Count stores fixed membership classifications once and excludes only the
remaining compatible assignments. Correlated contingent memberships still
require joint reasoning, with potentially exponential evidence. Unique Direction
training includes a consistency witness, since a proof alone need not establish
that the premise set has a model. Training witnesses use qualitative axis ranks;
audit views expose coordinates for all query kinds. Automatic proof construction
is incomplete and unsupported candidates fail closed.

The workload derives checked-trace, answer-only, and Symbolic-only corrupted
controls. Corruption mutates semantic evidence while preserving syntax and the
gold decision, and is admitted only if reasoning replay rejects it. Answer-only
has absent evidence, not invalid evidence. Exact tokenizer/template admission
checks full training chats and evaluation prompts plus generation reserve, then
rejects entire paired groups on overflow. Inference receives the admitted prompt
without retemplating. The bounded pilot and unrun research comparisons are
specified in Part II.

== Failure and Output Contracts

The pipeline does not silently continue when a boundary fails. Unsupported or
unparsed text, invalid object references, inconsistent premises, unresolved
menus, solving failures, and generation round-trip mismatches are returned as
explicit failures or rejected samples. Successful outputs therefore carry a
clear provenance chain from source input, through the structured problem and
semantic analysis, to the final answer, trace, or audit record.
