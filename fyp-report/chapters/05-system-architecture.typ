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

The implementation maps these responsibilities to `spatial_text_v2.py` and
`spatialeval_adapter_v2.py` for reading; `spatial_solver_v2.py` for the shared
contract and reasoning core; `spatial_grading_v2.py` for answer policy and menu
encoding; `spatial_proofs_v2.py` and its renderer for checked derivations;
`spatial_model_certificates_v2.py` for constructive evidence;
`spatial_answer_certificates_v2.py` for complete Direction candidate sets;
`spatial_which_certificates_v2.py` for complete three-valued membership sets;
`spatial_explanations_v2.py` for current audit compatibility; and
`audit_spatialeval_v2.py` or `spatial_generation_v2.py` for the two pipelines.

== End-to-End Lifecycle

@solver-system-lifecycle shows the shared processing path and the two supported
entry points. A released dataset row is parsed through a dataset adapter. A
synthetic workload instead begins from an explicit generation policy and builds
the same structured problem directly. Both routes then use the same solver and
produce the same typed semantic analysis.

#system-lifecycle <solver-system-lifecycle>

The output path deliberately branches only after semantic analysis. Answer
policy can map the result into a single answer or a set-valued answer. The
proof layer can produce coordinate-free training traces. The audit layer
can retain witness coordinates and provenance for researcher inspection. This
separation prevents audit-only coordinates from leaking into supervised
training targets.

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

The target SpatialEntail path renders Natural or Symbolic reasoning from one
replayable certificate. The proof-first generator migration is incomplete:
Direction certificates support spatial chains and bounded Boolean case proofs;
complete Direction and Which certificates support unique and ambiguous answer
sets. Correlated Count certificates remain to be implemented before the
post-hoc training renderer is removed. Training rows exclude coordinate
witnesses by default. Coordinates can be added only as separate audit metadata
for diagnostic use.

== Failure and Output Contracts

The pipeline does not silently continue when a boundary fails. Unsupported or
unparsed text, invalid object references, inconsistent premises, unresolved
menus, solving failures, and generation round-trip mismatches are returned as
explicit failures or rejected samples. Successful outputs therefore carry a
clear provenance chain from source input, through the structured problem and
semantic analysis, to the final answer, trace, or audit record.
