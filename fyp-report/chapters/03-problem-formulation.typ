#pagebreak(weak: true)
#import "../figures/solver-correctness.typ": paired-countermodels
#import "../figures/spatialeval-modalities.typ": modality-alignment

= Problem Formulation and Benchmark Diagnosis

== Task and Query Contract

A SpatialEntail problem contains a finite set of named objects, a Boolean
formula of qualitative spatial premises, and one typed query. Atomic relations
use the eight compass directions North, Northeast, East, Southeast, South,
Southwest, West, and Northwest. `NOT`, `AND`, `OR`, implication, and equivalence
compose atoms into finite formulas.

The three query families are:

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 7pt, y: 5pt),
    table.header([*Query*], [*Semantic question*]),
    [`DIR` — Direction Identification], [Which exact direction can hold between one target and reference?],
    [`SEL` — Entity Selection], [Which declared candidates can or must satisfy a requested relation?],
    [`CNT` — Cardinality Determination], [Which joint cardinalities are possible over the declared candidates?],
  ),
  caption: [SpatialEntail query families. Count is evaluated jointly rather than by summing independent candidate possibilities.],
) <problem-query-families>

`SEL` is the generic research name for SpatialMap's source _Which_ questions.
Answer modes are explicit and query-specific. Under `SINGLE`, Direction and
Count require exactly one possible value, whereas Selection requires exactly
one entailed entity and no other possible entity. `ALL_POSSIBLE` returns every
possible candidate value; for Selection this is the union of individually
possible entities, not possible complete membership sets. `VISIBLE_POSSIBLE`
is a menu-relative diagnostic. The option menu cannot change which spatial
values are possible.

== Directions as Axis Comparisons

Each exact direction denotes one X comparison and one Y comparison. Northeast,
for example, means that the subject is east and north of the reference;
North means equal X and greater Y. The solver translates these statements into
integer-order constraints. Only relative order matters: translation, scale,
and arbitrary spacing do not affect qualitative direction.

This representation is finite. Any finite real-coordinate map can be rank
compressed independently on X and Y while preserving every `<`, `=`, and `>`
comparison. Conversely, integer ranks satisfying the no-co-location constraint
define a valid qualitative world. Appendix A states the correspondence and
query-correctness results formally.

== Possibility, Entailment, and Ambiguity

Let $P$ be the visible premise formula and $A$ a candidate answer. The solver
uses standard model-theoretic distinctions @russellNorvig2020aima:

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    inset: (x: 7pt, y: 5pt),
    table.header([*Status*], [*Check*], [*Interpretation*]),
    [Consistent], [$"SAT"(P)$], [At least one spatial world satisfies the premises.],
    [Possible], [$"SAT"(P and A)$], [At least one valid world supports the candidate.],
    [Entailed], [$"UNSAT"(P and not A)$], [No valid world provides a counterexample.],
    [Impossible], [$"UNSAT"(P and A)$], [No valid world supports the candidate.],
  ),
  caption: [Semantic decisions made relative to the visible premises.],
) <problem-semantic-status>

One witness proves possibility, not entailment. Entailment requires eliminating
every counterexample. If two valid witnesses give different query answers, a
single-answer question is underdetermined even when the published answer remains
possible.

== SpatialEval Release Boundary

SpatialEval evaluates Spatial-Map, Spatial-Grid, Spatial-Real, and Maze-Nav
through textual, visual, and combined modalities @wang2024spatialeval. This
work studies the 1,500 released Spatial-Map TQA rows: 500 each of Direction,
Which, and Count. The official release contains questions, answers, inference
code, and aligned modality records, but not the Spatial-Map generator or the
source coordinates for those rows.

A row-level check confirms that all 4,635 TQA, VQA, and VTQA identifiers align
in the same order after changing only the modality component, and that their
oracle fields match @wang2024spatialevaldataset. Spatial-Map contributes 1,500
of these aligned triples. The same latent instance therefore supplies several
information views and one shared answer.

#modality-alignment <spatialeval-modality-alignment>

That alignment does not establish that the TQA text uniquely identifies the
latent answer. Let $K$ be the released text, $cal(M)(K)$ the maps satisfying it,
$m^*$ the source map, and $f(m)$ the query answer. The release records

$ a^* = f(m^*), $

while textual entailment requires

$ forall m in cal(M)(K), f(m) = a^*. $

The first statement establishes hidden-world truth. The second establishes that
the evaluated model received enough information to recover one answer.

== Constructive Diagnosis

Consider released item `spatialmap.tqa.2003.0`. Its premises force Recycle
Center to be north of Nightingale Novelties but leave their horizontal order
undetermined. The released menu includes Northeast and Northwest, while the
published answer selects Northeast.

The two diagrams below are not reconstructions of the unavailable source map.
They are independently checked countermodels satisfying the same visible text.

#paired-countermodels <spatialeval-paired-countermodels>

One witness makes Recycle Center northeast of Nightingale Novelties; the other
makes it northwest. Both southern candidates are impossible because the
premises transitively force Recycle Center above Nightingale Novelties. Thus the
published answer is possible but not uniquely entailed.

This distinction is constructive and auditable. Candidate satisfiability is
checked without consulting the oracle, and each returned coordinate assignment
is revalidated against every parsed premise. Coordinates demonstrate existence;
they are never presented to the evaluated text-only model.

== Audit Protocol

The audit:

+ reads the untouched SpatialMap-TQA release;
+ parses each row through a dataset-specific adapter;
+ solves the visible premises under the declared query and answer mode;
+ checks complete candidate domains;
+ revalidates every witness;
+ records the published answer as entailed, possible but non-unique,
  contradicted, inconsistent, or unparseable; and
+ writes the audit and corrected evaluation view separately from the source.

SpatialMap-TQA-Corr preserves the original rows and ordinary options while
adding an explicit undetermined response under `SINGLE`. The current result is
reported in Chapter 6 rather than embedded in the method.

== Scope and Threats

The diagnosis is conditional on the declared ontology, candidate scope, text
parser, and no-co-location assumption. It does not recover the authors' source
coordinates or prove their generator incorrect. It asks a narrower question:
whether the published answer follows uniquely from the published text under an
explicit qualitative semantics. Representative manual checks and assumption
sensitivity remain necessary before treating a corrected benchmark as frozen.
