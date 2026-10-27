#pagebreak(weak: true)
= Evidence and Discussion

== Historical SpatialMap Audit

The earlier semantic audit classified the published answer as uniquely
entailed in 667 of 1,500 SpatialMap-TQA questions and possible but non-unique
in 833 under its declared SINGLE semantics.

#figure(
  table(
    columns: (1fr, auto, auto),
    table.header([*Query*], [*Exact oracle*], [*Possible but non-unique*]),
    [Direction], [332], [168],
    [Which], [140], [360],
    [Count], [195], [305],
    [*Total*], [*667*], [*833*],
  ),
  caption: [Historical SpatialMap-TQA semantic audit. These are not results
    from the new six-option contract views or composition adapters.],
)

The result is conditional on the declared interpretation of the text. It does
not recover the source map, prove the original generator incorrect, or
establish the original authors' intended semantics. The existing Corr SINGLE
release remains unchanged.

== Composition Study Status

The composition study has a specification and fourteen hand-authored evidence
fixtures with executable semantic checks. They cover spatial derivation,
ambiguity, menu coverage, Selection membership cases, joint and invariant
Count, contradictions, consistent-cycle controls, and distinct-point effects.

Those fixtures are not a production dataset release. Automatic validation of
all generated evidence and arbitrary model-written explanations has not been
established. Matched renderers, full prompt round trips, release receipts, and
the production audit remain requirements.

No composition-adapter training or final evaluation result is reported.
Earlier training tables, benchmark allocations, and mixed-rule calibration
results are retained in the historical experiment records and the pre-pivot
report archive. They are not relabelled as evidence for the new study.

== Limitations

Controlled language and explanation-supported generation restrict the scope
of the synthetic dataset. Excluded categories and rejection counts must be
reported rather than hidden. Solver checks depend on correct spatial encoding,
and an encoding argument does not formally verify the Python implementation
or Z3.

Selection and Count can require evidence beyond simple axis paths. Distinct
points couple the axes, and independently possible memberships do not determine
joint counts. A failed proof search cannot be treated as a semantic label.

The six-option interface intentionally merges several semantic reasons into
E or F. Letter accuracy therefore does not establish correct reason
classification or knowledge of unlisted values. Matched explanations permit
a representation comparison but do not reveal a model's internal computation.

One initial training seed limits conclusions about stability. Shared story
views reduce the number of independent observations. External SpatialMap
results cannot establish broad generalization, and prior inspection of its
failures makes the study direction post-hoc. These boundaries remain part of
the report regardless of whether the supporting training results are positive.
