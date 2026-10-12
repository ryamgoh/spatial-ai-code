# SpatialEntail trace formats

The existing Natural and replayable Symbolic formats are renderings of one
checked answer certificate. They are data artifacts, not observations of an
LLM's hidden reasoning.

## Responsibility boundary

An answer certificate is the machine-checkable artifact used to generate,
verify, and audit a solution. It is not the default model response or central
SFT target. The replayable NDJSON format documented below remains useful for
pipeline tests and audits, but it must not be
called Chain-of-Symbol or treated as the normal Symbolic SFT representation.

Central SFT uses answer-only or a readable Natural worked solution rendered from
verified reasoning. A future optional Symbolic arm may use a compact CoS-like
notation only to test token efficiency. That notation is a separate format: it
must be shorter than Natural on the admitted corpus and need not reproduce the
audit certificate.

## Natural trace

The Natural view uses controlled prose with premise, derivation, dependency,
and branch identifiers. It is derivation-first: a unique Direction trace gives
the proof before its consistency witness and final domain/answer decision.
Model evidence uses qualitative X/Y rank orders rather than arbitrary numeric
coordinates. Count prints fixed memberships once and retains evidence for the
remaining correlated assignments.

Natural gold is trustworthy because it is rendered from a checked certificate.
Arbitrary Natural model output has no parser or process replay checker. Report
Natural process validity as not applicable, not false.

## Replayable Symbolic certificate view

The Symbolic view is versioned NDJSON for machine parsing and replay. It is not
the compact CoS-like representation proposed for possible future SFT. A
complete serialized certificate view contains exactly two records in this order:

1. a query-specific reasoning certificate; and
2. an answer-decision record.

Offline validation parses the reasoning, replays its dependencies and witnesses
against the visible problem, validates the decision, compares their
possible-value domains, and checks the canonical final answer footer.

## Dataset views and audit fixtures

- `checked-trace` is the implemented container for valid Natural or Symbolic
  renderings. Central SFT consumes its Natural rendering only.
- `answer-only` omits the worked solution; its process status is absent.
- `corrupted-trace` is a Symbolic-only verifier fixture. It preserves valid
  syntax, record order, decision, and final answer while changing a typed
  assertion until replay rejects it. It is not an SFT control.
- Audit rendering uses `spatial-audit-certificate-v1` and may contain numeric
  coordinate witnesses for Direction, Which, and Count. Coordinates are not
  ordinary training targets.

Natural and replayable Symbolic expose the intended checked dependencies, but
serve different responsibilities. The first can be a worked-solution target;
the second is a machine-checkable release artifact. They are not two central
training packages or a notation-only comparison.

## Symbolic proof grammar

SpatialEntail V2 serializes each proof or refutation segment as one compact JSON
object. JSON supplies the lexical grammar, while the versioned `schema` field
selects the typed record grammar.

Supported schemas are:

- `spatial-direction-proof-v1`
- `spatial-direction-refutation-v1`
- `spatial-formula-refutation-v1`
- `spatial-direction-trace-v2`
- `spatial-which-trace-v1`
- `spatial-count-trace-v2`
- `spatial-answer-decision-v1`

A Direction proof has exactly these fields:

```json
{
  "schema": "spatial-direction-proof-v1",
  "conclusion_step": "Q-DIR",
  "steps": []
}
```

Each step has exactly `id`, `rule`, `conclusion`, `inputs`, `premise_index`, and
`branch`. Conclusions carry an explicit `kind`: `relation`, `not`, `and`, `or`,
`implies`, `iff`, `axis`, `direction`, or `contradiction`. Direction and rule
tokens use their canonical enum names or values as defined by the codec.
Unknown fields, missing fields, unknown enum values, empty identifiers, and
incorrect primitive types are rejected.

The symbolic renderers emit this grammar. Their matching `parse_symbolic_*`
functions reconstruct the typed certificate against the original
`SpatialProblem` and invoke the local replay checker. Parsing is therefore not
sufficient for acceptance: every dependency, scope, rule application, model,
candidate domain, and conclusion must also pass its checker.

The grammar deliberately uses certificate object identifiers rather than
display labels. This keeps round trips unambiguous when human-readable labels
contain spaces or punctuation.

The Direction trace schema wraps either a unique positive proof or exhaustive
ambiguous candidate evidence. The Which and Count schemas preserve their typed
membership and joint-assignment evidence. Model evidence contains ordered
equality groups for each axis rather than numeric coordinates; the parser
reconstructs a rank model and passes it to the model checker. Each parser also
checks that the declared possibility domain agrees with the reconstructed
proof, model, and refutation evidence.

A complete Symbolic certificate view contains exactly two newline-delimited
JSON records: one query-specific reasoning envelope and one answer-decision record.
`parse_symbolic_training_trace` checks both records and verifies that their
possible-value domains agree. The decision record is also checked against the
semantic menu answer, including answer mode, resolution status, and selected
letters.

Unique Direction certificate records include a checked consistency witness as well
as the derivation. Count records store fixed memberships once; assignment
coverage is checked only over subsets compatible with those memberships. Count
model claims omit an expanded exact-count formula: parsing reconstructs it from
the enclosing query and count value. Remaining contingent assignments preserve
joint dependencies and can still require exponential evidence.

This grammar covers the replayable Symbolic certificate view only. Natural gold
is rendered from checked source evidence, but arbitrary Natural model prose is
not parsed or replayed. Offline validation additionally checks the final answer
footer against the decision and gold answer.

Audit output uses `spatial-audit-certificate-v1` with coordinate-bearing witness
records for Direction, Which, and Count. It is separate from both Natural SFT
targets and any future compact CoS-like notation.
