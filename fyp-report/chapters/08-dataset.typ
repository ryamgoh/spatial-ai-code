#pagebreak(weak: true)
= SpatialEntail Dataset Construction

== Generation and Provenance

The implementation distinguishes generation modes by construction order.
World-first proposals use a sampled map, discard its proposed answer, and solve
the visible premises. Proof-template-first proposals instantiate a chosen rule
structure before constructing a certificate. Premise-first proposals sample
visible relation premises and a query without a coordinate map, then solve and
classify the result. Witness coordinates are requested only after solving.
Certificate-first would require the typed derivation to precede the problem;
that stronger provenance is not claimed here.

Every accepted problem is solved, checked against its requested semantic cell,
and accompanied by replayed evidence. The rendered question must parse back to
the same structured problem and answer contract. Unsupported proof construction,
inconsistency, and unmet policy constraints cause rejection. Accepted rows
therefore represent a selected construction fragment; rejection rates are part
of the dataset description, not evidence about unrestricted sampling.

== Difficulty and Menus

Controls include query family, ambiguity size, positive proof depth, and source
premise and entity budgets. Positive axis depth measures the retained positive
derivation; it does not measure all exclusion or Count reasoning. Semantic
premise-deletion checks include the effect of exclusions and memberships on the
answer domain. The checker greedily removes premises while retaining the
positive proof support and rechecking the remaining set. Thus the retained
support is jointly sufficient for the measured semantics. It is order-dependent
and need not be a minimum support or a measure of human reasoning difficulty.

Special options and ordinary options are shuffled together using a reproducible
seed. Answer-position distributions and semantic-cell frequencies must be
reported after admission. This prevents a fixed undetermined-answer letter
from serving as the synthetic task's label shortcut.

== Splits and Admission

All answer, format, and control variants share a base identity and remain in one
train, development, or test split. Canonical source signatures ignore object
names and premise or commutative-operand ordering. Canonicalization is exact
within a permutation budget and otherwise uses a conservative invariant that
may group distinct problems. Signature separation prevents overlap under that
identity; it does not itself withhold a rule composition or establish an
out-of-distribution test. Such claims require explicit cells reserved before
model selection, with overlap checks across all three splits.

Admission uses the chosen tokenizer and chat template on the actual prompt and
assistant target. It checks both the full training sequence and the evaluation
prompt plus generation reserve. If any required variant exceeds the limit, the
entire paired base problem is rejected. The manifest records tokenizer identity,
budgets, observed lengths, and rejection counts. Evaluation consumes the admitted
prompt without applying another template. Truncation is not a dataset repair.

== Bounded Pilot Scope

The first configured pilot covers unique and ambiguous Direction queries with
matched premise and entity budgets. It compares four supervision arms on paired
base problems. Which, Count, broader Boolean curricula, and additional transfer
sets are supported or planned at different levels of the pipeline; their
presence in APIs does not make them measured pilot conditions.

A tokenizer/generator smoke admitted 80 base problems and 320 paired rows,
with 48/16/16 training/development/test bases. This validates the bounded
configuration's preparation path, not a difficulty trend or training benefit.

A separate holdout preparation matrix reserves premise-first IFF cells for
development and premise-first implication cells for test, with atomic training
cells. Its eight/four/four base counts and common three-entity, two-premise
budget make this a small coverage check. It demonstrates explicit cell routing;
it does not measure generalisation or replace a calibrated final benchmark.
