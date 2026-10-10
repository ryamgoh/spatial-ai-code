#pagebreak(weak: true)
= SpatialEntail Certificate Supervision

== Motivation

A correct answer can be supported by an invalid explanation, and a valid
example world need not justify entailment. SpatialEntail therefore separates
semantic answers, local proof evidence, model witnesses, and menu decisions.
The research question is whether supervision built from these checked objects
improves accuracy or generalisation beyond task exposure alone.

== Evidence Contract

Direction evidence combines positive proofs, possibility witnesses, and
exclusions. A unique Direction training trace includes a consistency witness as
well as its derivation: a classical proof alone could be vacuous for inconsistent
premises. Which evidence classifies every declared candidate as entailed,
contingent, or impossible. Count evidence retains joint membership constraints;
individually contingent candidates cannot be counted as independent choices.

Count certificates store fixed membership classifications once and consider
only assignments compatible with those classifications. Each remaining
impossible assignment requires checked exclusion evidence. This removes
repeated fixed facts, but the number of assignments can still be exponential
in the number of contingent candidates. Context admission therefore remains
necessary even after compression.

== Natural and Symbolic Views

Natural targets narrate each checked step, its identifier, dependencies, and
branch scope, together with query-specific witnesses and exclusions.
Symbolic targets serialize query evidence and an answer-decision record using
a strict typed grammar. Qualitative rank orders express model evidence without
numeric coordinate choices; audit renderers retain coordinate-bearing witnesses.
Both targets expose the intended checked dependencies. This replaces the
earlier compact Natural summaries, but a mapping across the frozen corpus is
still needed to establish matched rendered information. Token lengths, cognitive
load, and process-checking capability can differ. The pilot therefore compares
supervision packages without attributing any effect to notation alone.

The checker validates generated evidence before either target is rendered.
For model-produced Symbolic output, it reconstructs and replays evidence against
the visible problem, checks the menu decision, and checks the final answer
footer. There is no corresponding parser for arbitrary Natural prose. Natural
answer accuracy is measurable; Natural process validity is reported as not
applicable rather than a failed proof.

A valid external certificate establishes a relation between premises and
conclusion. It does not reveal whether the model used those steps internally,
whether the explanation caused its answer, or whether a reader can follow it.
Those questions require separate intervention or human-evaluation studies.
