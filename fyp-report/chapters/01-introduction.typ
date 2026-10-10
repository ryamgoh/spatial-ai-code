#pagebreak(weak: true)
= Introduction

== Motivation

Spatial reasoning from language requires a model to identify objects, preserve
their relative positions, and compose relationships across several statements.
Unlike visual spatial intelligence, this setting removes object detection and
image grounding: the answer must follow from the information stated in text.
It is therefore useful for studying reasoning, intermediate representations,
and generalisation without conflating them with perception.

Reliable evaluation requires more than an answer that is true in one hidden
map. A benchmark may generate a complete world, expose only part of it as text,
and retain an answer derived from the complete world. That answer is possible,
but it is not necessarily entailed by the information available to the model.
Scoring it as uniquely correct rewards guessing omitted state rather than
reasoning from evidence.

This project studies one formal response to that problem. It defines
qualitative two-dimensional relations over finite named objects, interprets
answers across every spatial world satisfying the visible premises, and uses
the resulting semantics both to audit an existing benchmark and to construct
checked supervision for large language models.

== Problem and Scope

SpatialEval Spatial-Map TQA is the motivating case study. Its 1,500 released
text questions comprise Direction, Which, and Count queries. The source map is
not available in the release, so this work does not attempt to recover it.
Instead, it asks what follows from the released text under a declared
open-world semantics: unstated relationships remain unknown, a candidate is
possible when some satisfying map supports it, and it is entailed only when
every satisfying map supports it.

The same semantics underlies SpatialEntail, a controlled data and evaluation
framework over all eight compass directions and finite Boolean combinations of
spatial atoms. SpatialEntail generates typed answer certificates, renders
Natural and Symbolic supervision, checks Symbolic model output by replay, and
records constructive witnesses for audit. The project is text-only and does not
claim unrestricted spatial language understanding, metric geometry, visual
grounding, or embodied navigation.

== Research Questions

The report addresses three questions.

#figure(
  table(
    columns: (auto, 1fr),
    inset: (x: 7pt, y: 5pt),
    stroke: .5pt + luma(120),
    fill: (_, y) => if y == 0 { luma(235) } else { none },
    table.header([*RQ*], [*Question*]),
    [RQ1], [How does SpatialMap evaluation change when answers are defined over all worlds satisfying the visible premises rather than one latent generating world?],
    [RQ2], [Do replay-checked reasoning traces improve answer accuracy and structural generalisation over answer-only and invalid-reasoning supervision?],
    [RQ3], [How do Natural versus Symbolic traces, and local proofs versus proof-plus-rank-witness traces, affect accuracy, process validity, cost, and transfer?],
  ),
  caption: [Research questions linking benchmark validity, checked supervision, and representation.],
) <research-questions>

== Contributions

First, the project separates latent-world truth from textual entailment. Its
solver represents each exact compass direction through X- and Y-axis
comparisons, supports `NOT`, `AND`, `OR`, implication, and equivalence, and
distinguishes consistency, possibility, entailment, ambiguity, and
impossibility.

Second, the semantics is applied to the untouched SpatialMap-TQA release. The
audit preserves every published question and oracle, checks complete candidate
domains, and records counter-witnesses where several answers remain possible.
The corrected evaluation view adds an explicit _Cannot be determined_ option
without overwriting the released benchmark.

Third, SpatialEntail turns the same formal contract into a generation and
supervision pipeline. Accepted problems are solved, certificate-checked,
rendered, parsed back, and re-solved. They carry explicit construction
provenance, structural signatures, measured difficulty, and exact tokenizer
admission metadata.

Finally, the planned learning study compares answer-only, checked Natural,
checked Symbolic, corrupted Symbolic, and query-local Symbolic supervision over
Qwen3.5-2B and Qwen3.5-4B. It separates matched interpolation, unseen-depth
extrapolation, held-out formula composition, and external transfer. The model
experiments remain unrun in this draft; implementation checks are not presented
as learning results.

== Report Organization

Chapter 2 positions the work against textual spatial benchmarks, spatial
traces, explicit maps, and verified reasoning. Chapter 3 defines the formal
problem and SpatialEval diagnosis. Chapter 4 presents SpatialEntail's solver,
certificates, generator, difficulty taxonomy, and supervision views. Chapter 5
predeclares the model, data, quality-review, and evaluation protocols. Chapter
6 reports completed audit and artifact evidence and reserves the frozen model
result analyses. Chapter 7 concludes. Formal proofs, complete artifact examples,
configuration details, and extended results are placed in appendices.
