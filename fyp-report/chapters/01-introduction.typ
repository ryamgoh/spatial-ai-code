#pagebreak(weak: true)
= Introduction

== Motivation

Spatial reasoning from text requires a model to compose stated relations and
distinguish what is guaranteed from what is merely possible. Partial premises
can admit several maps. An answer that holds in one such map need not follow
from the information available to the reader.

SpatialEntail addresses this distinction through controlled spatial problems,
solver-verified semantic answers, and auditable explanations. The dataset is
the primary contribution. A supporting learning study asks how two matched
explanation representations affect answer accuracy and generated-token cost.

== Scope

The study uses atomic pairwise relations over distinct points in a shared
two-dimensional plane. Its reasoning core decomposes exact directions into
axis comparisons, propagates strict order and equality, and recombines the
resulting comparisons. Unstated relations remain unknown. Coordinates do not
encode unit movement, distances, or a bounded grid.

Direction, Selection, and Count are separate query families. Each supports
Unique Answer, Listed Possibilities, and Exhaustive Possibilities contracts.
Consistent and inconsistent stories are included explicitly. Selection also
distinguishes no match, multiple certain matches, and uncertain membership.

This work does not evaluate a mixed logical-proof curriculum, broad structural
generalization campaign, model sweep, visual grounding system, or navigation
agent. Retired experiments remain archived for provenance rather than becoming
required comparisons.

== Research Objectives

The primary objective is to construct a dataset whose labels, selected evidence,
rendered questions, and matched explanation views pass reproducible quality
checks. Controlled chain lengths, verified distractors, semantic cases, and
menu coverage make the intended comparisons inspectable.

The supporting empirical question is whether Natural and Symbolic explanations
of the same evidence produce different answer accuracy and token cost when
used for matched supervised fine-tuning. Symbolic notation is not assumed to
be shorter or more accurate. The dataset remains useful if neither adapter
improves on its style-matched untuned baseline.

== Contribution and Evidence Boundary

The intended contribution combines controlled atomic spatial composition,
three matched answer contracts, and checkable evidence for definite answers,
possibilities, underdetermination, and inconsistency. Natural and Symbolic
supervision must preserve the same selected evidence and conclusion.

The study currently has a written specification and executable semantic
fixtures. These establish covered examples, not production readiness or
learning gains. This report distinguishes the intended method from completed
evidence throughout.

The pivot was informed by inspected SpatialMap failures. This post-hoc history
must be disclosed. SpatialMap stories and contract variants are external
evaluation only, never training, development, or checkpoint-selection data.

== Report Organization

Chapter 2 gives the research context. Chapter 3 defines spatial semantics and
answer contracts. Chapter 4 specifies evidence construction and release checks.
Chapter 5 describes the bounded matched training and evaluation protocol.
Chapter 6 records the current evidence and limitations; Chapter 7 concludes.
Appendices record the spatial correctness argument and release requirements.
