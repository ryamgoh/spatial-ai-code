#pagebreak(weak: true)
#import "../figures/solver-correctness.typ": paired-countermodels
#import "../figures/spatialeval-modalities.typ": modality-alignment

= SpatialEval as a Case Study

== Benchmark and Release Boundary

SpatialEval evaluates four broad dimensions of spatial intelligence through
Spatial-Map, Spatial-Grid, Spatial-Real, and Maze-Nav. Each task is presented in
text-only, vision-only, and combined vision-text modalities. This work focuses
on the 1,500 released Spatial-Map TQA questions: 500 Direction questions, 500
Which questions, and 500 Count questions @wang2024spatialeval[Section 2.1].

The SpatialEval paper describes a configurable synthetic construction process
in which objects are assigned to a map before textual and visual inputs are
derived. The official release provides the generated datasets, inference code,
and exact-match evaluation code. At the time of this study, the official
repository still described the dataset generator as forthcoming, and neither
the repository nor its linked dataset release contained that generator.

The unavailable generator limits the claim that can be made. This work does not
reproduce the authors' latent coordinates or diagnose a software defect in
their generator. It instead evaluates whether each published answer follows
from the text that was released to the LLM.

== Aligned Modalities and Shared Oracles

SpatialEval describes each problem as having an image and a textual
representation and evaluates the modalities on the same question set
@wang2024spatialeval[Section 2.1]. TQA provides the textual scene description
and question, VQA provides the image and textual question, and VTQA provides
the image, description, and question.

A row-level check of the official dataset release confirms this alignment
@wang2024spatialevaldataset. After replacing only the modality component of
each ID, all 4,635 TQA, VQA, and VTQA keys align in the same order, and
`oracle_answer`, `oracle_option`, and `oracle_full_answer` match for every
aligned triple. Spatial-Map contributes 1,500 of those aligned triples. Their
complete text fields differ by design because VQA omits the textual scene
description.

@spatialeval-modality-alignment shows the resulting evidence boundary.
The same latent benchmark instance supplies each modality and one shared
oracle, but the information exposed to the evaluated model differs.

#modality-alignment <spatialeval-modality-alignment>

== Latent-World Supervision and Open-World Evaluation

SpatialEval does not state that it adopts the logical closed-world assumption.
Under that assumption, a proposition absent from the knowledge base is treated
as false. A fully specified map is instead one complete world: every object has
a fixed position in that world, including relationships that may not appear in
its textual rendering. The shared cross-modal oracle is therefore evidence of
*privileged latent-world supervision*, not evidence that SpatialEval defines
absence from the TQA description as negation.

The distinction can be stated precisely. Let $K$ be the released TQA premises,
$cal(M)(K)$ the set of spatial maps satisfying them, $m^*$ the particular map
used to construct the benchmark instance, and $f(m)$ the answer to its query in
map $m$. SpatialEval records

$ a^* = f(m^*). $

That establishes the answer in one selected map. The TQA text uniquely entails
an answer $a$ only when

$ forall m in cal(M)(K), f(m) = a. $

Consequently, reusing $a^*$ across TQA, VQA, and VTQA does not by itself show
that $K$ entails $a^*$. If two maps in $cal(M)(K)$ satisfy every released
premise but give different answers, the oracle may remain possible while the
text-only question is underdetermined.

The V2 audit makes this evidence boundary explicit by interpreting $K$ under an
open-world semantics. Unstated relations remain unknown, and the solver
considers every map in $cal(M)(K)$. If those maps disagree on the query answer,
`SINGLE` returns _Cannot be determined_. Explicit negation remains informative;
mere absence does not become negation. Under this declared semantics, 833 of
the 1,500 released Spatial-Map TQA questions admit more than one answer, while
667 uniquely entail the published oracle. Chapter 5 reports the audit by query
family and preserves the original benchmark separately from the corrected
evaluation view.

== Hidden-World Truth and Textual Entailment

Consider the abstract pair of statements:

```text
A is Northeast of B.
B is Northwest of C.
```

Both statements place A north of C, but their horizontal effects oppose one
another. Without distances, A may lie northwest or northeast of C. A complete
generator map chooses one arrangement, whereas the two statements do not.

This produces two different questions:

+ *What is true in the sampled map?*
+ *What is entailed by the released description?*

The first question requires access to the latent map. The second can be answered
from the TQA input. A four-option single-answer evaluation silently treats them
as equivalent when it uses the sampled-map answer as the only accepted label.

@spatialeval-paired-countermodels gives a concrete instance from the
released benchmark. For compactness, let R denote Recycle Center, S Sally's
Salon, A Andy's Autos, U Unicorn Umbrellas, N Nightingale Novelties, and T Trail
Hiking Gear. Apart from declaring R to be in the map, the released item contains
the following nine premises.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    table.header([*No.*], [*Subject*], [*Relation*], [*Reference*]),
    [1], [S], [Southeast], [R],
    [2], [A], [Northwest], [S],
    [3], [A], [Southeast], [R],
    [4], [U], [Northwest], [S],
    [5], [U], [Southwest], [A],
    [6], [N], [Southwest], [A],
    [7], [N], [Southwest], [S],
    [8], [T], [Southeast], [R],
    [9], [T], [Southwest], [S],
  ),
  caption: [Released relational premises for item
    `spatialmap.tqa.2003.0`, written using the aliases above.],
) <spatialeval-counterexample-premises>

The question asks for R relative to N and offers Southeast, Southwest,
Northeast, and Northwest. The published answer is Northeast. The premises
force R to be north of N, but they do not order R and N on the X-axis.

The two diagrams below are not reconstructions of the authors' hidden map.
They are countermodels showing that the same textual premises admit different
query answers.

#paired-countermodels <spatialeval-paired-countermodels>

Their exact coordinates are shown in @spatialeval-counterexample-coordinates.

#figure(
  table(
    columns: (auto, auto, auto),
    table.header([*Location*], [*Witness 1*], [*Witness 2*]),
    [R], [$ (1,5) $], [$ (0,5) $],
    [S], [$ (4,2) $], [$ (5,2) $],
    [A], [$ (3,4) $], [$ (4,4) $],
    [U], [$ (2,3) $], [$ (3,3) $],
    [N], [$ (0,0) $], [$ (1,0) $],
    [T], [$ (2,1) $], [$ (2,1) $],
  ),
  caption: [Coordinate witnesses for the released counterexample. Coordinates
    are existence certificates, not recovered generator coordinates.],
) <spatialeval-counterexample-coordinates>

Each premise can now be checked as two inequalities. For example, in Witness 1,
S is southeast of R because $4 > 1$ and $2 < 5$; in Witness 2, the same premise
holds because $5 > 0$ and $2 < 5$. Applying the same check to the remaining
eight rows confirms that both witnesses satisfy the entire released text.
However, Witness 1 gives R northeast of N because $1 > 0$ and $5 > 0$, whereas
Witness 2 gives R northwest of N because $0 < 1$ and $5 > 0$. Thus Northeast is
possible, but it is not uniquely entailed by the text.

== Scope of the Diagnosis

The audit treats the original file as immutable. It preserves every question,
option, and published oracle, then compares that oracle with the complete set of
answers admitted by the text under the declared ontology. A published answer
may be uniquely entailed, possible but non-unique, contradicted, or associated
with inconsistent or unparseable premises.

The central evidence for non-uniqueness is constructive. If two coordinate maps
satisfy every premise but produce different query answers, neither answer is
entailed as the unique result. The coordinates are used only to demonstrate
existence; they are not shown to the evaluated LLM.

== Questions for the Case Study

The case study asks:

+ How many published answers are uniquely entailed by the released text?
+ Which query families are most affected by underdetermination?
+ Are the alternative answers supported by valid counter-witness maps?
+ How sensitive are the results to candidate scope and other modelling choices?
+ Does adding an explicit undetermined option change conclusions about LLM
  performance?
