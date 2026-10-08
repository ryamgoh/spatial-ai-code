#pagebreak(weak: true)
= Introduction

== Background

Spatial reasoning requires a model to represent objects, preserve their
relative positions, and compose relationships across multiple statements. In a
language-only setting, the model cannot inspect the environment directly. It
must reason from the information expressed in the prompt. This makes textual
spatial tasks useful for studying reasoning independently of visual perception,
object detection, and image grounding.

Recent spatial-intelligence research increasingly includes images, video, 3D
scenes, and embodied interaction. These settings are important, but they make
it difficult to determine whether an error arose from perception or reasoning.
This project focuses on large language models and qualitative spatial
descriptions. Its formal domain uses eight compass directions over a finite set
of named locations. Problems do not provide coordinates, distances, or metric
measurements.

Reliable evaluation requires more than a plausible answer. A benchmark may be
constructed from a complete latent map while exposing only a partial textual
description to the model. An answer can be true in the latent map without being
entailed by the text. Treating that answer as uniquely correct conflates
spatial reasoning with guessing information that was never presented. This is
not the ordinary closed-world rule that unrecorded claims are false; it is a
mismatch between truth in one privileged latent world and entailment from the
evidence available to the model.

== Problem and Objectives

SpatialEval Spatial-Map TQA is the first case study in this work. Its released
questions describe map relationships in text and provide four answer options.
The public release contains generated data and evaluation code, but not the
Spatial-Map generator. Consequently, the original latent coordinates and any
undocumented generation assumptions cannot be reconstructed directly.

This project instead asks what follows from the released text under an explicit
formal semantics. Unstated relationships are treated as unknown rather than
false. Every complete spatial model satisfying the premises is considered. A
candidate answer is possible when it holds in at least one such model and is
entailed when it holds in all of them. If two satisfying models give different
answers, the question does not have one textually determined answer under a
single-answer contract.

The project has four objectives:

+ Define a qualitative spatial semantics for exact compass relations,
  incomplete information, and finite propositional combinations of spatial
  claims.
+ Validate a solver that decides consistency, possibility, and entailment and
  returns coordinate witnesses for audit purposes.
+ Audit the untouched SpatialMap-TQA release and derive
  SpatialMap-TQA-Corr by adding an explicit _Cannot be determined_ option where
  the text does not entail one answer.
+ Use the same semantics to construct SpatialEntail and test whether prompting,
  supervised fine-tuning, reasoning representation, and optional
  solver-verifiable reinforcement learning improve spatial reasoning in LLMs.

The main research question is:

#quote[
  Can formal spatial semantics improve both the validity of spatial-reasoning
  benchmarks and the spatial reasoning learned by large language models?
]

== Contributions

First, this work separates truth in one selected map from entailment across all
maps consistent with a prompt. The formal model represents each compass
relation through X- and Y-axis comparisons and supports negation, conjunction,
disjunction, implication, and equivalence. Satisfiability determines whether a
claim is possible; unsatisfiability of its negation determines whether it is
entailed.

Second, the formal model is applied to SpatialMap-TQA without modifying the
original benchmark. The audit preserves each published question and oracle,
classifies its answer status, and constructs alternative witnesses where the
text admits more than one answer. SpatialMap-TQA-Corr is stored separately and
adds a fifth option for questions that are undetermined under the declared
single-answer semantics.

Third, the project introduces SpatialEntail as a controlled benchmark for LLM
spatial reasoning. It extends the task to all eight compass directions,
Direction, Which, and Count queries, explicit answer contracts, and measured
structural difficulty. Generated questions are accepted only after they are
rendered, parsed back into the formal representation, and solved again.

Finally, the project uses SpatialEntail to compare direct and structured
prompting, answer-only supervision, Natural reasoning traces, Symbolic
reasoning traces, and proof-structure ablations. These comparisons are designed
to determine whether intermediate reasoning supervision improves held-out
spatial structure rather than only teaching answer format. A later
reinforcement-learning experiment is conditional on the supervised model
leaving sufficient headroom and producing usable reward variance.

== Report Structure

The report is organised in two parts. The first develops the formal semantics,
validates the solver, documents the end-to-end system architecture, and uses
SpatialEval as a case study in benchmark auditing and correction. The second
constructs SpatialEntail and evaluates prompting, supervised fine-tuning,
reasoning representations, structural generalisation, and, if justified by the
supervised results, reinforcement learning.

This organisation follows the central methodological link of the project: the
same semantics first determines what a benchmark may validly score and then
determines what labels, traces, and rewards may be used to train an LLM.
