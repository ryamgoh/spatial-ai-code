#pagebreak(weak: true)
= Spatial Reasoning in Large Language Models

== Spatial Reasoning from Language

Spatial reasoning covers several related capabilities: identifying the
relationship between two entities, locating entities relative to a reference,
counting entities in a spatial region, maintaining a spatial state across
multiple statements, and composing relations that are not stated directly.
Language models must perform these operations from symbolic descriptions rather
than direct access to the represented environment.

This separation is useful experimentally. A visual or embodied system may fail
because it did not detect an object, grounded a relation incorrectly, lost
information across views, or reasoned incorrectly after perception. A textual
task provides the relevant relations directly and therefore isolates the
reasoning component. This project consequently evaluates LLMs only. Multimodal,
video, three-dimensional, and embodied spatial systems remain relevant context
but are not combined with the text-only results.

== Existing Evaluation Settings

Textual spatial benchmarks differ in what they treat as the problem. Some use
short controlled stories to test multi-hop direction composition. Others
include richer relations, containment, coreference, or explicit unknown
answers. SpatialEval compares text-only, vision-only, and combined modalities
across map, grid, real-image, and navigation tasks. These settings cannot be
placed on a single leaderboard without preserving their ontology, modality,
answer contract, split, prompt, and metric.

Recent work has expanded spatial evaluation toward images, egocentric video,
three-dimensional scenes, robotics, and embodied action. Text-only research has
continued in parallel. Its value is not that language replaces perception, but
that it permits controlled tests of relational inference, incomplete
information, state tracking, and logical composition.

== DRAFT — Text-Only Spatial Benchmarks

#block(
  fill: rgb("#fff7ed"),
  stroke: .6pt + rgb("#c2410c"),
  inset: 10pt,
  radius: 3pt,
)[
  *Draft status.* This related-work section records the present literature
  interpretation before the SpatialEntail suites and proof schema are frozen.
  Its novelty language is intentionally conservative. The underlying search
  ledger is `docs/research/frontier-assessment.md`.
]

Controlled text-only spatial reasoning predates current LLMs. StepGame provides
synthetic stories requiring one to ten relation-composition steps and separates
short-hop training from longer-hop tests @shi2022stepgame. SpartQA broadens the
task to relation, block, object, and yes/no/unknown questions over richer spatial
language @mirzaee2021spartqa. SpaRTUN then supplies synthetic formal
representations and reasoning annotations, while ReSQ tests transfer to
human-written descriptions @mirzaee2022spartun. These benchmarks establish
multi-hop composition, explicit unknown answers, formal spatial rules, and
synthetic-to-natural transfer as existing research directions.

SpatialEval differs in modality and answer format. Its Spatial-Map TQA subset
provides textual pairwise relations followed by Direction, Which, or Count
multiple-choice questions, while aligned VQA and VTQA views retain images
@wang2024spatialeval. The benchmark is the external case study in this report,
but it is not the only recent pure-text setting. By 2026, SODA trains LLMs with
143,000 pure-text spatial control examples using SFT and GRPO @bai2026soda;
SiT-Bench spans coordinate-aware textual variants of navigation, perspective,
manipulation, and geometric tasks @guo2026sitbench; SpatialText combines human
descriptions with programmatically generated scenes and explicitly includes
non-omniscient cases @jiang2026spatialtext; and MentalMap evaluates multilingual
textual world-model construction over structured indoor scenes
@pan2026mentalmap. The defensible gap is therefore not “text-only spatial
reasoning,” which is already crowded, but a narrower form of qualitative
spatial entailment with explicit proof and generation provenance.

== DRAFT — Spatial Paths, Symbols, and Solvers

Several works already produce or exploit structured spatial reasoning. PistaQ
separates neural extraction from deterministic Prolog inference, showing that
formal reasoning can be reliable once the text has been grounded correctly
@mirzaee2023pistaq. SpaRC characterises the spatial properties that affect
composition, and SpaRP constructs symbolic reasoning paths over StepGame and
SpaRTUN, verbalises those paths, and uses them for LLM fine-tuning
@rizvi2024sparp. SpaRP is the closest precedent for solver-derived spatial
traces: a claim that this project is the first to generate verified spatial
reasoning paths would be false.

The neuro-symbolic Q-Chain approach uses 79 spatial rules to forward-chain over
SpaRTUN annotations and converts the resulting consistency relationships into
training constraints @premsri2025neurosymbolic. Graph-based synthetic training
likewise constructs relational graphs and samples reasoning chains for StepGame
SFT @zhou2024graphsynthetic. These methods show that rule- or graph-derived
spatial supervision is established, even though they do not use the exact
proof-checker and all-model semantics proposed here.

Representation is also established as an experimental variable. Chain-of-Symbol
rewrites corrected natural-language demonstrations into compact symbolic
sequences and reports large prompting gains on spatial reasoning and planning
tasks @hu2024chainofsymbol. This motivates a controlled Natural-versus-Symbolic
comparison; it does not support claiming symbolic traces themselves as a new
contribution.

The strongest recent counterclaim is `interwhen`. It monitors intermediate
claims produced during text-only SpatialMap reasoning, translates directional
claims into Z3 constraints, and feeds detected contradictions back to the model
@bhat2026interwhen. Consequently, neither “first use of Z3 in textual spatial
reasoning” nor “first verification of intermediate SpatialMap claims” is
defensible. The remaining distinction is that consistency monitoring of
free-form claims is not the same artifact as a dependency-explicit proof whose
local steps are replayed by a small checker and whose global semantics are
separately decided by an SMT encoding.

== DRAFT — Proof-First and Proof-Carrying Reasoning Outside Space

The generation order discussed in this project also has clear logical-reasoning
precedents. ProofWriter generates synthetic theories and proof DAGs, supports an
open-world `Unknown` class, and trains models to emit natural-language proofs
@tafjord2021proofwriter. PrOntoQA goes further in the direction relevant here:
it constructs an ontology, walks it to generate a proof, and then translates
the ontology and proof into a context, query, chain of thought, and answer
@saparov2023prontoqa. Proof-first generation is therefore not a new generic
idea.

Faithful Chain-of-Thought interleaves natural-language decomposition with
executable symbolic programs so that a deterministic solver, rather than the
free-form rationale, produces the answer @lyu2023faithfulcot. LogicGuide
constrains language-model deduction through a theorem-proving environment,
making accepted reasoning steps independently checkable @poesia2024logicguide.
PCRLLM explicitly requires each generated step to identify its premises,
inference rule, and conclusion @li2025pcrllm. These works establish executable,
certified, and proof-carrying reasoning outside the spatial domain.

The relevant lesson is also a limitation: formal checking proves a conclusion
relative to the formalised premises, but does not by itself prove that those
premises faithfully represent the natural-language question. SpatialEntail can
reduce this gap because its text is rendered deterministically from typed atoms,
then reparsed and re-solved. It cannot claim that the general autoformalisation
problem has been solved.

== DRAFT — Generation Provenance and the Remaining Gap

This project distinguishes four construction orders:

+ *World-first:* sample a complete coordinate world, render some relations, and
  read the answer from that same world. This proves possibility in one selected
  model, not entailment from the visible text.
+ *Proof-template-first:* choose a rule/path template, instantiate a problem,
  then construct and replay the resulting certificate. This is the current
  controlled generator.
+ *Certificate-first:* sample a complete typed derivation before constructing
  the problem text, then replay and independently re-solve it. This stricter
  provenance is not claimed for the current generator.
+ *Premise-first:* sample visible premises and a query without selecting an
  answer, discover the complete semantic class after solving, and admit the
  problem to the resulting evaluation bucket.

The implemented design uses proof-template-first problems for controlled trace
supervision and records world-first rows separately. A distinct premise-first
route now proposes visible relations without coordinates, then solves and
classifies them. Its value for structural generalisation remains untested.
Natural and Symbolic targets originate from accepted certificates and expose
checked dependencies through prose or typed records. Their corpus-wide evidence
mapping and token costs remain to be audited.
A small rule checker validates local steps; a separately implemented SMT
encoding checks global possibility, entailment, ambiguity, and inconsistency.

#figure(
  table(
    columns: (1.25fr, auto, auto, auto, auto),
    table.header(
      [*Prior line*],
      [*Spatial*],
      [*Proof-first*],
      [*Premise-first*],
      [*Dual check*],
    ),
    [SpaRP], [Yes], [Partial], [No], [No],
    [Q-Chain training], [Yes], [Partial], [No], [No],
    [`interwhen`], [Yes], [No], [No], [Partial],
    [ProofWriter], [No], [Partial], [Partial], [No],
    [PrOntoQA], [No], [Yes], [No], [No],
    [PCRLLM / LogicGuide], [No], [Partial], [No], [Partial],
    [Proposed SpatialEntail], [Yes], [Yes], [Yes], [Yes],
  ),
  caption: [Conservative mechanism-level overlap with the proposed SpatialEntail
    design. “Partial” indicates a related mechanism, not an identical protocol.],
) <spatialentail-prior-overlap>

The individual cells in the final row are not individually novel. The bounded
absence finding from the present primary-source search is that no located work
combines all of them in one text-only qualitative spatial benchmark and learning
pipeline. In particular, no located source simultaneously uses replayable
proof-first spatial certificates, premise-first open-world evaluation, local
proof checking independent of a global SMT oracle, and paired Natural/Symbolic
renderings of the same derivation.

The working novelty claim must therefore concern integration and controlled
evaluation, not invention of proofs, symbols, solvers, spatial paths, text-only
training, or verifiable rewards. A suitably cautious formulation is:

#quote[
  SpatialEntail studies an underexplored intersection of proof-carrying data,
  model-theoretic spatial evaluation, independent verification, and controlled
  trace representation.
]

Any eventual priority claim should use “to our knowledge,” name SpaRP,
PrOntoQA, Chain-of-Symbol, `interwhen`, and proof-carrying reasoning as the
nearest precedents, and be rechecked before submission.

== Benchmark Validity

A generated benchmark can possess a correct internal world and still expose an
invalid evaluation item. The generator may select an answer from a complete
latent map, then render a description that omits a relation needed to recover
that answer. The selected answer remains true in the latent map, but it is no
longer the only answer supported by the text.

This distinction requires an explicit answer semantics. Under an open-world
interpretation, an unstated relation is unknown rather than false. A
single-answer question is determined only when every spatial model satisfying
the prompt agrees on the same answer. Questions that admit several answers
must either request the complete possibility set or provide an explicit
undetermined response.

== Position of This Work

The project studies formal semantics as both an evaluation tool and a source of
LLM supervision. SpatialEval Spatial-Map TQA is the first case study because it
provides parallel textual and visual views of generated maps and exposes the
difference between latent-map truth and textual entailment. The later
SpatialEntail benchmark uses the same semantics constructively to generate
controlled reasoning problems, explanations, and outcome checks.
