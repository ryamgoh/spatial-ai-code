#pagebreak(weak: true)
= Related Work

== Text-Only Spatial Benchmarks

StepGame established controlled multi-hop directional reasoning over eight
relations. Its models train on chains of length one through five and are tested
through length ten, providing the closest direct precedent for SpatialEntail's
held-out depth ladder @shi2022stepgame. Its questions nevertheless request one
relation from one generated chain; they do not represent open-world answer
sets, Which queries, or correlated Count domains.

SpartQA broadens text-only spatial reasoning to relation, block, object, and
yes/no/unknown questions @mirzaee2021spartqa. Its human-authored portion is a
particularly useful language and epistemic transfer test. SpaRTUN supplies
formal relation annotations and multi-hop synthetic supervision, while ReSQ
tests transfer to human-written descriptions @mirzaee2022spartun. These
benchmarks motivate reporting native task contracts rather than forcing every
external label into one compass-only metric.

SpatialEval evaluates spatial reasoning in textual, visual, and combined
modalities @wang2024spatialeval. Its released Spatial-Map TQA text is the case
study in this report because a shared latent-world oracle need not be uniquely
recoverable from a partial textual description. SpatialText later makes
non-omniscient textual scenes an explicit diagnostic condition
@jiang2026spatialtext, while MentalMap evaluates progressively richer textual
world models and structured graph outputs @pan2026mentalmap.

== Spatial Paths and Symbolic Supervision

SpaRP constructs deductively verified, query-directed paths with symbolic
spatial reasoners, verbalizes them, and fine-tunes Llama models on the resulting
traces @rizvi2024sparp. It is the closest precedent for solver-derived spatial
proof-path supervision. SpatialEntail differs by requiring complete
premise-relative answer evidence, a separate replay checker, constructive model
certificates, and explicit open-world answer contracts.

Chain-of-Symbol replaces verbose natural-language reasoning with compact
relation symbols and reports improvements from few-shot prompting
@hu2024chainofsymbol. Graph-based StepGame training similarly asks models to
extract ordered relation triples before answering @zhou2024graphsynthetic.
Q-Chain derives formal spatial chains and converts their consistency relations
into differentiable training constraints @premsri2025neurosymbolic. These works
show that symbolic spatial structure is established prior art; the research
question here concerns checked evidence, representation cost, and held-out
generalisation rather than the novelty of symbols.

== Procedural Traces and Explicit Maps

SODA trains an Observe–Orient–Decide–Act procedure over SPOD-143k with SFT and
GRPO @bai2026soda. Some tasks calculate coordinate deltas or maintain a local
grid position, demonstrating that structured spatial state tracking can be
learned. Its task tiers and external SPACE/application evaluations are broad
transfer tests, not a controlled train-short/test-long depth split.

Text2Space directly fine-tunes an LLM to construct ASCII layouts from spatial
descriptions @huang2026text2space. The paper identifies a read–write asymmetry:
models read supplied layouts more reliably than they construct them, generated
layout errors can harm answers, and explicit construction training improves
later text-only reasoning. This motivates SpatialEntail's secondary comparison
between a query-local proof and the same proof plus a qualitative rank witness.
SpatialEntail's setting is additionally underdetermined: several rank-compressed
maps may satisfy the same visible premises, so semantic witness checking is more
appropriate than exact matching to one arbitrary completion.

== Verified Reasoning and Systematic Generalisation

ProofWriter generates natural-language proofs and shows that iterative
single-rule proof construction extrapolates to unseen proof depths much better
than emitting an entire proof at once @tafjord2021proofwriter. PrOntoQA
constructs an ontology and proof before translating them to language and finds
that locally valid deductions do not guarantee successful global proof planning
@saparov2023prontoqa. Faithful Chain-of-Thought delegates final execution to a
deterministic solver @lyu2023faithfulcot, while LogicGuide constrains deduction
inside a theorem-proving environment @poesia2024logicguide.

`interwhen` monitors intermediate claims on the same SpatialMap task family by
translating them into Z3 constraints and steering generation after detected
violations @bhat2026interwhen. This rules out broad novelty claims about using
Z3 or checking intermediate spatial claims. SpatialEntail's narrower distinction
is a typed dependency-explicit certificate replayed locally while a separate
SMT encoding decides global possibility and entailment.

Outside space, CLUTRR trains on shorter kinship chains and tests longer chains,
held-out rule combinations, and unseen paraphrases @sinha2019clutrr. CFQ's
maximum-compound-divergence splits expose primitive operations while holding
out their compounds @keysers2020cfq. Recent shortest-path work distinguishes
transfer to unseen maps from length scaling to paths longer than training
@tong2026shortestpath. These distinctions motivate separate SpatialEntail
results for interpolation, depth extrapolation, formula composition, language
shift, and external transfer.

== Position of This Work

The individual components are not independently novel: textual spatial tasks,
solver-derived paths, symbolic traces, map construction, proof generation, and
SMT-backed monitoring all have strong precedents. SpatialEntail studies their
integration under one explicit contract: visible-premise all-model semantics,
proof-template-first and premise-first provenance, replay-checked answer
certificates, paired Natural/Symbolic projections, and independently frozen
generalisation suites. Any priority claim is therefore limited to this
combination and must use _to our knowledge_ language.
