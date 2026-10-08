# Is SpatialEntail a new research frontier?

*Primary-source research note. Literature checked through 7 October 2026.*

## Executive answer

**The individual ingredients are not new. The exact combination appears to be
new within the primary-source literature found for text-only qualitative spatial
reasoning, but that is an absence-of-evidence claim, not proof of priority.**

The strongest defensible position is that SpatialEntail occupies an
**underexplored integration frontier**:

1. proof-first generation produces a replayable certificate before prose;
2. premise-first generation samples the visible premises and query without a
   target answer, then solves and buckets the actual semantics;
3. a small proof checker replays local derivations while an independently
   implemented SMT encoding checks global possibility, entailment, ambiguity,
   and inconsistency; and
4. Natural and Symbolic traces are deterministic views of the same checked
   proof object, enabling a controlled representation experiment.

No source found in this search implements all four for a **text-only, static,
qualitative spatial entailment** task. The nearest spatial precedent, SpaRP,
builds deductively verified spatial paths and verbalizes them for fine-tuning.
The nearest direct SMT precedent, *interwhen*, monitors intermediate claims on
text-only SpatialMap with Z3. The nearest proof-first precedent, PrOntoQA,
generates a symbolic ontology and proof before translating both to natural
language. The nearest explicitly proof-carrying work, PCRLLM and Proof-Carrying
CoT, is domain-general logic or arithmetic rather than spatial reasoning. These
precedents make a broad claim such as “the first proof-carrying reasoning
framework” untenable. They leave room for a precise **first-combination in
text-only qualitative spatial entailment** claim after a final systematic review
and public artifact comparison.

“New research frontier” should therefore be used as a forward-looking framing,
not a historical fact. The evidence supports “a new benchmark and experimental
programme at the intersection of established lines,” not “a wholly new field.”

## Research question and scope

### Question

Do proof-first or proof-carrying generation, premise-first evaluation,
independent proof checking plus SMT validation, and paired Natural/Symbolic
proof traces together constitute a novel research direction for text-only LLM
spatial reasoning?

### Included

- Text-only spatial question answering and spatial-world reasoning.
- Static qualitative relations, especially compass direction, containment,
  topology, transitivity, partial information, and multi-hop composition.
- Synthetic data generation, trace supervision, neuro-symbolic inference,
  independent verification, and formal answer semantics.
- Adjacent logical and mathematical reasoning only where it calibrates the
  novelty of proof generation, checking, or verifier-filtered data.

### Excluded or boundary-only

- Vision-language, image-grounded, video, 3D perception, robotics, embodied
  interaction, and navigation as primary comparators.
- Euclidean geometry theorem proving, because its diagrams, constructions, and
  theorem libraries define a materially different problem.
- Dynamic text-only control and navigation appear only as boundary evidence
  where a paper also makes a broad pure-text spatial-training claim.

The distinction matters. SiT-Bench is evaluated with text alone, for example,
but much of its content is converted from visual benchmarks and covers
navigation, robotic manipulation, perspective shifts, and metric geometry
([Guo et al., 2026](https://arxiv.org/abs/2601.03590)). SODA is also pure-text,
but its OODA traces are built around dynamic control tasks
([Bai et al., 2026](https://aclanthology.org/2026.acl-long.1382/)). Neither is a
like-for-like benchmark for static qualitative entailment.

## What the proposed combination means

The four terms are used operationally here, not asserted as standard names.

| Component | Required property |
|---|---|
| **Proof-first generation** | Sample a typed proof structure first; derive the exact visible premises, query, answer, and certificate from it; replay the certificate; then re-solve the resulting problem to detect unintended alternatives or shorter proofs. |
| **Premise-first evaluation** | Sample visible premises `P` and query `Q` without selecting an answer; classify every candidate under declared open-world semantics; search for proofs or countermodels; admit the instance to the semantic and difficulty bucket discovered after solving. |
| **Independent proof checker + SMT** | A small checker validates the certificate's local rule applications and dependencies. A separately implemented SMT encoding decides global semantics, including `SAT(P & A)` and `UNSAT(P & not A)`. Neither component merely calls the other. |
| **Paired Natural/Symbolic traces** | Both traces are deterministic renderings of one proof object, with identical base problem, supporting premises, conclusion, and split assignment. They are experimental views, not separately authored rationales. |

This is stronger than “solver generated the label,” “an LLM produced CoT,” or
“a verifier liked the steps.” It also differs from executing a symbolic program
to obtain an answer: the proposal separately tests the local certificate and
the complete model-theoretic answer semantics.

## Search method

The search was conducted from 4–7 October 2026 and refreshed on 7 October. It
used original papers, official proceedings pages, arXiv originals, official
project pages, and official repositories. Surveys and search-result summaries
were used only to discover candidates; no substantive claim below rests on a
secondary source.

Search families included:

- `text-only spatial reasoning LLM benchmark`, `textual spatial reasoning
  symbolic proof`, `spatial reasoning path generation`, `StepGame SFT graph
  synthetic data`, and exact searches for SpatialEval, SpaRP, SODA/SPOD-143k,
  SiT-Bench, SpatialText, MentalMap, and FloorplanQA;
- `proof-first generation`, `proof-carrying reasoning LLM`, `natural symbolic
  proof traces`, `independent proof checker LLM`, `SMT spatial reasoning LLM`,
  and `verifier-guided synthetic reasoning data`; and
- exact searches for RuleTaker, ProofWriter, PrOntoQA, NLProofS, Logic-LM,
  LINC, Faithful CoT, PCRLLM, Typed/Proof-Carrying CoT, LogicGuide, PRoSFI,
  Lean Workbook, and Herald.

For each candidate, the audit asked: What is generated first? What is the gold
label based on? Is there a proof object? Who or what checks each step? Is the
global answer independently decided? Are natural and symbolic traces paired
from one semantic object? Is the task text-only qualitative spatial reasoning?

### Limits of the search

- No literature search can establish universal non-existence. “Not found” below
  means not found in the primary sources surfaced by these searches by the
  cutoff date.
- Terminology is unstable. Prior work may implement a similar ordering without
  calling it “proof-first” or “premise-first.” The overlap analysis therefore
  compares mechanisms, not keywords.
- Several 2025–2026 items are arXiv preprints or submissions. They are evidence
  of disclosed prior work, but not peer-reviewed findings unless explicitly
  marked as published.

## Text-only spatial landscape, 2021–2026

| Work | Status / venue | What is verified in the primary source | Relevance to the proposed combination |
|---|---|---|---|
| **SpartQA** ([Mirzaee et al., 2021](https://aclanthology.org/2021.naacl-main.364/)) | NAACL 2021 | Text stories support multi-label relation, block, object, and yes/no/unknown questions; synthetic data are derived from scene graphs and spatial rules. | Establishes rich text-only spatial QA and an explicit unknown class. It does not report proof-carrying examples or a dual checker/SMT architecture. |
| **StepGame** ([Shi et al., 2022](https://ojs.aaai.org/index.php/AAAI/article/view/21383)) | AAAI 2022 | Synthetic textual stories test 1–10-hop relative-direction composition and distractor robustness. | Establishes controlled proof-depth scaling. Its target is a relation label from a generated graph, not open-world all-model entailment with replayable certificates. |
| **SpaRTUN / ReSQ** ([Mirzaee & Kordjamshidi, 2022](https://arxiv.org/abs/2210.16952)) | arXiv preprint | SpaRTUN supplies synthetic formal representations and reasoning annotations; ReSQ tests transfer to human-written spatial language. | Establishes formal spatial supervision and synthetic-to-natural transfer, but not the proposed generation and independent-validation stack. |
| **PistaQ** ([Mirzaee & Kordjamshidi, 2023](https://aclanthology.org/2023.findings-emnlp.221/)) | Findings of EMNLP 2023 | Separates neural extraction from deterministic Prolog reasoning over spatial rules. | Strong neuro-symbolic precedent. The formal reasoner produces answers after parsing; examples do not carry independently checked proof objects. |
| **SpaRC / SpaRP** ([Rizvi et al., 2024](https://aclanthology.org/2024.acl-long.261/)) | ACL 2024 | Builds symbolic paths over SpaRTUN and StepGame contexts, composes them with spatial reasoners, verbalizes each path link-by-link, and fine-tunes LLMs on the resulting reasoning paths. | **Closest trace precedent.** It has deductively verified spatial paths and natural verbalizations. It starts from existing context-question-answer data, uses the spatial reasoner for ground-truth path production, and does not report premise-first all-candidate semantics, a second SMT oracle, or a paired Natural-versus-Symbolic trace ablation from one certificate. |
| **Chain-of-Symbol (CoS)** ([Hu et al., 2024](https://openreview.net/forum?id=Hvq9RtSoHG)) | COLM 2024 | Rewrites corrected natural-language CoT demonstrations into condensed symbols and evaluates few-shot prompting on spatial QA and planning tasks. | Establishes Natural-versus-Symbolic representation as a spatial reasoning variable. The demonstrations are not proof certificates and the two forms are not deterministic renderings of an independently checked proof object. |
| **SpatialEval / Spatial-Map TQA** ([Wang et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89cc5e613d34f90de90c21e996e60b30-Abstract-Conference.html); [official repo](https://github.com/jiayuww/SpatialEval)) | NeurIPS 2024 | Defines text-only, visual-only, and combined variants; Spatial-Map TQA contains textual pairwise relations and direction/object/count multiple-choice questions. | Exact external case-study family. The publication does not expose proof certificates or an open-world entailment audit; its generator is not public in the official repository. |
| **Graph-based synthetic StepGame SFT** ([Zhou et al., 2024](https://arxiv.org/abs/2409.12437)) | arXiv preprint | Builds relational graphs, samples non-repeating random-walk chains, removes one edge as the target, verbalizes the remainder, and tunes models with standard answer or Extract-Then-Answer prompting. | Strong precedent for graph-first synthetic SFT and answer derivation from a sampled chain. It does not check a proof certificate independently, apply SMT all-model semantics, or compare paired proof renderings. |
| **Neuro-symbolic Training for Reasoning over Spatial Language** ([Premsri & Kordjamshidi, 2025](https://aclanthology.org/2025.findings-naacl.128/)) | Findings of NAACL 2025 | Uses 79 spatial rules to build forward-chained Q-Chains from SpaRTUN annotations, then turns consistency relationships into differentiable training constraints. | Establishes rule-derived spatial process supervision. The trained model is not a certified reasoner at inference, and Q-Chain construction plus neural loss is not the proposed dual validation. |
| **DSPy + ASP spatial pipeline** ([Wang & Sun, 2025](https://doi.org/10.1016/j.neunet.2025.108022); [arXiv](https://arxiv.org/abs/2411.18564)) | *Neural Networks* 2025 | An LLM translates text into ASP, iteratively refines it with solver feedback, and Clingo computes answers on StepGame and SpartQA. | Establishes solver-in-the-loop text-only spatial reasoning. Its focus is inference-time formalization, not certified dataset provenance or controlled paired traces. |
| **SODA / SPOD-143k / SPOD-Bench** ([Bai et al., 2026](https://aclanthology.org/2026.acl-long.1382/)) | ACL 2026 | Embeds the OODA loop in pure-text spatial control tasks, releases a 143K training set and a three-level benchmark, and uses a two-stage SFT-plus-GRPO recipe. | **Defeats “first pure-text spatial SFT/GRPO” claims.** Its focus is dynamic control and OODA cognition, not formal entailment certificates or proof-checker/SMT agreement. |
| **SiT-Bench** ([Guo et al., 2026](https://arxiv.org/abs/2601.03590)) | arXiv preprint | Provides 3,800+ text-input examples across 17 tasks. The pipeline converts images or visual benchmarks to coordinate-aware descriptions, uses DeepSeek-R1 filtering, and ends with expert review of R1 rationales. | Broad text-input evidence, but much of the task ontology is navigation, perspective transformation, manipulation, depth, and metric geometry. Its rationale review is not formal proof checking. |
| **SpatialText** ([Jiang et al., 2026](https://arxiv.org/abs/2603.03002)) | arXiv preprint | Combines human descriptions with 80 programmatically generated 2D/3D scenes. The synthetic arm derives text from coordinates and explicitly includes non-omniscient cases where answers can be undecidable. | **Closest partial-information precedent.** It tests epistemic uncertainty and says coordinate-derived descriptions are mathematically verifiable, but it remains world/coordinate-first and reports neither proof-carrying traces nor independent replay-checker-plus-SMT validation. |
| **MentalMap** ([Pan et al., 2026](https://arxiv.org/abs/2605.28277)) | arXiv preprint | Uses 100 ProcTHOR houses, canonical world states, action trajectories, and transition logs across eight languages plus structured text; levels progress to generated world graphs. | Valuable for multilingual world-model diagnostics. It is scene/trajectory-first and partly dynamic, not qualitative entailment from deliberately partial premises. |
| **FloorplanQA** ([Rodionov et al., 2026](https://openreview.net/forum?id=bcN12UkftY); [arXiv](https://arxiv.org/abs/2507.07644)) | ICML 2026 | Tests LLM reasoning over JSON/XML floorplans, including distance, visibility, placement, and constraints. | Text/structured-input boundary comparator; it is metric floorplan reasoning, not proof-carrying qualitative entailment. |
| **interwhen** ([Bhat et al., 2026](https://arxiv.org/abs/2602.11202)) | arXiv preprint | On text-only SpatialMap, forks an evolving CoT, elicits structured intermediate direction claims, maps them into Z3 constraints, rejects contradictions/impossible answers, and feeds corrections back to the model. | **Strongest direct counterclaim.** It establishes SMT-monitored intermediate spatial reasoning on the same task family. It checks claims from an unconstrained trace rather than a dependency-explicit proof certificate, uses Z3 as the operative spatial verifier rather than an independent checker plus semantic oracle, and provides no proof-first/premise-first corpus or paired trace views. |

### What changed by 2026

The 2026 literature makes “text-only spatial reasoning” itself crowded. SODA,
SiT-Bench, SpatialText, MentalMap, and FloorplanQA cover pure-text training,
spatial-world diagnostics, epistemic incompleteness, multilingual world models,
and structured floorplans. `interwhen` goes further and directly brings Z3 into
text-only SpatialMap inference. The remaining opening is consequently narrow:
**certificate-bearing qualitative entailment data and evaluation, with explicit
construction provenance and two genuinely separate validation paths.**

## Adjacent proof, verifier, and synthetic-data precedents

| Work | Status / venue | Mechanism that matters here | Why it does not subsume the proposed spatial study |
|---|---|---|---|
| **RuleTaker** ([Clark et al., 2020](https://arxiv.org/abs/2002.05867)) | IJCAI 2020 | Samples a logic theory first, computes its forward closure and proofs, then selects true and closed-world-false questions before verbalization. | Strong precedent for theory/premise-first logical data, but not open-world qualitative spatial semantics or dual checking. |
| **ProofWriter** ([Tafjord et al., 2021](https://aclanthology.org/2021.findings-acl.317/); [arXiv](https://arxiv.org/abs/2012.13048)) | Findings of ACL 2021 | Defines proofs as fact/rule DAGs, supports open-world `Unknown`, and trains models to generate full or iterative one-step natural-language proofs. | Establishes proof-producing language models and open-world proof datasets. Its neural proof writer is not an independent formal checker, and the domain is synthetic Datalog rather than space. |
| **PrOntoQA** ([Saparov & He, 2023](https://openreview.net/forum?id=qFVVBzXxR2V); [arXiv](https://arxiv.org/abs/2210.01240)) | ICLR 2023 | **Generates an ontology, then a proof by walking it, then translates ontology and proof into context, query, CoT, and label.** | Direct prior art for proof-first generation. Its linear modus-ponens ontology and natural-language CoT do not provide spatial calculus, premise-first evaluation, paired renderings, or an independent SMT oracle. |
| **NLProofS** ([Yang et al., 2022](https://aclanthology.org/2022.emnlp-main.7/); [arXiv](https://arxiv.org/abs/2205.12443)) | EMNLP 2022 | Uses an independently trained neural verifier to score generated proof steps and guide search on EntailmentBank and RuleTaker. | Establishes verifier-guided proof generation, but its verifier is learned rather than a sound replay checker and it is not spatial. |
| **Faithful CoT** ([Lyu et al., 2023](https://aclanthology.org/2023.ijcnlp-main.20/)) | IJCNLP-AACL 2023 | Interleaves natural-language decomposition with executable symbolic programs; an external deterministic solver obtains the final answer. | Establishes mixed Natural/Symbolic reasoning and executable faithfulness, but not two paired renderings of one proof nor an independent second semantics checker. |
| **Logic-LM / LINC** ([Pan et al., 2023](https://aclanthology.org/2023.findings-emnlp.248/); [Olausson et al., 2023](https://aclanthology.org/2023.emnlp-main.313/)) | Findings of EMNLP / EMNLP 2023 | Translate natural-language premises and goals into solver languages, then use deterministic symbolic solvers or a first-order prover for the answer. | Strong solver-backed reasoning precedent. Correctness remains conditional on formalization; neither paper constructs a checked spatial proof corpus with independent local/global validation. |
| **Deductive Verification / Natural Program** ([Ling et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html)) | NeurIPS 2023 | Structures CoT as numbered assumptions and premise-citing steps, then performs local deduction checks. | Very close proof shape, but the step verifier is itself model-based rather than a formal spatial checker. |
| **LogicGuide** ([Poesia et al., 2024](https://openreview.net/forum?id=yXnwrs2Tl6)) | TMLR 2024 | Constrains LM generation through a theorem-proving environment so deductive steps are formally checkable. | Establishes independently checkable LM deduction, but not the spatial domain, construction-provenance split, or paired trace experiment. |
| **PCRLLM** ([Li et al., 2025](https://arxiv.org/abs/2511.08392)) | arXiv preprint | Generates logic first, naturalizes the input, and requires JSON outputs with premises, rules, and conclusions. It grades local and inter-step conformity. | Direct prior art for the phrase and concept “proof-carrying reasoning.” Its data generator and grader use the same NAL engine, it studies two-step NAL rather than space, and it has no independent SMT answer oracle. |
| **Typed / Proof-Carrying CoT** ([Perrier, 2025](https://arxiv.org/abs/2510.01069)) | arXiv preprint | Emits a typed JSON program, constructs a typed reasoning graph, applies certification gates, and deterministically renders a textual view. | Direct prior art for typed proof-carrying CoT and dual machine/human views. Its evaluation is GSM8K arithmetic; certification is type/dataflow checking, not a separate proof replay plus spatial SMT entailment test. |
| **Lean-STaR** ([Lin et al., 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a5357781c204d4412e44ed9cbcdb08d5-Abstract-Conference.html)) | ICLR 2025 Spotlight | Retrospectively generates informal thoughts from ground-truth Lean tactics, trains on proof-state/thought/tactic triples, and keeps successful expert-iteration proofs. | Strong precedent for interleaved Natural/Symbolic proof supervision, but it is theorem proving and does not offer spatial premise-first evaluation or a separate SMT semantics oracle. |
| **Theorem Prover as a Judge** ([Leang et al., 2025](https://aclanthology.org/2025.acl-long.1448/)) | ACL 2025 | Autoformalizes intermediate mathematical reasoning into Lean and uses theorem-prover feedback to filter or train reasoning. | Strong precedent for formally judged synthetic natural-language reasoning, but it is mathematical and does not separate a small domain proof checker from an SMT model oracle. |
| **Herald** ([Gao et al., 2025](https://openreview.net/forum?id=Se6MgCtRhz)) | ICLR 2025 | Translates Lean proof steps into hierarchical natural-language annotations and releases paired informal/formal theorem-proof data. | Strong precedent for paired natural/formal proof artifacts. It is Mathlib theorem proving, not matched spatial trace representations for LLM ablation. |
| **PRoSFI** ([Chen et al., 2026](https://arxiv.org/abs/2603.29500); [OpenReview](https://openreview.net/forum?id=Tsag0RrOW7)) | arXiv preprint / submission | Requires dependency-explicit atomic steps in a structured intermediary and gives high process reward only to chains accepted by a formal prover. | Very close proof-carrying supervision architecture, but it evaluates first-order logic and Knights-and-Knaves, not spatial entailment or paired spatial renderings. |
| **FoVer** ([Kamoi et al., 2026](https://aclanthology.org/2026.findings-acl.403/)) | Findings of ACL 2026 | Uses Z3 and Isabelle to attach formal step-level error labels to logic and theorem-proving traces, then trains process reward models. | Establishes formal-verifier-generated process data. The deployed verifier is learned, examples do not carry replayable certificates, and the tasks are not spatial. |
| **ProofSketcher** ([Kommuru et al., 2026](https://arxiv.org/abs/2604.06401)) | arXiv preprint | An LLM emits a typed proof-sketch DSL; a small trusted kernel generates obligations, and external SMT/ATP results are accepted only through certificates checked by the kernel. | **Closest architecture to checker-plus-SMT separation.** It is generic math/logic, not a spatial benchmark, and does not add premise-first evaluation or paired spatial trace arms. |

Learned process reward models are relevant but weaker comparators. *Let's Verify
Step by Step* trains a process reward model from human step labels on MATH
([Lightman et al., 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html));
Math-Shepherd derives step labels from sampled continuations and final-answer
correctness ([Wang et al., 2024](https://aclanthology.org/2024.acl-long.510/)).
Neither provides an independently replayable certificate. They should be cited
as verifier-guided reasoning, not as proof-carrying data.

## Nearest prior work

### 1. SpaRP: closest spatial proof-trace precedent

SpaRP explicitly calls its paths “deductively verified,” obtains a symbolic
path by traversing a graph over the context, applies dataset-specific spatial
composition rules, and verbalizes the path link-by-link. It also reports
fine-tuning with those paths. Any SpatialEntail claim about “first verified
spatial reasoning paths” or “first solver-generated spatial CoT” would conflict
with this paper.

SpatialEntail can remain distinct if its artifact demonstrably adds all of:

- proof-first problem construction rather than post-hoc paths over inherited
  context-question-answer tuples;
- premise-first solve-and-bucket evaluation under explicit open-world,
  all-model semantics;
- a replay checker whose implementation and failure modes differ from the SMT
  semantic oracle; and
- matched Natural/Symbolic renderings from the exact same certificate.

### 2. interwhen: closest same-benchmark SMT precedent

`interwhen` uses the text-only `spatial_map_text_only` split, supplies parsed
given relations to a structured side stream, extracts intermediate direction
claims, and checks each against Z3 constraints. It therefore establishes both
the feasibility and prior disclosure of Z3-backed checking on SpatialMap.

Its object of verification is a claim extracted from a free-form reasoning
stream. It does not require a proof step to cite dependencies or a rule, does
not replay a complete certificate, and does not build the training/evaluation
instance by proof-first or premise-first provenance. SpatialEntail should cite
it prominently and define “proof checking” more narrowly than “claim
consistency checking.”

### 3. PrOntoQA: closest generation-order precedent

PrOntoQA is unambiguous proof-first prior art: ontology, proof, then natural
language context/query/CoT/label. The novelty cannot be the ordering alone.
SpatialEntail's difference is the combination of a qualitative spatial theory,
open-world candidate semantics, independent replay and SMT checks, and paired
trace views.

### 4. PCRLLM and Proof-Carrying CoT: closest certificate vocabulary

Both works explicitly use proof-carrying language and machine-readable step
structures. PCRLLM emits premises, rule names, conclusions, and evidential
metadata; Proof-Carrying CoT emits typed programs and certifies their dataflow.
SpatialEntail should not present “proof-carrying reasoning” as a coined term or
general conceptual novelty. It can claim a domain-specific instantiation with
stronger separation between local proof replay and global semantic validation.

### 5. SpatialText: closest partial-information benchmark precedent

SpatialText's non-omniscient synthetic condition deliberately omits relations
so some questions are undecidable. This is important prior art for epistemic
uncertainty in a 2026 text-only spatial benchmark. Its synthetic pipeline still
assigns coordinates first and derives pairwise relations from them. A single
coordinate world certifies possibility, not entailment across every world
consistent with the visible text. SpatialEntail's contribution must be the
explicit model-theoretic contract and provenance controls, not merely adding an
“unknown” answer.

## Overlap matrix

Legend: **Y** = clearly present; **P** = partial or analogous mechanism; **N** =
not reported in the cited primary source. “Dual validation” means two
substantively independent paths, not two calls to one engine.

| Work | Text-only qualitative spatial | Proof-first generation | Premise-first solve-and-bucket | Replay checker + independent SMT semantics | Paired Natural/Symbolic views of one proof |
|---|:---:|:---:|:---:|:---:|:---:|
| SpaRP | Y | N | N | N | P |
| Graph-based StepGame SFT | Y | P | N | N | N |
| Neuro-symbolic spatial training | Y | N | N | N | P |
| SODA / SPOD-143k | P | N | N | N | N |
| SpatialText synthetic arm | P | N | P | N | N |
| interwhen on SpatialMap | Y | N | N | P | N |
| RuleTaker / ProofWriter | N | P | P | N | P |
| PrOntoQA | N | Y | N | N | P |
| Faithful CoT / Logic-LM / LINC | N | N | N | P | P |
| PCRLLM | N | Y | N | N | P |
| Proof-Carrying CoT | N | N | N | P | P |
| LogicGuide / PRoSFI | N | N | N | P | P |
| ProofSketcher | N | N | N | Y | N |
| **Proposed SpatialEntail design** | **Y** | **Y** | **Y** | **Y** | **Y** |

The matrix is deliberately conservative. For example, RuleTaker is
“premise-first-like” because it samples a theory and computes its closure before
selecting questions, but it does not use the proposed term or solve-and-bucket
an open-world spatial candidate set. Likewise, `interwhen` receives **P** for
dual validation because its Z3 check is formally grounded, but it is not paired
with a separate certificate replay checker.

## Novelty assessment

### High-confidence conclusions

1. **Proof-first generation is not new in LLM reasoning.** PrOntoQA and PCRLLM
   are explicit precedents; graph-chain sampling is an additional spatially
   adjacent analogue.
2. **Verified spatial reasoning paths are not new.** SpaRP generates and
   verbalizes paths with spatial reasoners, while neuro-symbolic spatial
   training builds rule-derived Q-Chains.
3. **Natural-versus-symbolic spatial reasoning is not new.** CoS directly
   contrasts natural-language CoT with symbolic spatial representations.
4. **Solver-backed text-only spatial reasoning is not new.** PistaQ, DSPy+ASP,
   and `interwhen` use deterministic symbolic machinery; `interwhen` directly
   uses Z3 on SpatialMap intermediate claims.
5. **Pure-text spatial SFT and GRPO are not new by late 2026.** SODA trains with
   SPOD-143k using SFT and GRPO.
6. **Partial-information spatial diagnosis is not wholly new.** SpartQA has an
   unknown class and SpatialText explicitly includes non-omniscient scenes.
7. **Proof-carrying and paired informal/formal reasoning are established
   outside spatial reasoning.** PCRLLM, Proof-Carrying CoT, LogicGuide, Herald,
   Lean-STaR, theorem-prover feedback, ProofSketcher, and PRoSFI make that clear.

### Calibrated absence finding

Within the searched primary sources, **no work was found that combines all four
components in one text-only qualitative spatial benchmark and learning
pipeline**. More specifically, none was found that simultaneously:

- creates controlled spatial training examples from typed, replayable proof
  certificates;
- creates evaluation examples by sampling only visible premises and discovering
  their open-world semantic class after solving;
- requires agreement between a local proof replay checker and a separately
  implemented SMT model oracle; and
- derives matched Natural and Symbolic trace variants from the same certificate
  for controlled prompting/SFT experiments.

This is meaningful white space, but it does not prove priority. The claim could
be falsified by an unindexed preprint, a non-English paper, a repository without
a paper, or a method described under different terminology.

### Recommended claim language

Use:

> To our knowledge, SpatialEntail is the first text-only qualitative spatial
> reasoning framework to combine proof-first certificate generation,
> premise-first open-world evaluation, independent proof replay and SMT
> semantics, and matched natural/symbolic renderings of the same derivation.

Immediately follow it with:

> Each ingredient has prior art in spatial reasoning, logical reasoning, or
> formal theorem proving; the contribution is their integration and controlled
> evaluation in qualitative spatial entailment.

Safer until the artifact and systematic review are complete:

> SpatialEntail studies an underexplored intersection of proof-carrying data,
> model-theoretic spatial evaluation, independent verification, and controlled
> trace representation.

Avoid:

- “the first text-only spatial reasoning benchmark”;
- “the first proof-first dataset”;
- “the first verified spatial chain-of-thought”;
- “the first use of SMT/Z3 for text-only spatial reasoning”;
- “the first Natural-versus-Symbolic spatial reasoning study”;
- “the first spatial SFT/GRPO method”; and
- “proof-carrying generation guarantees natural-language correctness.”

## Risks and counterclaims

| Counterclaim or risk | Why it is credible | Required response |
|---|---|---|
| “SpaRP already did solver-verified spatial paths and fine-tuning.” | Correct at the component level. | Claim the stronger provenance/validation/paired-view combination; run a SpaRP-style path baseline. |
| “PrOntoQA already did proof-first natural-language data.” | Correct. Its pipeline explicitly generates ontology and proof before language. | Cite it as the generation-order precedent and narrow novelty to spatial open-world semantics plus independent validation. |
| “interwhen already checks SpatialMap reasoning with Z3.” | Correct and directly on the task family. | Distinguish consistency monitoring of extracted claims from dependency-explicit proof replay; compare against monitor-only inference if feasible. |
| “SpatialText already tests unknowns under incomplete text.” | Correct. | Emphasize all-model candidate classification, countermodels, and controlled provenance rather than the mere presence of undecidable items. |
| “PCRLLM or PC-CoT already owns proof-carrying reasoning.” | They establish the term and general architecture. | Do not coin or claim the generic concept; claim a spatial instantiation with dual validation. |
| “The proof checker and SMT solver are not really independent.” | They may share parser, IR, relation tables, or tests, allowing correlated bugs. | Document separate implementations and trust boundaries; use mutation, differential, metamorphic, witness, and countermodel tests. |
| “A checked proof can still formalize the English incorrectly.” | Formal proof systems guarantee derivability from the formalization, not semantic fidelity of the translation. | Deterministically generate text from typed atoms where possible; reparse every emitted prompt; reject non-round-tripping cases; manually audit held-out templates. |
| “Proof-first data teaches templates, not reasoning.” | Proof skeleton, premise order, surface form, or answer position can leak construction. | Use premise-first evaluation, held-out rule compositions, proof-shape splits, premise permutations, distractors, and canonical-world deduplication. |
| “Natural and Symbolic traces differ in more than representation.” | Length, tokenization, instruction wording, and error exposure can confound results. | Pair by base problem and checked certificate; report target tokens and compute; vary one representation factor at a time. |
| “SMT says entailment but does not provide a human proof.” | Unsatisfiability is a semantic decision, not automatically an explanatory derivation. | Keep the replayable proof as the explanation artifact and SMT as the independent global oracle; publish countermodels for non-entailments. |
| “The benchmark is only another synthetic micro-world.” | Controlled logic may not transfer to natural language. | Include held-out templates and names, ReSQ/SpaRTUN transfer where semantically compatible, and the corrected SpatialMap external case study. |

## Concrete positioning recommendations

1. **Lead with the problem, not the machinery.** The central problem is that a
   label can be true in a hidden generating world without being entailed by the
   visible text. Proof-carrying construction and model-theoretic evaluation are
   the remedy.

2. **Name the contribution as an integration.** A suitable phrase is
   “proof-carrying qualitative spatial entailment under partial information.”
   Treat `proof-first` and `premise-first` as clearly defined provenance classes,
   not claims that the terms themselves are novel.

3. **Make the trust boundary an artifact.** Publish the certificate schema,
   small replay checker, SMT encoding, agreement tests, parser round trips,
   witnesses/countermodels, and a manifest recording construction provenance.
   The novelty claim is much weaker if “independent” cannot be audited.

4. **Use proof-first for controlled supervision and premise-first for the main
   generalisation test.** This directly answers the template-leakage objection.
   A secondary premise-first training pool is defensible only when extracted
   proofs also replay successfully.

5. **Run matched trace ablations.** Answer-only, Natural, and Symbolic targets
   should share the same base IDs and checked proof supports.

6. **Include the nearest baselines, not only generic CoT.** At minimum compare
   direct answer, natural CoT, CoS-like symbolic prompting, SpaRP-like verbalized
   paths, solver-only inference, and—if implementable—`interwhen`-style claim
   monitoring. SODA is the relevant pure-text SFT/GRPO training comparator even
   though its tasks are dynamic.

7. **Report semantics rather than one accuracy number.** Separate uniquely
   entailed, ambiguous, impossible, and inconsistent cases; report by proof
   depth, independent-axis support, propositional depth, distractor type, and
   answer contract.

8. **Treat RL as secondary.** SODA already establishes spatial SFT+GRPO and the
   general RLVR literature establishes outcome-verifiable training. The sharper
   question is whether certificate-aware or exact semantic rewards add value
   beyond strong answer-only and trace-SFT baselines.

9. **Use a two-level novelty statement.** The paper-level claim should be the
   exact four-part integration. Individual sections should acknowledge their
   closest precedent: PrOntoQA for proof-first generation, RuleTaker/ProofWriter
   for theory-first open-world logic, SpaRP for spatial proof paths, CoS for
   symbols, `interwhen` for Z3 monitoring, and PCRLLM/PC-CoT for proof-carrying
   structure.

10. **Re-run the priority search before submission.** Search the exact title,
    core mechanism terms, citations to SpaRP and `interwhen`, and 2026–submission
    venue proceedings. Preserve the search log as supplementary material.

## Bottom line

SpatialEntail should not be sold as inventing proofs, symbolic spatial
reasoning, SMT-backed checking, trace supervision, or text-only spatial
training. Those are established. The credible frontier is the **joint research
design**: distinguish proof-first training from premise-first evaluation;
attach replayable certificates to qualitative spatial problems; validate local
proof structure separately from global SMT semantics; and compare Natural and
Symbolic views without changing the underlying derivation.

That combination is not present in the primary sources found through 7 October
2026. It is specific enough to be defensible, important enough to motivate a
benchmark, and narrow enough that it must be stated with “to our knowledge” and
an explicit closest-work comparison.

## Bibliography

- Bai, S., et al. (2026). *One Cognitive Loop Is Enough: SODA unlocks
  Pure-Text Spatial Reasoning in Large Language Models*. ACL 2026.
  [ACL Anthology](https://aclanthology.org/2026.acl-long.1382/),
  [doi:10.18653/v1/2026.acl-long.1382](https://doi.org/10.18653/v1/2026.acl-long.1382).
- Clark, P., Tafjord, O., & Richardson, K. (2020). *Transformers as Soft
  Reasoners over Language*. IJCAI 2020.
  [arXiv:2002.05867](https://arxiv.org/abs/2002.05867).
- Perrier, E. (2025). *Typed Chain-of-Thought: A Curry-Howard Framework
  for Verifying LLM Reasoning*. Preprint.
  [arXiv:2510.01069](https://arxiv.org/abs/2510.01069).
- Guo, Z., et al. (2026). *Can LLMs See Without Pixels? Benchmarking Spatial
  Intelligence from Textual Descriptions*. Preprint.
  [arXiv:2601.03590](https://arxiv.org/abs/2601.03590).
- Hu, Y., et al. (2024). *Chain-of-Symbol Prompting for Spatial Reasoning in
  Large Language Models*. COLM 2024.
  [OpenReview](https://openreview.net/forum?id=Hvq9RtSoHG),
  [arXiv:2305.10276](https://arxiv.org/abs/2305.10276).
- Leang, J. O. J., Hong, G., Li, W., & Cohen, S. B. (2025). *Theorem Prover as a Judge for Synthetic Data
  Generation*. ACL 2025.
  [ACL Anthology](https://aclanthology.org/2025.acl-long.1448/),
  [doi:10.18653/v1/2025.acl-long.1448](https://doi.org/10.18653/v1/2025.acl-long.1448).
- Jiang, P., Qin, Z., & Li, X. (2026). *SpatialText: A Pure-Text Cognitive
  Benchmark for Spatial Understanding in Large Language Models*. Preprint.
  [arXiv:2603.03002](https://arxiv.org/abs/2603.03002).
- Kamoi, R., et al. (2026). *Efficient PRM Training Data Synthesis via Formal
  Verification*. Findings of ACL 2026.
  [ACL Anthology](https://aclanthology.org/2026.findings-acl.403/),
  [doi:10.18653/v1/2026.findings-acl.403](https://doi.org/10.18653/v1/2026.findings-acl.403).
- Kommuru, K., Khanvilkar, K., & Parekh, G. (2026). *ProofSketcher: Hybrid LLM
  + Lightweight Proof Checker for Reliable Math/Logic Reasoning*. Preprint.
  [arXiv:2604.06401](https://arxiv.org/abs/2604.06401).
- Poesia, G., Gandhi, K., Zelikman, E., & Goodman, N. D. (2024). *Certified Deductive Reasoning with Language
  Models*. TMLR.
  [OpenReview](https://openreview.net/forum?id=yXnwrs2Tl6).
- Li, T., et al. (2025). *PCRLLM: Proof-Carrying Reasoning with Large Language
  Models under Stepwise Logical Constraints*. Preprint.
  [arXiv:2511.08392](https://arxiv.org/abs/2511.08392).
- Chen, L., Zhou, Y., & Zhang, H. (2026). *Learning to Generate Formally Verifiable Step-by-Step
  Logic Reasoning via Structured Formal Intermediaries*. Preprint/submission.
  [arXiv:2603.29500](https://arxiv.org/abs/2603.29500),
  [OpenReview](https://openreview.net/forum?id=Tsag0RrOW7).
- Gao, G., et al. (2025). *Herald: A Natural Language Annotated Lean 4 Dataset*.
  ICLR 2025. [OpenReview](https://openreview.net/forum?id=Se6MgCtRhz).
- Lin, H., Sun, Z., Welleck, S., & Yang, Y. (2025). *Lean-STaR: Learning to
  Interleave Thinking and Proving*. ICLR 2025 Spotlight.
  [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a5357781c204d4412e44ed9cbcdb08d5-Abstract-Conference.html).
- Lightman, H., et al. (2024). *Let's Verify Step by Step*. ICLR 2024.
  [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html),
  [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).
- Ling, Z., et al. (2023). *Deductive Verification of Chain-of-Thought
  Reasoning*. NeurIPS 2023.
  [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html),
  [arXiv:2306.03872](https://arxiv.org/abs/2306.03872).
- Lyu, Q., et al. (2023). *Faithful Chain-of-Thought Reasoning*.
  IJCNLP-AACL 2023.
  [ACL Anthology](https://aclanthology.org/2023.ijcnlp-main.20/),
  [doi:10.18653/v1/2023.ijcnlp-main.20](https://doi.org/10.18653/v1/2023.ijcnlp-main.20).
- Mirzaee, R., et al. (2021). *SpartQA: A Textual Question Answering Benchmark
  for Spatial Reasoning*. NAACL 2021.
  [ACL Anthology](https://aclanthology.org/2021.naacl-main.364/),
  [doi:10.18653/v1/2021.naacl-main.364](https://doi.org/10.18653/v1/2021.naacl-main.364).
- Mirzaee, R., & Kordjamshidi, P. (2022). *Transfer Learning with Synthetic
  Corpora for Spatial Role Labeling and Reasoning*. Preprint.
  [arXiv:2210.16952](https://arxiv.org/abs/2210.16952).
- Mirzaee, R., & Kordjamshidi, P. (2023). *Disentangling Extraction and
  Reasoning in Multi-hop Spatial Reasoning*. Findings of EMNLP 2023.
  [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.221/),
  [doi:10.18653/v1/2023.findings-emnlp.221](https://doi.org/10.18653/v1/2023.findings-emnlp.221).
- Olausson, T., et al. (2023). *LINC: A Neurosymbolic Approach for Logical
  Reasoning by Combining Language Models with First-Order Logic Provers*.
  EMNLP 2023.
  [ACL Anthology](https://aclanthology.org/2023.emnlp-main.313/),
  [doi:10.18653/v1/2023.emnlp-main.313](https://doi.org/10.18653/v1/2023.emnlp-main.313).
- Pan, L., et al. (2023). *Logic-LM: Empowering Large Language Models with
  Symbolic Solvers for Faithful Logical Reasoning*. Findings of EMNLP 2023.
  [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.248/),
  [arXiv:2305.12295](https://arxiv.org/abs/2305.12295).
- Pan, Z., et al. (2026). *Do LLMs Build World Models From Text? A Multilingual
  Diagnostic of Spatial Reasoning*. Preprint.
  [arXiv:2605.28277](https://arxiv.org/abs/2605.28277).
- Premsri, T., & Kordjamshidi, P. (2025). *Neuro-symbolic Training for
  Reasoning over Spatial Language*. Findings of NAACL 2025.
  [ACL Anthology](https://aclanthology.org/2025.findings-naacl.128/),
  [arXiv:2406.13828](https://arxiv.org/abs/2406.13828).
- Rizvi, M. I. H., Zhu, X., & Gurevych, I. (2024). *SpaRC and SpaRP: Spatial
  Reasoning Characterization and Path Generation for Understanding Spatial
  Reasoning Capability of Large Language Models*. ACL 2024.
  [ACL Anthology](https://aclanthology.org/2024.acl-long.261/),
  [doi:10.18653/v1/2024.acl-long.261](https://doi.org/10.18653/v1/2024.acl-long.261).
- Rodionov, F., et al. (2026). *FloorplanQA: A Benchmark for Spatial Reasoning
  in LLMs using Structured Representations*. ICML 2026.
  [OpenReview](https://openreview.net/forum?id=bcN12UkftY),
  [arXiv:2507.07644](https://arxiv.org/abs/2507.07644).
- Saparov, A., & He, H. (2023). *Language Models Are Greedy Reasoners: A
  Systematic Formal Analysis of Chain-of-Thought*. ICLR 2023.
  [OpenReview](https://openreview.net/forum?id=qFVVBzXxR2V),
  [arXiv:2210.01240](https://arxiv.org/abs/2210.01240).
- Shi, Z., et al. (2022). *StepGame: A New Benchmark for Robust Multi-Hop
  Spatial Reasoning in Texts*. AAAI 2022.
  [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/21383),
  [arXiv:2204.08292](https://arxiv.org/abs/2204.08292).
- Bhat, V. K., et al. (2026). *interwhen: A Generalizable Framework for
  Verifiable Reasoning with Test-time Monitors*. Preprint.
  [arXiv:2602.11202](https://arxiv.org/abs/2602.11202).
- Tafjord, O., Dalvi, B., & Clark, P. (2021). *ProofWriter: Generating
  Implications, Proofs, and Abductive Statements over Natural Language*.
  Findings of ACL 2021.
  [ACL Anthology](https://aclanthology.org/2021.findings-acl.317/),
  [doi:10.18653/v1/2021.findings-acl.317](https://doi.org/10.18653/v1/2021.findings-acl.317),
  [arXiv:2012.13048](https://arxiv.org/abs/2012.13048).
- Wang, P., et al. (2024). *Math-Shepherd: Verify and Reinforce LLMs
  Step-by-step without Human Annotations*. ACL 2024.
  [ACL Anthology](https://aclanthology.org/2024.acl-long.510/),
  [doi:10.18653/v1/2024.acl-long.510](https://doi.org/10.18653/v1/2024.acl-long.510).
- Wang, R., & Sun, K. (2025). *DSPy-based Neural-Symbolic Pipeline to Enhance
  Spatial Reasoning in LLMs*. *Neural Networks*, 193, 108022.
  [doi:10.1016/j.neunet.2025.108022](https://doi.org/10.1016/j.neunet.2025.108022),
  [arXiv:2411.18564](https://arxiv.org/abs/2411.18564).
- Wang, J., et al. (2024). *Is A Picture Worth A Thousand Words? Delving Into
  Spatial Reasoning for Vision Language Models*. NeurIPS 2024.
  [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89cc5e613d34f90de90c21e996e60b30-Abstract-Conference.html),
  [arXiv:2406.14852](https://arxiv.org/abs/2406.14852).
- Yang, K., Deng, J., & Chen, D. (2022). *Generating Natural Language Proofs
  with Verifier-Guided Search*. EMNLP 2022.
  [ACL Anthology](https://aclanthology.org/2022.emnlp-main.7/),
  [doi:10.18653/v1/2022.emnlp-main.7](https://doi.org/10.18653/v1/2022.emnlp-main.7),
  [arXiv:2205.12443](https://arxiv.org/abs/2205.12443).
- Zhou, J., et al. (2024). *Enhancing Logical Reasoning in Large Language
  Models through Graph-based Synthetic Data*. Preprint.
  [arXiv:2409.12437](https://arxiv.org/abs/2409.12437).
