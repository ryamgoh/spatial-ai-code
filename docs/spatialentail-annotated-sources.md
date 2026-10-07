# SpatialEntail annotated primary-source ledger

*Checked through 7 October 2026. This is a report-writing ledger, not a claim
that the search proves universal priority.*

## How to use this ledger

Each entry separates five things that are easy to collapse in related work:

1. what the source actually did;
2. which SpatialEntail idea it supports or limits;
3. a sentence that is safe to use in the report;
4. a stronger sentence that the source does **not** justify; and
5. where the citation belongs in the report.

Only original papers, official proceedings pages, official project pages, and
official repositories are linked. `Published` means a venue publication was
verified. `Preprint` means that this ledger verified an arXiv disclosure but not
a peer-reviewed publication. An absence statement means “not found in the
bounded primary-source search through the date above,” never “does not exist.”

## Coverage of the current research-landscape chapter

All 19 keys currently cited in
`fyp-report/chapters/02-research-landscape.typ` resolve in
`fyp-report/references.bib`.

| Current key | Source | Group below |
|---|---|---|
| `shi2022stepgame` | StepGame | Text-only benchmarks |
| `mirzaee2021spartqa` | SpartQA | Text-only benchmarks |
| `mirzaee2022spartun` | SpaRTUN / ReSQ | Text-only benchmarks |
| `wang2024spatialeval` | SpatialEval | Text-only benchmarks |
| `bai2026soda` | SODA / SPOD-143k | Text-only benchmarks |
| `guo2026sitbench` | SiT-Bench | Text-only benchmarks |
| `jiang2026spatialtext` | SpatialText | Text-only benchmarks |
| `pan2026mentalmap` | MentalMap | Text-only benchmarks |
| `mirzaee2023pistaq` | PistaQ | Verified paths and traces |
| `rizvi2024sparp` | SpaRC / SpaRP | Verified paths and traces |
| `premsri2025neurosymbolic` | Neuro-symbolic spatial training | Verified paths and traces |
| `zhou2024graphsynthetic` | Graph-based synthetic reasoning data | Verified paths and traces |
| `hu2024chainofsymbol` | Chain-of-Symbol | Verified paths and traces |
| `tafjord2021proofwriter` | ProofWriter | Proof-first / theory-first generation |
| `saparov2023prontoqa` | PrOntoQA | Proof-first / theory-first generation |
| `lyu2023faithfulcot` | Faithful Chain-of-Thought | Formal verification / proof-carrying |
| `poesia2024logicguide` | LogicGuide | Formal verification / proof-carrying |
| `li2025pcrllm` | PCRLLM | Formal verification / proof-carrying |
| `bhat2026interwhen` | `interwhen` | Newest direct counterclaims |

## Reference-file metadata corrections applied

The annotation audit exposed several metadata errors in
`fyp-report/references.bib`; these were corrected against the primary sources
before the report was recompiled.

| Current key | Correction applied | Primary-source metadata |
|---|---|---|
| `shi2022stepgame` | Replaced the incorrect ten-author list. | Zhengxiang Shi, Qiang Zhang, and Aldo Lipani. |
| `mirzaee2022spartun` | Changed the entry from arXiv-only to a conference publication. | EMNLP 2022; DOI `10.18653/v1/2022.emnlp-main.413`. |
| `hu2024chainofsymbol` | Replaced the incorrect lead author and abbreviated list. | Hanxu Hu, Hongyuan Lu, Huajian Zhang, Yun-Ze Song, Wai Lam, and Yue Zhang. |
| `zhou2024graphsynthetic` | Corrected the lead author's given name and added the complete list. | The arXiv paper's lead author is Jiaming Zhou. |
| `bai2026soda` | Corrected the lead author's given name and added the complete list. | The ACL record says Shunwen Bai. |
| `guo2026sitbench` | Changed the entry from arXiv-only to a conference publication and added all authors. | Findings of ACL 2026; DOI `10.18653/v1/2026.findings-acl.90`. |
| `bhat2026interwhen` | Updated the authors and title to the latest arXiv record. | Vishak K Bhat et al., *interwhen: A Generalizable Framework for Steering Reasoning Models with Test-time Verification*. |
| `wang2024spatialeval` | Added the proceedings DOI. | DOI `10.52202/079017-2400`. |

## Text-only spatial benchmarks

### StepGame

- **Full citation and status:** Zhengxiang Shi, Qiang Zhang, and Aldo Lipani.
  “StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts.”
  AAAI 2022. **Published.** [Official proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/21383),
  [arXiv:2204.08292](https://arxiv.org/abs/2204.08292),
  [DOI](https://doi.org/10.1609/aaai.v36i10.21383).
- **What it did:** Generated controlled textual stories whose query relation
  requires one to ten composition steps, with dedicated longer-hop and noisy
  evaluations.
- **Supports:** Proof-depth stratification; held-out composition depth;
  distractor tests; a clean text-only spatial baseline.
- **Safe report claim:** “StepGame established controlled one-to-ten-hop
  directional composition as a text-only spatial benchmark.”
- **Does not support / avoid:** It does not establish open-world all-model
  entailment, proof-carrying examples, or an independent proof checker. Do not
  treat its relation label as evidence for the correctness of SpatialEntail's
  answer semantics.
- **Relationship to SpatialEntail:** Nearest standard benchmark for depth, but
  narrower in answer contract and provenance.
- **Suggested placement:** Research landscape; evaluation-suite motivation;
  external-transfer limitations.
- **BibTeX key:** Existing, corrected `shi2022stepgame`.

### SpartQA

- **Full citation and status:** Roshanak Mirzaee, Hossein Rajaby Faghihi, Qiang
  Ning, and Parisa Kordjamshidi. “SPARTQA: A Textual Question Answering
  Benchmark for Spatial Reasoning.” NAACL-HLT 2021. **Published.**
  [ACL Anthology](https://aclanthology.org/2021.naacl-main.364/),
  [DOI](https://doi.org/10.18653/v1/2021.naacl-main.364).
- **What it did:** Introduced human and automatically generated textual spatial
  QA with find-relation, find-block, choose-object, and yes/no/unknown tasks over
  directional, topological, containment, and distance relations.
- **Supports:** Rich spatial language; multi-label answers; explicit unknown
  handling; synthetic-to-human transfer.
- **Safe report claim:** “SpartQA demonstrates that text-only spatial QA can
  require richer relation algebras and an explicit unknown response.”
- **Does not support / avoid:** Its `unknown` label is not by itself proof that
  SpatialEntail's open-world semantics or candidate-set policy is correct.
- **Relationship to SpatialEntail:** Broader linguistic and relational scope;
  useful transfer comparator, but not a certificate-first benchmark.
- **Suggested placement:** Benchmark landscape and open-world motivation.
- **BibTeX key:** Existing `mirzaee2021spartqa`.

### SpaRTUN and ReSQ

- **Full citation and status:** Roshanak Mirzaee and Parisa Kordjamshidi.
  “Transfer Learning with Synthetic Corpora for Spatial Role Labeling and
  Reasoning.” EMNLP 2022. **Published.**
  [ACL Anthology](https://aclanthology.org/2022.emnlp-main.413/),
  [DOI](https://doi.org/10.18653/v1/2022.emnlp-main.413),
  [arXiv:2210.16952](https://arxiv.org/abs/2210.16952).
- **What it did:** Released SpaRTUN, a synthetic corpus with formal spatial
  representations and reasoning annotations, and ReSQ, human-written spatial
  QA used to test transfer from controlled synthetic language.
- **Supports:** Formal spatial annotations; rule-aware supervision;
  synthetic-to-natural transfer and parser stress testing.
- **Safe report claim:** “SpaRTUN supplies formal spatial supervision, while
  ReSQ tests whether models trained on controlled language transfer to
  human-written descriptions.”
- **Does not support / avoid:** The paper does not establish SpatialEntail's
  proof-first provenance, independent dual validation, or exact compass-eight
  ontology.
- **Relationship to SpatialEntail:** Important source dataset for SpaRP and the
  2025 Q-Chain work; natural-language transfer comparator.
- **Suggested placement:** Benchmark landscape; language-transfer evaluation;
  limitations of template-generated text.
- **BibTeX key:** Existing, corrected `mirzaee2022spartun`.

### SpatialEval and Spatial-Map TQA

- **Full citation and status:** Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav
  Vineet, Xin Wang, Yixuan Li, and Neel Joshi. “Is a Picture Worth a Thousand
  Words? Delving Into Spatial Reasoning for Vision Language Models.” NeurIPS
  2024. **Published.**
  [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89cc5e613d34f90de90c21e996e60b30-Abstract-Conference.html),
  [arXiv:2406.14852](https://arxiv.org/abs/2406.14852),
  [official repository](https://github.com/jiayuww/SpatialEval),
  [official dataset](https://huggingface.co/datasets/MilaWang/SpatialEval).
- **What it did:** Compared text-only, image-only, and redundant text-plus-image
  inputs across four spatial tasks. Spatial-Map TQA asks Direction, Which, and
  Count multiple-choice questions over textual pairwise map relations.
- **Supports:** The external case study; the three question families; strict
  modality separation; the need to report protocol-specific results.
- **Safe report claim:** “SpatialEval provides aligned textual and visual
  representations and makes Spatial-Map TQA a suitable case study for isolating
  textual spatial reasoning.”
- **Does not support / avoid:** The original paper does **not** establish the
  report's later hidden-world audit, the `833/1500` ambiguity count, or
  SpatialMap-TQA-Corr. Cite the project's released-data audit and implementation
  evidence for those claims, not the original paper alone.
- **Relationship to SpatialEntail:** External diagnosis and transfer set, not
  SpatialEntail training data. The official repository releases data and
  evaluation code but no public generator.
- **Suggested placement:** Motivation; benchmark diagnosis; corrected-benchmark
  evaluation; modality boundary.
- **BibTeX keys:** Existing `wang2024spatialeval`; existing dataset key
  `wang2024spatialevaldataset` is available but not cited in Chapter 2.

### SODA, SPOD-143k, and SPOD-Bench

- **Full citation and status:** Shunwen Bai, Jiahuan Zhang, Haoran Huang, Yurun
  Wang, Jiale Liu, Yanxi Wu, Ningzhe Yu, Yudong Gao, and Mingjun Cheng. “One
  Cognitive Loop Is Enough: SODA unlocks Pure-Text Spatial Reasoning in Large
  Language Models.” ACL 2026. **Published.**
  [ACL Anthology](https://aclanthology.org/2026.acl-long.1382/),
  [DOI](https://doi.org/10.18653/v1/2026.acl-long.1382).
- **What it did:** Embedded an Observe–Orient–Decide–Act loop in multiple
  pure-text spatial control tasks, built SPOD-143k, introduced a three-level
  SPOD-Bench, and trained with SFT followed by GRPO.
- **Supports:** A recent pure-text spatial training comparator; structured
  reasoning supervision; the need to treat SFT and GRPO as existing rather than
  intrinsically novel.
- **Safe report claim:** “By ACL 2026, SODA had already combined large-scale
  pure-text spatial data with SFT and GRPO.”
- **Does not support / avoid:** It does not establish proof certificates,
  premise-first open-world evaluation, or local-checker/SMT agreement. Its
  dynamic control focus should not be collapsed into static qualitative
  entailment.
- **Relationship to SpatialEntail:** Strong training-method counterclaim but a
  boundary task family.
- **Suggested placement:** Newest benchmark landscape; RL/SFT related work;
  scope boundary.
- **BibTeX key:** Existing, corrected `bai2026soda`.

### SiT-Bench

- **Full citation and status:** Zhongbin Guo, Zhen Yang, Yushan Li, Xinyue
  Zhang, Wenyu Gao, Jiacheng Wang, Chengzhi Li, Xiangrui Liu, and Ping Jian.
  “Can LLMs See Without Pixels? Benchmarking Spatial Intelligence from Textual
  Descriptions.” Findings of ACL 2026. **Published.**
  [ACL Anthology](https://aclanthology.org/2026.findings-acl.90/),
  [DOI](https://doi.org/10.18653/v1/2026.findings-acl.90),
  [arXiv:2601.03590](https://arxiv.org/abs/2601.03590).
- **What it did:** Collected more than 3,800 text-input examples over 17 tasks
  spanning mapping, navigation, perspective shifts, geometry, manipulation,
  state tracking, and anomaly detection. Its construction includes captions or
  structured descriptions derived from visual data, automated filtering, and
  expert review.
- **Supports:** The claim that text-only spatial evaluation is broad and
  crowded by 2026; the importance of separating input modality from task
  provenance.
- **Safe report claim:** “SiT-Bench evaluates spatial reasoning without pixels,
  but many tasks retain coordinate-aware, navigation, manipulation, or
  image-derived structure.”
- **Does not support / avoid:** Text-only input does not make SiT-Bench a direct
  qualitative-entailment comparator. Its reviewed rationales are not formal
  proof certificates.
- **Relationship to SpatialEntail:** Boundary comparator rather than a score
  leaderboard peer.
- **Suggested placement:** Scope and newest-benchmark paragraph.
- **BibTeX key:** Existing, corrected `guo2026sitbench`.

### SpatialText

- **Full citation and status:** Peiyao Jiang, Zequn Qin, and Xi Li.
  “SpatialText: A Pure-Text Cognitive Benchmark for Spatial Understanding in
  Large Language Models.” 2026. **Preprint.**
  [arXiv:2603.03002](https://arxiv.org/abs/2603.03002).
- **What it did:** Combined human descriptions of indoor scenes with 80
  programmatically generated 2D/3D scenes. The synthetic arm assigns
  coordinates first, derives pairwise relations deterministically, and includes
  non-omniscient settings in which some answers are undecidable.
- **Supports:** Pure-text epistemic uncertainty; controlled complete versus
  incomplete descriptions; the value of checking whether a model admits
  underdetermination.
- **Safe report claim:** “SpatialText explicitly tests non-omniscient textual
  scenes and recognition of formally undecidable spatial relations.”
- **Does not support / avoid:** Coordinate-derived consistency is not the same
  as proof that a label is entailed by every world satisfying the visible text.
  The pipeline is world-first, not proof-first or premise-first in
  SpatialEntail's operational sense.
- **Relationship to SpatialEntail:** Closest 2026 partial-information
  benchmark; important counterclaim to novelty based only on an unknown class.
- **Suggested placement:** Benchmark landscape; partial-information motivation;
  world-first contrast.
- **BibTeX key:** Existing `jiang2026spatialtext`.

### MentalMap

- **Full citation and status:** Zhikai Pan, Chih-Ting Liao, Chunrui Liu, Xi
  Xiao, Yitong Qiao, Chunlei Meng, Zhangquan Chen, and Xin Cao. “Do LLMs Build
  World Models From Text? A Multilingual Diagnostic of Spatial Reasoning.”
  2026. **Preprint.**
  [arXiv:2605.28277](https://arxiv.org/abs/2605.28277).
- **What it did:** Derived tasks from 100 ProcTHOR houses and AI2-THOR action
  traces, evaluating six levels from atomic facts to generated world graphs in
  eight languages plus structured text.
- **Supports:** Multilingual spatial world-model diagnosis; strict versus
  partial graph metrics; separation of language effects from structured-state
  reasoning.
- **Safe report claim:** “MentalMap extends pure-text spatial diagnosis to
  multilingual descriptions and explicit world-graph construction.”
- **Does not support / avoid:** It is scene/trajectory-first and partly dynamic;
  it does not validate proof-first certificates or open-world qualitative
  entailment.
- **Relationship to SpatialEntail:** Language and world-model boundary
  comparator, not a direct semantics benchmark.
- **Suggested placement:** Newest benchmarks; multilingual limitations and
  future work.
- **BibTeX key:** Existing `pan2026mentalmap`.

## Verified spatial paths, traces, and symbolic supervision

### PistaQ

- **Full citation and status:** Roshanak Mirzaee and Parisa Kordjamshidi.
  “Disentangling Extraction and Reasoning in Multi-hop Spatial Reasoning.”
  Findings of EMNLP 2023. **Published.**
  [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.221/),
  [DOI](https://doi.org/10.18653/v1/2023.findings-emnlp.221).
- **What it did:** Separated spatial-relation and entity extraction from
  deterministic reasoning and used Prolog spatial rules after formalization.
- **Supports:** A neural-parser/symbolic-reasoner architecture; the claim that
  extraction and reasoning have different error modes.
- **Safe report claim:** “PistaQ shows that deterministic spatial reasoning can
  be separated from the language-extraction problem.”
- **Does not support / avoid:** Solver-backed answers are not proof-carrying
  training traces, and correctness remains conditional on extraction and rule
  coverage.
- **Relationship to SpatialEntail:** Architectural predecessor for typed
  parsing plus solving; SpatialEntail adds provenance and independent checks.
- **Suggested placement:** Spatial solvers and parser trust boundary.
- **BibTeX key:** Existing `mirzaee2023pistaq`.

### SpaRC and SpaRP

- **Full citation and status:** Md Imbesat Rizvi, Xiaodan Zhu, and Iryna
  Gurevych. “SpaRC and SpaRP: Spatial Reasoning Characterization and Path
  Generation for Understanding Spatial Reasoning Capability of Large Language
  Models.” ACL 2024. **Published.**
  [ACL Anthology](https://aclanthology.org/2024.acl-long.261/),
  [DOI](https://doi.org/10.18653/v1/2024.acl-long.261).
- **What it did:** Characterized spatial reasoning properties, constructed
  symbolic paths over SpaRTUN and StepGame contexts, applied spatial
  composition rules, verbalized paths link by link, and fine-tuned LLMs on the
  resulting data.
- **Supports:** Solver-derived spatial paths; proof/path verbalization; spatial
  trace fine-tuning; property-aware benchmark design.
- **Safe report claim:** “SpaRP is the closest published precedent for
  solver-derived and verbalized spatial reasoning paths used in LLM
  fine-tuning.”
- **Does not support / avoid:** Do not claim SpatialEntail is the first to
  generate verified spatial paths. SpaRP starts from inherited
  context-question-answer data and does not report proof-first construction,
  premise-first all-candidate semantics, or an independent SMT cross-check.
- **Relationship to SpatialEntail:** Nearest trace baseline and the most
  important limiting citation for the trace contribution.
- **Suggested placement:** Spatial paths/traces; novelty qualification; SFT
  baselines.
- **BibTeX key:** Existing `rizvi2024sparp`.

### Neuro-symbolic training with Q-Chains

- **Full citation and status:** Tanawan Premsri and Parisa Kordjamshidi.
  “Neuro-symbolic Training for Reasoning over Spatial Language.” Findings of
  NAACL 2025. **Published.**
  [ACL Anthology](https://aclanthology.org/2025.findings-naacl.128/),
  [DOI](https://doi.org/10.18653/v1/2025.findings-naacl.128),
  [arXiv:2406.13828](https://arxiv.org/abs/2406.13828).
- **What it did:** Used 79 spatial rules to forward-chain from SpaRTUN's formal
  annotations, created example-specific Q-Chains, derived consistency
  constraints, and incorporated them into neural training losses.
- **Supports:** Rule-derived intermediate supervision; spatial logical
  constraints as a training signal; multi-hop generalization motivation.
- **Safe report claim:** “Rule-derived spatial supervision predates
  SpatialEntail: the Q-Chain approach generates forward-chained facts and uses
  their consistency relationships during training.”
- **Does not support / avoid:** The fine-tuned model is not a certified solver
  at inference, and the Q-Chain loss is not a replayable proof certificate plus
  independent SMT oracle.
- **Relationship to SpatialEntail:** Closest neuro-symbolic training comparator;
  differs in artifact and verification boundary.
- **Suggested placement:** Trace supervision and neuro-symbolic training.
- **BibTeX key:** Existing `premsri2025neurosymbolic`.

### Graph-based synthetic reasoning data

- **Full citation and status:** Jiaming Zhou, Abbas Ghaddar, Ge Zhang, Liheng
  Ma, Yaochen Hu, Soumyasundar Pal, Mark Coates, Bin Wang, Yingxue Zhang, and
  Jianye Hao. “Enhancing Logical Reasoning in Large Language Models through
  Graph-based Synthetic Data.” 2024. **Preprint.**
  [arXiv:2409.12437](https://arxiv.org/abs/2409.12437).
- **What it did:** Constructed relational graphs, sampled non-repeating
  random-walk chains, removed an edge as the prediction target, verbalized the
  remaining graph, and trained on CLUTRR and StepGame with standard or
  Extract-Then-Answer prompting.
- **Supports:** Graph-structured spatial data synthesis; chain-length control;
  task-specific SFT; extracting a relational structure before answering.
- **Safe report claim:** “Graph-based StepGame SFT shows that structured
  synthetic chains can improve task-specific spatial reasoning.”
- **Does not support / avoid:** A sampled graph chain is not an independently
  checked proof certificate, and a removed edge label is not an open-world
  all-model entailment decision.
- **Relationship to SpatialEntail:** Spatially adjacent proof-structure and SFT
  precedent; motivates stronger provenance checks.
- **Suggested placement:** Synthetic training data and graph baselines.
- **BibTeX key:** Existing, corrected `zhou2024graphsynthetic`.

### Chain-of-Symbol prompting

- **Full citation and status:** Hanxu Hu, Hongyuan Lu, Huajian Zhang, Yun-Ze
  Song, Wai Lam, and Yue Zhang. “Chain-of-Symbol Prompting for Spatial Reasoning
  in Large Language Models.” COLM 2024. **Published.**
  [OpenReview](https://openreview.net/forum?id=Hvq9RtSoHG),
  [arXiv:2305.10276](https://arxiv.org/abs/2305.10276).
- **What it did:** Generated and manually corrected natural-language CoT
  demonstrations, replaced spatial relations with compact symbols, and used
  the resulting demonstrations for few-shot prompting on spatial QA and
  planning tasks.
- **Supports:** Symbolic trace form as a causal representation variable; token
  efficiency; a direct Natural-versus-Symbolic spatial comparator.
- **Safe report claim:** “Chain-of-Symbol establishes compact symbolic
  intermediate representations as a strong spatial-prompting baseline.”
- **Does not support / avoid:** Its natural and symbolic demonstrations are not
  independently checked renderings of one typed proof certificate, and the
  result does not predict which form will win under matched SFT.
- **Relationship to SpatialEntail:** Closest representation baseline;
  SpatialEntail's proposed advance is exact semantic pairing and controlled
  training, not symbolic traces themselves.
- **Suggested placement:** Trace representation; prompting baselines; novelty
  limitation.
- **BibTeX key:** Existing, corrected `hu2024chainofsymbol`.

## Proof-first and theory-first generation

### RuleTaker

- **Full citation and status:** Peter Clark, Oyvind Tafjord, and Kyle
  Richardson. “Transformers as Soft Reasoners over Language.” IJCAI 2020.
  **Published.** [arXiv:2002.05867](https://arxiv.org/abs/2002.05867),
  [official IJCAI proceedings PDF](https://www.ijcai.org/Proceedings/2020/0537.pdf).
- **What it did:** Sampled a small formal theory first, computed its exhaustive
  forward consequences and proof depths, then selected true and closed-world
  false queries before rendering facts, rules, and questions in synthetic
  English.
- **Supports:** Theory-first generation; solve-then-select evaluation items;
  proof-depth control; the general principle that labels should be computed
  from the visible theory.
- **Safe report claim:** “RuleTaker is an early theory-first precedent: it
  generates a formal theory, computes its consequences, and only then selects
  questions.”
- **Does not support / avoid:** It uses a closed-world assumption and does not
  implement SpatialEntail's premise-first open-world bucketing or qualitative
  spatial semantics.
- **Relationship to SpatialEntail:** Closest generic precedent for the order of
  premise sampling followed by semantic classification.
- **Suggested placement:** Generation-provenance section immediately before
  ProofWriter and PrOntoQA.
- **BibTeX key:** Proposed `clark2020ruletaker`.

### ProofWriter

- **Full citation and status:** Oyvind Tafjord, Bhavana Dalvi, and Peter Clark.
  “ProofWriter: Generating Implications, Proofs, and Abductive Statements over
  Natural Language.” Findings of ACL-IJCNLP 2021. **Published.**
  [ACL Anthology](https://aclanthology.org/2021.findings-acl.317/),
  [DOI](https://doi.org/10.18653/v1/2021.findings-acl.317),
  [arXiv:2012.13048](https://arxiv.org/abs/2012.13048).
- **What it did:** Defined proofs as fact/rule DAGs, supported closed- and
  open-world datasets including `Unknown`, and trained T5 to generate complete
  proofs or iterative one-step implications that are assembled into deeper
  proofs.
- **Supports:** Explicit proof objects; open-world unknown; proof depth;
  premise/rule/conclusion dependencies; theory-first semantics.
- **Safe report claim:** “ProofWriter establishes natural-language proof
  generation over synthetic theories, including open-world `Unknown` cases.”
- **Does not support / avoid:** Its model-generated proof is evaluated against
  synthetic gold structure; it is not accepted by an independent formal
  runtime checker. The work is Datalog reasoning, not spatial reasoning.
- **Relationship to SpatialEntail:** Generic proof-object and open-world
  precedent; SpatialEntail specializes these ideas to a spatial calculus and
  adds dual validation.
- **Suggested placement:** Generation provenance; proof representation; formal
  answer semantics.
- **BibTeX key:** Existing `tafjord2021proofwriter`.

### PrOntoQA

- **Full citation and status:** Abulhair Saparov and He He. “Language Models
  Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought.” ICLR
  2023. **Published.**
  [OpenReview](https://openreview.net/forum?id=qFVVBzXxR2V),
  [arXiv:2210.01240](https://arxiv.org/abs/2210.01240).
- **What it did:** Generated a symbolic ontology, walked it to construct a
  controllable modus-ponens proof, and translated the ontology and proof into
  a natural-language context, query, CoT, and label.
- **Supports:** **Direct proof-first generation precedent**; proof-length and
  distractor controls; symbolic-to-natural trace generation.
- **Safe report claim:** “PrOntoQA is clear prior art for proof-first synthetic
  reasoning data: the ontology and proof precede the natural-language example.”
- **Does not support / avoid:** Proof-first generation is therefore not a new
  generic contribution. PrOntoQA does not provide a qualitative spatial
  calculus, premise-first evaluation, paired trace arms, or an independent SMT
  oracle.
- **Relationship to SpatialEntail:** Nearest generation-order precedent and a
  mandatory qualifier for any priority language.
- **Suggested placement:** Definition of proof-first generation and novelty
  qualification.
- **BibTeX key:** Existing `saparov2023prontoqa`.

### EntailmentBank

- **Full citation and status:** Bhavana Dalvi, Peter Jansen, Oyvind Tafjord,
  Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark.
  “Explaining Answers with Entailment Trees.” EMNLP 2021. **Published.**
  [ACL Anthology](https://aclanthology.org/2021.emnlp-main.585/),
  [DOI](https://doi.org/10.18653/v1/2021.emnlp-main.585).
- **What it did:** Introduced multistep entailment trees and separated a setting
  with exactly the gold proof premises from settings with distractors or
  full-corpus retrieval.
- **Supports:** Separating inference quality from premise selection/retrieval;
  explicit tree-shaped dependencies; evaluation decomposition.
- **Safe report claim:** “EntailmentBank demonstrates the value of evaluating
  proof construction separately from premise selection and retrieval.”
- **Does not support / avoid:** It is not premise-first synthetic generation in
  SpatialEntail's sense, does not use an SMT oracle, and is not spatial.
- **Relationship to SpatialEntail:** Supports a premise-given diagnostic slice,
  not the core novelty claim.
- **Suggested placement:** Evaluation design and error decomposition.
- **BibTeX key:** Proposed `dalvi2021entailmentbank`.

## Formal verification and proof-carrying reasoning

### Formal entailment and satisfiability semantics

- **Full citation and status:** Stuart Russell and Peter Norvig. *Artificial
  Intelligence: A Modern Approach*, 4th ed. Pearson, 2020. **Published book.**
  [Official companion site](https://aima.cs.berkeley.edu/).
- **What it did:** Gives the standard model-theoretic definitions of
  satisfaction, entailment, satisfiability, soundness, completeness, and
  entailment-by-refutation.
- **Supports:** `Possible(P,A) iff SAT(P and A)` and `Entailed(P,A) iff
  UNSAT(P and not A)` as standard logical relationships, once the project's
  spatial encoding and ontology are declared.
- **Safe report claim:** “Entailment requires a claim to hold in every model of
  the premises; satisfiability requires at least one model.”
- **Does not support / avoid:** AIMA does not prove this project's encoding,
  parser, checker, or solver implementation sound and complete.
- **Relationship to SpatialEntail:** Foundational semantics, not novelty
  evidence.
- **Suggested placement:** Formal-method definitions and proof obligations.
- **BibTeX key:** Existing `russellNorvig2020aima`.

### Faithful Chain-of-Thought

- **Full citation and status:** Qing Lyu, Shreya Havaldar, Adam Stein, Li
  Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch.
  “Faithful Chain-of-Thought Reasoning.” IJCNLP-AACL 2023. **Published.**
  [ACL Anthology](https://aclanthology.org/2023.ijcnlp-main.20/),
  [DOI](https://doi.org/10.18653/v1/2023.ijcnlp-main.20).
- **What it did:** Had an LLM translate natural-language problems into
  interleaved natural-language decompositions and task-specific symbolic
  programs, then used a deterministic executor to produce the final answer.
- **Supports:** Moving answer production into an executable symbolic layer;
  mixed natural/symbolic representations; explicit dependency structure.
- **Safe report claim:** “Faithful CoT makes the final answer a deterministic
  result of an executable symbolic component rather than a free-form rationale.”
- **Does not support / avoid:** Execution faithfulness is conditional on the
  translation. The paper does not provide two matched views of one spatial
  proof or an independent second semantic checker.
- **Relationship to SpatialEntail:** Precedent for executable faithfulness;
  SpatialEntail adds a domain proof object and dual semantic validation.
- **Suggested placement:** Formal verification and trace representation.
- **BibTeX key:** Existing `lyu2023faithfulcot`.

### LogicGuide

- **Full citation and status:** Gabriel Poesia, Kanishk Gandhi, Eric Zelikman,
  and Noah D. Goodman. “Certified Deductive Reasoning with Language Models.”
  TMLR 2024. **Published.**
  [OpenReview](https://openreview.net/forum?id=yXnwrs2Tl6).
- **What it did:** Had models formalize natural-language assumptions and used
  the Peano theorem-proving environment to guide and certify subsequent
  deductive generation.
- **Supports:** Independently checkable language-model deduction; small trusted
  formal environments; explicit formalization/checking boundary.
- **Safe report claim:** “LogicGuide shows that language-model deduction can be
  constrained by an independently checkable theorem-proving environment.”
- **Does not support / avoid:** Certification is relative to the formalized
  assumptions and does not prove semantic faithfulness to the source text. It
  is not a spatial checker-plus-SMT design.
- **Relationship to SpatialEntail:** Strong checker precedent outside space.
- **Suggested placement:** Proof checking, trust boundary, and
  autoformalization limitation.
- **BibTeX key:** Existing `poesia2024logicguide`.

### PCRLLM

- **Full citation and status:** Tangrui Li, Pei Wang, Hongzheng Wang, Christian
  Hahm, Matteo Spatola, and Justin Shi. “PCRLLM: Proof-Carrying Reasoning with
  Large Language Models under Stepwise Logical Constraints.” 2025.
  **Preprint.** [arXiv:2511.08392](https://arxiv.org/abs/2511.08392).
- **What it did:** Generated a logical NAL instance before naturalizing the
  input and required a JSON response whose steps explicitly name premises,
  rules, conclusions, truth values, and evidential bases. It grades local and
  inter-step conformity.
- **Supports:** The term and mechanism “proof-carrying reasoning”; explicit
  premise/rule/conclusion schemas; chain-level machine grading.
- **Safe report claim:** “PCRLLM is direct prior art for requiring an LLM output
  to carry the premises, rules, and conclusions needed for stepwise checking.”
- **Does not support / avoid:** SpatialEntail must not claim to coin
  “proof-carrying reasoning.” PCRLLM uses the same NAL reasoning engine in
  construction and grading and does not add an independent spatial SMT oracle.
- **Relationship to SpatialEntail:** Closest vocabulary and structured-output
  precedent; different logic and validation independence.
- **Suggested placement:** Proof-carrying definition and novelty caveat.
- **BibTeX key:** Existing `li2025pcrllm`.

### Typed / Proof-Carrying Chain-of-Thought

- **Full citation and status:** Elija Perrier. “Typed Chain-of-Thought: A
  Curry-Howard Framework for Verifying LLM Reasoning.” 2025. **Preprint.**
  [arXiv:2510.01069](https://arxiv.org/abs/2510.01069).
- **What it did:** Required a typed JSON program, constructed a typed reasoning
  graph, applied certification gates, filtered self-consistency samples by
  certification, and deterministically rendered a textual proof-like view.
- **Supports:** Typed proof objects; complete dataflow; machine and human views
  from one representation; certification as a selection signal.
- **Safe report claim:** “Proof-Carrying CoT shows that typed dataflow can turn
  an LLM rationale into a machine-checkable program with a deterministic
  textual rendering.”
- **Does not support / avoid:** Its GSM8K type/dataflow checks are not a spatial
  proof calculus and are not a second SMT model-theoretic oracle.
- **Relationship to SpatialEntail:** Very close certificate-and-rendering
  precedent outside space.
- **Suggested placement:** Certificate schema and paired trace design.
- **BibTeX key:** Proposed `perrier2025typedcot`.

### ProofSketcher

- **Full citation and status:** Kranthi Kommuru, Kunal Khanvilkar, and Gaurav
  Parekh. “ProofSketcher: Hybrid LLM + Lightweight Proof Checker for Reliable
  Math/Logic Reasoning.” 2026. **Preprint.**
  [arXiv:2604.06401](https://arxiv.org/abs/2604.06401).
- **What it did:** Had an LLM emit a typed proof-sketch DSL; a lightweight
  trusted kernel expands it into proof obligations; external SMT/ATP results
  are admitted only through certificates validated by a trusted checker.
- **Supports:** A small proof checker plus separately invoked automated solvers;
  explicit trust boundaries; localized proof-repair feedback.
- **Safe report claim:** “ProofSketcher is a close non-spatial precedent for
  combining LLM proof sketches with a lightweight trusted kernel and
  certificate-checked external automation.”
- **Does not support / avoid:** It does not establish novelty for checker-plus-
  SMT architecture in general, and it provides no spatial benchmark,
  premise-first protocol, or paired Natural/Symbolic spatial data.
- **Relationship to SpatialEntail:** Strongest limiting source for the claimed
  architecture; SpatialEntail's novelty can only be domain integration and the
  exact local/global division.
- **Suggested placement:** Independent checker plus SMT; trust model; closest
  non-spatial architecture.
- **BibTeX key:** Proposed `kommuru2026proofsketcher`.

### Lean-STaR

- **Full citation and status:** Haohan Lin, Zhiqing Sun, Sean Welleck, and
  Yiming Yang. “Lean-STaR: Learning to Interleave Thinking and Proving.” ICLR
  2025 Spotlight. **Published.**
  [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a5357781c204d4412e44ed9cbcdb08d5-Abstract-Conference.html).
- **What it did:** Generated retrospective natural-language thoughts from
  ground-truth Lean tactics, trained on proof-state/thought/tactic triples, and
  used expert iteration that retains Lean-successful proofs.
- **Supports:** Interleaving natural reasoning with formal proof actions;
  verifier-filtered trace training; natural text grounded in formal states.
- **Safe report claim:** “Lean-STaR provides a theorem-proving precedent for
  pairing informal thoughts with formal proof states and tactics.”
- **Does not support / avoid:** Retrospective thoughts are not the same as two
  controlled renderings of a spatial proof certificate, and Lean acceptance
  does not validate natural-language equivalence by itself.
- **Relationship to SpatialEntail:** Important trace-pairing and
  verifier-filtering precedent outside space.
- **Suggested placement:** Natural/Symbolic trace supervision and formal
  verification.
- **BibTeX key:** Proposed `lin2025leanstar`.

### Herald

- **Full citation and status:** Guoxiong Gao, Yutong Wang, Jiedong Jiang, Qi
  Gao, Zihan Qin, Tianyi Xu, and Bin Dong. “Herald: A Natural Language
  Annotated Lean 4 Dataset.” ICLR 2025. **Published.**
  [OpenReview](https://openreview.net/forum?id=Se6MgCtRhz).
- **What it did:** Translated Mathlib tactic proofs line by line into natural
  language while preserving proof hierarchy and proof-state context, producing
  paired formal/informal theorem-proof data.
- **Supports:** Large-scale paired natural and symbolic proof traces;
  deterministic alignment targets; proof hierarchy in natural explanations.
- **Safe report claim:** “Herald is strong prior art for aligned natural-language
  annotations of formal proof traces.”
- **Does not support / avoid:** Paired Natural/Symbolic proofs are not new in
  general. Herald does not study qualitative space, matched spatial SFT arms,
  or independent SMT semantics.
- **Relationship to SpatialEntail:** Strongest broad limitation on the pairing
  claim; SpatialEntail's contribution is the spatial instantiation and
  controlled representation experiment.
- **Suggested placement:** Paired trace design and novelty qualification.
- **BibTeX key:** Proposed `gao2025herald`.

### FoVer

- **Full citation and status:** Ryo Kamoi, Yusen Zhang, Nan Zhang, Sarkar
  Snigdha Sarathi Das, Ranran Haoran Zhang, Wenpeng Yin, and Rui Zhang.
  “Efficient PRM Training Data Synthesis via Formal Verification.” Findings of
  ACL 2026. **Published.**
  [ACL Anthology](https://aclanthology.org/2026.findings-acl.403/),
  [DOI](https://doi.org/10.18653/v1/2026.findings-acl.403),
  [official repository](https://github.com/psunlpgroup/FoVer).
- **What it did:** Used Z3 to label formal-logic steps and Isabelle to label
  theorem-proving steps, then trained learned process reward models from those
  formally generated labels.
- **Supports:** Formal tools for scalable process-label synthesis; distinction
  between formally labeled training data and a learned verifier deployed later.
- **Safe report claim:** “FoVer shows that Z3 and Isabelle can supply efficient,
  exact step labels for training process reward models.”
- **Does not support / avoid:** A learned PRM trained on formal labels is not a
  proof-carrying artifact or an independent checker at inference. FoVer does
  not study spatial reasoning.
- **Relationship to SpatialEntail:** Adjacent verifier-guided data precedent;
  reinforces the need to name the verifier actually used at evaluation time.
- **Suggested placement:** Verifier-guided supervision and optional RL.
- **BibTeX key:** Proposed `kamoi2026fover`.

## Newest direct counterclaims

### `interwhen`

- **Full citation and status:** Vishak K Bhat, Prateek Chanda, Vijval Ekbote,
  Ashmit Khandelwal, Maitreyi Swaroop, Vineeth N. Balasubramanian, Subbarao
  Kambhampati, Nagarajan Natarajan, and Amit Sharma. “interwhen: A Generalizable
  Framework for Steering Reasoning Models with Test-time Verification.” 2026.
  **Preprint.** [arXiv:2602.11202](https://arxiv.org/abs/2602.11202).
- **What it did:** On the text-only SpatialMap split, forked a model's evolving
  reasoning trace, prompted a side stream to emit structured directional
  claims, checked each extracted claim against Z3 constraints derived from the
  problem, and fed contradiction or answer feedback back into generation.
- **Supports:** Direct feasibility of Z3-backed intermediate checking on the
  same benchmark family; consistency feedback during generation; a monitor-only
  baseline.
- **Safe report claim:** “`interwhen` already applies Z3-backed monitoring to
  intermediate claims on text-only SpatialMap.”
- **Does not support / avoid:** SpatialEntail cannot claim the first use of Z3,
  SMT, or intermediate verification for textual SpatialMap. `interwhen` checks
  extracted claim consistency, not a complete dependency-explicit certificate;
  it does not use proof-first or premise-first data construction and does not
  release paired Natural/Symbolic proof views.
- **Relationship to SpatialEntail:** Strongest direct counterclaim and required
  baseline/citation. The residual distinction is certificate replay plus a
  separate global semantic oracle and provenance-controlled data.
- **Suggested placement:** End of spatial solver/path related work; novelty
  qualification; evaluation baselines.
- **BibTeX key:** Existing, corrected `bhat2026interwhen`.

### Other 2026 counterclaims already annotated above

- **SODA** blocks “first pure-text spatial SFT/GRPO.”
- **SpatialText** blocks novelty based only on incomplete text or an
  undecidable/unknown category.
- **SiT-Bench and MentalMap** block broad claims that text-only spatial
  benchmarking or textual world-model diagnostics are new.
- **ProofSketcher** blocks a generic claim that checker-plus-SMT/ATP separation
  is itself new.
- **FoVer** blocks a generic claim that formal tools have not been used to
  synthesize process supervision.

## Contribution-to-source synthesis

This section maps each proposed contribution to both supportive and limiting
sources. The limiting source is often the more important citation for calibrated
novelty.

| Proposed contribution | Supporting / limiting primary sources | Report-ready synthesis |
|---|---|---|
| **Open-world, model-theoretic spatial evaluation under partial information** | AIMA for entailment/satisfiability definitions; ProofWriter for open-world `Unknown`; SpartQA for explicit unknown spatial QA; SpatialText for non-omniscient spatial cases | “SpatialEntail applies standard all-model entailment semantics to a text-only qualitative spatial benchmark; unknown and incomplete spatial cases exist in prior work, so the contribution is the exact formal contract and audit, not uncertainty itself.” |
| **World-first / proof-first / premise-first provenance taxonomy** | SpatialText illustrates coordinate/world-first construction; PrOntoQA is direct proof-first prior art; RuleTaker computes consequences from a sampled theory before selecting questions; EntailmentBank separates proof construction from premise retrieval | “The three labels organize established construction mechanisms. The taxonomy and their controlled comparison in qualitative space may be new; none of the mechanisms should be claimed as invented here.” |
| **Proof-first spatial training data with replayable certificates** | PrOntoQA for proof-first generation; ProofWriter for explicit proof DAGs; SpaRP for solver-derived spatial paths; PCRLLM for premise/rule/conclusion-carrying outputs | “The proposed contribution is a spatial synthesis: proof-first construction plus a replayable certificate and post-construction semantic re-solving. Proof-first data, proof DAGs, spatial paths, and proof-carrying outputs each have prior art.” |
| **Premise-first solve-and-bucket evaluation** | RuleTaker for theory-first closure; ProofWriter for theory-relative open-world classification; SpatialText for deliberately incomplete descriptions; EntailmentBank for premise-given versus retrieval-separated evaluation | “No located source uses the exact premise-first solve-and-bucket protocol for open-world qualitative spatial candidate sets. This remains a bounded absence finding, not proof of priority.” |
| **Small proof checker plus independent SMT semantic oracle** | LogicGuide for independently checked deduction; ProofSketcher for a lightweight kernel plus certificate-checked external automation; `interwhen` for Z3 on SpatialMap; FoVer for Z3-derived step labels | “Formal checkers and SMT-backed verification are established, including on SpatialMap. The narrower proposed contribution is the auditable separation of local certificate replay from global spatial possibility/entailment semantics.” |
| **Paired Natural and Symbolic traces from one proof object** | SpaRP for symbolic paths verbalized link-by-link; Chain-of-Symbol for a direct spatial Natural/Symbolic representation comparison; Proof-Carrying CoT for typed JSON plus deterministic text; Lean-STaR and Herald for aligned informal/formal proof traces | “Natural/symbolic pairing and symbolic spatial traces are established. SpatialEntail's proposed contribution is a matched spatial ablation in which both forms are deterministic views of the identical checked certificate.” |
| **Trace-supervised spatial SFT, with optional verifier-based RL** | SpaRP and Q-Chains for spatial trace supervision; graph-based StepGame for synthetic SFT; SODA for pure-text spatial SFT+GRPO; FoVer for formally synthesized process labels | “Neither spatial trace SFT nor spatial GRPO is new. The research question is whether certificate-grounded supervision or rewards improve structural generalization beyond matched answer-only and trace baselines.” |

## Calibrated overall statement

The primary sources support the following conclusion:

> Each major SpatialEntail ingredient has precedent in spatial reasoning,
> logical reasoning, or formal theorem proving. In the bounded primary-source
> search through 7 October 2026, no source was found that combines proof-first
> spatial certificates, premise-first open-world evaluation, a local replay
> checker independent of a global SMT semantic oracle, and paired Natural and
> Symbolic renderings of one derivation in a single text-only qualitative
> spatial benchmark and learning pipeline.

Use “to our knowledge” for any priority claim. Describe SpatialEntail as an
integration and controlled evaluation of established ideas. Re-run the search
before submission, especially forward citations to SpaRP, SpatialText,
`interwhen`, PCRLLM, ProofSketcher, and the 2026 ACL proceedings.
