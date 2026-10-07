# Textual spatial reasoning: benchmark and method landscape

*Literature note, checked 4 October 2026. It prioritises papers and official
dataset repositories that define a benchmark or method. Results are reported
only with their original task, split, model, and metric; there is no defensible
single "SOTA" ranking across these protocols.*

## Executive finding

The closest external comparators for this project are **StepGame** for
controlled multi-hop directional reasoning, **SpartQA / SpaRTUN / ReSQ** for
richer spatial language and open-world multi-label QA, and **SpatialEval
Spatial-Map TQA** for the exact text-only map-style task used in this repository.
SpatialSense is important evidence about relation definitions, annotation
ambiguity, and shortcut-resistant collection, but it is an image-based binary
relation-recognition benchmark rather than a text-reasoning comparator.

The strongest defensible thesis position is therefore not "a new overall SOTA
in spatial reasoning." It is a validity-first study of one explicit qualitative
calculus: audit inherited labels against formal semantics; generate
solver-round-tripped training and evaluation data; compare controlled reasoning
trace representations; and test whether outcome-verifiable RL adds anything to
a strong SFT policy on a frozen hard holdout.

## 1. Benchmark landscape

| Benchmark | Input and target | Construction / scale | Original evaluation and result to retain | Relevance and non-comparability |
|---|---|---|---|---|
| [StepGame](https://arxiv.org/abs/2204.08292) (AAAI 2022) | Text stories; classify the relation between two entities after 1--10 hops | Programmatically generated from 26 entity names and paraphrase templates. For each hop count: 10,000 train, 1,000 validation, and 10,000 test examples; training covers 1--5 hops and systematic-generalisation tests cover 6--10, with distractors at test time. | The paper's TP-MANN reaches 31.25% at 5 hops and 21.46% at 10 hops under its noisy-test protocol (means over five runs). Later numbers must be matched to the same split and noise setup. | Closest standard benchmark for proof-depth scaling, but its labels, language, and single-relation classification differ from SpatialMap's direction / which / count MCQs. |
| [SpartQA](https://aclanthology.org/2021.naacl-main.364/) ([arXiv](https://arxiv.org/abs/2104.05832)) | Text stories with find-relation (FR), find-block (FB), choose-object (CO), and yes/no/unknown (YN) questions; FR can be multi-label | SpartQA-Human contains about 1.1K expert-authored QA pairs grounded in rearrangeable NLVR scenes. SpartQA-Auto uses grammars, scene graphs, and spatial rules to generate distant supervision. | The original paper reports 92% expert accuracy on 100 human-test examples and evaluates each question family separately; automatic-data pretraining improves transfer to SpartQA-Human. | Stronger test of reference, containment, topology, quantifiers, and open-world `DK`; not the same ontology or answer contract as compass-only SpatialMap. |
| [SpaRTUN and ReSQ](https://arxiv.org/abs/2210.16952) | SpaRTUN is controlled synthetic spatial QA with formal facts/rules; ReSQ is human-written realistic spatial QA | SpaRTUN provides formal representations and reasoning annotations; ReSQ tests transfer to less controlled language. | The source paper evaluates YN and FR separately and tests transfer from synthetic pretraining to several target datasets. | Useful train-on-symbolic / transfer-to-natural pair. ReSQ also exposes extraction and commonsense limitations that are mostly absent from template-generated maps. |
| [SpatialEval: Spatial-Map TQA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89cc5e613d34f90de90c21e996e60b30-Abstract-Conference.html) ([paper](https://arxiv.org/abs/2406.14852), [official repository](https://github.com/jiayuww/SpatialEval)) | Text-only pairwise map relations plus one of three MCQs per map (direction, object satisfying a relation, count). TQA means the text contains all information needed; VQA and VTQA are separate modalities. | Configurable synthetic Spatial-Map is one of four SpatialEval tasks. The released local TQA slice used here has 1,500 rows: 500 per question family. | The paper uses four-choice accuracy and a step-by-step prompt. Its main finding is modality-level: text input often helps more than vision input; it did not publish this repository's later solver-corrected protocol. | The exact benchmark family for this FYP. Original accuracy, corrected-label accuracy, fifth-option results, and two-pass results are different protocols and must never be merged into one number. |
| [SpatialSense](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_SpatialSense_An_Adversarially_Crowdsourced_Benchmark_for_Spatial_Relation_Recognition_ICCV_2019_paper.html) ([arXiv](https://arxiv.org/abs/1908.02660)) | **Visual**, not text-only: given an image, two boxes, and one of nine predicates, decide whether the relation holds | 17,498 positive/negative relations over 11,569 real images, collected adversarially against language-only and 2D-only baselines | Humans achieve 94.6%; the paper uses binary relation accuracy and explicitly studies shortcut bias. | Valuable precedent for benchmark validity and relation ambiguity, not a direct score comparator. |

Two immediate rules follow.

1. Report StepGame by hop count and exact noise/split protocol; report SpartQA
   by question family and answer semantics; report SpatialMap by dataset version,
   modality, label policy, option set, and decoding protocol.
2. Put visual, embodied, and 3D benchmarks in related work, but do not use them
   to establish a text-only leaderboard. SpatialSense, Rel3D, and later VLM
   benchmarks measure perception plus spatial inference, whereas the FYP isolates
   reasoning from a complete textual world description.

### SpatialEval modality alignment: facts and inference boundary

**Official paper facts.** SpatialEval defines the unit at the *problem* level:
"each problem" has an image and a text representation sufficient to answer its
spatial question. Section 2.1 then says that TQA is purely textual, VQA supplies
an image without its textual description (the question itself remains text),
and VTQA supplies both representations, deliberately making their information
redundant; all are evaluated on "the same set of questions"
([paper, Introduction](https://arxiv.org/html/2406.14852v2#S1),
[Section 2.1 and Table 1](https://arxiv.org/html/2406.14852v2#S2.SS1)).
For Spatial-Map specifically, the authors construct a map with configurable
$K$, unique location names, and a textual representation made of pairwise
relations. The appendix says that three questions, Q1--Q3, are associated with
each Spatial-Map sample. Independently, the official evaluator's three
zero-based Spatial-Map question branches expect a direction answer, a
location-name answer, and a numeric answer
([paper, Spatial-Map construction](https://arxiv.org/html/2406.14852v2#S2.SS1),
[Appendix C.4](https://arxiv.org/html/2406.14852v2#A3.SS4),
[evaluator Q1--Q3 branches](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/evals/evaluation.py#L26-L136)).

**Official repository facts.** The repository exposes TQA, VQA, and VTQA as
separate dataset configurations and as a command-line `mode`, rather than using
a modality value in the inference paths
([README](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L44-L60),
[mode configuration](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/configs/inference_configs.py#L33-L42)).
Those paths always build the prompt from the row's `text`; the VLM path omits
the image for TQA and reads `image` for VQA/VTQA
([VLM inference](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/inference_vlm.py#L100-L145)).
The README documents `text` and `image` as input columns and `oracle_answer`,
`oracle_option`, and `oracle_full_answer` as correct-answer columns; both
inference paths carry those oracles and `id` into their output
([documented columns](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L57-L60),
[LM output fields](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/inference_lm.py#L79-L85),
[VLM output fields](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/inference_vlm.py#L243-L250)).
The code interprets the final dot-separated component of `id` as the question
ID and the first as the task, but does not document the remaining components
or perform a cross-modality join. Exact-match evaluation compares only
`oracle_answer`, not the other two oracle fields
([grouping](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/inference_lm.py#L34-L47),
[evaluation](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/evals/evaluation.py#L272-L290)).

**Released-data verification (official Hugging Face revision
`59ce045`, checked 5 October 2026).** A row-level comparison of every non-image
field in the three official test configurations establishes exact release
alignment, not merely equal aggregate size. Each configuration has 4,635 rows:
1,500 Spatial-Map, 1,500 Maze-Nav, 1,500 Spatial-Grid, and 135 Spatial-Real
([official size endpoint](https://datasets-server.huggingface.co/size?dataset=MilaWang%2FSpatialEval),
[revision-pinned dataset tree](https://huggingface.co/datasets/MilaWang/SpatialEval/tree/59ce0450bf65ae8dfa3590062442cf642474d7fe)).
The literal IDs differ in their modality component, for example
`spatialmap.tqa.2000.0` versus `spatialmap.vqa.2000.0`. After replacing only
that component, however, all 4,635 keys are shared and occur in the same row
order. `oracle_answer`, `oracle_option`, and `oracle_full_answer` match exactly
for all 4,635 aligned triples. TQA has five columns (`id`, `text`, and the three
oracles); VQA and VTQA add `image`. Their `image.path` values match on all 4,635
aligned rows and are non-null, with 1,635 distinct paths in each visual
configuration: 500 for each synthetic task and 135 for Spatial-Real
([official Parquet manifest](https://datasets-server.huggingface.co/parquet?dataset=MilaWang%2FSpatialEval),
[TQA viewer](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/tqa/test),
[VQA viewer](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/vqa/test),
[VTQA viewer](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/vtqa/test)).

Spatial-Map is especially clear: all 1,500 normalized IDs and all three oracle
fields align. The question-and-options suffix is literally identical between
VQA and VTQA in 1,500/1,500 rows and between TQA and either visual mode in
1,491/1,500; the nine TQA differences are only extra spaces around the same
location name, `Patty's Pet Shop`. Removing the fixed modality introductions,
the complete Spatial-Map description-plus-question text is literally identical
between TQA and VTQA in 1,488/1,500 rows. Thus the release contains aligned
versions of the same Spatial-Map problems and oracles, while the whole `text`
field is intentionally modality-specific. Across all tasks, whole-text equality
is 0/4,635 for TQA--VQA, 135/4,635 for TQA--VTQA (the Spatial-Real rows), and
0/4,635 for VQA--VTQA. Question wording also changes by modality in some other
tasks--for example, Maze-Nav uses `X` in textual paths and `Blue` in visual
paths--so oracle/key alignment is the robust cross-modality criterion, not raw
full-text equality.

**Inference boundary.** These are verified facts about the released dataset;
they do not reveal the implementation of the unreleased generator. The audit
also compared visual path metadata, not roughly 1.4 GB of image blobs, so it
does not independently prove byte-for-byte VQA/VTQA image equality. The paper's
construction description plus the released keys, paths, questions, and oracles
strongly establish paired benchmark instances, but claims about private
generation code, sampling order before export, or future generator publication
would remain inference.

### Release audit and the scope of the research gap

**SpatialEval released the generated benchmark, not its generator (checked 4
October 2026).** The paper says that its synthetic tasks are configurable and
scalable and that Spatial-Map is created with a configurable number of objects
([paper, Section 2.1](https://arxiv.org/html/2406.14852v2#S2.SS1)).
That describes the authors' construction process; it is not evidence that the
construction code was published. The current official repository contains
inference and exact-match evaluation code, but no dataset-construction module,
and its README still says, "Stay tuned! The dataset generation script will be
released in Feburary"
([README at current commit](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L138-L140),
[complete tree](https://github.com/jiayuww/SpatialEval/tree/d82ba382805265f205693cb83a30c4866ea6c35b)).
The linked official Hugging Face release contains generated TQA, VQA, and VTQA
test Parquet files (4,635 rows per modality), rather than generator source
([dataset card and tree](https://huggingface.co/datasets/MilaWang/SpatialEval/tree/59ce0450bf65ae8dfa3590062442cf642474d7fe)).
The repository's only public branch and its history through that commit contain
no generator. The precise conclusion is therefore **no generator is publicly
available in the official release locations checked**, not that no private
generator exists or that it will never be released.

The literature evidence also does **not** justify a field-wide claim that
spatial reasoning has "shifted" away from text-only LLMs. A bounded observation
is supportable: three 2024--2025 benchmark papers reviewed here make perception,
video/3D reconstruction, or robotics part of the task.
SpatialEval jointly studies TQA, VQA, and VTQA;
[VSI-Bench](https://arxiv.org/abs/2412.14171) evaluates MLLMs on more than 5,000
questions derived from egocentric videos; and
[RoboSpatial](https://arxiv.org/abs/2411.16537) trains and evaluates 2D/3D VLMs
for robot-centric spatial understanding. This establishes active recent
multimodal/3D benchmark development, not its share of the whole field or a
temporal trend. Text-only LLM work continued concurrently: ACL 2024's
[SpaRC/SpaRP](https://aclanthology.org/2024.acl-long.261/) characterises LLM
spatial relations and reasoning paths, and the 2025 neuro-symbolic study above
trains over spatial language. The defensible gap for this thesis is consequently
narrower: controlled, LLM-only reasoning from a complete textual map remains a
useful isolation setting, especially for solver-audited labels, explicit answer
semantics, and matched training ablations. These sources do not support the
broader claim that textual spatial-reasoning research is absent.

## 2. Annotation validity is part of the reasoning task

### Confirmed findings

- **Relation words need operational definitions.** SpatialSense already noted
  inherent annotation noise (94.6%, not 100%, human accuracy). A later audit,
  [SpatialSense+](https://arxiv.org/abs/2403.00729), found that the original data
  mixed viewer-centric and object-centric frames, used polysemous relations such
  as *on*, and inherited idiomatic language bias. The authors wrote explicit
  physical definitions and relabelled 7,254 triplets / 4,418 images, yielding
  5,346 train, 808 validation, and 1,100 test examples. This is direct evidence
  that a benchmark label is meaningful only relative to a stated relation
  calculus.
- **Unknown is not false.** SpartQA explicitly adopts an open-world assumption:
  YN questions use `Yes`, `No`, or `DK`, because an unmentioned relation is not
  automatically false. FR is multi-label. Changing either convention changes
  the task, not merely the metric.
- **Human-written spatial annotations contain omissions and errors.**
  [Mirzaee and Kordjamshidi (2023)](https://aclanthology.org/2023.findings-emnlp.221/)
  report missing/noisy SpartQA-Human spatial-role and coreference annotations.
  Their Prolog reasoner produces no reasoning errors on SpaRTUN, which was
  generated by that reasoner, but extraction noise and incomplete rule coverage
  remain on human language.
- **This is a general benchmark problem, not a SpatialMap-only concern.** The
  5,700-item expert audit in
  [MMLU-Redux](https://arxiv.org/abs/2406.04127) first asks whether a question is
  unambiguous, then whether exactly one option is valid, then whether that option
  matches the key. It estimates a 6.49% overall MMLU error rate and shows model
  rankings can change after filtering invalid items. A broader
  ["platinum benchmark" audit](https://arxiv.org/abs/2502.03461) separately
  distinguishes wrong labels from contradictory, ambiguous, and underspecified
  questions; for most studied benchmarks, over half of apparent frontier-model
  failures disappeared after cleaning.

### Repository evidence (not an external published leaderboard)

The current V2 solver audit makes the label issue concrete for the 1,500
released Spatial-Map TQA rows. Under a strict `SINGLE` contract over all worlds
consistent with the text, the published option is the unique entailed answer in
667 rows and merely one possible answer in 833 rows:

| Query family | Unique published answer | Published answer possible but not unique |
|---|---:|---:|
| Direction | 332 | 168 |
| Which object | 140 | 360 |
| Count | 195 | 305 |
| **Total** | **667** | **833** |

This does **not** prove that the SpatialEval authors intended the same
model-theoretic `SINGLE` semantics. It proves that original exact-match accuracy
and this thesis's invariant-answer accuracy measure different contracts. The
audit must therefore be presented as an explicit semantic reinterpretation and
correction, with original and corrected results kept side by side.

## 3. Solver-backed and neuro-symbolic spatial reasoning

There are three distinct patterns in the literature.

| Pattern | Primary example | What is verified | Main limitation |
|---|---|---|---|
| Programmatic benchmark generation | StepGame; SpartQA-Auto; SpaRTUN | The sampled world, direct facts, inferred relations, and answer can be derived from generation state/rules | A correct generator can still encode the wrong task semantics or permit shortcut-heavy language. |
| Neural extraction then symbolic solving | [PistaQ](https://aclanthology.org/2023.findings-emnlp.221/) and the GPT-3 + ASP comparison in [Premsri and Kordjamshidi](https://arxiv.org/abs/2406.13828) | Prolog/ASP applies converse, inverse, transitive, and topological rules deterministically after text is formalised | Extraction, entity resolution, and rule coverage become the bottleneck; the pipeline is strongest on controlled synthetic language. |
| Logic-constrained neural training | [Neuro-symbolic Training for Spatial Reasoning over Natural Language](https://arxiv.org/abs/2406.13828) | A Q-Chain is generated from 79 spatial rules; differentiable soft-logic penalties discourage inconsistent intermediate predictions during fine-tuning | The model is not a certified solver at inference, and the gains are task/model specific. |

The neuro-symbolic training paper is particularly useful because it reports
depth-resolved StepGame results. Its BERT transferred from SpaRTUN and trained
with Q-Chain constraints rises from 28.16% to 44.05% at 10 hops relative to the
plain BERT baseline, while a GPT-3 extraction + ASP solver reports 88.30% on the
same hop column. This is evidence for two narrower claims: logical supervision
helps longer compositions, and a correct external reasoner can still be much
stronger when extraction is reliable. It is not evidence that one architecture
dominates realistic spatial language.

[Faithful Chain-of-Thought](https://aclanthology.org/2023.ijcnlp-main.20/)
provides the relevant general design principle: translate language into an
executable symbolic chain, then let a deterministic solver produce the answer.
Its guarantee is answer--trace faithfulness *conditional on correct
translation*. The FYP's round-trip rule (render, reparse, re-solve, and reject
on any mismatch) is a stronger data-generation safeguard than generating prose
rationales and trusting them.

## 4. SFT traces, process supervision, and outcome supervision

These terms should not be collapsed.

| Setting | Training signal | What a positive result would establish |
|---|---|---|
| Answer-only SFT | Gold final answer tokens | The model can imitate the answer mapping; no claim that its intermediate text is correct. |
| Trace SFT / scratchpad training | Gold intermediate tokens plus final answer | A particular serialized computation is useful to learn under the tested length, model, and data regime. |
| Process reward model (PRM) | Step-level correctness labels used to score candidate traces | The verifier can rank solutions using intermediate correctness. This is not the same experiment as trace SFT. |
| Outcome reward model or RLVR | One score from the final answer/test result | Credit is sparse, but labels can be cheap and exact in verifiable domains. Correct answers do not certify faithful traces. |

[Scratchpads](https://arxiv.org/abs/2112.00114) provide early supervised
evidence that emitting intermediate computation can improve length
generalisation on algorithmic tasks. For spatial reasoning specifically,
[Chain-of-Symbol](https://openreview.net/forum?id=Hvq9RtSoHG) reports that
compact symbolic prompts outperform natural-language CoT across its three
spatial reasoning/planning settings while using fewer tokens (including
31.8% to 92.6% on its Brick World setting). This supports testing symbolic
representations, not assuming they will win under SFT on SpatialMap.

A current inference-time preprint,
[Spatial Reasoning via Modality Switching](https://arxiv.org/abs/2606.31285),
likewise reports accuracy improvement of up to 42% on StepGame by converting
text into explicit grid representations, while tracing most residual error to
relation extraction. It is useful corroboration that representation is a real
experimental variable, but it is neither an SFT result nor directly comparable
with the FYP's coordinate-free axis traces.

For process versus outcome supervision, the best-known direct comparison is
[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
(ICLR 2024): on a representative MATH subset, its process-supervised verifier
outperforms an outcome-supervised verifier and reaches 78% with best-of-N
selection. That result concerns verifier training and mathematical solutions;
it should motivate a spatial process-verification experiment, not be cited as
proof that process-supervised spatial SFT must outperform answer rewards.

The most direct trace evidence for this thesis remains its own controlled
Experiment 13 comparison: with the same 8K prompts, order, base model, recipe,
and context length, delta-state SFT scores 97.8% on the V13.1 development suite
versus 85.6% for full-state SFT, and 97.5% versus 63.2% at depth/length 10.
Because V13.1 informed later curriculum design, this is a strong internal
ablation rather than an untouched external-benchmark SOTA result.

## 5. Verifiable rewards and GRPO: relevant but secondary

[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced Group Relative
Policy Optimisation (GRPO), which estimates relative advantages within a group
of sampled completions without training a separate value critic.
[DeepSeek-R1](https://arxiv.org/abs/2501.12948) demonstrates the relevant reward
pattern at scale: rule-based accuracy rewards for deterministic math answers or
code tests, plus format rewards, without a learned neural reward model for the
reasoning task.

SpatialMap is suitable for the same *mechanism* because a solver can parse an
answer set and assign an exact reward. The appropriate thesis experiment is
therefore narrow: start from the strongest delta-state SFT policy, use the same
solver and answer semantics for reward and evaluation, calibrate prompts so
sample groups contain both successes and failures, and compare SFT against
SFT+GRPO on a frozen disjoint hard holdout plus retention suites. In ordinary
GRPO, an all-correct or all-wrong group has zero within-group reward variance
and supplies no relative advantage; this is an experimental-design constraint,
not evidence that GRPO itself improves spatial reasoning.

Until a completed run shows holdout gain and retention, the supported statement
is **"solver-verifiable GRPO is implemented as a marginal post-SFT probe,"**
not **"RL improves SpatialMap."** Outcome reward can optimise any behaviour
correlated with the answer, so trace correctness should be audited separately
if the thesis makes a process-quality claim.

## 6. Suggested thesis positioning (interpretation, not a literature fact)

### Claims the evidence supports

1. Existing spatial benchmarks test materially different capabilities:
   perceptual relation recognition, controlled graph composition, rich
   language extraction, and map-question answering.
2. Formal answer semantics and label audits are prerequisites for meaningful
   exact-match evaluation, especially under partial information and multi-label
   questions.
3. Solver-backed generation can make answers and difficulty controls auditable,
   but parser coverage, world-model assumptions, and train/test separation must
   also be checked.
4. Intermediate representation is a causal experimental variable. Compact
   symbolic or delta-state traces can remove an avoidable serialization burden;
   the exact benefit must be measured under controlled SFT ablations.
5. Exact-answer RLVR/GRPO is justified only as a marginal test after SFT, with
   reward variance, a frozen holdout, and retention reporting.

### Claims to avoid

- A single "state of the art" number spanning SpatialSense, StepGame,
  SpartQA/SpaRTUN/ReSQ, and SpatialEval.
- Treating SpatialSense as a text-only reasoning benchmark.
- Treating a published SpatialMap label, a possible answer, a uniquely entailed
  answer, and an all-possible answer set as interchangeable.
- Claiming that a generated reasoning trace is faithful merely because its
  final answer is correct.
- Generalising the MATH process-supervision result to spatial SFT without a
  matched spatial experiment.
- Claiming a GRPO benefit before a completed, retained, seed-aware comparison.

## Primary sources

- Yang et al. (2019), *SpatialSense: An Adversarially Crowdsourced Benchmark
  for Spatial Relation Recognition*, ICCV.
  [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_SpatialSense_An_Adversarially_Crowdsourced_Benchmark_for_Spatial_Relation_Recognition_ICCV_2019_paper.html)
- Shi et al. (2022), *StepGame: A New Benchmark for Robust Multi-Hop Spatial
  Reasoning in Texts*, AAAI. [arXiv:2204.08292](https://arxiv.org/abs/2204.08292)
- Mirzaee et al. (2021), *SpartQA: A Textual Question Answering Benchmark for
  Spatial Reasoning*, NAACL, doi:[10.18653/v1/2021.naacl-main.364](https://doi.org/10.18653/v1/2021.naacl-main.364)
- Mirzaee and Kordjamshidi (2022), *Transfer Learning with Synthetic Corpora
  for Spatial Role Labeling and Reasoning*.
  [arXiv:2210.16952](https://arxiv.org/abs/2210.16952)
- Mirzaee and Kordjamshidi (2023), *Disentangling Extraction and Reasoning in
  Multi-hop Spatial Reasoning*, Findings of EMNLP,
  doi:[10.18653/v1/2023.findings-emnlp.221](https://doi.org/10.18653/v1/2023.findings-emnlp.221)
- Wang et al. (2024), *Is A Picture Worth A Thousand Words? Delving Into
  Spatial Reasoning for Vision Language Models*, NeurIPS,
  doi:[10.52202/079017-2400](https://doi.org/10.52202/079017-2400)
- Rizvi, Zhu, and Gurevych (2024), *SpaRC and SpaRP: Spatial Reasoning
  Characterization and Path Generation for Understanding Spatial Reasoning
  Capability of Large Language Models*, ACL,
  doi:[10.18653/v1/2024.acl-long.261](https://doi.org/10.18653/v1/2024.acl-long.261)
- Yang et al. (2025), *Thinking in Space: How Multimodal Large Language Models
  See, Remember, and Recall Spaces*, CVPR.
  [arXiv:2412.14171](https://arxiv.org/abs/2412.14171)
- Song et al. (2025), *RoboSpatial: Teaching Spatial Understanding to 2D and 3D
  Vision-Language Models for Robotics*, CVPR.
  [arXiv:2411.16537](https://arxiv.org/abs/2411.16537)
- Premsri and Kordjamshidi (2025), *Neuro-symbolic Training for Reasoning over
  Spatial Language*, Findings of NAACL.
  [arXiv:2406.13828](https://arxiv.org/abs/2406.13828)
- Hu et al. (2024), *Chain-of-Symbol Prompting for Spatial Reasoning in Large
  Language Models*, COLM. [OpenReview](https://openreview.net/forum?id=Hvq9RtSoHG)
- Lyu et al. (2023), *Faithful Chain-of-Thought Reasoning*, IJCNLP-AACL,
  doi:[10.18653/v1/2023.ijcnlp-main.20](https://doi.org/10.18653/v1/2023.ijcnlp-main.20)
- Lightman et al. (2024), *Let's Verify Step by Step*, ICLR.
  [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- Shao et al. (2024), *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models*. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Guo et al. (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
  via Reinforcement Learning*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
