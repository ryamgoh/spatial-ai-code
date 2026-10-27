# SpatialEntail Pivot: Primary-Source Check

Supporting literature note for [spatialentail-composition-v1](../README.md).
Design recommendations below record the discussion at the time of research.
The [main pivot](../pivot_20261026.md) supersedes them for current decisions,
including the final answer interface; this is not a second experiment plan.

2026-10-26. Scope: original Chain-of-Symbol (CoS), plus three closely relevant works: SpatialEval/Spatial-Map, SPARTQA, and SPARTUN. **Documented** reports source content; **inference/recommendation** denotes this note's analysis, not an author claim.

## 1. Chain-of-Symbol: What Was Actually Tested

**Documented.** Hu et al.'s [May 2023 original](https://arxiv.org/html/2305.10276v1) is titled *Chain-of-Symbol Prompting Elicits Planning in Large Langauge Models*. The [2024 v7 full text](https://arxiv.org/html/2305.10276v7) and [COLM record](https://openreview.net/forum?id=Hvq9RtSoHG) use *Chain-of-Symbol Prompting for Spatial Reasoning in Large Language Models*. These are versions of the same work, not independent replications.

- **Prompting, not SFT:** natural-language problems remain shared; manually constructed symbolic intermediate demonstrations replace natural-language CoT demonstrations. Models generate intermediate steps at inference. v7 sections 3-4 describe five-shot matched examples; Appendix A.1 discusses training with CoS only as a future possibility.
- **Actual notation:** Appendix A.3 uses `A//C//B` for brick stacks; `C->B->A` for horizontal/front ordering; `(large, round, black)` for object attributes; and `bank A / bank C / ...` with explicitly supplied road lengths summed for navigation. Spatial-QA examples use `<`, `>`, up/down arrows, `=`, and infinity-like distance markers. These are task-specific encodings, not a universal displacement calculus. The [official prompt file](https://github.com/hanxuhu/chain-of-symbol-planning/blob/main/demo/cos_demo_2d_1.txt) also contains inconsistent arrow glosses, so copying its notation without a formal legend is unsafe.
- **Geometry:** Brick World concerns stacking/accessibility; NLVR manipulation concerns objects in three boxes; navigation uses a binary-tree road network with explicit 100/200-meter edges. Adding those road lengths is legitimate there, not evidence for unit-step addition of qualitative directions.
- **Accuracy correction:** v7 Table 1's 31.8% to 92.6% is **+60.8 percentage points versus zero-shot CoT**, on 1D Shuffle Label. Matched five-shot CoT is 75.1%, making that comparison **+17.5 points**. Table 3 reports smaller spatial-QA gains: 47.1% to 49.4% for GPT-3.5 and 69.8% to 72.6% for GPT-4.
- **Token scope:** 407 to 139 is 65.8% fewer **demonstration intermediate-step tokens**, not total request, generated-answer, or SFT-training tokens. Appendix A.2.1 corrects earlier whitespace-counting language to OpenAI Playground tokenization. The prose describing 653 to 534 as "nearly by half" is arithmetically wrong: approximately 18.2%.
- **Limits:** v7 section 4.4 reports that smaller Llama-2 models do not consistently benefit. Its main text calls the QA dataset SPARTUN, but appendix captions call it SPARTQA; the original SPARTUN paper identifies SPARTUN as synthetic, not the human-annotated dataset described by CoS. Treat that QA attribution cautiously.

**Inference:** CoS motivates a representation comparison, but establishes neither a matched Natural/Symbolic SFT result nor correctness of symbolic traces.

## 2. SpatialMap-TQA: Direction Is Not Displacement

**Documented.** Wang et al.'s [*Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models*](https://arxiv.org/html/2406.14852), NeurIPS 2024, introduces SpatialEval. Spatial-Map is one task; TQA is its text-only modality, not a separate paper. Section 2 describes arbitrarily positioned named objects and pairwise directional descriptions, contrasting Spatial-Map with rigid Spatial-Grid. Questions cover direction, object selection, and counting. Evaluation uses four-option questions and answer-plus-explanation prompting of existing checkpoints, not a new matched SFT intervention.

The [official release README](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md) points to [MilaWang/SpatialEval](https://huggingface.co/datasets/MilaWang/SpatialEval), with `text`, `oracle_answer`, `oracle_option`, and `oracle_full_answer`. At checked commit `d82ba382805265f205693cb83a30c4866ea6c35b`, generation scripts are still advertised as forthcoming and absent from the repository tree.

**Release evidence and inference.** [Released rows 0-3](https://datasets-server.huggingface.co/rows?dataset=MilaWang/SpatialEval&config=tqa&split=test&offset=0&length=4) include:

- `2000.0`: Narwhal is NW of Police, Coral is NW of Narwhal, and Coral is NW of Police. Assigning every NW edge `(-1,+1)` contradicts the direct edge; qualitative inequalities do not. Thus uniform unit-displacement semantics cannot satisfy this released story.
- `2003.0`: the oracle says Recycle Center is NE of Nightingale Novelties. Under SpatialEntail's proposed sign-based semantics, both NE and NW are possible. Set Recycle `R=(0,0)`, Sally `S=(3,-3)`, Andy `A=(2,-2)`, Unicorn `U=(1,-2.5)`, Trail `T=(0.5,-5)`. Nightingale `N=(-1,-4)` yields NE; `N=(1,-4)` yields NW. Both layouts have distinct points and satisfy all nine released directional constraints, checked directly.

**Uncertainty/world assumptions:** the inspected paper and release documentation do not explicitly specify open-world versus closed-world relation semantics, an unknown-answer policy, or a complete formal direction calculus. The examples use definite oracle answers; this does **not** establish closed-world semantics. The counterexample establishes underdetermination under *our* semantics, not the authors' intended labeling algorithm. Exact geometric thresholds and generator behavior remain unverified because the generator was unavailable. Do not describe SpatialMap as a unit-step or explicitly closed-world benchmark; audit/filter it before semantic-transfer claims.

## 3. Closest Dataset Precedents

**SPARTQA, documented:** [Mirzaee et al., NAACL 2021, full paper](https://aclanthology.org/2021.naacl-main.364.pdf), sections 3-6, explicitly adopts an **open-world assumption**: unmentioned information is not false. Its Yes/No/DK generation checks a relation, its incompatible reverse, or neither. It includes rule-generated questions, candidate object/block selection, multirelation answers, scene-graph annotations, and contrast/consistency tests. Supervised QA training/further pretraining improves BERT's overall human-set accuracy from 30.17% to 47.25%, but does not uniformly improve YN performance. This is classifier training, not symbolic generative-CoT SFT. Geometry includes extended objects, containment and distance, not exclusively exact eight-direction points.

**SPARTUN, documented:** [Mirzaee and Kordjamshidi, EMNLP 2022, full paper](https://aclanthology.org/2022.emnlp-main.413.pdf), section 4, already uses predicate notation such as `ABOVE(a,b)`, Prolog/Horn-clause reasoning, selected reasoning paths, **supporting-fact annotations**, and irrelevant extra facts. It expands directional, topological, and qualitative-distance coverage and trains BERT classifiers with synthetic supervision. Section 3.1 uses binary YN and multilabel FR; that binary interface alone does not establish closed-world semantics. Table 7 reports ReSQ accuracy 57.37% to 63.60% after SPARTUN supervision; this is not a token-efficiency or Natural/Symbolic comparison.

**Inference:** neither "synthetic spatial data," "logical verification," "evidence," nor "open-world spatial QA" is independently supportable as novel here. A narrower contribution may be the particular rigor and controlled contract comparisons of SpatialEntail; this bounded review cannot establish priority.

## 4. Contradiction Recognition

**Documented distinctions:** SPARTQA already recognizes contradicted *queries* through its No versus DK rule. CoS's appendix includes a spatial query answered No because the opposite direction follows. SPARTUN includes a `Not` rule and removes incompatible predicted relations during postprocessing. SpatialEval section 4.1 tests mismatched images that may conflict with text.

**Limit:** none of these inspected evaluations establishes a dedicated benchmark for detecting globally unsatisfiable textual premise sets. Query refutation, output-consistency filtering, and cross-modal conflict are not equivalent to premise-contradiction detection. Absence in these sources is not a novelty proof.

**Our design decision:** include contradictory premise sets in both training and evaluation. A shared `INCONSISTENT` status precedes all three answer contracts; its evidence is a verified conflicting premise subset, with no claim of minimality or minimum size. This adds premise-consistency recognition and checkable conflict evidence to the dataset's positioning, distinct from query refutation. Report inconsistency detection and conflict-evidence validity separately from performance on consistent Unique/Listed/Exhaustive examples. This is a planned design contribution, not a prior-work claim or established novelty.

**Further user decisions:** consistency applies to the whole map, including contradictions disconnected from the query. Include matched consistent-cycle controls so that cycle presence alone cannot predict `INCONSISTENT`. Keep each original/modified problem family within one split. A conflicting subset is checked together with the declared spatial axioms, including distinct-point assumptions; identify any such axiom used by its explanation.

## 5. Recommended Three-Week Scope

**Recommendations, not literature claims:**

1. Lead with the **solver-verified dataset**: controlled difficulty, evidence, and matched **Unique/Listed/Exhaustive** contracts. Split semantic instances before rendering; publish verification checks and failure cases.
2. Define distinct 2D points and exact sign-based directions: `NE(a,b) := x_a>x_b AND y_a>y_b`; `N(a,b) := x_a=x_b AND y_a>y_b`; `E(a,b) := x_a>x_b AND y_a=y_b`. Derive the other five symmetrically. NE decomposes into **axis inequalities**, not exact `N(a,b) AND E(a,b)`, which is inconsistent. Prefer `NE(a,b)` with an explicit legend and inequality evidence; never encode NE as displacement `(1,1)`.
3. Keep relations open-world while bounding explicit candidate domains. Specify each contract's quantifiers and selection domain. A singleton listed answer is not necessarily globally unique; exhaustiveness concerns the declared full candidate domain, not unstated relations becoming false.
4. After successful analysis, check premise satisfiability before applying any contract. Unsatisfiable facts return shared `INCONSISTENT`; analysis errors must not be classified as inconsistency. Verify that conflict evidence references actual premises and that the cited subset is itself unsatisfiable; do not require or advertise a minimum-size subset. For consistent facts `F`, entailment requires `F AND NOT q` unsatisfiable; refutation requires `F AND q` unsatisfiable; ambiguity requires satisfying witnesses for both. This separates inconsistent premises from vacuous entailment. Use checked evidence, not only generator-authored chains.
5. Keep matched **Natural vs Symbolic SFT** supporting: identical semantic instances, evidence content, backbone, training exposure, and canonical value outputs, not option letters. Report actual tokenizer counts, truncation, accuracy by difficulty/contract, and evidence validity. Do not conflate representation with additional solver assistance or claim CoS's prompting savings as SFT savings.

Primary papers were read beyond abstracts, including relevant appendices; GitHub inspection used `gh`. No full-release SpatialMap audit or training reproduction was performed.
