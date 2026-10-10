# Capability-framework literature review

**Checked 2026-10-10.** Primary-source critique of the
[task taxonomy](task-taxonomy.md), using the
[SODA naming note](soda-task-taxonomy.md), [source ledger](annotated-sources.md),
and existing report bibliography as leads, not substitutes for papers.
Recommendations only: no implemented algorithm, executed transfer result,
allocation change, or novelty claim. Proceedings locators below give printed
page, then PDF page in parentheses.

Terminology note: this review evaluates the earlier `MR` shorthand. The final
working proposal uses **MTR (Model-Theoretic Reasoning)** because SODA already
uses MR for Mental Rotation; the critique of overlap remains applicable.

## 1. Verdict: scope-bound grouping, not a standard taxonomy

**RR/LR/MTR is defensible as a local experimental organization** for finite named
objects, fixed-frame qualitative 2D compass relations, ground Boolean formulas,
and Direction/Which/Count queries. None of the papers below establishes these
three labels as a standardized, exhaustive, disjoint taxonomy or an empirical
hardness hierarchy.

| Group | Useful organizing question | Important overlap |
|---|---|---|
| Relational reasoning (RR) | Which spatial consequences follow by inversion, axis composition, and transitivity? | These are logical inferences too; equality and exact/coarse atom interpretation govern their validity. |
| Logical reasoning (LR) | How do negation, implication, equivalence, and case scope control inference? | Boolean deductions can require spatial closure and global candidate exclusion. |
| Model-theoretic reasoning (MTR) | Which query claims are necessary, possible, or impossible across satisfying worlds? | This is also a semantic perspective on RR/LR, not a separate substance or a mandatory map-building operation. |

In particular, EXC can be solved by refutation without constructing maps; NEG
and EXC overlap. CSR combines branches, while AMB supplies contrasting possible
answers; the same formula can need both. QAG is determined-membership aggregation,
whereas JCR preserves joint constraints: `(East(A,R) AND West(B,R)) OR
(West(A,R) AND East(B,R))` leaves each membership contingent but fixes the count
at one. Uncertainty of an individual claim does not imply uncertainty of the
complete answer. These are scope/semantic distinctions, not evidence for three
independent cognitive modules.

Use a frozen primary-label policy plus secondary rule/evidence tags. Operator
presence or one chosen proof does not establish capability necessity; review
alternate supports and predeclared semantic interventions. MLC needs enumerated
compounds, not a miscellaneous “hard” bucket. Separate query family, proof depth,
formula nesting, branching, distractor/dependency structure, context/token length,
answer-domain uncertainty, consistency, and representation from capability labels.

**Missing dimensions matter in two ways.** Within this scope, expose equality,
exact versus coarse direction meanings, parsing/scope errors, branch dependencies,
and necessity versus possibility versus complete-answer uniqueness. Beyond it,
reference-frame changes/rotation, topology/containment, extended objects, metric
geometry, navigation/planning, time/state updates, and quantification are genuine
scope extensions—not coverage supplied by RR/LR/MTR names. SpaRC makes object
extent, relation completeness, and quantitative specification explicit
(§3.1, pp. 4752–4753 / PDF 3–4) [5]; SODA adds transformations and multi-turn
control (§3.2.1) [1]. Neither should be imported wholesale into a static
qualitative-entailment task.

**Naming:** SODA's **MR means Mental Rotation**, not model-based reasoning [1].
Spell out the proposed group in comparisons; avoid an unexplained bare MR column.

## 2. SODA: what OODA actually does and what was evaluated

The official ACL paper distinguishes three mechanisms [1]:

- **Training-free prompt orchestration:** a question extractor supplies a
  four-stage blueprint; a self-instructing prompt generator creates stage-specific
  prompts; LLMs execute Observe → Orient → Decide → Act sequentially. The authors
  acknowledge latency from multiple agent calls (§3.1, p. 29976 / PDF 3). This is
  an explicit inference-time prompting workflow, not a model-architecture change.
- **Training:** SPOD-143k includes OODA traces, then Qwen3 models receive SFT and
  GRPO (§3.3, p. 29978 / PDF 5). Seed/template expansion uses scripts and tools;
  strong LLMs generate traces conditioned on questions **and known answers**.
  Batches receive sampled double-blind manual checks, including stage order and
  reasoning/answer consistency (§3.2.2–3.2.3, pp. 29977–29978 / PDF 4–5).
  Answer-conditioned rationales and sampled review are not independently replayed
  proofs. GRPO rewards answer matching and OODA format/order; the description
  mentions logical jumps but supplies no formal step-verification calculus.
- **Actual environment feedback:** robotic control encodes current state and
  rules, queries the LLM for an action, executes it, then re-encodes the new state
  (Appendix D.0.2, pp. 29994–29995 / PDF 21–22). That is a genuine external closed
  loop. A static QA response's “Act” can simply output an option (Figure 2), so
  four headings do not guarantee iterative environmental interaction.

**Evaluation:** Table 1 tests training-free OODA prompting on five tasks; Table 2
reports native accuracy across SPOD-Bench's 13 tasks in Basic Single-Turn,
Complex Single-Turn, and Multi-Turn tiers; Table 3 adds SPACE transfer; Table 5
compares SFT with SFT+GRPO. The tables show benefits but also task-level regressions
and cross-tier reversals: neither uniform improvement nor a model-independent
difficulty ordering is warranted. Chess evaluates legal-move continuation and
completion, not simply wins against Stockfish (Appendix C). Robotic evidence is
limited; the main text calls it real-arm validation, whereas D.0.4 describes
simulated experiments and differs in setup details. Do not inflate it into broad
real-world transfer.

**Teachable lesson:** staged extraction/orientation and explicit state updates are
useful procedural precedents. However, outcome accuracy, stage-format reward, and
stage token counts do not prove that the trained model executes a faithful hidden
OODA algorithm. The cited experiments do not isolate OODA from all effects of
additional task data, trace supervision, and RL using a fully matched no-OODA
training control. This limits causal attribution, not the reported task results.

## 3. Adjacent logic and spatial-trace evidence

### ProofWriter: explicit proofs and open-world unknown

ProofWriter supplies synthetic English facts/rules, hypothesis labels, and
fact/rule proof DAGs; separate tasks generate implications and restricted
single-fact abductions (§3.1, p. 3623 / PDF 3) [2]. Its all-at-once and iterative
one-step approaches teach different inference procedures; the iterative system
repeatedly derives implications and assembles proofs (§3.6). Evaluation examines
answers and proof correctness, including deeper inference (§4), rather than only
fluent explanations.

**Its contract is Datalog-style supported-fact reasoning**, not unrestricted
classical FOL. CWA uses True/False and negation-as-failure; OWA includes explicit
negative facts/conclusions and True/False/Unknown (§3.2). Iterative answering finds
the query (True), its negation (False), or neither (OWA Unknown; §3.6.2,
p. 3625 / PDF 5). **Unknown is not contradictory premises.** Appendix A.2
(p. 3631 / PDF 11) says OWA theories were regenerated to be stratifiable to avoid
negation cycles; stratifiability alone is not a general satisfiability guarantee.
The paper defines no fourth inconsistent label. Do not silently reinterpret its
rules as allowing all classical contraposition or conflate simultaneous support
with Unknown.

### FOLIO: richer quantified scope

FOLIO's native reasoning task predicts True/False/Unknown; NL-to-FOL translation
is separate (§4.1–4.2, p. 22022 / PDF 6) [3]. Its FOL annotations include negation,
conjunction, disjunction, implication, universal/existential quantification, and
equality; modal/temporal logic is excluded (Appendix E.2, p. 22029 / PDF 13).
Necessary intrinsic/commonsense axioms are made explicit and annotations receive
syntax/label checks (§3.4–3.5, p. 22021 / PDF 5). This is stronger scope than the
current **ground propositional spatial fragment**, not merely another compass
benchmark. Failure can reflect quantifier scope or NL formalization, not spatial
LR weakness. Dataset verification does not certify a free-running model's
reasoning or establish arbitrary premise consistency.

### PrOntoQA: controlled synthetic deduction

PrOntoQA constructs an ontology and proof before translating them into a context,
query, CoT, and label (§3, PDF 3–4) [4]. Its linear ontology and repeated
modus-ponens fragment allow fine-grained proof-step analysis (§4, PDF 4–5;
Appendix A.1, Figure 6, PDF 13),
including valid steps that do not advance the target proof. This supports testing
useful proof progress separately from local validity and final correctness.
Native queries are **True/False**, not T/F/Unknown: false examples query the
negation of a proved conclusion, rather than treating failed derivation as falsity.
Synthetic ontology/proof controls and distractors are useful for teachability,
but they do not establish general Boolean, quantified FOL, uncertainty, or spatial
model-construction competence. These locators refer to the final author version
v4 (2 March 2023).

### SpaRP and Chain-of-Symbol: teachable representation, not hidden cognition

SpaRP obtains annotated/regex-extracted relations, selects graph traversal paths,
composes links with property-specific spatial reasoners, and verbalizes them
link-by-link (§4, pp. 4754–4755 / PDF 5–6) [5]. Fine-tuning and analysis evaluate
answers and generated-path errors (§§5–6). This is direct precedent for
solver-generated spatial trace supervision; inherited contexts and task-specific
composition assumptions are not a general proof-first, all-model certificate
system.

Chain-of-Symbol generates CoT demonstrations, manually corrects them, substitutes
compact spatial symbols, and few-shot prompts models (§3, PDF 4–5; §4.1,
PDF 5–6; Appendix A.2.2, PDF 14) [6]. Its results motivate representation and
length controls, not a guarantee that symbolic SFT will win or that demonstrations
are independently certified. Neither spatial paths nor symbols establish the
hidden causal computation of an LLM.

## 4. Candidate teachable policy: defensible, with conditions

Keep three claims distinct: **dataset taxonomy** groups evaluation targets;
**data-generation/verification** checks supervision artifacts; **inference-time
procedure** selects and produces evidence. Hidden cognition is a fourth, stronger
claim that output traces alone cannot establish.

The proposed **interpret premises → derive/refute relevant claims → construct
witnesses for unresolved cases/consistency → decide under the explicit query
contract** is a reasonable evidence policy, not yet an implemented general
algorithm or a mandatory four-paragraph trace:

1. **Interpret:** fix objects, atom semantics, Boolean/branch scope, candidate
   domain, and answer mode. Do not flatten OR into facts, activate unsupported
   antecedents, or turn NOT East into West.
2. **Derive/refute:** use relevant scoped dependencies. Exclude a candidate by
   assuming it and deriving contradiction; a case consequence must survive every
   live case. A missing proof or failed bounded search proves neither falsity nor
   ambiguity. Solver decidability exceeds the automatic certificate constructor's
   calculus; typed contraposition/De Morgan conversions are currently absent.
3. **Witness as needed:** a satisfying world establishes consistency/possibility;
   two worlds with different **complete query answers** establish non-uniqueness.
   One witness never establishes necessity; sampled worlds never exhaust the
   answer domain. Proof and witness construction may alternate. Global maps or
   rank tables are a representation choice, not mandatory on every task; retaining
   independent validity checks is essential when comparing local-proof targets.
4. **Decide:** preserve native Direction/Which/Count projections and existing
   SINGLE/ALL_POSSIBLE contracts. Count requires joint membership constraints;
   Which depends on whether a singular object or a complete set was requested.
   Ambiguity, impossible candidates, inconsistent premises, and incomplete menus
   are different states. Current generation rejects inconsistent premises; this
   review proposes no new inconsistency answer class.

For clarity, the **classical model-theoretic diagnostic**, conditional on the
project's declared encoding, is:

| SAT(P ∧ q) | SAT(P ∧ ¬q) | Interpretation |
|---|---|---|
| yes | no | entailed / True |
| no | yes | refuted / False |
| yes | yes | contingent / Unknown |
| no | no | inconsistent P, not Unknown |

This is an alignment explanation, **not a replacement for ProofWriter's supported
closure semantics or native spatial outputs**. False means explicit negative
entailment, not failure to prove q. Inconsistent classical premises entail both
sides vacuously; check consistency before making non-vacuous necessity claims.

**Z3 boundary:** use it for generation, semantic labels, witness validation, and
independent audit/scoring without leaking answers or evidence into test prompts.
That does not give the tested LLM a solver. A model-plus-Z3 test-time oracle is a
separately labelled tool-assisted system. Likewise, verified gold traces do not
make arbitrary model traces verified; current Symbolic outputs have replay
support, whereas arbitrary Natural outputs lack a process checker. See the local
[semantic contract](../spatialentail/semantics.md) and
[certificate limits](../spatialentail/certificates.md).

Test teachability with matched answer-only/Natural/Symbolic controls, legal step
and scope use, candidate coverage, primitive retention, and held-out logical-spatial
compounds, ambiguity, and correlated counts—not mnemonic compliance alone.
Current Boolean/depth/axis-generation combinations and the proposed local-proof
comparison still require feasibility/coverage checks; no learning benefit is
established by naming the policy.

## 5. Secondary validation recommendation

**One optional nonspatial check is useful: ProofWriter OWA**, retaining its native
labels, inference semantics, released split, and a frozen prompt/scoring protocol.
It can test rule chaining, explicit negative evidence, and unknown recognition
outside spatial vocabulary. It cannot validate every Boolean family, geometry,
JCR, or the spatial witness policy. FOLIO is a broader language/quantifier transfer
stress test, not necessary additional scope for this study.

Keep native logical accuracy separate from native spatial scores; no blanket
cross-dataset direction score. A predeclared compatible/tagged subset or derived
atom-level T/F/Unknown probe must be labelled **adapted diagnostic**, with inclusion
rules and counts reported, never presented as the original benchmark. Preserve
existing entailment and menu outputs. Do not select subsets using final model
failures or pool incompatible tasks into one capability macro-score.

## Primary references

1. **Shunwen Bai et al. (2026).** *One Cognitive Loop Is Enough: SODA unlocks Pure-Text
   Spatial Reasoning in Large Language Models*. ACL, pp. 29974–29996.
   [Official record](https://aclanthology.org/2026.acl-long.1382/),
   [PDF](https://aclanthology.org/2026.acl-long.1382.pdf).
   DOI **10.18653/v1/2026.acl-long.1382**. Existing key `bai2026soda`.
2. **Oyvind Tafjord, Bhavana Dalvi, and Peter Clark (2021).** *ProofWriter:
   Generating Implications, Proofs, and Abductive Statements over Natural
   Language*. Findings of ACL-IJCNLP, pp. 3621–3634.
   [Official record](https://aclanthology.org/2021.findings-acl.317/),
   [PDF](https://aclanthology.org/2021.findings-acl.317.pdf).
   DOI **10.18653/v1/2021.findings-acl.317**. Existing key `tafjord2021proofwriter`.
3. **Simeng Han et al. (2024).** *FOLIO: Natural Language Reasoning with
   First-Order Logic*. EMNLP, pp. 22017–22031.
   [Official record](https://aclanthology.org/2024.emnlp-main.1229/),
   [PDF](https://aclanthology.org/2024.emnlp-main.1229.pdf).
   DOI **10.18653/v1/2024.emnlp-main.1229**. No bibliography edit made.
4. **Abulhair Saparov and He He (2023).** *Language Models Are Greedy Reasoners:
   A Systematic Formal Analysis of Chain-of-Thought*. ICLR.
   [Official record](https://openreview.net/forum?id=qFVVBzXxR2V),
   [author version v4](https://arxiv.org/abs/2210.01240v4),
   [PDF](https://arxiv.org/pdf/2210.01240v4).
   Existing key `saparov2023prontoqa`. OpenReview required a browser check;
   claims were checked against the final author version, not a different paper
   or a later PrOntoQA-derived benchmark.
5. **Md Imbesat Rizvi, Xiaodan Zhu, and Iryna Gurevych (2024).** *SpaRC and SpaRP:
   Spatial Reasoning Characterization and Path Generation for Understanding
   Spatial Reasoning Capability of Large Language Models*. ACL, pp. 4750–4767.
   [Official record](https://aclanthology.org/2024.acl-long.261/),
   [PDF](https://aclanthology.org/2024.acl-long.261.pdf).
   DOI **10.18653/v1/2024.acl-long.261**. Existing key `rizvi2024sparp`.
6. **Hanxu Hu, Hongyuan Lu, Huajian Zhang, Yun-Ze Song, Wai Lam, and Yue Zhang
   (2024).** *Chain-of-Symbol Prompting for Spatial Reasoning in Large Language
   Models*. COLM. [Official record](https://openreview.net/forum?id=Hvq9RtSoHG),
   [author PDF, arXiv v7](https://arxiv.org/pdf/2305.10276v7).
   Existing key `hu2024chainofsymbol`. OpenReview required a browser check;
   locators were verified in the COLM-labelled v7 PDF. Its PDF title differs
   from the arXiv landing-page title; the PDF title is used here.
