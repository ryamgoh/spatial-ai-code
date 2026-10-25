# External relational-capability review

Primary-source review, 2026-10-10. This compares native benchmark obligations with
`task-taxonomy.md`; it is not a generated-task catalogue or evidence of
capability necessity in individual examples. Text2Space and StepGame capabilities
requiring fresh row audits are **not verified**. SpatialMap observations below are
explicitly a lexical/answer-distribution scan, not a fresh solver audit.

Terminology note: this review used the proposal's earlier `MR` shorthand. The
consolidated proposal now uses **MTR (Model-Theoretic Reasoning)** to avoid
collision with SODA's Mental Rotation task; the underlying alignment judgments
are unchanged.

## Evidence and alignment definitions

- **D — Direct:** the native task-level obligation aligns with the family. This
  does not establish universal row necessity, a minimal proof, or identical
  semantics to SpatialEntail.
- **P — Partial:** a related operation exists, but semantics, output contract, or
  support control differs. It is not validated coverage of the proposed family.
- **A — Absent:** the examined native source contract supplies no validated
  obligation for this family. This is not a claim that the model or an extended
  generator could never perform it.

The comparison must distinguish exact compass relations (a cardinal direction
also fixes equality on the orthogonal axis), projected/coarse comparisons (only
one axis matters), and fixed-unit offsets (quantitative constraints whose final
output may nevertheless be qualitative). These are not interchangeable. Nor are
overlap, an uncertain layout, and an uncertain complete query answer equivalent.

## 1. Original StepGame and its 2024 correction

### Native contract and semantics

Shi, Zhang and Lipani's AAAI 2022 StepGame takes a natural-language story and a
question asking the relative position of two named entities. The basic story
has `k` binary relations joining `k+1` entities in a chain, with shuffled
sentences. Eight directions generate premises; the answer has nine labels,
including **overlap**. The queried pair can be separated by fewer than `k`
edges: `k` is a maximum/story-chain setting, not guaranteed minimal query-proof
depth [SG1, §3.2].

The released generator adds offsets `(0,±1)`, `(±1,0)`, and `(±1,±1)`, then
classifies the queried displacement by X/Y signs and equality tests. Its output
spellings are `above`, `below`, `left`, `right`, `upper-left`, `upper-right`,
`lower-left`, `lower-right`, and `overlap` [SG2, `generate_one_story`]. Li, Hogg
and Cohn's corrected reasoner likewise adds fixed offsets in a **2D grid** and
returns the direction of the resulting displacement [SG3, Methods; SG4,
`asp_solution.py`, `location` and `stepgame` program strings]. Thus the operational
contract is unit-grid displacement composition with qualitative output, not
unrestricted qualitative entailment. Diagonal offsets change each axis by one;
they are not Euclidean unit length. Opposing components can cancel under this
prior when unrestricted positive distances would not.

Overlap means coincident coordinates of different named entities, not unknown
or topological region overlap. There is no native Which, Count, Boolean yes/no,
unknown/abstention, or complete ALL_POSSIBLE answer contract [SG1, §3.2]. Premises
are positive binary relation descriptions. English clock-face templates starting
with “If” map directly to a relation; they do not validate implication deduction
[SG4, `sentence_to_relation.py`]. ToT's “sure/likely/impossible” evaluates graph
search extensions, not global impossibility of spatial answer candidates [SG3,
Methods, Tree-of-Thoughts].

### Versions, splits and noise

The original paper trains jointly on **clean k=1–5**, with 10,000 examples per k
(50,000 total), validates on 1,000 per k (5,000), and tests on noisy k=1–10 with
10,000 per k (100,000). k=6–10 is the larger-depth generalisation setting. Clean
test results are also discussed separately [SG1, §§5.1, 5.3–5.4]. Noise comprises
irrelevant branching, disconnected chains, and supporting alternative paths;
all are intended to preserve the answer. Supporting noise is not necessarily
irrelevant to a derivation [SG1, §3.3].

Keep the paper protocol separate from the current original repository's
`CompleteVersion`: its README claims 30,000 train, 1,000 validation and 30,000
test examples **per k=1–10 per clean/noise condition** [SG2, README, Version].
Current generator/source inspection is not a frozen 2022 release audit.

The 2024 correction identifies **14 erroneous templates among 214 examined**;
labels had followed intended relations rather than necessarily the sentences'
meanings [SG3, Problems with the Dataset, Table 1]. Its repository describes
filtering out whole examples containing incorrect-template sentences.
`data/correct_clean` and `data/correct_noise` each list
`qa{1…10}_{train,valid,test}.txt` [SG4]. The author-deposited Leeds README §4
claims 30,000 training, 1,000 validation and 10,000 test examples per file, but
§5 also describes filtering. The paper's Table 2 reports substantial erroneous
fractions in the 10,000-example source tests: clean 7.64% at k=1 to 54.29% at
k=10; noisy 20.43% to 74.21% [SG3; SG5]. **Exact retained corrected split counts
have not been verified; these metadata counts must not be represented as
freshly counted retained sizes.** Current `correct.py` also comments out writing
and train/validation paths despite README instructions; execution was not tested.

The correction paper evaluates clean filtered subsets of 30, 100 and 1,000
examples per k and several few-shot regimes, including k-matched training
examples. These are evaluation settings, not a replacement 50K-training protocol
[SG3, Experimental Design]. The original repository separately announces
February 2024 template fixes/regeneration and a March Hugging Face recommendation.
Equivalence to Li et al.'s filtered release is unverified; republications are not
independent fresh test evidence [SG2, README news].

## 2. Text2Space

Text2Space generates connected directional graphs with eight relations, 2–8
entities and 1–12 relations, rejects constraint conflicts, and selects queried
pairs for unique inferability. It converts descriptions to/from ASCII maps and
rendered images and evaluates direction QA with text-only or ASCII-only input,
as well as answer/map orderings [T1, §§3.1–3.2, 4.2; Appendix F, Algorithm 1].
`Desc + gold ASCII` is explicitly an oracle upper bound, not equivalent to
text-only reasoning.

Its schema contains **full, vertical and horizontal** queries. Full queries use
eight compass labels; vertical answers are above/below/same level, horizontal
answers left/right/same column. Appendix H projects a vertical comparison onto
Y and a horizontal comparison onto X, ignoring the other axis. Appendix J.3
calls full labels mutually exclusive and pure directions aligned; projected
queries are weaker than exact compass relations [T1, Appendix A, Table 5;
Appendix H, Algorithm 3; J.3]. `VectorToDir` is not fully defined in the examined
algorithm, so every full-query edge case is not resolved by public pseudocode.

**Fixed-unit graph semantics are unverified.** §3.1 and Algorithm 1 do not provide
an edge displacement equation. J.1's neighboring-cell/diagonal rendering
instructions cannot be elevated to universal premise distances; its example
itself contains unequal diagonal spans. Stacking labels such as `AF` represents
coincident map glyphs, but overlap is not a native relation/query label in Table
5. The paper explicitly excludes continuous geometry and 3D environments in
its Limitations [T1, J.1; Limitations].

The reported conversion-training selection is **4,000 train / 500 validation /
1,000 test from 20,000 generated examples**. Selection mechanics and
scene/isomorphism-disjoint deduplication are not established by the examined
sources. Test metadata reports 53.2% directly stated queries and 42.4% unique
layouts [T1, §4.4; Appendix C, Tables 7–8]. Nonunique layouts do **not** establish
AMB coverage: queried pairs are admitted for unique answers. Algorithm 2 checks
map relation agreement and entity inventory, not exact gold-string identity or
all-model entailment [T1, Appendix G]. Conversion-only SFT and text-only transfer
results must be distinguished from gold-map QA [T1, §5.3, Table 4].

No native Which, Count, unknown, complete-answer enumeration, or validated
Boolean-premise evaluation is reported in the examined contract. Multiple
positive declarative clauses and implementation control-flow operators do not
establish logical-premise families. No author-owned release URL was located in
the examined abstract/HTML; this is a search limit, not proof none exists. No
Text2Space row audit was performed.

## 3. SpatialEval SpatialMap TQA

SpatialEval defines TQA as text plus question, VQA as image plus question, and
VTQA as both. The paper says the same questions are evaluated across modalities
and claims each representation supplies the necessary information. SpatialMap
objects are positioned arbitrarily and text describes pairwise relations [SM1,
§3 and Figure 1]. These statements do not define absence-as-negation or a formal
open-/closed-world semantics. The checked repository still advertises a future
generator release, so its coordinate sampling and label-generation inequalities
are not established [SM3, README lines 138–140].

The native SpatialMap release includes Direction, single-choice Which, and
Count questions. Single-choice Which is not SpatialEntail's complete candidate
set, and a map-oracle Count is not automatically determined-membership
aggregation from the supplied premises. The pinned dataset card/tree supports
release structure; TQA has 4,635 rows across the whole SpatialEval suite, not
4,635 SpatialMap rows [SM2].

A **current, non-revision-pinned datasets-server rows-API scan on 2026-10-10**
retrieved 1,500 unique SpatialMap TQA IDs, with 500 in each question family and
500 distinct map-ID middle components. It observed **262 cardinal Count
questions, all with oracle zero** [SM4]. This is a lexical and oracle-distribution
observation, not an inequality-policy proof, semantic consistency audit, or
capability-necessity check. It is not attributable to the separately pinned
`59ce045…` revision. It cautions against treating cardinal Count as a coarse
half-plane count and exposes a possible cardinal→zero answer shortcut within
this subset; neither inference establishes the generator's formal semantics.
No native validated Boolean or uncertainty obligation is established by the
source contract; inspected lexical patterns alone cannot prove universal syntax
absence.

The pinned inference script passes `item['text']` to the model; oracle fields
are retained for scoring/results, not fed as the prompt [SM3,
`inference_lm.py`, lines 34–85]. Native labels should therefore be described as
released map-oracle answers, not automatically as necessary consequences of
all text-satisfying worlds.

## 4. Thirteen-family matrix

| Family | StepGame, including correction | Text2Space | SpatialMap TQA |
|---|---|---|---|
| ARC | D | D | D |
| TRC | D | D | P |
| AXC | P | P | P |
| QAG | A | A | P |
| IMP | A | A | A |
| EQV | A | A | A |
| NEG | A | A | A |
| DJE | A | A | A |
| CSR | A | A | A |
| MLC | A | A | A |
| EXC | A | A | A |
| AMB | A | A | A |
| JCR | A | A | A |

StepGame explicitly separates one-hop extraction/converse and multi-hop
composition [SG3, Spatial Reasoning Types]; Text2Space admits stated and inferred
queries [T1, §3.1; Appendix F]. These justify D without universal row necessity.
SpatialMap's relation recovery is native, but controlled, necessarily transitive
support is not established. All AXC entries remain P: two-axis outputs or map
arithmetic are not controlled **separately supported X/Y composition**. SpatialMap
QAG is P because Which/Count exist with different answer and oracle obligations.
EXC requires nontrivial global negative evidence, not classification alternatives
or search dead ends; AMB requires contrasting complete answers, not overlap or
nonunique layouts. No native correlated possible-count task establishes JCR.

SpatialMap-TQA-Corr is **our derived correction**, not an independent external
source. Its unknown-answer contract can target AMB, and its Direction/Which/Count
contracts can host the proposal's families, but logical operators, candidate
refutations and joint-count obligations require separately validated generated
examples. Neither corrected labels nor a chosen proof certify family necessity.
Prior project audit totals are not repeated here as primary-paper findings.

## 5. Evaluation boundaries and source limits

For StepGame, provide story and query only, not generator coordinates, solved
ASP layouts or answers. Declare the fixed-unit adapter prior, separate overlap
from SpatialEntail's noncoincident domain, report hop-conditioned clean/noise
results, and do not call one-hop overlap with training primitives independent
length extrapolation. For Text2Space, separate text-only, generated-map and
gold-map conditions: a supplied gold map can fix unspecified relations even
when the paper intends the chosen query to be unique. No leakage accusation or
clean split audit is established here. Republished releases may duplicate tests.

Primary evidence does not verify fresh StepGame/Text2Space capability audits,
retained corrected counts, controlled AXC necessity, external Boolean/MTR
coverage, or universal agreement with SpatialEntail semantics. Access limits:
the correction-repo URL without the username hyphen and arXiv correction `v3`
returned 404; the correct repository and paper v1 were accessible. No code tests
were run for this documentation-only review.

## Primary references

- **SG1:** [StepGame original paper](https://arxiv.org/html/2204.08292v1), §§3.1–3.3,
  5.1, 5.3–5.4; [arXiv record](https://arxiv.org/abs/2204.08292).
- **SG2:** [Official repository/README](https://github.com/ShiZhengyan/StepGame),
  News and Version; [generator](https://github.com/ShiZhengyan/StepGame/blob/main/Code/parameterized_step_game_8relation.py).
- **SG3:** [2024 correction paper](https://arxiv.org/html/2401.03991v1), The StepGame
  Benchmark, Methods, Experimental Design, Tables 1–2;
  [arXiv record](https://arxiv.org/abs/2401.03991).
- **SG4:** [Correction repository](https://github.com/Fangjun-Li/SpatialLM-StepGame):
  README; `correct.py`, `sentence_to_relation.py`, `asp_solution.py`;
  [`data/correct_clean`](https://github.com/Fangjun-Li/SpatialLM-StepGame/tree/main/data/correct_clean)
  and [`data/correct_noise`](https://github.com/Fangjun-Li/SpatialLM-StepGame/tree/main/data/correct_noise).
- **SG5:** [Author-deposited corrected dataset](https://doi.org/10.5518/1468);
  [README §§4–5](https://archive.researchdata.leeds.ac.uk/1321/1/README_Fangjun-etal_2024.txt).
- **T1:** [Text2Space v1](https://arxiv.org/html/2604.14641v1):
  [§3.1](https://arxiv.org/html/2604.14641v1#S3.SS1), §4.2, §4.4, §5.3;
  Appendices [A](https://arxiv.org/html/2604.14641v1#A1),
  [C](https://arxiv.org/html/2604.14641v1#A3),
  [F](https://arxiv.org/html/2604.14641v1#A6),
  [G](https://arxiv.org/html/2604.14641v1#A7),
  [H](https://arxiv.org/html/2604.14641v1#A8),
  [J.1](https://arxiv.org/html/2604.14641v1#A10.SS1),
  [J.3](https://arxiv.org/html/2604.14641v1#A10.SS3).
  PDF pages: §3.1 p3; §§4.2–4.4 pp4–5; Tables 7–8 pp12–13;
  Algorithms 1–3 pp16–17; J.1/J.3 pp20–21.
- **SM1:** [SpatialEval paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/89cc5e613d34f90de90c21e996e60b30-Paper-Conference.pdf),
  §3 and Figure 1.
- **SM2:** [Official pinned dataset card](https://huggingface.co/datasets/MilaWang/SpatialEval/blob/59ce0450bf65ae8dfa3590062442cf642474d7fe/README.md)
  and [release tree](https://huggingface.co/datasets/MilaWang/SpatialEval/tree/59ce0450bf65ae8dfa3590062442cf642474d7fe).
- **SM3:** [Pinned official repository](https://github.com/jiayuww/SpatialEval/tree/d82ba382805265f205693cb83a30c4866ea6c35b),
  README lines 138–140; `inference_lm.py` lines 34–85.
- **SM4:** [Current TQA rows API](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=0&length=100),
  paginated by increasing `offset`, filtering SpatialMap IDs. Observation date
  2026-10-10; not revision-pinned and not a semantic audit.
