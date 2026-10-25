# Spatial-Map Answer Generation and Answer Contract

Checked: 21 October 2026. Scope: Wang et al. (2024), arXiv:2406.14852,
SpatialEval's Spatial-Map task, official paper/appendix, released dataset, and
`jiayuww/SpatialEval` code. This note extends the existing
[modality and release review](literature-landscape.md#spatialeval-modality-alignment-facts-and-inference-boundary)
and [generation-provenance note](../archive/research/generation-provenance-validity.md#spatialeval-likely-map-first-but-not-fully-auditable);
it does not repeat their full cross-modality audit.

## Bottom Line

- **Official answer format:** four-choice questions with one released gold
  answer, not a multi-select/set-valued gold contract. The three oracle columns
  are representations of that answer, not three alternative answers. [P2, R1, D1]
- **Which:** choose one of four offered object names. Those names can be a
  subset of the map's non-reference objects. A unique offered answer is not
  necessarily a unique qualifying object across the whole map. [P3, D2]
- **Count:** count qualifying map objects, then select a numeric answer. The
  numeric menu is not a list of objects to count and need not contain every
  possible count. This reading follows the task wording and illustrations,
  not a published generator implementation. [P1, P3, D3]
- **Generation correctness:** the paper claims text sufficiency, but the
  checked primary sources do not publish an algorithm or verification step
  establishing that each gold answer follows uniquely from the emitted text.
  Neither “they check only one option” nor “they enumerate all entailed answers”
  is a disclosed author method. [P1, P3, R1, R2]
- **Separate local result:** under the explicitly declared qualitative,
  text-only `SINGLE` semantics below, the fresh 1,500-row audit finds 667 exact
  answers and 833 underdetermined but gold-compatible answers. This is our
  audit result, not an author-reported property or proof of the private
  generator's algorithm. [L1-L4]
  This full-domain contract is stronger than assuming that the intended
  answer must be one of the offered choices. The counts must not be reported
  as an unconditional number of erroneous SpatialEval labels.

## What the Authors Actually Disclose

| Source | Exact disclosure | What it establishes |
|---|---|---|
| Introduction | “each problem in our benchmark has an image and a text representation that is sufficient for answering each spatial understanding question.” | Intended answerability from either representation; not a formal entailment certificate. [P0] |
| Section 2.1 | TQA “contains all necessary information for a person to answer the questions”; VTQA renders “the information in both modalities redundant.” | An explicit text-sufficiency claim, stronger than merely providing some clues. [P1] |
| Section 2.1, Spatial-Map | “we create a map-like dataset ... with K objects, where K is configurable”; unique location names; textual representation “consists of pairwise relations”. | Map-based construction and relation-based textual representation. [P1] |
| Section 2.1, Spatial-Grid comparison | “Spatial-Map where objects are positioned arbitrarily.” | Spatial-Map is not presented as a rigid grid or unit-displacement task. [P1] |
| Section 2.2, Evaluation | “As each question contains four options, we use accuracy as the main evaluation metric.” | Four-choice evaluation. The accompanying prompt asks for a concise answer followed by reasoning. [P2] |
| Appendix C.4 | “Three questions (termed Q1 to Q3) are associated with each sample”; Figure 14 supplies complete descriptions. | Direction, Which, and Count illustrations, not generation pseudocode. [P3] |
| README, generation section | “Stay tuned! The dataset generation script will be released in Feburary” [sic]. | An announced script, not published construction code at the checked revision. [R1] |

The construction section, appendix, and checked official release do **not**
specify coordinate sampling, pairwise-edge selection, query selection,
distractor selection, whether labels are computed from coordinates or text,
all-model uniqueness testing, rejection of text-ambiguous items, or an
enumeration of all qualifying/entailed answers. The current complete official
repository tree is non-truncated and contains inference/evaluation code but
no Spatial-Map generator; its only branch is `main`. The linked dataset
release contains generated Parquet files and documentation, not generator
source. These are bounded observations about the checked release locations,
not claims that the authors lack a private algorithm. [P1, P3, R1, R2, D1]

**Map-first versus text-first:** map/scene-based construction is explicitly
described. It is reasonable to infer a map-first workflow, but it remains
unverified whether the gold was read from a privileged complete map or
re-solved from the textual premises. Sharing a gold across modalities is
consistent with either implementation. Do not turn that inference into a
claim that labels were generated from coordinates without checking text. [P1;
existing local provenance review linked above]

## Single Gold, Candidate Subsets, and Semantic Answer Sets

### Three Different Meanings of “Subset”

1. **Offered object subset:** Which's A-D menu contains four object names drawn
   from the larger map. This is directly visible in the official examples.
2. **Multi-select response subset:** an answer such as `{B, D}`, selecting
   several letters simultaneously. This is not the released gold format or
   the official evaluator's set-valued contract.
3. **Semantic possible/entailed subset:** the set of answers compatible with
   some text-consistent world, or true in every such world. These sets are
   products of our local analysis; the official release does not supply them.

The official dataset schema makes `oracle_answer`, `oracle_option`, and
`oracle_full_answer` strings. For example, row `spatialmap.tqa.2000.1` stores
`Narwhal's Novelties`, `B`, and `B. Narwhal's Novelties`. The official inference
code copies these fields from the dataset; it does not generate or verify the
gold. The official extraction branches return a direction, one object-name
string, or one number, and scoring consults `oracle_answer`, not an answer
set. [D1, D2, R3, R4]

**Scoring caveat:** the README calls evaluation “exact match”, but the checked
implementation actually uses lowercased substring containment
(`ref_ans.lower() in model_answer.lower()`) after heuristic extraction.
Consequently, it should not be described as strict string equality. This
implementation detail neither proves gold correctness nor constitutes an
entailment check. [R1, R4]

### Which and Count Have Different Universes

| Family | What is selected/countable | What the four options mean | Evidence boundary |
|---|---|---|---|
| Direction | Direction of the queried pair; released examples offer the four diagonal directions. | Candidate direction labels. | Official examples and evaluator branches. [P3, D2, R4] |
| Which | An offered object satisfying the queried relation. The map can have additional unoffered objects. | Four candidate object names, not all qualifying map objects. | Directly observed single-choice menu. No published rule proves global uniqueness or explains distractor sampling. [P3, D2] |
| Count | Objects in the described map satisfying the relation, excluding the reference itself under a strict directional predicate. | Four candidate numeric totals. | Wording/illustration supports map-wide counting. Exclusion follows the strict qualitative reading, not an explicit author counting-loop disclosure. [P1, P3, D3] |

Appendix Figure 14 has six named locations. Its Which question offers four
objects and asks which is southwest of Gale Gifts, highlighting only
`B. Unicorn Umbrellas`. Its Count question asks how many objects are southwest
of Himalayan Hot Springs, with numeric choices `3, 4, 1, 0`, highlighting `3`.
These are plainly different domains: four offered object candidates for
Which versus a numeric answer to a map-wide count. [P3]

The official release provides an especially clean scope example:
`spatialmap.tqa.2002.1` asks which object is southwest of Iris's Ice Skates.
Both Dolphin's Dresses and Art Supplies are **explicitly** stated southwest
of Iris in the text, but only Art Supplies is offered; gold is `A. Art
Supplies`. Thus the single gold cannot mean “the only qualifying object in the
entire map.” This example does not require a solver or an assumption about
the hidden map. [D2, row index 502]

Row `spatialmap.tqa.2000.1` similarly has six map objects but offers Coral,
Narwhal, Planetarium, and Police; Oz Oddities is unoffered. Its omission alone
does not establish that Oz is entailed southwest of Ice Queen. [D2]

## Fresh Local Audit, Not Author Claims

### Assumptions and Provenance

The main-agent run was completed on 21 October 2026; this research pass read
its summary and the two example audit records. The main agent also reported
17 passing audit/grading tests and independently checked both coordinate
completions below against all nine parsed premises.

- Input: `fyp-report/data/spatialeval_org.jsonl`, 1,500 rows;
  SHA-256 `53f4cff9d3baef1a1495fbf028f2a5ec80b85de2afb2ed7031f721f271982766`.
- Backend/contract: Z3, `answer_mode=single`; no use of the image or hidden map.
- Every stated diagonal relation is a conjunction of strict x/y comparisons;
  north is increasing y. All emitted positive premises are required jointly.
- Coordinates are unbounded integers and objects have distinct positions
  (not necessarily distinct individual x or y values). No fixed grid,
  equal diagonal step lengths, or unstated closed-world negations are added.
- Direction answers range over the four ordinal directions in this adapter.
  Which ranges over its four offered objects. Count ranges over all five
  non-reference map objects and checks totals `0..5` before menu encoding.
- `SINGLE` requires a unique possible Direction/Count value. For Which it
  requires one entailed object and no other possible candidate; one merely
  possible object is insufficient if some worlds have no offered match.

These are **our operational semantics**, implemented in the local adapter,
text parser, solver, and grading code, not a formal semantics specified by
Wang et al. [L2-L4]

Reproduction command, from `fyp-report` (use a new output directory if these
artifacts already exist; the original input remains unchanged):

```bash
uv --system-certs run --python 3.12 --no-project --with typer --with z3-solver \
  python -m spatial.v2.audit_spatialeval \
  --input data/spatialeval_org.jsonl \
  --output-dir /tmp/spatialmap-research-audit-20261021
```

The command is a reproduction recipe, not a claim to have captured the
main agent's exact shell history. The audit validates each saved witness
against the premises and requested answer. [L2]

### Results

| Query | Exact Match | Underdetermined, Gold Possible | Total |
|---|---:|---:|---:|
| Direction | 332 | 168 | 500 |
| Which | 140 | 360 | 500 |
| Count | 195 | 305 | 500 |
| Total | 667 (44.47%) | 833 (55.53%) | 1,500 |

No other audit statuses occur: every gold is compatible with at least one
text-consistent world; none is contradicted. “Underdetermined” therefore does
not mean impossible gold or incorrect relative to an intended complete map.
It means the text does not establish the required `SINGLE` answer under the
declared contract. [L1]

The main agent's fresh structural scan also reports four options and one gold
letter in every row; gold frequencies are `A=376, B=378, C=369, D=377`.
All rows have six map objects; the 500 Which parses have four offered
candidates and the 500 Count parses have five non-reference candidates.
These are local scan observations, not author-reported sampling rules.

**Do not equate 833 with 833 rows having multiple possible displayed letters.**
The summary reports possible-value cardinalities `1:778, 2:463, 3:140, 4:119`.
In particular, some Which rows have one possible object but no entailed
object: the queried region can contain no offered candidate in another world.
Possible counts also include values omitted from the menu. [L1, L4]

### Reproducible Which Counterexample

Official row `spatialmap.tqa.2000.1` asks which object is southwest of Ice Queen
Ice Cream; gold is Narwhal (`B`). Both completions below satisfy every one of
its nine textual premises. [D2; locally verified completions in L5]

| Object | Completion 1 | Completion 2 |
|---|---:|---:|
| Police Supply Store | `(0, 0)` | `(0, 0)` |
| Narwhal's Novelties | `(-1, 1)` | `(-1, 4)` |
| Coral Crafts | `(-3, 4)` | `(-3, 5)` |
| Planetarium Prints | `(3, 2)` | `(3, 1)` |
| Oz Oddities | `(-2, -2)` | `(-2, -2)` |
| Ice Queen Ice Cream | `(-0.5, 3)` | `(1, 2)` |

Completion 1 has Narwhal as the unique **offered** southwest object;
Completion 2 has Police (`D`). Multiply every coordinate by two to obtain
integer witnesses for the local encoding. All x and y values are distinct in
these particular witnesses, so imposing distinct axes would not remove this
counterexample. Oz qualifies in these two completions but is not offered;
its qualification in witnesses is not an entailment claim. The fresh audit
independently reports possible offered objects `{Narwhal, Police}` and no
entailed offered object. [L1, L5]

### Sole Compatible Displayed Count Is Not Full-Domain Uniqueness

Official row `spatialmap.tqa.2002.2` asks how many objects are southeast of
Mordor Supplies. Menu: `A=0, B=4, C=2, D=5`; gold: `C=2`. [D3, row index 1002]

The fresh local audit finds full-domain possible counts `{2, 3}`. Hence only
one displayed option is compatible, but the count is not uniquely determined:
`3` is possible and absent from the menu. The saved witnesses differ in
Mordor's and Narwhal's y coordinates:

| Object | Count-2 World | Count-3 World |
|---|---:|---:|
| Andy's Autos | `(4, 2)` | `(4, 2)` |
| Art Supplies | `(0, 1)` | `(0, 1)` |
| Dolphin's Dresses | `(0, 0)` | `(0, 0)` |
| Iris's Ice Skates | `(2, 3)` | `(2, 3)` |
| Mordor Supplies | `(1, 4)` | `(1, 5)` |
| Narwhal's Novels | `(3, 5)` | `(3, 4)` |

Both satisfy the nine premises; Narwhal's Novels additionally lies southeast
of Mordor in the second world. Thus restricting verification to displayed
numeric choices would miss the alternative count. This demonstrates a
semantic/menu distinction, **not** that the authors' unpublished generator
performed such restricted verification. [L1-L3]

### Follow-Up: Options Can Be Part of the Task Contract

If a multiple-choice question implicitly guarantees that the correct count is
listed, the solver may condition on that guarantee. With premises `P`, count
function `C`, and listed counts `M`, this means solving
`P AND (C in M)` rather than just `P`. In the preceding `{2, 3}` example,
`M = {0, 4, 2, 5}` removes count 3 and leaves a unique offered answer, 2.
That example alone therefore does not establish a defect in the authors'
multiple-choice task. It establishes a difference from our stronger
full-domain count-entailment contract.

A follow-up projection of all 500 count audit records onto their listed
numeric options gives:

| Compatible Listed Counts | Questions |
|---|---:|
| 1 | 282 |
| 2 | 181 |
| 3 | 35 |
| 4 | 2 |

Every count row has exactly one official gold letter. There are no released
multi-gold count labels. However, 218 rows permit at least two listed counts
under our qualitative semantics. Of the 305 full-domain ambiguous count
rows, 87 have only one compatible listed count; these should be separated
from displayed-option ambiguity.

For example, `spatialmap.tqa.2003.2` asks how many objects are southeast of
Nightingale Novelties and offers `A=0, B=1, C=2, D=5`, with gold `B=1`.
The audit permits both 0 and 1, which are both listed. The main agent
independently rechecked the following witnesses against all nine premises
and recomputed the count:

| Object | Count-0 World | Count-1 World |
|---|---:|---:|
| Andy's Autos | `(2, 4)` | `(2, 4)` |
| Nightingale Novelties | `(1, 0)` | `(1, 1)` |
| Recycle Center | `(0, 5)` | `(0, 5)` |
| Sally's Salon | `(3, 2)` | `(3, 2)` |
| Trail Hiking Gear | `(2, 1)` | `(2, 0)` |
| Unicorn Umbrellas | `(0, 3)` | `(0, 3)` |

These are alternative maps, not two simultaneously true exact counts in one
map. Both worlds also satisfy a guarantee that exactly one listed count is
correct. This is stronger evidence of text-only underdetermination, though
still conditional on the spatial interpretation and absence of additional
documented map constraints. Source: the same input and audit artifacts in
L1-L3; row 1003 of the local input.

The paper does not explicitly declare open-world or closed-world relational
semantics. A finite, fully listed object domain does not imply that an
unmentioned relation is false, and labeling a concrete scene is not itself a
closed-world inference rule. Our all-model relation analysis should be
identified as our operational contract, not attributed to the authors.
See the follow-up [contract alignment note](../archive/research/spatialmap-contract-alignment.md).

## Safe Report Wording

> SpatialEval's Spatial-Map release uses four-choice questions with one gold
> label. Which selects among offered object names, whereas Count asks for a
> total over the map's non-reference objects. The authors claim text-only
> sufficiency, but the checked publication and official release do not disclose
> a procedure verifying unique entailment from the emitted text. Under our
> declared qualitative text-only SINGLE contract, a fresh audit finds 667
> exact answers and 833 underdetermined but gold-compatible answers. These
> findings concern released-text sufficiency, not the correctness of labels
> relative to an intended complete map or the private generator's algorithm.
> The full-domain SINGLE contract is stronger than a menu-conditioned
> multiple-choice contract. Count results should additionally distinguish
> 218 rows with multiple compatible listed counts from 87 rows whose only
> alternative counts are unlisted.

## Sources

### Official Primary Sources

- **P0:** [Paper v2, Introduction](https://arxiv.org/html/2406.14852v2#S1).
- **P1:** [Paper v2, Section 2.1](https://arxiv.org/html/2406.14852v2#S2.SS1).
  Also checked [v1, Section 3.1](https://arxiv.org/html/2406.14852v1#S3.SS1);
  use v2 section numbering in this note.
- **P2:** [Paper v2, Section 2.2](https://arxiv.org/html/2406.14852v2#S2.SS2).
- **P3:** [Appendix C.4](https://arxiv.org/html/2406.14852v2#A3.SS4),
  [Figure 14](https://arxiv.org/html/2406.14852v2#A3.F14), and
  [full-resolution figure](https://arxiv.org/html/2406.14852v2/1-map-detailed.png).
  The figure was visually inspected, not inferred from its caption.
- **R1:** [Official README](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L57-L60),
  [evaluation description](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/README.md#L123-L140).
- **R2:** [Official complete tree](https://github.com/jiayuww/SpatialEval/tree/d82ba382805265f205693cb83a30c4866ea6c35b),
  checked using `gh api 'repos/jiayuww/SpatialEval/git/trees/main?recursive=1'`
  and `gh api repos/jiayuww/SpatialEval/branches`.
  Current `main` resolved to `d82ba382805265f205693cb83a30c4866ea6c35b`.
- **R3:** [Official inference, oracle copying](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/inference_lm.py#L79-L85).
- **R4:** [Official scalar answer extraction](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/evals/evaluation.py#L26-L136),
  [actual scoring](https://github.com/jiayuww/SpatialEval/blob/d82ba382805265f205693cb83a30c4866ea6c35b/evals/evaluation.py#L272-L290).
- **D1:** [Official dataset revision/card](https://huggingface.co/datasets/MilaWang/SpatialEval/tree/59ce0450bf65ae8dfa3590062442cf642474d7fe),
  [metadata API](https://huggingface.co/api/datasets/MilaWang/SpatialEval).
  Current revision resolved to `59ce0450bf65ae8dfa3590062442cf642474d7fe`.
- **D2:** [Official Which rows 500-502](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=500&length=3),
  [TQA viewer](https://huggingface.co/datasets/MilaWang/SpatialEval/viewer/tqa/test).
- **D3:** [Official Count rows 1000-1002](https://datasets-server.huggingface.co/rows?dataset=MilaWang%2FSpatialEval&config=tqa&split=test&offset=1000&length=3).
  Dataset-server links are live; revision identification above records the
  release checked.

### Local Analysis Sources

- **L1:** [Fresh audit summary](file:///tmp/spatialmap-research-audit-20261021/spatialeval_audit_summary.json)
  and [audit records](file:///tmp/spatialmap-research-audit-20261021/spatialeval_audit.jsonl).
  Temporary artifacts are referenced, not copied into this note.
- **L2:** [Audit implementation](../../spatial/v2/audit_spatialeval.py)
  and [classification adapter](../../spatial/v2/spatialeval_adapter.py#L157-L194).
- **L3:** [Candidate-domain parsing](../../spatial/v2/text.py#L312-L360)
  and [Z3 coordinate/Which/Count checks](../../spatial/v2/solver.py#L950-L1100).
- **L4:** [Answer resolution](../../spatial/v2/grading.py#L252-L293).
- **L5:** [Existing world-assumption audit and coordinate completions](../archive/audits/spatialeval-world-assumption.md#row-evidence-the-released-tqa-text-can-underdetermine-its-oracle).

Local links and test reports are evidence for our analysis only. They must not
be cited as disclosures by the SpatialEval authors.
