# Spatial-Map Contract Alignment

Checked: 21 October 2026. Bounded follow-up to
[the answer-generation review](../../research/spatialmap-answer-generation-review.md).
Primary-source check: paper v2 Sections 2.1/2.2 and Appendix C.4, using two
fetches; no repository/generator rescan. Numerical evidence below was supplied
by the main agent, not independently rerun in this paper-only check.

## Intended Task Versus Audit Contract

Section 2.1 describes a map with a configurable number of uniquely named
objects, represented textually by pairwise relations. It says TQA "contains
all necessary information for a person to answer the questions" and calls
the vision/text information redundant. Section 2.2 evaluates four-option
questions by accuracy. Appendix C.4 supplies complete illustrative
descriptions, not formal semantic or generation rules. These support intended
text answerability for concrete-map MCQs, but do not explicitly prescribe
open-world reasoning, relation CWA, or all-model entailment. [P1-P3]

The following distinctions are analytical, not author-defined terminology:

- **Closed entity domain:** count the objects in the described map, not
  arbitrarily many unnamed objects. This matches the finite-map setup; closing
  the object inventory does not close the relation facts. [P1]
- **Relation closed-world assumption (CWA):** treating an unstated relation as
  false is a separate assumption. The checked passages do not instruct this;
  absence of a pairwise statement alone is not evidence of its negation. [P1, P3]
- **Concrete scene:** the intended map has actual positions and an actual
  count. A gold answer can be correct for that scene even when qualitative
  text admits other scenes. Map-based construction supports this interpretation,
  but does not establish the private label-generation procedure. [P1]
- **All-model entailment:** our stronger text-only audit asks whether the
  answer is invariant across every scene satisfying the premises under its
  declared qualitative encoding. The paper's sufficiency claim motivates
  checking answerability, but does not itself specify that exact encoding or
  require count uniqueness independently of the answer menu. [P1, P2]
- **One correct listed option:** the four-choice format reasonably supports
  assuming the intended answer is among the options and exactly one option
  answers the question in the intended scene. This is not an explicit formal
  axiom in the checked passages, nor does it make all other listed counts
  impossible across every text-consistent scene. [P2, P3]

## Options Can Constrain the Intended Answer

The Count **object domain** remains the map inventory; numeric options are
candidate totals, not objects to count. Nevertheless, under the offered-options
contract they can constrain which scene/count is intended. Let `P` be the
textual premises, `S` the counts possible under the audit semantics, and `O`
the displayed numeric values. If the intended scene satisfies `P` and its
count must belong to `O`, then a singleton `S intersect O` suffices to identify
the answer even when `S` contains other, unlisted counts. This is entailment
conditional on the menu assumption, not full-domain count entailment.

Thus an unlisted alternative count alone does **not** demonstrate an MCQ
defect. Stronger underdetermination requires at least two distinct listed
counts with valid witnesses satisfying every premise and the applicable scene
constraints. For Count, each such scene still has exactly one correct listed
numeric option; the ambiguity is between scenes, so the single-correct-option
assumption does not remove it. These are logical consequences of the
distinguished contracts, not disclosed author verification methods.

## Main-Agent Evidence

The main agent reports one official gold per each of the 500 Count rows, with
no multi-gold rows. Its previous full-domain audit classified 195 as unique
and 305 as ambiguous. Projecting possible counts onto the four displayed
values gives 282 rows with one compatible listed count, 181 with two, 35 with
three, and 2 with four. Therefore 218 have multiple compatible listed counts;
87 of the 305 full-domain-ambiguous rows have only one compatible listed count.
These figures concern the local audit semantics, not author-reported results.

Reported example: `spatialmap.tqa.2003.2`, count southeast of Nightingale;
options `A=0, B=1, C=2, D=5`, official gold `B=1`, possible listed counts
`0` and `1`. The main agent supplied these witnesses:

| Object | Count-0 Scene | Count-1 Scene |
|---|---|---|
| Andy | `(2, 4)` | `(2, 4)` |
| Nightingale | `(1, 0)` | `(1, 1)` |
| Recycle | `(0, 5)` | `(0, 5)` |
| Sally | `(3, 2)` | `(3, 2)` |
| Trail | `(2, 1)` | `(2, 0)` |
| Unicorn | `(0, 3)` | `(0, 3)` |

The main agent independently validated both saved witnesses against all nine
released premises and recomputed their counts as 0 and 1. Under the declared
directional semantics, this example survives the offered-options objection;
the aggregate 218 should likewise be presented as local semantic findings,
not established generator failures. The input is
`fyp-report/data/spatialeval_org.jsonl`; the prior validated audit records are
at `/tmp/spatialmap-research-audit-20261021/spatialeval_audit.jsonl`.

## Reporting Limit

The concern about task mismatch is justified for full-domain ambiguity with
only one compatible displayed answer. Qualify the original criticism
accordingly. Multiple witnessed displayed answers provide stronger evidence
against text-only MCQ sufficiency under the stated semantics, but still do not
prove the gold wrong for the intended map or identify a private generator bug.
Missing formal definitions alone are insufficient evidence of a defect.

## Primary Sources

- **P1:** [Paper v2, Section 2.1](https://arxiv.org/html/2406.14852v2#S2.SS1).
- **P2:** [Paper v2, Section 2.2, Evaluation](https://arxiv.org/html/2406.14852v2#S2.SS2).
- **P3:** [Paper v2, Appendix C.4](https://arxiv.org/html/2406.14852v2#A3.SS4).
  The appendix text was checked directly; Figure 14's pixels were not
  re-fetched in this follow-up.
