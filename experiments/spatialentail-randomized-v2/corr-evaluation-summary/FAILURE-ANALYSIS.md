# Why SFT Scores Below Untuned on Corr

## Bottom Line

The evidence supports a combination of answer-policy distribution shift,
incorrect generated deductions, and question/menu grounding failures.
Stage 2 is mostly preserving incorrect adapter answers, not destroying
otherwise correct ones. This is negative transfer under the frozen two-stage
protocol, not proof of a single causal mechanism such as catastrophic forgetting.

## 1. The Training Menu Did Not Teach This Abstention Task

The candidate training archive was checked against the per-file SHA-256 hashes
in `research-analysis/provenance.json`. Each of the three training files contains
8,000 examples. Both seeds of an arm use the corresponding training preparation.
This verifies candidate contents, not actual cluster consumption.

| Training file | Rows | Targets containing letter E | User questions containing "Cannot be determined" |
|---|---:|---:|---:|
| SINGLE Natural | 8,000 | 1,652 | 0 |
| SINGLE answer-only | 8,000 | 1,652 | 0 |
| MULTI Natural | 8,000 | 4,069 | 0 |

**The model did see E. It did not see E with this abstention meaning in these
training menus.** For example, one Count training menu ends `D. 0, E. 1`;
another Direction menu ends `D. East, E. Southwest`.

SINGLE training has G=1 throughout: all 8,000 rows have `semantic_shape=unique`
and `resolution_status=exact`. MULTI has 2,000 unique SINGLE examples and 6,000
ambiguous ALL-POSSIBLE examples, with 2,000 examples each at G=1,2,3,4.
Its ambiguity training asks for sets of possible substantive answers rather
than the Corr policy of selecting E for underdetermination. Correct behavior
on the former does not guarantee transfer to the latter.

Corr instead has 833/1,500 gold-E examples (55.53%). This makes an untrained
answer-policy transition central to the external score, not a rare edge case.
The internal G1 ceiling did not test this abstention behavior.

The system prompt does explain the SINGLE policy, so this is not absence of
instructions at evaluation. The question is whether the adapted model follows
that policy on unfamiliar problems.

## 2. Where the Score Deficit Comes From

These are net correct-answer deficits relative to untuned, pairing the same
source IDs. A negative deficit means the adapter gains correct answers.

| Adapter | Deficit on 667 determinate | Deficit on 833 underdetermined | Underdetermined share of total deficit |
|---|---:|---:|---:|
| SINGLE Natural 42 | 46 | 199 | 81.22% |
| SINGLE Natural 43 | 111 | 228 | 67.26% |
| SINGLE answer-only 42 | 190 | 265 | 58.24% |
| SINGLE answer-only 43 | -18 | 249 | 107.79% |
| MULTI Natural 42 | 102 | 369 | 78.34% |
| MULTI Natural 43 | 141 | 354 | 71.52% |

MULTI E recall is only 5.28%/7.08%, versus 49.58% untuned. It predicts E on
59/126 of all 1,500 questions, versus 465 for untuned. This is not simply a
general inability to emit letter E: E is common in its training targets.

The determinate deficits also matter. MULTI accuracy there is 68.22%/62.37%
versus 83.51% untuned. Fixing abstention alone would not remove every deficit.
SINGLE answer-only seed 43 is a useful exception: it gains 18 determinate
answers but loses 249 underdetermined answers.

## 3. Concrete Reasoning and Grounding Failures

Six illustrative source IDs were selected deterministically: the lexicographically
first ID in each query x status cell where untuned is correct and both MULTI
seeds are wrong. The examples below inspect seed 42. They illustrate mechanisms;
they are deliberately failure-selected and do not estimate their prevalence.
All are in the original archive's
`results/<model>/responses_spatialentail_randomized_v2_corr_single.jsonl`.

### Invalid transitivity and an unsupported uniqueness claim

ID `spatialmap.tqa.2004.0`, Direction, gold E:

The adapter derives:

> Mantis Maternity is east of Raven's Records.
> Raven's Records is west of Arctic Apparel.
> By X-axis transitivity ... Mantis Maternity is west of Arctic Apparel.

That deduction is invalid: two objects east of the same reference are not
ordered relative to each other. The model then claims a unique Southwest
answer and selects A. Its stated axis combination is also inconsistent with
that direction label. Untuned instead constructs different possible placements
and selects E.

This is proof-shaped text, not a verified proof. Correct solver-rendered
training explanations do not guarantee valid model-generated deductions on
new examples.

### Missing a direct inverse relation

ID `spatialmap.tqa.2021.0`, Direction, gold C:

The prompt explicitly says Stingray Shoes is Northwest of Marina Market.
The question asks for Marina Market relative to Stingray Shoes, so Southeast
follows directly. Untuned identifies this. The adapter overlooks that relation,
uses two Southwest relations to a shared reference as if transitive, and
answers Southwest (D).

### Answering the wrong question type and mapping the wrong value

ID `spatialmap.tqa.2029.1`, Selection, gold B:

The question asks which offered object is Southeast of Falcon's Florist.
The adapter instead analyzes the direction of Lighthouse Luminary, concludes
"Southeast," and selects C. But C names Riverbank Relaxation, not a direction.
Untuned selects the offered object in B using the inverse of a stated relation.

This is question/menu grounding failure, not just failure to output E.

### Claiming a count that contradicts its own description

ID `spatialmap.tqa.2005.2`, Count, gold C (zero):

The question asks for the number of objects Southeast of Zephyr Zucchini.
The adapter's witness places all other objects north of Zephyr, yet it later
marks Factory Finds as an entailed member and answers one (B). Untuned
identifies that all named objects are north and answers zero.

Other selected cases: `spatialmap.tqa.2001.2` (underdetermined Count) and
`spatialmap.tqa.2011.1` (underdetermined Selection). In the latter, the adapter
uses a supporting placement of Coral Crafts to claim a unique answer; untuned
recognizes its east/west position relative to the reference remains unknown.

## 4. Why Stage 2 Does Not Repair These Errors

The same checkpoint generates both stages. Stage 2 sees the original prompt
plus its own stage-1 reasoning and produces a constrained letter; it is not an
independent solver or verifier. See `eval/eval_new.py`, `generate_until`.

| Model | Stage-1 accuracy | Stage-2 accuracy | Wrong to correct | Correct to wrong |
|---|---:|---:|---:|---:|
| Untuned | 2.73% | 64.67% | 929 | 0 |
| MULTI Natural 42 | 31.80% | 33.27% | 23 | 1 |
| MULTI Natural 43 | 30.80% | 31.67% | 15 | 2 |

For MULTI, the poor answer is usually already present before stage 2.
Only five responses per seed have an extracted multi-letter native answer,
four of them on gold-E questions. Therefore, the observed collapse is not
primarily stage 2 arbitrarily selecting one option from thousands of correct
native answer sets.

Untuned's very low native extraction score is misleading as a measure of
reasoning quality. Its reasoning can contain useful correct deductions without
ending in an extractable answer. The inspected examples show stage 2 recovering
appropriate answers from such text. This does not establish that all 929
rescues are mere extraction; stage 2 can itself make a new decision.

Untuned averages about 10,897 stage-1 characters; MULTI averages 4,731/5,257.
These are character counts, not token counts or a causal explanation. Untuned
closes `</think>` in only 4.80% of responses; finish reasons are unavailable.
It often repeats reasoning, so longer output is not automatically better.

## 5. Interpretation and Next Steps

The strongest working hypothesis is specialization to the generated training
distribution, with weak transfer of answer policy and semantic grounding.
Training uses logical-rule scaffolds and controlled proof-style explanations;
Corr uses shop names, direct narrative relations, different menus, and, for
Count, no explicit `Considering only [...]` candidate list.

The untuned checkpoint retains a different response strategy that works better
on this external task under stage 2. There is no evidence here proving benchmark
pretraining exposure, an optimizer defect, or a specific forgetting mechanism.
Training logs and checkpoint hashes remain needed.

Recommended order:

1. Report this negative transfer alongside the internal gains; do not replace
   the original scores or redefine Corr after observing failures.
2. Use a separate diagnostic set to compare E-abstention and all-possible
   contracts on matched content, and narrative versus structured rendering.
   This distinguishes policy transfer from reasoning/grounding failures.
3. Audit generated premises, axis deductions, and menu mapping on a balanced
   sample with a predefined rubric, including successes as well as failures.
4. Only then consider a new training arm containing ambiguous SINGLE examples,
   validated E targets, and varied narrative forms. Keep Corr out of its
   training and tuning data. This is a new experiment, not a retroactive repair
   of an incomplete original SFT.

No model reruns, training changes, or scoring changes were made for this analysis.
