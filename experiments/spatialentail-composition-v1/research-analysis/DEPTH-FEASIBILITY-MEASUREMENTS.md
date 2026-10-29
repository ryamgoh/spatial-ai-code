# Depth Feasibility: Measured Results

Exploratory bounded engineering pilot. Not a training dataset, final benchmark, completed release audit, or model evaluation.

## Admission

| Measure | Count |
| --- | --- |
| Attempted stories | 94 |
| Attempted query bases | 282 |
| accepted | 247 |
| token-rejected | 35 |
| Admitted views per style | 861 |
| Admitted stories | 94 |
| Admitted source families | 45 |
| Admitted canonical story groups | 94 |

An admission failure quarantines the entire query base: all menu, contract and style views. Other queries on that story may remain. No rejection is replaced with another random draw.

## Controlled Depth

Each cell below starts with six constructed bases. Before-token counts require evidence verification, actual requested depth, and the applicable answer-dependence check. These are not six independent families per depth: source families group deliberately related depth variants.

| Depth | Direction Before/After | Selection Before/After | Count Before/After |
| --- | --- | --- | --- |
| 1 | 6 / 6 | 6 / 6 | 6 / 6 |
| 2 | 6 / 6 | 6 / 6 | 6 / 6 |
| 3 | 6 / 6 | 6 / 6 | 6 / 6 |
| 4 | 6 / 6 | 6 / 6 | 6 / 6 |
| 5 | 6 / 1 | 6 / 6 | 6 / 6 |
| 6 | 6 / 0 | 6 / 6 | 6 / 6 |
| 7 | 6 / 0 | 6 / 6 | 6 / 6 |
| 8 | 6 / 0 | 6 / 6 | 6 / 6 |
| 9 | 6 / 0 | 6 / 6 | 6 / 6 |
| 10 | 6 / 0 | 6 / 6 | 6 / 6 |

### Admitted Bases by Provisional Role

| Role / Query / Requested Depth | Bases |
| --- | --- |
| dev-pilot/count/1 | 2 |
| dev-pilot/count/2 | 2 |
| dev-pilot/count/3 | 2 |
| dev-pilot/count/4 | 2 |
| dev-pilot/count/5 | 2 |
| dev-pilot/count/6 | 2 |
| dev-pilot/count/7 | 2 |
| dev-pilot/count/8 | 2 |
| dev-pilot/direction/1 | 2 |
| dev-pilot/direction/2 | 2 |
| dev-pilot/direction/3 | 2 |
| dev-pilot/direction/4 | 2 |
| dev-pilot/which/1 | 2 |
| dev-pilot/which/2 | 2 |
| dev-pilot/which/3 | 2 |
| dev-pilot/which/4 | 2 |
| dev-pilot/which/5 | 2 |
| dev-pilot/which/6 | 2 |
| dev-pilot/which/7 | 2 |
| dev-pilot/which/8 | 2 |
| extrapolation-pilot/count/10 | 6 |
| extrapolation-pilot/count/9 | 6 |
| extrapolation-pilot/which/10 | 6 |
| extrapolation-pilot/which/9 | 6 |
| in-range-test-pilot/count/1 | 2 |
| in-range-test-pilot/count/2 | 2 |
| in-range-test-pilot/count/3 | 2 |
| in-range-test-pilot/count/4 | 2 |
| in-range-test-pilot/count/5 | 2 |
| in-range-test-pilot/count/6 | 2 |
| in-range-test-pilot/count/7 | 2 |
| in-range-test-pilot/count/8 | 2 |
| in-range-test-pilot/direction/1 | 2 |
| in-range-test-pilot/direction/2 | 2 |
| in-range-test-pilot/direction/3 | 2 |
| in-range-test-pilot/direction/4 | 2 |
| in-range-test-pilot/which/1 | 2 |
| in-range-test-pilot/which/2 | 2 |
| in-range-test-pilot/which/3 | 2 |
| in-range-test-pilot/which/4 | 2 |
| in-range-test-pilot/which/5 | 2 |
| in-range-test-pilot/which/6 | 2 |
| in-range-test-pilot/which/7 | 2 |
| in-range-test-pilot/which/8 | 2 |
| train-pilot/count/1 | 2 |
| train-pilot/count/2 | 2 |
| train-pilot/count/3 | 2 |
| train-pilot/count/4 | 2 |
| train-pilot/count/5 | 2 |
| train-pilot/count/6 | 2 |
| train-pilot/count/7 | 2 |
| train-pilot/count/8 | 2 |
| train-pilot/direction/1 | 2 |
| train-pilot/direction/2 | 2 |
| train-pilot/direction/3 | 2 |
| train-pilot/direction/4 | 2 |
| train-pilot/direction/5 | 1 |
| train-pilot/which/1 | 2 |
| train-pilot/which/2 | 2 |
| train-pilot/which/3 | 2 |
| train-pilot/which/4 | 2 |
| train-pilot/which/5 | 2 |
| train-pilot/which/6 | 2 |
| train-pilot/which/7 | 2 |
| train-pilot/which/8 | 2 |

Non-controlled random/targeted cases do not fill the table above, even when a depth descriptor happens to be applicable.

## Gold Mix After Admission

| Contract | Views | E | F | E/F Share |
| --- | --- | --- | --- | --- |
| single | 287 | 115 | 34 | 51.9% |
| all-possible | 287 | 18 | 75 | 32.4% |
| visible-possible | 287 | 18 | 44 | 21.6% |

Rates describe this accepted pilot only. Token rejection can bias the mixture; a lower E/F rate does not certify the sampling policy.

| Query | Contract | Observed Substantive Sizes | Possible-Set Sizes |
| --- | --- | --- | --- |
| direction | single | 1 | 1, 2, 3, 8 |
| direction | all-possible | 1, 2, 3 | 1, 2, 3, 8 |
| direction | visible-possible | 1, 2, 3, 4 | 1, 2, 3, 8 |
| which | single | 1 | 0, 1, 2, 3, 4, 5, 6, 7 |
| which | all-possible | 1, 2, 3, 4 | 0, 1, 2, 3, 4, 5, 6, 7 |
| which | visible-possible | 1, 2, 3, 4 | 0, 1, 2, 3, 4, 5, 6, 7 |
| count | single | 1 | 1, 2, 3, 4, 5, 6, 7 |
| count | all-possible | 1, 2, 3, 4 | 1, 2, 3, 4, 5, 6, 7 |
| count | visible-possible | 1, 2, 3, 4 | 1, 2, 3, 4, 5, 6, 7 |

E/F are escapes, not size-one substantive answers. A count value of four is one answer value. Possible-set size and returned size are different measurements. Inapplicable combinations are not quota failures.

### Missing Literal Strings

This is a coverage diagnostic, not a requirement to force all strings into each query or the determinate extrapolation cohort.

| Role | Contract | Not Observed |
| --- | --- | --- |
| train-pilot | single | none |
| train-pilot | all-possible | none |
| train-pilot | visible-possible | none |
| dev-pilot | single | none |
| dev-pilot | all-possible | AB, BC, BD, CD, BCD, E |
| dev-pilot | visible-possible | AB, BC, BD, BCD, E |
| in-range-test-pilot | single | none |
| in-range-test-pilot | all-possible | AB, AD, BC, ABC, ACD |
| in-range-test-pilot | visible-possible | AB, AD, BC, ABC, ACD |
| extrapolation-pilot | single | none |
| extrapolation-pilot | all-possible | AB, AC, AD, BC, ABD, ACD, BCD, ABCD, E |
| extrapolation-pilot | visible-possible | AB, AC, AD, BC, ABD, ACD, BCD, E |

## Premise-Blind Baseline

For each contract choose its most frequent training-role answer, breaking ties alphabetically. Apply unchanged to other provisional roles. These are exploratory pilot diagnostics, not final benchmark scores.

| Contract | Training-Selected Prediction |
| --- | --- |
| single | E |
| all-possible | F |
| visible-possible | ABCD |

| Role | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| train-pilot | 88 | 306 | 28.8% |
| dev-pilot | 61 | 219 | 27.9% |
| in-range-test-pilot | 64 | 246 | 26.0% |
| extrapolation-pilot | 17 | 90 | 18.9% |

## Token Maxima Before Admission

| Style | Target | Full Training | Stage-2 Reservation |
| --- | --- | --- | --- |
| natural | 9062 | 9690 | 4868 |
| symbolic | 7920 | 8547 | 4867 |

Pinned Qwen3.5-4B tokenizer; Stage-1 cap 4,096, context 8,192, Stage-2 cap 16. No truncation, model loading or GPU inference.

## Structural Scope

| Cohort | Stories | Canonical Groups | Degree Profiles | Object Range |
| --- | --- | --- | --- | --- |
| controlled | 60 | 60 | 50 | 8-22 |
| random/targeted | 34 | 34 | 28 | 5-8 |

Degree profiles and canonical identities are descriptive, not proof of rich structural diversity. The controlled constructor uses one backbone with optional leaf branches and disconnected distractors. It does not yet randomize independent x/y networks, multiple essential deep members, or general query-relevant branching. Gold labels can correlate with these scaffolds; shortcut baselines beyond contract-majority remain needed.

## Rejections

| Story | Query | Status | Reason |
| --- | --- | --- | --- |
| depth-5-sample-0 | direction | token-rejected | natural depth-5-sample-0-direction-m0-all-possible: 4344 supervised tokens |
| depth-5-sample-2 | direction | token-rejected | natural depth-5-sample-2-direction-m0-all-possible: 4142 supervised tokens |
| depth-5-sample-3 | direction | token-rejected | natural depth-5-sample-3-direction-m0-all-possible: 4495 supervised tokens |
| depth-5-sample-4 | direction | token-rejected | natural depth-5-sample-4-direction-m0-all-possible: 4529 supervised tokens |
| depth-5-sample-5 | direction | token-rejected | natural depth-5-sample-5-direction-m0-all-possible: 5154 supervised tokens; symbolic depth-5-sample-5-direction-m0-all-possible: 4512 supervised tokens |
| depth-6-sample-0 | direction | token-rejected | natural depth-6-sample-0-direction-m0-all-possible: 5249 supervised tokens; symbolic depth-6-sample-0-direction-m0-all-possible: 4571 supervised tokens |
| depth-6-sample-1 | direction | token-rejected | natural depth-6-sample-1-direction-m0-all-possible: 5264 supervised tokens; symbolic depth-6-sample-1-direction-m0-all-possible: 4606 supervised tokens |
| depth-6-sample-2 | direction | token-rejected | natural depth-6-sample-2-direction-m0-all-possible: 5008 supervised tokens; symbolic depth-6-sample-2-direction-m0-all-possible: 4390 supervised tokens |
| depth-6-sample-3 | direction | token-rejected | natural depth-6-sample-3-direction-m0-all-possible: 5553 supervised tokens; symbolic depth-6-sample-3-direction-m0-all-possible: 4859 supervised tokens |
| depth-6-sample-4 | direction | token-rejected | natural depth-6-sample-4-direction-m0-all-possible: 5038 supervised tokens; symbolic depth-6-sample-4-direction-m0-all-possible: 4416 supervised tokens |
| depth-6-sample-5 | direction | token-rejected | natural depth-6-sample-5-direction-m0-all-possible: 5316 supervised tokens; symbolic depth-6-sample-5-direction-m0-all-possible: 4670 supervised tokens |
| depth-7-sample-0 | direction | token-rejected | natural depth-7-sample-0-direction-m0-all-possible: 6412 supervised tokens; symbolic depth-7-sample-0-direction-m0-all-possible: 5594 supervised tokens |
| depth-7-sample-1 | direction | token-rejected | natural depth-7-sample-1-direction-m0-all-possible: 6060 supervised tokens; symbolic depth-7-sample-1-direction-m0-all-possible: 5302 supervised tokens |
| depth-7-sample-2 | direction | token-rejected | natural depth-7-sample-2-direction-m0-all-possible: 5961 supervised tokens; symbolic depth-7-sample-2-direction-m0-all-possible: 5203 supervised tokens |
| depth-7-sample-3 | direction | token-rejected | natural depth-7-sample-3-direction-m0-all-possible: 5668 supervised tokens; symbolic depth-7-sample-3-direction-m0-all-possible: 4958 supervised tokens |
| depth-7-sample-4 | direction | token-rejected | natural depth-7-sample-4-direction-m0-all-possible: 5800 supervised tokens; symbolic depth-7-sample-4-direction-m0-all-possible: 5066 supervised tokens |
| depth-7-sample-5 | direction | token-rejected | natural depth-7-sample-5-direction-m0-all-possible: 6451 supervised tokens; symbolic depth-7-sample-5-direction-m0-all-possible: 5633 supervised tokens |
| depth-8-sample-0 | direction | token-rejected | natural depth-8-sample-0-direction-m0-all-possible: 6597 supervised tokens; symbolic depth-8-sample-0-direction-m0-all-possible: 5763 supervised tokens |
| depth-8-sample-1 | direction | token-rejected | natural depth-8-sample-1-direction-m0-all-possible: 6854 supervised tokens; symbolic depth-8-sample-1-direction-m0-all-possible: 5996 supervised tokens |
| depth-8-sample-2 | direction | token-rejected | natural depth-8-sample-2-direction-m0-all-possible: 6629 supervised tokens; symbolic depth-8-sample-2-direction-m0-all-possible: 5795 supervised tokens |
| depth-8-sample-3 | direction | token-rejected | natural depth-8-sample-3-direction-m0-all-possible: 6708 supervised tokens; symbolic depth-8-sample-3-direction-m0-all-possible: 5862 supervised tokens |
| depth-8-sample-4 | direction | token-rejected | natural depth-8-sample-4-direction-m0-all-possible: 6622 supervised tokens; symbolic depth-8-sample-4-direction-m0-all-possible: 5776 supervised tokens |
| depth-8-sample-5 | direction | token-rejected | natural depth-8-sample-5-direction-m0-all-possible: 6634 supervised tokens; symbolic depth-8-sample-5-direction-m0-all-possible: 5788 supervised tokens |
| depth-9-sample-0 | direction | token-rejected | natural depth-9-sample-0-direction-m0-all-possible: 7928 supervised tokens; symbolic depth-9-sample-0-direction-m0-all-possible: 6934 supervised tokens; natural depth-9-sample-0-direction-m0-visible-possible: 4258 supervised tokens |
| depth-9-sample-1 | direction | token-rejected | natural depth-9-sample-1-direction-m0-all-possible: 7663 supervised tokens; symbolic depth-9-sample-1-direction-m0-all-possible: 6697 supervised tokens; natural depth-9-sample-1-direction-m0-visible-possible: 4108 supervised tokens |
| depth-9-sample-2 | direction | token-rejected | natural depth-9-sample-2-direction-m0-all-possible: 8331 supervised tokens; symbolic depth-9-sample-2-direction-m0-all-possible: 7265 supervised tokens; natural depth-9-sample-2-direction-m0-visible-possible: 4484 supervised tokens |
| depth-9-sample-3 | direction | token-rejected | natural depth-9-sample-3-direction-m0-all-possible: 7929 supervised tokens; symbolic depth-9-sample-3-direction-m0-all-possible: 6935 supervised tokens; natural depth-9-sample-3-direction-m0-visible-possible: 4259 supervised tokens |
| depth-9-sample-4 | direction | token-rejected | natural depth-9-sample-4-direction-m0-all-possible: 7695 supervised tokens; symbolic depth-9-sample-4-direction-m0-all-possible: 6709 supervised tokens; natural depth-9-sample-4-direction-m0-visible-possible: 4137 supervised tokens |
| depth-9-sample-5 | direction | token-rejected | natural depth-9-sample-5-direction-m0-all-possible: 7602 supervised tokens; symbolic depth-9-sample-5-direction-m0-all-possible: 6636 supervised tokens |
| depth-10-sample-0 | direction | token-rejected | natural depth-10-sample-0-direction-m0-all-possible: 7413 supervised tokens; symbolic depth-10-sample-0-direction-m0-all-possible: 6475 supervised tokens |
| depth-10-sample-1 | direction | token-rejected | natural depth-10-sample-1-direction-m0-all-possible: 8494 supervised tokens; symbolic depth-10-sample-1-direction-m0-all-possible: 7424 supervised tokens; natural depth-10-sample-1-direction-m0-visible-possible: 4584 supervised tokens |
| depth-10-sample-2 | direction | token-rejected | natural depth-10-sample-2-direction-m0-all-possible: 8498 supervised tokens; symbolic depth-10-sample-2-direction-m0-all-possible: 7428 supervised tokens; natural depth-10-sample-2-direction-m0-visible-possible: 4652 supervised tokens |
| depth-10-sample-3 | direction | token-rejected | natural depth-10-sample-3-direction-m0-all-possible: 8681 supervised tokens; symbolic depth-10-sample-3-direction-m0-all-possible: 7587 supervised tokens; natural depth-10-sample-3-direction-m0-visible-possible: 4651 supervised tokens |
| depth-10-sample-4 | direction | token-rejected | natural depth-10-sample-4-direction-m0-all-possible: 9064 supervised tokens; symbolic depth-10-sample-4-direction-m0-all-possible: 7922 supervised tokens; natural depth-10-sample-4-direction-m0-visible-possible: 4718 supervised tokens; symbolic depth-10-sample-4-direction-m0-visible-possible: 4120 supervised tokens |
| depth-10-sample-5 | direction | token-rejected | natural depth-10-sample-5-direction-m0-all-possible: 8603 supervised tokens; symbolic depth-10-sample-5-direction-m0-all-possible: 7521 supervised tokens; natural depth-10-sample-5-direction-m0-visible-possible: 4624 supervised tokens |

## Reproduction

Run `python -m spatial.v2.composition_feasibility` with the pinned tokenizer and a new output directory using the dependencies recorded in `manifest.json`. The fixed pre-run schedule and every attempted problem are preserved. Run `python -m spatial.v2.composition_feasibility_report --candidate <directory> --output <report.md>` to recompute these tables.

The manifest records generation source and tokenizer hashes; it is not a dataset seal. No final training files or benchmark release were created.
