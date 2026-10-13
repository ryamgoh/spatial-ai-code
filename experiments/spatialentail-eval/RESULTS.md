# SpatialEntail frozen benchmark: working results matrix

This is the living record for model-evaluation results. It deliberately keeps
observations separate from conclusions while the answer-extraction protocol is
being audited.

## Evaluation status

- Snapshot: first complete seed-42 run supplied on 2026-10-13.
- Benchmark: 4,300 frozen bases: CORE 1,300, COMP 1,200, and SHIFT 1,800.
- Evaluation mode: single-stage generation followed by the current strict
  answer extractor.
- Metric: micro accuracy.
- Status: provisional. The untuned result is invalid because the extractor can
  reject a semantically correct final bare letter such as `E` when it is not
  written as `Answer: E`. Tuned-model scores should also be retained as
  provisional until the same extraction audit is complete.

## Main comparison

| Model | Seed | Overall | CORE | COMP | SHIFT | Status |
|---|---:|---:|---:|---:|---:|---|
| Untuned Qwen3.5-4B | — | N/A | N/A | N/A | N/A | Recorded as 0% by the current scorer, but invalid pending fair rescoring |
| Answer-only SFT | 42 | 52.63% | 22.15% | 88.50% | 50.72% | Provisional |
| Verified Natural SFT | 42 | 93.14% | 99.54% | 88.75% | 91.44% | Provisional |

The public benchmark hierarchy is CORE, COMP, and SHIFT. No subgroup is given
an extra headline column.

## Reporting hierarchy

| Top-level group | Size | Within-group diagnostics |
|---|---:|---|
| CORE | 1,300 | 13 capability/semantic cells and Direction/Selection/Count summaries |
| COMP | 1,200 | 12 matched-composition cells and Direction/Selection/Count summaries |
| SHIFT | 1,800 | SHIFT cells, query-family summaries, and optional depth diagnostics |

Cell-level results should be reported for all three groups. SHIFT slice labels
are metadata for interpreting cells, not a separate scoring tier.

## Cell-name legend

| Code | Meaning |
|---|---|
| `DIR` | Direction Identification query |
| `SEL` | Entity Selection query |
| `CNT` | Cardinality Determination query |
| `ARC` | Atomic Relation Comprehension |
| `TRC` | Transitive Relation Composition |
| `AXC` | Axis Composition |
| `MP` | Modus Ponens |
| `IFF` | Equivalence Elimination |
| `NEG` | Negation |
| `DJE` | Disjunctive Elimination |
| `CSR` | Case-Split Reasoning |
| `JCR` | Joint Count Reasoning obligation |
| `UNI` | Unique/invariant answer |
| `ALT` | Alternative answers across valid worlds |
| `MUL` | Multiple certain Selection matches under the singular contract |
| `NOM` | No possible listed match |
| `NOISE` | Solver-verified removable distractor premises |

Codes joined in one cell name describe the required reasoning sequence and
answer condition. For example, `CNT-CSR-JCR-UNI` is a Count problem requiring
case-split and joint-count reasoning with a unique result. A composition such as
`DIR-MP-DJE-TRC-UNI` requires the named operations in the declared chain.

## Query-family matrix by top-level group

| Benchmark group | Answer-only Direction | Answer-only Selection | Answer-only Count | Natural Direction | Natural Selection | Natural Count |
|---|---:|---:|---:|---:|---:|---:|
| CORE | 26.80% | 22.00% | 14.67% | 99.80% | 99.00% | 100.00% |
| COMP | 92.00% | 87.75% | 85.75% | 100.00% | 99.75% | 66.50% |
| SHIFT | 57.00% | 49.50% | 45.67% | 99.83% | 91.67% | 82.83% |

## Full per-cell breakdown

These are the provisional scores from the original single-stage seed-42 run.
The untuned row is omitted because its recorded zero scores are invalid pending
fair rescoring.

### CORE

| Cell | N | Answer-only | Natural |
|---|---:|---:|---:|
| **CORE cumulative** | **1,300** | **22.15%** | **99.54%** |
| `CNT-AXC-UNI` | 100 | 23% | 100% |
| `CNT-CSR-JCR-UNI` | 100 | 21% | 100% |
| `CNT-NEG-UNI` | 100 | 0% | 100% |
| `DIR-ARC-ALT` | 100 | 3% | 99% |
| `DIR-ARC-UNI` | 100 | 0% | 100% |
| `DIR-DJE-UNI` | 100 | 71% | 100% |
| `DIR-MP-UNI` | 100 | 20% | 100% |
| `DIR-TRC-UNI` | 100 | 40% | 100% |
| `SEL-ARC-MUL` | 100 | 2% | 99% |
| `SEL-ARC-NOM` | 100 | 0% | 100% |
| `SEL-CSR-UNI` | 100 | 89% | 100% |
| `SEL-IFF-UNI` | 100 | 14% | 100% |
| `SEL-TRC-UNI-NOISE` | 100 | 5% | 96% |

### COMP

| Cell | N | Answer-only | Natural |
|---|---:|---:|---:|
| **COMP cumulative** | **1,200** | **88.50%** | **88.75%** |
| `CNT-DJE-TRC-UNI` | 100 | 86% | 54% |
| `CNT-IFF-TRC-UNI` | 100 | 84% | 62% |
| `CNT-MP-TRC-UNI` | 100 | 79% | 56% |
| `CNT-TRC-MP-UNI` | 100 | 94% | 94% |
| `DIR-DJE-TRC-UNI` | 100 | 98% | 100% |
| `DIR-IFF-TRC-UNI` | 100 | 89% | 100% |
| `DIR-MP-TRC-UNI` | 100 | 91% | 100% |
| `DIR-TRC-MP-UNI` | 100 | 90% | 100% |
| `SEL-DJE-TRC-UNI` | 100 | 86% | 100% |
| `SEL-IFF-TRC-UNI` | 100 | 91% | 99% |
| `SEL-MP-TRC-UNI` | 100 | 92% | 100% |
| `SEL-TRC-MP-UNI` | 100 | 82% | 100% |

### SHIFT

| Cell | Slice | N | Answer-only | Natural |
|---|---|---:|---:|---:|
| **SHIFT cumulative** | **All SHIFT** | **1,800** | **50.72%** | **91.44%** |
| `CNT-MP-DJE-TRC-UNI` | Held-out composition | 100 | 91% | 72% |
| `CNT-MP-TRC-MP-UNI` | Held-out composition | 100 | 98% | 96% |
| `CNT-TRC-MP-TRC-UNI` | Held-out composition | 100 | 84% | 73% |
| `DIR-MP-DJE-TRC-UNI` | Held-out composition | 100 | 98% | 100% |
| `DIR-MP-TRC-MP-UNI` | Held-out composition | 100 | 90% | 100% |
| `DIR-TRC-MP-TRC-UNI` | Held-out composition | 100 | 95% | 99% |
| `SEL-MP-DJE-TRC-UNI` | Held-out composition | 100 | 98% | 100% |
| `SEL-MP-TRC-MP-UNI` | Held-out composition | 100 | 94% | 100% |
| `SEL-TRC-MP-TRC-UNI` | Held-out composition | 100 | 98% | 100% |
| `count-depth-3` | Depth | 50 | 0% | 100% |
| `count-depth-4` | Depth | 50 | 0% | 98% |
| `count-depth-5` | Depth | 50 | 0% | 92% |
| `count-depth-6` | Depth | 50 | 0% | 76% |
| `count-depth-7` | Depth | 50 | 2% | 80% |
| `count-depth-8` | Depth | 50 | 0% | 66% |
| `direction-depth-3` | Depth | 50 | 22% | 100% |
| `direction-depth-4` | Depth | 50 | 28% | 100% |
| `direction-depth-5` | Depth | 50 | 16% | 100% |
| `direction-depth-6` | Depth | 50 | 18% | 100% |
| `direction-depth-7` | Depth | 50 | 20% | 100% |
| `direction-depth-8` | Depth | 50 | 14% | 100% |
| `which-depth-3` | Depth | 50 | 4% | 96% |
| `which-depth-4` | Depth | 50 | 2% | 76% |
| `which-depth-5` | Depth | 50 | 2% | 90% |
| `which-depth-6` | Depth | 50 | 2% | 86% |
| `which-depth-7` | Depth | 50 | 0% | 82% |
| `which-depth-8` | Depth | 50 | 4% | 70% |

## Findings matrix

| Finding | Evidence in this snapshot | Current interpretation | Confidence / next detail needed |
|---|---|---|---|
| Natural SFT is the strongest trained model overall | 93.14% overall versus 52.63% for answer-only | Natural supervision is associated with much stronger generalisation in this seed | Promising, but only one seed; complete the scoring audit and add seed 43 |
| Natural SFT is nearly saturated on CORE | 99.54% CORE; every query family is at least 99% | CORE may be easy for Natural after training, but this alone does not establish leakage or overfitting | Check train/benchmark structural overlap and the second seed before drawing that conclusion |
| Natural still has identifiable weaknesses | COMP Count is 66.50%; depth-8 Selection is 70% and Count is 66% | The benchmark is not uniformly trivial; compositional counting and deeper Selection/Count remain discriminative | Add cell-level errors and response examples |
| Natural transfers well to the SHIFT group | 93.33% on held-out composition and 89.56% on depth | Strong evidence of transfer beyond the matched CORE/COMP conditions for this seed | Confirm after fair rescoring and replication |
| Answer-only performance is highly uneven | Strong COMP and several non-depth SHIFT cells, but very weak depth-labelled SHIFT cells | Answer-only learned some composition patterns but did not acquire robust depth reasoning | Audit prompts, output validity, and cell-level failure modes |
| Answer-only CORE is unexpectedly weak | 22.15% CORE despite 88.50% COMP | This inversion is suspicious and should not yet be explained as task difficulty alone | Check formatting failures, label distribution, and per-cell completions |
| Answer-only depth failure is broad | Direction 19.67%, Selection 2.33%, Count 0.33%; little change from depth 3 to 8 | This looks more like failure to execute the depth task at all than gradual degradation with depth | Inspect representative responses and answer distributions |
| The untuned baseline has not yet been measured fairly | A known correct response ending in bare `E` was scored zero | The displayed all-zero row measures extractor incompatibility, not model capability | Rescore saved outputs with the agreed answer-selection protocol |
| Overfitting cannot be concluded from this run | One seed, no corrected untuned baseline, and no train-versus-benchmark overlap analysis here | High CORE accuracy is a signal to investigate, not proof of overfitting | Add seed replication, overlap audit, and final extraction protocol |

## Results still to add

| Item | Status |
|---|---|
| Fairly rescored untuned baseline | Pending |
| Final answer-selection protocol | Two-stage: same checkpoint for both passes, with constrained stage-2 selection; pending rerun |
| Seed 43 for answer-only and Natural | Pending evaluation |
| Cell-level error counts and representative failures | Pending review |
| Output-format failure rate by model | Pending extraction audit |
| Train-to-benchmark structural-overlap audit | Pending analysis |
| External-dataset transfer results | Pending evaluation |
