# SpatialEntail-SINGLE-Bench: working results matrix

This is the living record for model-evaluation results. It deliberately keeps
observations separate from conclusions while replication, external transfer, and
error analysis are still being filled in.

## Evaluation status

- Snapshot: first complete structured-decoding seed-42 run supplied on
  2026-10-13.
- Benchmark: `SpatialEntail-SINGLE-Bench`, 4,300 frozen bases: CORE 1,300,
  COMP 1,200, and SHIFT 1,800.
- Evaluation mode: two-stage evaluation with constrained stage-2 answer
  selection from the same evaluated checkpoint. The frozen benchmark uses the
  `SINGLE` answer contract, so stage 2 emits exactly one option letter.
- Primary metric: stage-2 constrained-choice micro accuracy.
- Diagnostic metrics: stage-1 native accuracy and stage-1 canonical answer
  format validity.
- Status: first successful `SpatialEntail-SINGLE` experiment. Results are still
  provisional until the second seed and external transfer runs are added.

## Main comparison

| Model | Seed | Overall | CORE | COMP | SHIFT | Status |
|---|---:|---:|---:|---:|---:|---|
| Untuned Qwen3.5-4B | — | 91.42% | 96.46% | 98.42% | 83.11% | Prompt-matched baseline |
| Answer-only SFT | 42 | 86.02% | 93.92% | 97.58% | 72.61% | First seed |
| Verified Natural SFT | 42 | 97.91% | 99.46% | 99.75% | 95.56% | First seed |

The public `SpatialEntail-SINGLE-Bench` hierarchy is CORE, COMP, and SHIFT. No
subgroup is given an extra headline column.

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
| CORE | 99.80% | 92.20% | 87.00% | 99.80% | 98.80% | 100.00% |
| COMP | 100.00% | 99.75% | 93.00% | 100.00% | 100.00% | 99.25% |
| SHIFT | 93.00% | 76.83% | 48.00% | 100.00% | 95.33% | 91.33% |

Untuned query-family values are retained in `summary.json`. Its headline
profile is high on CORE/COMP and drops mainly on SHIFT Count.

## Stage comparison

| Model | Stage 1 native | Stage 1 canonical format | Stage 2 constrained |
|---|---:|---:|---:|
| Untuned Qwen3.5-4B | 76.47% | 0.00% | 91.42% |
| Answer-only SFT | 73.44% | 59.79% | 86.02% |
| Verified Natural SFT | 95.23% | 94.28% | 97.91% |

## Full per-cell breakdown

These are the first structured-decoding scores from the seed-42 run.

### CORE

| Cell | N | Untuned | Answer-only | Natural |
|---|---:|---:|---:|---:|
| **CORE cumulative** | **1,300** | **96.46%** | **93.92%** | **99.46%** |
| `CNT-AXC-UNI` | 100 | 68% | 61% | 100% |
| `CNT-CSR-JCR-UNI` | 100 | 100% | 100% | 100% |
| `CNT-NEG-UNI` | 100 | 100% | 100% | 100% |
| `DIR-ARC-ALT` | 100 | 99% | 100% | 99% |
| `DIR-ARC-UNI` | 100 | 100% | 100% | 100% |
| `DIR-DJE-UNI` | 100 | 100% | 100% | 100% |
| `DIR-MP-UNI` | 100 | 100% | 100% | 100% |
| `DIR-TRC-UNI` | 100 | 100% | 99% | 100% |
| `SEL-ARC-MUL` | 100 | 97% | 100% | 95% |
| `SEL-ARC-NOM` | 100 | 98% | 100% | 100% |
| `SEL-CSR-UNI` | 100 | 100% | 100% | 100% |
| `SEL-IFF-UNI` | 100 | 100% | 100% | 100% |
| `SEL-TRC-UNI-NOISE` | 100 | 92% | 61% | 99% |

### COMP

| Cell | N | Untuned | Answer-only | Natural |
|---|---:|---:|---:|---:|
| **COMP cumulative** | **1,200** | **98.42%** | **97.58%** | **99.75%** |
| `CNT-DJE-TRC-UNI` | 100 | 99% | 94% | 99% |
| `CNT-IFF-TRC-UNI` | 100 | 96% | 90% | 98% |
| `CNT-MP-TRC-UNI` | 100 | 99% | 88% | 100% |
| `CNT-TRC-MP-UNI` | 100 | 100% | 100% | 100% |
| `DIR-DJE-TRC-UNI` | 100 | 98% | 100% | 100% |
| `DIR-IFF-TRC-UNI` | 100 | 98% | 100% | 100% |
| `DIR-MP-TRC-UNI` | 100 | 98% | 100% | 100% |
| `DIR-TRC-MP-UNI` | 100 | 100% | 100% | 100% |
| `SEL-DJE-TRC-UNI` | 100 | 99% | 99% | 100% |
| `SEL-IFF-TRC-UNI` | 100 | 95% | 100% | 100% |
| `SEL-MP-TRC-UNI` | 100 | 99% | 100% | 100% |
| `SEL-TRC-MP-UNI` | 100 | 100% | 100% | 100% |

### SHIFT

| Cell | Slice | N | Untuned | Answer-only | Natural |
|---|---|---:|---:|---:|---:|
| **SHIFT cumulative** | **All SHIFT** | **1,800** | **83.11%** | **72.61%** | **95.56%** |
| `count-depth-3` | Depth | 50 | 38% | 0% | 100% |
| `count-depth-4` | Depth | 50 | 34% | 2% | 98% |
| `count-depth-5` | Depth | 50 | 24% | 2% | 92% |
| `count-depth-6` | Depth | 50 | 34% | 0% | 90% |
| `count-depth-7` | Depth | 50 | 32% | 0% | 76% |
| `count-depth-8` | Depth | 50 | 40% | 2% | 78% |
| `direction-depth-3` | Depth | 50 | 98% | 92% | 100% |
| `direction-depth-4` | Depth | 50 | 98% | 90% | 100% |
| `direction-depth-5` | Depth | 50 | 96% | 86% | 100% |
| `direction-depth-6` | Depth | 50 | 100% | 90% | 100% |
| `direction-depth-7` | Depth | 50 | 98% | 84% | 100% |
| `direction-depth-8` | Depth | 50 | 100% | 74% | 100% |
| `which-depth-3` | Depth | 50 | 84% | 54% | 98% |
| `which-depth-4` | Depth | 50 | 64% | 44% | 90% |
| `which-depth-5` | Depth | 50 | 72% | 54% | 94% |
| `which-depth-6` | Depth | 50 | 70% | 66% | 92% |
| `which-depth-7` | Depth | 50 | 64% | 54% | 88% |
| `which-depth-8` | Depth | 50 | 72% | 50% | 82% |
| `CNT-MP-DJE-TRC-UNI` | Held-out composition | 100 | 100% | 95% | 94% |
| `CNT-MP-TRC-MP-UNI` | Held-out composition | 100 | 99% | 100% | 97% |
| `CNT-TRC-MP-TRC-UNI` | Held-out composition | 100 | 100% | 90% | 90% |
| `DIR-MP-DJE-TRC-UNI` | Held-out composition | 100 | 95% | 100% | 100% |
| `DIR-MP-TRC-MP-UNI` | Held-out composition | 100 | 100% | 100% | 100% |
| `DIR-TRC-MP-TRC-UNI` | Held-out composition | 100 | 98% | 100% | 100% |
| `SEL-MP-DJE-TRC-UNI` | Held-out composition | 100 | 99% | 100% | 100% |
| `SEL-MP-TRC-MP-UNI` | Held-out composition | 100 | 100% | 100% | 100% |
| `SEL-TRC-MP-TRC-UNI` | Held-out composition | 100 | 96% | 100% | 100% |

## Findings matrix

| Finding | Evidence in this snapshot | Current interpretation | Confidence / next detail needed |
|---|---|---|---|
| Natural SFT is the strongest model overall | 97.91% overall versus 91.42% untuned and 86.02% answer-only | Verified Natural supervision improves the already-strong base model on `SpatialEntail-SINGLE-Bench` | First seed only; add seed 43 |
| Stage-2 structured decoding fixed the untuned false-negative problem | Untuned rises from 0% under strict stage-1 formatting to 91.42% stage-2 accuracy; its canonical stage-1 format validity is 0% | The earlier untuned zero was an extractor/format artifact, not model incapability | Keep stage-1 format as diagnostic only |
| CORE and COMP are mostly solved by the base model | Untuned scores 96.46% CORE and 98.42% COMP | These sections are useful sanity checks but not the main discriminative signal under structured decoding | Report cell-level errors rather than overclaiming difficulty |
| SHIFT carries most of the learning signal | Natural scores 95.56% SHIFT versus 83.11% untuned and 72.61% answer-only | Natural improves structural generalisation; answer-only degrades it in this seed | Add seed 43 and external transfer |
| Answer-only SFT is a negative result in this run | Answer-only is below untuned overall and especially on SHIFT, 72.61% versus 83.11% | Answer exposure alone does not improve the model and may hurt robustness | Audit training/decoding mismatch and response distributions |
| Depth Count remains the clearest weakness | Answer-only is 0–2% on count-depth cells; untuned is 24–40%; Natural is 76–100% | Natural helps substantially on deeper Count, while answer-only fails this slice | Inspect representative depth Count outputs |
| Overfitting cannot be concluded from this run | Natural is very high on CORE/COMP, but untuned is also high after structured decoding | High matched-section accuracy mostly shows these cells are easy for the base model under this protocol | Use seed replication, SHIFT, and external datasets for claims |

## Results still to add

| Item | Status |
|---|---|
| Fairly rescored untuned baseline | Complete for `SpatialEntail-SINGLE-Bench` |
| Final answer-selection protocol | Two-stage: same checkpoint for both passes, constrained stage-2 selection |
| Seed 43 for answer-only and Natural | Pending evaluation |
| Cell-level error counts and representative failures | Pending review |
| Output-format failure rate by model | Pending extraction audit |
| Train-to-benchmark structural-overlap audit | Pending analysis |
| External-dataset transfer results | Pending evaluation |
