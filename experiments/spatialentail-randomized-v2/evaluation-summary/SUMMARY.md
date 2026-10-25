# Randomized-v2 Evaluation Summary

Accuracy values are percentages. Stage 2 exact-set accuracy is primary; stage 1 native extraction and canonical-format validity are diagnostics.

Overall is example-weighted. Partition macro is the unweighted mean of the four partition accuracies. SINGLE is G1 (900 rows); ALL-POSSIBLE is G2-G4 (2,700 rows). L is measured semantic proof length; X is measured distractor count. Transition slices can overlap.

Validation checks completion markers, full task/sample counts, unique base IDs, common labeled examples, and agreement of sample metrics with results.json. Receipt hashes bind the claimed benchmark; source data, training receipts, adapter hashes and the cluster environment were not independently audited.

Raw responses remain in each package's results directory. The JSON report retains per-task metrics, input hashes, runtime settings, per-contract slices and diagnostic failure counts. Those categories describe stage outcomes, not causal reasoning-error diagnoses.

Compare response regimes cautiously: the P0 untuned baseline and fresh reasoning runs use different evaluation configurations. Seed ranges below are descriptive, not confidence intervals.

## prepared-general-eval

| Model | N | Stage 1 | Format | Stage 2 | Partition macro | SINGLE | ALL-POSSIBLE |
|---|---|---|---|---|---|---|---|
| qwen3-5-4b-thinking-off | 3600 | 48.81 | 1.58 | 73.19 | 72.20 | 88.22 | 68.19 |
| qwen3-5-9b-thinking-off | 3600 | 48.14 | 1.19 | 77.64 | 76.61 | 91.89 | 72.89 |
| gemma-4-e4b-it | 3600 | 1.33 | 0.03 | 67.64 | 66.84 | 91.67 | 59.63 |
| gemma-3-4b-it | 3600 | 0.00 | 0.00 | 18.94 | 18.72 | 71.56 | 1.41 |
| ministral-3-8b-instruct | 3600 | 5.22 | 0.00 | 34.97 | 34.22 | 67.56 | 24.11 |
| phi-4-mini-instruct | 3600 | 0.06 | 0.00 | 17.94 | 17.71 | 61.89 | 3.30 |

### partition breakdown

| partition | N per model | qwen3-5-4b-thinking-off | qwen3-5-9b-thinking-off | gemma-4-e4b-it | gemma-3-4b-it | ministral-3-8b-instruct | phi-4-mini-instruct |
|---|---|---|---|---|---|---|---|
| composition-ood | 720 | 77.92 | 80.14 | 74.17 | 19.72 | 37.50 | 18.47 |
| length-ood | 720 | 61.67 | 67.78 | 58.47 | 16.53 | 30.42 | 16.67 |
| matched | 1440 | 77.15 | 81.74 | 70.83 | 19.86 | 37.99 | 18.89 |
| noise-ood | 720 | 72.08 | 76.81 | 63.89 | 18.75 | 30.97 | 16.81 |

### query breakdown

| query | N per model | qwen3-5-4b-thinking-off | qwen3-5-9b-thinking-off | gemma-4-e4b-it | gemma-3-4b-it | ministral-3-8b-instruct | phi-4-mini-instruct |
|---|---|---|---|---|---|---|---|
| count | 1200 | 76.00 | 83.33 | 77.92 | 11.67 | 30.17 | 13.08 |
| direction | 1200 | 72.83 | 70.50 | 60.00 | 21.67 | 39.00 | 18.00 |
| which | 1200 | 70.75 | 79.08 | 65.00 | 23.50 | 35.75 | 22.75 |

### G breakdown

| G | N per model | qwen3-5-4b-thinking-off | qwen3-5-9b-thinking-off | gemma-4-e4b-it | gemma-3-4b-it | ministral-3-8b-instruct | phi-4-mini-instruct |
|---|---|---|---|---|---|---|---|
| 1 | 900 | 88.22 | 91.89 | 91.67 | 71.56 | 67.56 | 61.89 |
| 2 | 900 | 63.56 | 71.56 | 55.89 | 2.22 | 24.00 | 4.22 |
| 3 | 900 | 69.33 | 73.78 | 60.11 | 1.33 | 24.67 | 3.67 |
| 4 | 900 | 71.67 | 73.33 | 62.89 | 0.67 | 23.67 | 2.00 |

### L breakdown

| L | N per model | qwen3-5-4b-thinking-off | qwen3-5-9b-thinking-off | gemma-4-e4b-it | gemma-3-4b-it | ministral-3-8b-instruct | phi-4-mini-instruct |
|---|---|---|---|---|---|---|---|
| 1 | 189 | 80.95 | 82.54 | 68.78 | 31.75 | 43.92 | 27.51 |
| 2 | 324 | 85.19 | 85.19 | 75.31 | 27.78 | 44.44 | 25.62 |
| 3 | 399 | 77.69 | 82.46 | 71.68 | 21.55 | 39.60 | 18.80 |
| 4 | 533 | 79.55 | 82.93 | 74.86 | 20.26 | 35.27 | 16.14 |
| 5 | 637 | 71.90 | 78.81 | 67.66 | 16.01 | 34.38 | 17.11 |
| 6 | 798 | 71.43 | 75.44 | 65.66 | 14.66 | 31.08 | 15.16 |
| 7 | 336 | 63.99 | 70.24 | 61.31 | 17.56 | 32.14 | 16.96 |
| 8 | 384 | 59.64 | 65.62 | 55.99 | 15.62 | 28.91 | 16.41 |

### X breakdown

| X | N per model | qwen3-5-4b-thinking-off | qwen3-5-9b-thinking-off | gemma-4-e4b-it | gemma-3-4b-it | ministral-3-8b-instruct | phi-4-mini-instruct |
|---|---|---|---|---|---|---|---|
| 0 | 300 | 78.33 | 79.00 | 73.00 | 23.00 | 42.00 | 21.00 |
| 1 | 336 | 75.00 | 82.14 | 72.32 | 21.43 | 38.69 | 21.13 |
| 2 | 372 | 75.27 | 78.76 | 71.24 | 19.89 | 37.37 | 16.13 |
| 3 | 420 | 75.00 | 78.10 | 68.10 | 19.76 | 39.05 | 19.05 |
| 4 | 444 | 74.10 | 76.58 | 67.57 | 17.79 | 33.56 | 16.67 |
| 5 | 480 | 70.42 | 77.71 | 67.50 | 17.92 | 33.33 | 17.08 |
| 6 | 528 | 69.51 | 74.81 | 64.02 | 15.91 | 31.82 | 17.99 |
| 7 | 348 | 73.56 | 77.87 | 64.94 | 19.54 | 32.18 | 18.39 |
| 8 | 372 | 70.70 | 75.81 | 62.90 | 18.01 | 29.84 | 15.32 |

### heldout_transition breakdown

| heldout_transition | N per model | qwen3-5-4b-thinking-off | qwen3-5-9b-thinking-off | gemma-4-e4b-it | gemma-3-4b-it | ministral-3-8b-instruct | phi-4-mini-instruct |
|---|---|---|---|---|---|---|---|
| disjunctive-syllogism->iff-elimination | 147 | 73.47 | 76.19 | 70.07 | 14.97 | 29.93 | 14.29 |
| double-negation->iff-elimination | 165 | 80.61 | 83.64 | 74.55 | 18.18 | 36.97 | 16.36 |
| iff-elimination->double-negation | 162 | 74.69 | 77.16 | 67.28 | 21.60 | 37.04 | 18.52 |
| iff-elimination->iff-elimination | 75 | 78.67 | 80.00 | 80.00 | 22.67 | 41.33 | 21.33 |
| iff-elimination->modus-ponens | 118 | 79.66 | 80.51 | 81.36 | 13.56 | 42.37 | 17.80 |
| modus-ponens->axis-transitivity | 53 | 86.79 | 88.68 | 81.13 | 41.51 | 45.28 | 33.96 |

## prepared-reasoning-eval

| Model | N | Stage 1 | Format | Stage 2 | Partition macro | SINGLE | ALL-POSSIBLE |
|---|---|---|---|---|---|---|---|
| qwen3-5-4b-thinking-on | 3600 | 2.78 | 0.06 | 35.86 | 34.39 | 42.22 | 33.74 |
| qwen3-5-9b-thinking-on | 3600 | 2.75 | 0.00 | 37.47 | 36.74 | 46.78 | 34.37 |
| ministral-3-8b-reasoning | 3600 | 1.28 | 0.08 | 21.25 | 21.13 | 58.44 | 8.85 |
| deepseek-r1-distill-qwen-7b | 3600 | 4.19 | 6.08 | 28.28 | 26.58 | 58.89 | 18.07 |

### partition breakdown

| partition | N per model | qwen3-5-4b-thinking-on | qwen3-5-9b-thinking-on | ministral-3-8b-reasoning | deepseek-r1-distill-qwen-7b |
|---|---|---|---|---|---|
| composition-ood | 720 | 38.06 | 39.58 | 20.14 | 30.56 |
| length-ood | 720 | 29.86 | 34.86 | 19.31 | 20.28 |
| matched | 1440 | 41.74 | 40.42 | 21.74 | 35.07 |
| noise-ood | 720 | 27.92 | 32.08 | 23.33 | 20.42 |

### query breakdown

| query | N per model | qwen3-5-4b-thinking-on | qwen3-5-9b-thinking-on | ministral-3-8b-reasoning | deepseek-r1-distill-qwen-7b |
|---|---|---|---|---|---|
| count | 1200 | 35.00 | 42.08 | 14.83 | 35.08 |
| direction | 1200 | 29.67 | 21.75 | 29.33 | 22.58 |
| which | 1200 | 42.92 | 48.58 | 19.58 | 27.17 |

### G breakdown

| G | N per model | qwen3-5-4b-thinking-on | qwen3-5-9b-thinking-on | ministral-3-8b-reasoning | deepseek-r1-distill-qwen-7b |
|---|---|---|---|---|---|
| 1 | 900 | 42.22 | 46.78 | 58.44 | 58.89 |
| 2 | 900 | 27.56 | 28.22 | 9.44 | 21.33 |
| 3 | 900 | 32.67 | 33.00 | 7.78 | 16.56 |
| 4 | 900 | 41.00 | 41.89 | 9.33 | 16.33 |

### L breakdown

| L | N per model | qwen3-5-4b-thinking-on | qwen3-5-9b-thinking-on | ministral-3-8b-reasoning | deepseek-r1-distill-qwen-7b |
|---|---|---|---|---|---|
| 1 | 189 | 42.86 | 38.10 | 32.80 | 35.98 |
| 2 | 324 | 48.46 | 41.98 | 28.40 | 37.04 |
| 3 | 399 | 43.36 | 40.35 | 22.31 | 35.09 |
| 4 | 533 | 35.83 | 36.40 | 21.39 | 31.33 |
| 5 | 637 | 34.54 | 38.15 | 20.72 | 28.41 |
| 6 | 798 | 31.83 | 36.59 | 17.17 | 24.56 |
| 7 | 336 | 28.87 | 33.93 | 21.13 | 22.32 |
| 8 | 384 | 30.73 | 35.68 | 17.71 | 18.49 |

### X breakdown

| X | N per model | qwen3-5-4b-thinking-on | qwen3-5-9b-thinking-on | ministral-3-8b-reasoning | deepseek-r1-distill-qwen-7b |
|---|---|---|---|---|---|
| 0 | 300 | 45.33 | 47.00 | 22.67 | 45.67 |
| 1 | 336 | 45.83 | 42.26 | 18.75 | 38.99 |
| 2 | 372 | 38.17 | 39.52 | 23.12 | 33.87 |
| 3 | 420 | 37.14 | 37.62 | 20.48 | 34.05 |
| 4 | 444 | 39.19 | 40.09 | 21.40 | 24.10 |
| 5 | 480 | 33.54 | 35.21 | 18.75 | 24.17 |
| 6 | 528 | 31.63 | 34.66 | 20.64 | 21.02 |
| 7 | 348 | 30.17 | 34.20 | 22.13 | 22.41 |
| 8 | 372 | 25.81 | 30.11 | 24.46 | 18.55 |

### heldout_transition breakdown

| heldout_transition | N per model | qwen3-5-4b-thinking-on | qwen3-5-9b-thinking-on | ministral-3-8b-reasoning | deepseek-r1-distill-qwen-7b |
|---|---|---|---|---|---|
| disjunctive-syllogism->iff-elimination | 147 | 36.73 | 39.46 | 19.73 | 29.93 |
| double-negation->iff-elimination | 165 | 35.15 | 41.21 | 15.15 | 28.48 |
| iff-elimination->double-negation | 162 | 41.98 | 40.74 | 22.22 | 27.78 |
| iff-elimination->iff-elimination | 75 | 36.00 | 29.33 | 18.67 | 36.00 |
| iff-elimination->modus-ponens | 118 | 41.53 | 45.76 | 19.49 | 31.36 |
| modus-ponens->axis-transitivity | 53 | 33.96 | 32.08 | 33.96 | 37.74 |

## prepared-eval

| Model | N | Stage 1 | Format | Stage 2 | Partition macro | SINGLE | ALL-POSSIBLE |
|---|---|---|---|---|---|---|---|
| untuned | 3600 | 4.83 | 0.00 | 65.11 | 63.72 | 88.44 | 57.33 |
| single-natural-seed-42 | 3600 | 25.06 | 62.28 | 25.11 | 25.12 | 100.00 | 0.15 |
| single-natural-seed-43 | 3600 | 25.14 | 71.19 | 25.28 | 25.28 | 99.78 | 0.44 |
| single-answer-only-seed-42 | 3600 | 24.92 | 72.53 | 25.00 | 25.00 | 100.00 | 0.00 |
| single-answer-only-seed-43 | 3600 | 24.03 | 13.28 | 24.92 | 24.91 | 99.67 | 0.00 |
| multi-natural-seed-42 | 3600 | 99.11 | 99.36 | 99.39 | 99.38 | 99.56 | 99.33 |
| multi-natural-seed-43 | 3600 | 99.89 | 96.94 | 98.78 | 98.82 | 100.00 | 98.37 |

### partition breakdown

| partition | N per model | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| composition-ood | 720 | 72.08 | 25.42 | 25.56 | 25.00 | 25.00 | 99.72 | 99.17 |
| length-ood | 720 | 53.06 | 25.00 | 25.00 | 25.00 | 24.86 | 99.17 | 98.75 |
| matched | 1440 | 70.69 | 25.07 | 25.28 | 25.00 | 24.93 | 99.44 | 98.61 |
| noise-ood | 720 | 59.03 | 25.00 | 25.28 | 25.00 | 24.86 | 99.17 | 98.75 |

### query breakdown

| query | N per model | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| count | 1200 | 71.00 | 25.00 | 25.08 | 25.00 | 24.75 | 98.50 | 96.58 |
| direction | 1200 | 58.75 | 25.08 | 25.17 | 25.00 | 25.00 | 99.75 | 99.75 |
| which | 1200 | 65.58 | 25.25 | 25.58 | 25.00 | 25.00 | 99.92 | 100.00 |

### G breakdown

| G | N per model | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| 1 | 900 | 88.44 | 100.00 | 99.78 | 100.00 | 99.67 | 99.56 | 100.00 |
| 2 | 900 | 59.44 | 0.22 | 0.44 | 0.00 | 0.00 | 99.11 | 98.67 |
| 3 | 900 | 58.78 | 0.11 | 0.22 | 0.00 | 0.00 | 99.44 | 99.11 |
| 4 | 900 | 53.78 | 0.11 | 0.67 | 0.00 | 0.00 | 99.44 | 97.33 |

### L breakdown

| L | N per model | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| 1 | 189 | 77.25 | 33.33 | 33.33 | 33.33 | 33.33 | 100.00 | 100.00 |
| 2 | 324 | 79.01 | 30.56 | 31.17 | 30.56 | 30.25 | 98.77 | 100.00 |
| 3 | 399 | 70.43 | 27.82 | 27.82 | 27.82 | 27.82 | 99.25 | 99.25 |
| 4 | 533 | 68.86 | 24.39 | 24.58 | 24.20 | 24.20 | 99.25 | 98.50 |
| 5 | 637 | 67.50 | 23.23 | 23.39 | 23.08 | 23.08 | 99.22 | 98.27 |
| 6 | 798 | 60.40 | 21.68 | 21.93 | 21.43 | 21.30 | 100.00 | 98.37 |
| 7 | 336 | 54.17 | 25.00 | 25.60 | 25.00 | 24.70 | 99.11 | 98.21 |
| 8 | 384 | 52.08 | 25.00 | 24.48 | 25.00 | 25.00 | 99.22 | 99.22 |

### X breakdown

| X | N per model | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| 0 | 300 | 75.67 | 25.33 | 26.33 | 25.00 | 25.00 | 99.33 | 97.67 |
| 1 | 336 | 75.60 | 25.30 | 25.00 | 25.00 | 25.00 | 99.70 | 99.40 |
| 2 | 372 | 70.97 | 25.27 | 25.54 | 25.00 | 25.00 | 98.92 | 98.66 |
| 3 | 420 | 63.33 | 25.00 | 24.76 | 25.00 | 25.00 | 99.52 | 98.81 |
| 4 | 444 | 64.86 | 25.00 | 25.23 | 25.00 | 25.00 | 99.55 | 98.65 |
| 5 | 480 | 62.08 | 25.21 | 25.21 | 25.00 | 25.00 | 99.79 | 98.75 |
| 6 | 528 | 60.98 | 25.00 | 25.19 | 25.00 | 24.62 | 99.24 | 99.24 |
| 7 | 348 | 57.76 | 25.00 | 25.57 | 25.00 | 25.00 | 99.43 | 98.56 |
| 8 | 372 | 60.22 | 25.00 | 25.00 | 25.00 | 24.73 | 98.92 | 98.92 |

### heldout_transition breakdown

| heldout_transition | N per model | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| disjunctive-syllogism->iff-elimination | 147 | 68.03 | 18.37 | 18.37 | 17.69 | 17.69 | 100.00 | 97.96 |
| double-negation->iff-elimination | 165 | 67.88 | 21.21 | 21.82 | 21.21 | 21.21 | 99.39 | 100.00 |
| iff-elimination->double-negation | 162 | 73.46 | 27.16 | 27.16 | 26.54 | 26.54 | 99.38 | 98.77 |
| iff-elimination->iff-elimination | 75 | 69.33 | 28.00 | 28.00 | 28.00 | 28.00 | 100.00 | 100.00 |
| iff-elimination->modus-ponens | 118 | 77.12 | 21.19 | 22.03 | 21.19 | 21.19 | 100.00 | 100.00 |
| modus-ponens->axis-transitivity | 53 | 84.91 | 58.49 | 56.60 | 56.60 | 56.60 | 100.00 | 98.11 |

### SFT Seed Summary

| Arm | Subset | Seed 42 | Seed 43 | Mean | Min | Max | Range (pp) |
|---|---|---|---|---|---|---|---|
| single-natural | overall | 25.11 | 25.28 | 25.19 | 25.11 | 25.28 | 0.17 |
| single-natural | single | 100.00 | 99.78 | 99.89 | 99.78 | 100.00 | 0.22 |
| single-natural | all-possible | 0.15 | 0.44 | 0.30 | 0.15 | 0.44 | 0.30 |
| single-answer-only | overall | 25.00 | 24.92 | 24.96 | 24.92 | 25.00 | 0.08 |
| single-answer-only | single | 100.00 | 99.67 | 99.83 | 99.67 | 100.00 | 0.33 |
| single-answer-only | all-possible | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| multi-natural | overall | 99.39 | 98.78 | 99.08 | 98.78 | 99.39 | 0.61 |
| multi-natural | single | 99.56 | 100.00 | 99.78 | 99.56 | 100.00 | 0.44 |
| multi-natural | all-possible | 99.33 | 98.37 | 98.85 | 98.37 | 99.33 | 0.96 |

### Primary Comparisons

Deltas are first minus second, in percentage points.

| Comparison | Subset | Seed 42 delta | Seed 43 delta | Mean delta |
|---|---|---|---|---|
| Natural minus answer-only | single | 0.00 | 0.11 | 0.06 |
| MULTI minus untuned | all-possible | 42.00 | 41.04 | 41.52 |
| MULTI minus SINGLE Natural | all-possible | 99.19 | 97.93 | 98.56 |
| SINGLE retention: MULTI minus SINGLE Natural | single | -0.44 | 0.22 | -0.11 |
