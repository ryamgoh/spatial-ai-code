# Randomized-v2 Corr Transfer Summary

External SINGLE-contract evaluation. Accuracy values are percentages. Stage 2 constrained A-E accuracy remains primary; native extraction is diagnostic.

Validated 7 complete runs, 1,500 unique source IDs per run, and 10,500 responses. Always choosing E scores 55.53% overall.

Archive receipt, prepared inputs and local environment source hashes match. Every saved document, target, prompt and generation setting matches the prepared data; runtime model arguments match configs. Original scorer replay agrees on every sample, and aggregated metrics agree with results.json. This does not independently verify cluster training consumption, adapter weight hashes, or all installed runtime packages.

## Scores

| Model | Stage 1 | Stage 2 | Determinate | E recall | False abstention |
|---|---|---|---|---|---|
| untuned | 2.73 | 64.67 | 83.51 | 49.58 | 7.80 |
| single-natural-seed-42 | 42.27 | 48.33 | 76.61 | 25.69 | 9.90 |
| single-natural-seed-43 | 34.20 | 42.07 | 66.87 | 22.21 | 16.19 |
| single-answer-only-seed-42 | 30.53 | 34.33 | 55.02 | 17.77 | 15.74 |
| single-answer-only-seed-43 | 42.93 | 49.27 | 86.21 | 19.69 | 2.85 |
| multi-natural-seed-42 | 31.80 | 33.27 | 68.22 | 5.28 | 2.25 |
| multi-natural-seed-43 | 30.80 | 31.67 | 62.37 | 7.08 | 10.04 |

E recall is accuracy on gold-E rows; false abstention is the rate of predicting E on determinate rows. These have different denominators.

## Seed Summary

| Arm | Seed 42 | Seed 43 | Mean | Min | Max |
|---|---|---|---|---|---|
| single-natural | 48.33 | 42.07 | 45.20 | 42.07 | 48.33 |
| single-answer-only | 34.33 | 49.27 | 41.80 | 34.33 | 49.27 |
| multi-natural | 33.27 | 31.67 | 32.47 | 31.67 | 33.27 |

## By Query

| Slice | N | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| count | 500 | 54.20 | 44.00 | 35.00 | 23.40 | 28.20 | 19.20 | 18.60 |
| direction | 500 | 86.80 | 56.40 | 50.20 | 41.40 | 68.40 | 49.40 | 47.40 |
| which | 500 | 53.00 | 44.60 | 41.00 | 38.20 | 51.20 | 31.20 | 29.00 |

## By Status

| Slice | N | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| determinate | 667 | 83.51 | 76.61 | 66.87 | 55.02 | 86.21 | 68.22 | 62.37 |
| underdetermined | 833 | 49.58 | 25.69 | 22.21 | 17.77 | 19.69 | 5.28 | 7.08 |

## By Query Status

| Slice | N | untuned | single-natural-seed-42 | single-natural-seed-43 | single-answer-only-seed-42 | single-answer-only-seed-43 | multi-natural-seed-42 | multi-natural-seed-43 |
|---|---|---|---|---|---|---|---|---|
| count/determinate | 195 | 47.69 | 60.51 | 50.77 | 42.05 | 68.72 | 38.97 | 38.46 |
| count/underdetermined | 305 | 58.36 | 33.44 | 24.92 | 11.48 | 2.30 | 6.56 | 5.90 |
| direction/determinate | 332 | 98.49 | 84.34 | 71.69 | 56.02 | 96.99 | 73.80 | 66.57 |
| direction/underdetermined | 168 | 63.69 | 1.19 | 7.74 | 12.50 | 11.90 | 1.19 | 9.52 |
| which/determinate | 140 | 97.86 | 80.71 | 77.86 | 70.71 | 85.00 | 95.71 | 85.71 |
| which/underdetermined | 360 | 35.56 | 30.56 | 26.67 | 25.56 | 38.06 | 6.11 | 6.94 |

## Stage Diagnostics

| Model | Stage 1 format | Normalized stage 1 | Closed think | Rescues | Regressions |
|---|---|---|---|---|---|
| untuned | 0.00 | 3.07 | 4.80 | 929 | 0 |
| single-natural-seed-42 | 85.33 | 42.27 | 90.40 | 93 | 2 |
| single-natural-seed-43 | 70.80 | 34.20 | 96.20 | 126 | 8 |
| single-answer-only-seed-42 | 84.73 | 30.53 | 89.93 | 57 | 0 |
| single-answer-only-seed-43 | 37.80 | 42.93 | 79.67 | 95 | 0 |
| multi-natural-seed-42 | 88.13 | 31.80 | 94.80 | 23 | 1 |
| multi-natural-seed-43 | 93.07 | 30.80 | 95.47 | 15 | 2 |

Normalized extraction strips markdown emphasis/backticks and a trailing </think> only. It is a sensitivity diagnostic, not a replacement score. A missing </think> is not proof of token-limit truncation; generation finish reasons are not present here.

## Paired Checkpoint Comparisons

Differences are second minus first. Wins/losses pair the same source IDs. 95% percentile bootstrap intervals resample IDs within each reported subset, with 2,000 repetitions and seed 42. They condition on these checkpoints, treat IDs as sampling units, and do not capture training-seed or unseen-map uncertainty. Intervals are descriptive and not multiplicity-adjusted.

| First | Second | Subset | Wins | Losses | Delta (pp) | 95% interval (pp) |
|---|---|---|---|---|---|---|
| untuned | single-natural-seed-42 | overall | 174 | 419 | -16.33 | -19.27 to -13.13 |
| untuned | single-natural-seed-42 | determinate | 66 | 112 | -6.90 | -11.09 to -3.00 |
| untuned | single-natural-seed-42 | underdetermined | 108 | 307 | -23.89 | -28.45 to -19.33 |
| untuned | single-answer-only-seed-42 | overall | 130 | 585 | -30.33 | -33.40 to -27.27 |
| untuned | single-answer-only-seed-42 | determinate | 51 | 241 | -28.49 | -33.28 to -23.84 |
| untuned | single-answer-only-seed-42 | underdetermined | 79 | 344 | -31.81 | -36.25 to -27.49 |
| untuned | multi-natural-seed-42 | overall | 57 | 528 | -31.40 | -34.07 to -28.60 |
| untuned | multi-natural-seed-42 | determinate | 41 | 143 | -15.29 | -19.19 to -11.54 |
| untuned | multi-natural-seed-42 | underdetermined | 16 | 385 | -44.30 | -47.90 to -40.58 |
| single-natural-seed-42 | multi-natural-seed-42 | overall | 106 | 332 | -15.07 | -17.67 to -12.40 |
| single-natural-seed-42 | multi-natural-seed-42 | determinate | 81 | 137 | -8.40 | -12.74 to -4.20 |
| single-natural-seed-42 | multi-natural-seed-42 | underdetermined | 25 | 195 | -20.41 | -23.53 to -17.17 |
| single-answer-only-seed-42 | single-natural-seed-42 | overall | 382 | 172 | 14.00 | 11.00 to 17.07 |
| single-answer-only-seed-42 | single-natural-seed-42 | determinate | 215 | 71 | 21.59 | 16.94 to 26.39 |
| single-answer-only-seed-42 | single-natural-seed-42 | underdetermined | 167 | 101 | 7.92 | 4.20 to 11.64 |
| untuned | single-natural-seed-43 | overall | 145 | 484 | -22.60 | -25.60 to -19.67 |
| untuned | single-natural-seed-43 | determinate | 56 | 167 | -16.64 | -20.84 to -12.59 |
| untuned | single-natural-seed-43 | underdetermined | 89 | 317 | -27.37 | -31.57 to -22.69 |
| untuned | single-answer-only-seed-43 | overall | 144 | 375 | -15.40 | -18.33 to -12.53 |
| untuned | single-answer-only-seed-43 | determinate | 75 | 57 | 2.70 | -0.60 to 6.00 |
| untuned | single-answer-only-seed-43 | underdetermined | 69 | 318 | -29.89 | -33.85 to -25.69 |
| untuned | multi-natural-seed-43 | overall | 65 | 560 | -33.00 | -35.60 to -30.27 |
| untuned | multi-natural-seed-43 | determinate | 42 | 183 | -21.14 | -25.49 to -17.24 |
| untuned | multi-natural-seed-43 | underdetermined | 23 | 377 | -42.50 | -46.10 to -38.78 |
| single-natural-seed-43 | multi-natural-seed-43 | overall | 138 | 294 | -10.40 | -12.93 to -7.73 |
| single-natural-seed-43 | multi-natural-seed-43 | determinate | 97 | 127 | -4.50 | -8.85 to -0.30 |
| single-natural-seed-43 | multi-natural-seed-43 | underdetermined | 41 | 167 | -15.13 | -18.61 to -11.88 |
| single-answer-only-seed-43 | single-natural-seed-43 | overall | 167 | 275 | -7.20 | -9.93 to -4.33 |
| single-answer-only-seed-43 | single-natural-seed-43 | determinate | 42 | 171 | -19.34 | -23.39 to -15.29 |
| single-answer-only-seed-43 | single-natural-seed-43 | underdetermined | 125 | 104 | 2.52 | -1.20 to 6.24 |

## Scope

Do not merge these scores with internal SpatialEntail scores or legacy 1,329-row Corr. The two SFT seeds are reported separately; their range is not a seed-level confidence interval. Gold labels inherit the current qualitative-coordinate correction assumptions; manual assumption sensitivity remains pending. Raw responses and logs stay in the original archive.

Archive SHA-256: `021d9cf7d84449042d947fae716a2cd3a7dff7f87df1608c535f5cc58ba70f75`.

Receipt SHA-256: `f352c1dd47b3436e2707d7613242892587cb15915dc4323db48e0afe0e0a65a0`.
