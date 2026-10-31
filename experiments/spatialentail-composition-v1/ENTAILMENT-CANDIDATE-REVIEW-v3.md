# Entailment Candidate Distribution Review

Status: unsealed candidate. This report is descriptive, not approval to train.
Candidate manifest SHA-256: `9b8a4f742205c593c8dc8a5858a2c494f7f2d8897509b681e6947a93426e18cd`.

## Retained Bases
| Split | Direction | Selection | Count | Total |
| --- | --- | --- | --- | --- |
| train | 2000 | 2000 | 2000 | 6000 |
| dev | 400 | 400 | 400 | 1200 |
| in-range-test | 800 | 800 | 800 | 2400 |
| extrapolation-test | 400 | 400 | 400 | 1200 |

## Correlated View Exposure
| Split | Original Views/Style | Final Views/Style | Augmented Bases | Added Views/Style |
| --- | --- | --- | --- | --- |
| train | 20568 | 21222 | 45 | 654 |
| dev | 4146 | 4344 | 12 | 198 |
| in-range-test | 8316 | 8622 | 17 | 306 |
| extrapolation-test | 4335 | 4335 | 0 | 0 |

Position views are not additional independent bases or families. Both styles
receive identical questions, menus and labels. Report base-level or
family-clustered evaluation uncertainty, not independence of all views.
Multi-entailed Selection receives more exposure under this augmentation.

## Escape Distribution
| Split | Contract | Original E/F | Final E/F | E | F | Final Views |
| --- | --- | --- | --- | --- | --- | --- |
| train | unique-entailed | 32.91% | 34.97% | 1628 | 846 | 7074 |
| train | listed-entailed | 32.00% | 31.01% | 288 | 1906 | 7074 |
| train | exhaustive-entailed | 32.15% | 31.71% | 288 | 1955 | 7074 |
| dev | unique-entailed | 32.78% | 35.84% | 340 | 179 | 1448 |
| dev | listed-entailed | 31.62% | 30.18% | 72 | 365 | 1448 |
| dev | exhaustive-entailed | 31.84% | 30.94% | 72 | 376 | 1448 |
| in-range-test | unique-entailed | 33.55% | 35.91% | 665 | 367 | 2874 |
| in-range-test | listed-entailed | 32.61% | 31.45% | 138 | 766 | 2874 |
| in-range-test | exhaustive-entailed | 32.79% | 32.43% | 138 | 794 | 2874 |
| extrapolation-test | unique-entailed | 16.96% | 16.96% | 0 | 245 | 1445 |
| extrapolation-test | listed-entailed | 16.96% | 16.96% | 0 | 245 | 1445 |
| extrapolation-test | exhaustive-entailed | 16.96% | 16.96% | 0 | 245 | 1445 |

Original means position variant zero, including its omission companion.
E/F prevalence alone is not a correctness or difficulty measure.

## Sources
| Split | Controlled | Random World | Random Atoms | Targeted |
| --- | --- | --- | --- | --- |
| train | 4200 | 900 | 600 | 300 |
| dev | 840 | 180 | 120 | 60 |
| in-range-test | 1680 | 360 | 240 | 120 |
| extrapolation-test | 1200 | 0 | 0 | 0 |

## Explanation Lengths
| Style | Min Tokens | Mean Tokens | Max Tokens |
| --- | --- | --- | --- |
| natural | 66 | 367.91 | 1139 |
| symbolic | 51 | 339.68 | 1002 |

Lengths use pinned-tokenizer measurements saved by full assembly.

| Depth | Style | Min Tokens | Mean Tokens | Max Tokens |
| --- | --- | --- | --- | --- |
| None | natural | 66 | 229.27 | 626 |
| None | symbolic | 51 | 203.90 | 635 |
| 1 | natural | 66 | 274.66 | 638 |
| 1 | symbolic | 51 | 244.72 | 584 |
| 2 | natural | 77 | 325.13 | 752 |
| 2 | symbolic | 62 | 292.35 | 694 |
| 3 | natural | 100 | 367.16 | 835 |
| 3 | symbolic | 96 | 334.70 | 785 |
| 4 | natural | 111 | 386.38 | 922 |
| 4 | symbolic | 108 | 357.07 | 840 |
| 5 | natural | 117 | 404.11 | 955 |
| 5 | symbolic | 119 | 374.24 | 903 |
| 6 | natural | 121 | 428.48 | 1028 |
| 6 | symbolic | 124 | 400.19 | 927 |
| 7 | natural | 131 | 446.21 | 1094 |
| 7 | symbolic | 136 | 421.01 | 971 |
| 8 | natural | 138 | 462.02 | 1087 |
| 8 | symbolic | 143 | 436.34 | 979 |
| 9 | natural | 148 | 484.82 | 1136 |
| 9 | symbolic | 143 | 454.46 | 1002 |
| 10 | natural | 151 | 489.39 | 1139 |
| 10 | symbolic | 143 | 464.19 | 1002 |

## Review Status
Automated quota and coverage checks are recorded in `summary.json`.
Independent evidence/input audit and human-readable paired sample review
must be recorded separately. Sample export is not itself a manual pass.
No seal or training authorization is granted by this report.
