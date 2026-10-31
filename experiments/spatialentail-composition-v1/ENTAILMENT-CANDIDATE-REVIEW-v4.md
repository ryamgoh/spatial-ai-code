# Entailment Candidate Distribution Review

Status: unsealed candidate. This report is descriptive, not approval to train.
Candidate manifest SHA-256: `fb2cee949e844b8cdfabe9ca51567a401c986e95bad4a17a69d73536a5fa4700`.

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
| natural | 66 | 370.13 | 1139 |
| symbolic | 51 | 341.96 | 1002 |

Lengths use pinned-tokenizer measurements saved by full assembly.

| Depth | Style | Min Tokens | Mean Tokens | Max Tokens |
| --- | --- | --- | --- | --- |
| None | natural | 77 | 229.68 | 626 |
| None | symbolic | 62 | 204.29 | 635 |
| 1 | natural | 66 | 276.75 | 638 |
| 1 | symbolic | 51 | 246.84 | 584 |
| 2 | natural | 77 | 327.05 | 752 |
| 2 | symbolic | 62 | 294.32 | 694 |
| 3 | natural | 100 | 369.47 | 835 |
| 3 | symbolic | 96 | 337.06 | 785 |
| 4 | natural | 111 | 388.85 | 922 |
| 4 | symbolic | 109 | 359.60 | 840 |
| 5 | natural | 117 | 407.23 | 955 |
| 5 | symbolic | 119 | 377.45 | 903 |
| 6 | natural | 121 | 431.92 | 1028 |
| 6 | symbolic | 124 | 403.73 | 927 |
| 7 | natural | 131 | 448.73 | 1094 |
| 7 | symbolic | 136 | 423.60 | 971 |
| 8 | natural | 138 | 464.87 | 1087 |
| 8 | symbolic | 143 | 439.28 | 979 |
| 9 | natural | 148 | 488.23 | 1136 |
| 9 | symbolic | 143 | 457.96 | 1002 |
| 10 | natural | 155 | 492.43 | 1139 |
| 10 | symbolic | 143 | 467.31 | 1002 |

## Review Status
Automated quota and coverage checks are recorded in `summary.json`.
Independent evidence/input audit and human-readable paired sample review
must be recorded separately. Sample export is not itself a manual pass.
No seal or training authorization is granted by this report.
