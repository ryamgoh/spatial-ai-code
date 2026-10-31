# Entailment Candidate Distribution Review

Status: unsealed candidate. This report is descriptive, not approval to train.
Candidate manifest SHA-256: `f326b91aeec015401f28390a453886a0ee8a195efd8fc2b4dd934a7bbbb0c452`.

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
| natural | 60 | 334.94 | 1002 |
| symbolic | 51 | 348.04 | 1019 |

Lengths use pinned-tokenizer measurements saved by full assembly.

| Depth | Style | Min Tokens | Mean Tokens | Max Tokens |
| --- | --- | --- | --- | --- |
| None | natural | 71 | 218.24 | 584 |
| None | symbolic | 62 | 217.59 | 704 |
| 1 | natural | 60 | 262.42 | 608 |
| 1 | symbolic | 51 | 260.62 | 616 |
| 2 | natural | 71 | 304.24 | 709 |
| 2 | symbolic | 62 | 307.53 | 723 |
| 3 | natural | 90 | 337.87 | 761 |
| 3 | symbolic | 99 | 345.96 | 790 |
| 4 | natural | 94 | 350.48 | 805 |
| 4 | symbolic | 107 | 363.91 | 836 |
| 5 | natural | 103 | 365.37 | 855 |
| 5 | symbolic | 118 | 380.94 | 903 |
| 6 | natural | 108 | 384.63 | 895 |
| 6 | symbolic | 125 | 405.57 | 953 |
| 7 | natural | 117 | 396.86 | 953 |
| 7 | symbolic | 136 | 422.91 | 959 |
| 8 | natural | 123 | 410.48 | 983 |
| 8 | symbolic | 143 | 438.22 | 1019 |
| 9 | natural | 129 | 429.20 | 995 |
| 9 | symbolic | 143 | 455.01 | 965 |
| 10 | natural | 133 | 433.20 | 1002 |
| 10 | symbolic | 143 | 465.01 | 976 |

## Review Status
Automated quota and coverage checks are recorded in `summary.json`.
Independent evidence/input audit and human-readable paired sample review
must be recorded separately. Sample export is not itself a manual pass.
No seal or training authorization is granted by this report.
