# Selected Subset Distributions

Measured from the actual v4 selected IDs, not from the surplus candidate pool.
This is a diagnostic selected subset: no replacement training package is exported or sealed.
Natural and Symbolic share the same bases, menus and answers; their token lengths differ.

## Interpretation

- The approved sizes, source mixtures, depth allocation, and joint quota/coverage/family constraints are satisfied by the selected subset.
- Multi-entailment Selection is now substantial: 345 bases have two entailments, 313 have three, and 242 have four. Another 16 have five; these do not count toward the exact 900 target.
- The 2-4 entailed-object share is 22.85% train, 23.00% dev, 24.75% in-range test, and 38.25% extrapolation. This is not equalized across splits.
- All-menu E/F exposure is 33.02% overall and 34.72% in training. Primary-only E/F is 22.57% overall. Missing-menu companions deliberately increase F.
- All controlled depth-2..10 examples use axis-split proofs. This exceeds the minimum nonhomogeneous requirement, but is not a balanced mixture of homogeneous, mixed-chain, and axis-split proofs.
- The current Symbolic references average 769.70 target tokens versus 705.24 Natural (about 9.1% longer). In training the means are 720.76 versus 664.15 (about 8.5% longer). Symbolic brevity is not established by these references.
- All 15 A-D subsets plus E/F occur for Listed and Exhaustive in train/dev/in-range test across all menus. This is presence coverage, not equal frequency.

- Bases: 10,800; unique stories: 6,695; unique families: 2,379.
- Views per style: 37,734 = 32,400 primary + 5,334 omission views.
- Bases with omission companions: 1,778. The seeded 20% applies only to eligible bases, not necessarily all bases.
- G1-G4 count selected substantive A-D letters. E and F are separate categories. Count value 4 is G1.
- Primary tables give one observation per base per contract. All-menu tables include correlated omission companions.
- The exact 900/3,600 multi-Selection target is overall, not per split/depth and not a balanced G2:G3:G4 target.

## Splits And Views

| split | bases | direction | which | count | stories | families | paired_omission_bases | views_per_style |
|---|---|---|---|---|---|---|---|---|
| train | 6000 | 2000 | 2000 | 2000 | 3644 | 1249 | 942 | 20826 |
| dev | 1200 | 400 | 400 | 400 | 735 | 210 | 194 | 4182 |
| in-range-test | 2400 | 800 | 800 | 800 | 1460 | 419 | 390 | 8370 |
| extrapolation-test | 1200 | 400 | 400 | 400 | 856 | 501 | 252 | 4356 |

## Sources

| split | controlled | random-world | random-atoms | targeted |
|---|---|---|---|---|
| train | 4200 | 900 | 600 | 300 |
| dev | 840 | 180 | 120 | 60 |
| in-range-test | 1680 | 360 | 240 | 120 |
| extrapolation-test | 1200 | 0 | 0 | 0 |

## Controlled Depth Allocation

| split | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | noncontrolled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 336 | 378 | 420 | 462 | 546 | 588 | 672 | 798 | 0 | 0 | 1800 |
| dev | 105 | 105 | 105 | 105 | 105 | 105 | 105 | 105 | 0 | 0 | 360 |
| in-range-test | 210 | 210 | 210 | 210 | 210 | 210 | 210 | 210 | 0 | 0 | 720 |
| extrapolation-test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 600 | 600 | 0 |

Depths above are requested controlled strata. Noncontrolled cases are a separate cohort, not all depth-null. Their measured depths appear below.

## Selection Entailed Cardinality

| split | bases | inconsistent | 0 | 1 | 2 | 3 | 4 | 5 | multi_2_4 |
|---|---|---|---|---|---|---|---|---|---|
| train | 2000 | 139 | 375 | 1018 | 190 | 147 | 120 | 11 | 457 |
| dev | 400 | 30 | 74 | 203 | 31 | 37 | 24 | 1 | 92 |
| in-range-test | 800 | 53 | 158 | 387 | 65 | 73 | 60 | 4 | 198 |
| extrapolation-test | 400 | 0 | 0 | 247 | 59 | 56 | 38 | 0 | 153 |

## Selection Cardinalities Above Four

| split | 5 |
|---|---|
| train | 11 |
| dev | 1 |
| in-range-test | 4 |
| extrapolation-test | 0 |

## Gold By Query And Contract (primary)

| query | contract | views | G1 | G2 | G3 | G4 | E | F |
|---|---|---|---|---|---|---|---|---|
| direction | unique-entailed | 3600 | 3091 | 0 | 0 | 0 | 509 | 0 |
| direction | listed-entailed | 3600 | 3091 | 0 | 0 | 0 | 197 | 312 |
| direction | exhaustive-entailed | 3600 | 3091 | 0 | 0 | 0 | 197 | 312 |
| which | unique-entailed | 3600 | 1855 | 0 | 0 | 0 | 1745 | 0 |
| which | listed-entailed | 3600 | 1855 | 345 | 313 | 258 | 222 | 607 |
| which | exhaustive-entailed | 3600 | 1855 | 345 | 313 | 242 | 222 | 623 |
| count | unique-entailed | 3600 | 2811 | 0 | 0 | 0 | 789 | 0 |
| count | listed-entailed | 3600 | 2811 | 0 | 0 | 0 | 211 | 578 |
| count | exhaustive-entailed | 3600 | 2811 | 0 | 0 | 0 | 211 | 578 |

## Gold By Split And Contract (primary)

| split | contract | views | G1 | G2 | G3 | G4 | E | F |
|---|---|---|---|---|---|---|---|---|
| train | unique-entailed | 6000 | 4191 | 0 | 0 | 0 | 1809 | 0 |
| train | listed-entailed | 6000 | 4191 | 190 | 147 | 131 | 393 | 948 |
| train | exhaustive-entailed | 6000 | 4191 | 190 | 147 | 120 | 393 | 959 |
| dev | unique-entailed | 1200 | 849 | 0 | 0 | 0 | 351 | 0 |
| dev | listed-entailed | 1200 | 849 | 31 | 37 | 25 | 85 | 173 |
| dev | exhaustive-entailed | 1200 | 849 | 31 | 37 | 24 | 85 | 174 |
| in-range-test | unique-entailed | 2400 | 1670 | 0 | 0 | 0 | 730 | 0 |
| in-range-test | listed-entailed | 2400 | 1670 | 65 | 73 | 64 | 152 | 376 |
| in-range-test | exhaustive-entailed | 2400 | 1670 | 65 | 73 | 60 | 152 | 380 |
| extrapolation-test | unique-entailed | 1200 | 1047 | 0 | 0 | 0 | 153 | 0 |
| extrapolation-test | listed-entailed | 1200 | 1047 | 59 | 56 | 38 | 0 | 0 |
| extrapolation-test | exhaustive-entailed | 1200 | 1047 | 59 | 56 | 38 | 0 | 0 |

## Selection Gold By Split (primary)

| split | contract | views | G1 | G2 | G3 | G4 | E | F |
|---|---|---|---|---|---|---|---|---|
| train | unique-entailed | 2000 | 1018 | 0 | 0 | 0 | 982 | 0 |
| train | listed-entailed | 2000 | 1018 | 190 | 147 | 131 | 139 | 375 |
| train | exhaustive-entailed | 2000 | 1018 | 190 | 147 | 120 | 139 | 386 |
| dev | unique-entailed | 400 | 203 | 0 | 0 | 0 | 197 | 0 |
| dev | listed-entailed | 400 | 203 | 31 | 37 | 25 | 30 | 74 |
| dev | exhaustive-entailed | 400 | 203 | 31 | 37 | 24 | 30 | 75 |
| in-range-test | unique-entailed | 800 | 387 | 0 | 0 | 0 | 413 | 0 |
| in-range-test | listed-entailed | 800 | 387 | 65 | 73 | 64 | 53 | 158 |
| in-range-test | exhaustive-entailed | 800 | 387 | 65 | 73 | 60 | 53 | 162 |
| extrapolation-test | unique-entailed | 400 | 247 | 0 | 0 | 0 | 153 | 0 |
| extrapolation-test | listed-entailed | 400 | 247 | 59 | 56 | 38 | 0 | 0 |
| extrapolation-test | exhaustive-entailed | 400 | 247 | 59 | 56 | 38 | 0 | 0 |

## Gold By Query And Contract (all-menus)

| query | contract | views | G1 | G2 | G3 | G4 | E | F |
|---|---|---|---|---|---|---|---|---|
| direction | unique-entailed | 4245 | 3091 | 0 | 0 | 0 | 509 | 645 |
| direction | listed-entailed | 4245 | 3091 | 0 | 0 | 0 | 197 | 957 |
| direction | exhaustive-entailed | 4245 | 3091 | 0 | 0 | 0 | 197 | 957 |
| which | unique-entailed | 4131 | 1855 | 0 | 0 | 0 | 1931 | 345 |
| which | listed-entailed | 4131 | 1925 | 414 | 360 | 258 | 222 | 952 |
| which | exhaustive-entailed | 4131 | 1855 | 345 | 313 | 242 | 222 | 1154 |
| count | unique-entailed | 4202 | 2811 | 0 | 0 | 0 | 789 | 602 |
| count | listed-entailed | 4202 | 2811 | 0 | 0 | 0 | 211 | 1180 |
| count | exhaustive-entailed | 4202 | 2811 | 0 | 0 | 0 | 211 | 1180 |

## Gold By Split And Contract (all-menus)

| split | contract | views | G1 | G2 | G3 | G4 | E | F |
|---|---|---|---|---|---|---|---|---|
| train | unique-entailed | 6942 | 4191 | 0 | 0 | 0 | 1907 | 844 |
| train | listed-entailed | 6942 | 4233 | 218 | 175 | 131 | 393 | 1792 |
| train | exhaustive-entailed | 6942 | 4191 | 190 | 147 | 120 | 393 | 1901 |
| dev | unique-entailed | 1394 | 849 | 0 | 0 | 0 | 369 | 176 |
| dev | listed-entailed | 1394 | 855 | 40 | 40 | 25 | 85 | 349 |
| dev | exhaustive-entailed | 1394 | 849 | 31 | 37 | 24 | 85 | 368 |
| in-range-test | unique-entailed | 2790 | 1670 | 0 | 0 | 0 | 772 | 348 |
| in-range-test | listed-entailed | 2790 | 1682 | 85 | 83 | 64 | 152 | 724 |
| in-range-test | exhaustive-entailed | 2790 | 1670 | 65 | 73 | 60 | 152 | 770 |
| extrapolation-test | unique-entailed | 1452 | 1047 | 0 | 0 | 0 | 181 | 224 |
| extrapolation-test | listed-entailed | 1452 | 1057 | 71 | 62 | 38 | 0 | 224 |
| extrapolation-test | exhaustive-entailed | 1452 | 1047 | 59 | 56 | 38 | 0 | 252 |

## Selection Gold By Split (all-menus)

| split | contract | views | G1 | G2 | G3 | G4 | E | F |
|---|---|---|---|---|---|---|---|---|
| train | unique-entailed | 2285 | 1018 | 0 | 0 | 0 | 1080 | 187 |
| train | listed-entailed | 2285 | 1060 | 218 | 175 | 131 | 139 | 562 |
| train | exhaustive-entailed | 2285 | 1018 | 190 | 147 | 120 | 139 | 671 |
| dev | unique-entailed | 459 | 203 | 0 | 0 | 0 | 215 | 41 |
| dev | listed-entailed | 459 | 209 | 40 | 40 | 25 | 30 | 115 |
| dev | exhaustive-entailed | 459 | 203 | 31 | 37 | 24 | 30 | 134 |
| in-range-test | unique-entailed | 912 | 387 | 0 | 0 | 0 | 455 | 70 |
| in-range-test | listed-entailed | 912 | 399 | 85 | 83 | 64 | 53 | 228 |
| in-range-test | exhaustive-entailed | 912 | 387 | 65 | 73 | 60 | 53 | 274 |
| extrapolation-test | unique-entailed | 475 | 247 | 0 | 0 | 0 | 181 | 47 |
| extrapolation-test | listed-entailed | 475 | 257 | 71 | 62 | 38 | 0 | 47 |
| extrapolation-test | exhaustive-entailed | 475 | 247 | 59 | 56 | 38 | 0 | 75 |

## Semantic Cases

| split | query | inconsistent | no-entailed-answer | unique-entailed | multiple-entailed |
|---|---|---|---|---|---|
| train | direction | 122 | 203 | 1675 | 0 |
| train | which | 139 | 375 | 1018 | 468 |
| train | count | 132 | 370 | 1498 | 0 |
| dev | direction | 26 | 39 | 335 | 0 |
| dev | which | 30 | 74 | 203 | 93 |
| dev | count | 29 | 60 | 311 | 0 |
| in-range-test | direction | 49 | 70 | 681 | 0 |
| in-range-test | which | 53 | 158 | 387 | 202 |
| in-range-test | count | 50 | 148 | 602 | 0 |
| extrapolation-test | direction | 0 | 0 | 400 | 0 |
| extrapolation-test | which | 0 | 0 | 247 | 153 |
| extrapolation-test | count | 0 | 0 | 400 | 0 |

## Controlled Count Values

| split | bases | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| train | 1400 | 284 | 281 | 279 | 279 | 277 |
| dev | 280 | 56 | 56 | 56 | 56 | 56 |
| in-range-test | 560 | 112 | 112 | 112 | 112 | 112 |
| extrapolation-test | 400 | 80 | 80 | 80 | 80 | 80 |

## Noncontrolled Count Values

| split | bases | inconsistent | no-entailed-count | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|---|
| train | 600 | 132 | 370 | 78 | 16 | 2 | 1 | 1 |
| dev | 120 | 29 | 60 | 18 | 10 | 3 | 0 | 0 |
| in-range-test | 240 | 50 | 148 | 33 | 8 | 1 | 0 | 0 |
| extrapolation-test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Measured Depths Including Noncontrolled

| split | query | None | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | direction | 325 | 291 | 219 | 143 | 154 | 182 | 196 | 224 | 266 | 0 | 0 |
| train | which | 590 | 112 | 134 | 141 | 155 | 182 | 196 | 224 | 266 | 0 | 0 |
| train | count | 503 | 117 | 177 | 175 | 160 | 182 | 196 | 224 | 266 | 0 | 0 |
| dev | direction | 65 | 68 | 54 | 38 | 35 | 35 | 35 | 35 | 35 | 0 | 0 |
| dev | which | 116 | 35 | 38 | 36 | 35 | 35 | 35 | 35 | 35 | 0 | 0 |
| dev | count | 90 | 36 | 53 | 45 | 36 | 35 | 35 | 35 | 35 | 0 | 0 |
| in-range-test | direction | 119 | 148 | 110 | 73 | 70 | 70 | 70 | 70 | 70 | 0 | 0 |
| in-range-test | which | 235 | 70 | 74 | 71 | 70 | 70 | 70 | 70 | 70 | 0 | 0 |
| in-range-test | count | 199 | 71 | 96 | 84 | 70 | 70 | 70 | 70 | 70 | 0 | 0 |
| extrapolation-test | direction | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 200 |
| extrapolation-test | which | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 200 |
| extrapolation-test | count | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 200 |

## Direction Coverage

| split | query | North | Northeast | East | Southeast | South | Southwest | West | Northwest | unresolved |
|---|---|---|---|---|---|---|---|---|---|---|
| train | direction | 175 | 227 | 176 | 256 | 177 | 245 | 168 | 251 | 325 |
| train | which | 220 | 262 | 207 | 273 | 201 | 244 | 198 | 256 | 139 |
| train | count | 195 | 268 | 203 | 271 | 188 | 267 | 202 | 274 | 132 |
| dev | direction | 39 | 48 | 29 | 51 | 37 | 44 | 39 | 48 | 65 |
| dev | which | 39 | 46 | 47 | 56 | 45 | 47 | 39 | 51 | 30 |
| dev | count | 41 | 50 | 52 | 53 | 54 | 44 | 36 | 41 | 29 |
| in-range-test | direction | 66 | 101 | 64 | 106 | 74 | 103 | 62 | 105 | 119 |
| in-range-test | which | 80 | 105 | 88 | 112 | 87 | 98 | 70 | 107 | 53 |
| in-range-test | count | 77 | 109 | 95 | 115 | 76 | 97 | 79 | 102 | 50 |
| extrapolation-test | direction | 39 | 61 | 51 | 61 | 36 | 59 | 36 | 57 | 0 |
| extrapolation-test | which | 40 | 58 | 49 | 68 | 35 | 56 | 37 | 57 | 0 |
| extrapolation-test | count | 39 | 68 | 37 | 58 | 42 | 56 | 41 | 59 | 0 |

For Direction, direction here is the entailed singleton; for Selection/Count it is the queried direction on consistent scenes. Unresolved includes inconsistent or non-singleton Direction cases.

## Family And Proof Diversity

| split | query | depth | bases | families | nonhomogeneous_bases |
|---|---|---|---|---|---|
| train | direction | 1 | 112 | 112 | 0 |
| train | direction | 2 | 126 | 126 | 126 |
| train | direction | 3 | 140 | 140 | 140 |
| train | direction | 4 | 154 | 154 | 154 |
| train | direction | 5 | 182 | 182 | 182 |
| train | direction | 6 | 196 | 196 | 196 |
| train | direction | 7 | 224 | 224 | 224 |
| train | direction | 8 | 266 | 266 | 266 |
| train | which | 1 | 112 | 112 | 0 |
| train | which | 2 | 126 | 126 | 126 |
| train | which | 3 | 140 | 140 | 140 |
| train | which | 4 | 154 | 154 | 154 |
| train | which | 5 | 182 | 182 | 182 |
| train | which | 6 | 196 | 196 | 196 |
| train | which | 7 | 224 | 224 | 224 |
| train | which | 8 | 266 | 266 | 266 |
| train | count | 1 | 112 | 112 | 0 |
| train | count | 2 | 126 | 126 | 126 |
| train | count | 3 | 140 | 140 | 140 |
| train | count | 4 | 154 | 154 | 154 |
| train | count | 5 | 182 | 182 | 182 |
| train | count | 6 | 196 | 196 | 196 |
| train | count | 7 | 224 | 224 | 224 |
| train | count | 8 | 266 | 266 | 266 |
| dev | direction | 1 | 35 | 35 | 0 |
| dev | direction | 2 | 35 | 35 | 35 |
| dev | direction | 3 | 35 | 35 | 35 |
| dev | direction | 4 | 35 | 35 | 35 |
| dev | direction | 5 | 35 | 35 | 35 |
| dev | direction | 6 | 35 | 35 | 35 |
| dev | direction | 7 | 35 | 35 | 35 |
| dev | direction | 8 | 35 | 35 | 35 |
| dev | which | 1 | 35 | 35 | 0 |
| dev | which | 2 | 35 | 35 | 35 |
| dev | which | 3 | 35 | 35 | 35 |
| dev | which | 4 | 35 | 35 | 35 |
| dev | which | 5 | 35 | 35 | 35 |
| dev | which | 6 | 35 | 35 | 35 |
| dev | which | 7 | 35 | 35 | 35 |
| dev | which | 8 | 35 | 35 | 35 |
| dev | count | 1 | 35 | 35 | 0 |
| dev | count | 2 | 35 | 35 | 35 |
| dev | count | 3 | 35 | 35 | 35 |
| dev | count | 4 | 35 | 35 | 35 |
| dev | count | 5 | 35 | 35 | 35 |
| dev | count | 6 | 35 | 35 | 35 |
| dev | count | 7 | 35 | 35 | 35 |
| dev | count | 8 | 35 | 35 | 35 |
| in-range-test | direction | 1 | 70 | 70 | 0 |
| in-range-test | direction | 2 | 70 | 70 | 70 |
| in-range-test | direction | 3 | 70 | 70 | 70 |
| in-range-test | direction | 4 | 70 | 70 | 70 |
| in-range-test | direction | 5 | 70 | 70 | 70 |
| in-range-test | direction | 6 | 70 | 70 | 70 |
| in-range-test | direction | 7 | 70 | 70 | 70 |
| in-range-test | direction | 8 | 70 | 70 | 70 |
| in-range-test | which | 1 | 70 | 70 | 0 |
| in-range-test | which | 2 | 70 | 70 | 70 |
| in-range-test | which | 3 | 70 | 70 | 70 |
| in-range-test | which | 4 | 70 | 70 | 70 |
| in-range-test | which | 5 | 70 | 70 | 70 |
| in-range-test | which | 6 | 70 | 70 | 70 |
| in-range-test | which | 7 | 70 | 70 | 70 |
| in-range-test | which | 8 | 70 | 70 | 70 |
| in-range-test | count | 1 | 70 | 70 | 0 |
| in-range-test | count | 2 | 70 | 70 | 70 |
| in-range-test | count | 3 | 70 | 70 | 70 |
| in-range-test | count | 4 | 70 | 70 | 70 |
| in-range-test | count | 5 | 70 | 70 | 70 |
| in-range-test | count | 6 | 70 | 70 | 70 |
| in-range-test | count | 7 | 70 | 70 | 70 |
| in-range-test | count | 8 | 70 | 70 | 70 |
| extrapolation-test | direction | 9 | 200 | 200 | 200 |
| extrapolation-test | direction | 10 | 200 | 200 | 200 |
| extrapolation-test | which | 9 | 200 | 200 | 200 |
| extrapolation-test | which | 10 | 200 | 200 | 200 |
| extrapolation-test | count | 9 | 200 | 200 | 200 |
| extrapolation-test | count | 10 | 200 | 200 | 200 |

## Full-Domain Negative Coverage

| split | query | contingent | impossible |
|---|---|---|---|
| train | direction | 203 | 1756 |
| train | which | 344 | 1843 |
| train | count | 370 | 1868 |
| dev | direction | 39 | 356 |
| dev | which | 55 | 368 |
| dev | count | 60 | 371 |
| in-range-test | direction | 70 | 713 |
| in-range-test | which | 136 | 741 |
| in-range-test | count | 148 | 750 |
| extrapolation-test | direction | 0 | 400 |
| extrapolation-test | which | 0 | 400 |
| extrapolation-test | count | 0 | 400 |

Negative categories count bases with at least one such full-domain non-entailment, and can overlap. They are diagnostic evidence, not positive supervised targets.

## Contract Differences On Any Matched Menu

| split | query | unique_listed_differ | unique_exhaustive_differ | listed_exhaustive_differ |
|---|---|---|---|---|
| train | direction | 203 | 203 | 0 |
| train | which | 843 | 843 | 109 |
| train | count | 370 | 370 | 0 |
| dev | direction | 39 | 39 | 0 |
| dev | which | 167 | 167 | 19 |
| dev | count | 60 | 60 | 0 |
| in-range-test | direction | 70 | 70 | 0 |
| in-range-test | which | 360 | 360 | 46 |
| in-range-test | count | 148 | 148 | 0 |
| extrapolation-test | direction | 0 | 0 | 0 |
| extrapolation-test | which | 153 | 153 | 28 |
| extrapolation-test | count | 0 | 0 | 0 |

## Reference Target Tokens

| split | style | n | min | mean | median | p95 | max | max_training_tokens |
|---|---|---|---|---|---|---|---|---|
| all | natural | 37734 | 85 | 705.24 | 436.0 | 2260 | 3037 | 3671 |
| all | symbolic | 37734 | 76 | 769.7 | 469.0 | 2508 | 3419 | 4059 |
| train | natural | 20826 | 85 | 664.15 | 415.0 | 2102 | 2879 | 3518 |
| train | symbolic | 20826 | 76 | 720.76 | 442.0 | 2327 | 3175 | 3835 |
| dev | natural | 4182 | 86 | 621.38 | 400.0 | 1977 | 2782 | 3417 |
| dev | symbolic | 4182 | 76 | 669.63 | 421.0 | 2167 | 3076 | 3713 |
| in-range-test | natural | 8370 | 86 | 621.86 | 405.0 | 1968 | 2827 | 3467 |
| in-range-test | symbolic | 8370 | 76 | 670.11 | 424.0 | 2176 | 3125 | 3758 |
| extrapolation-test | natural | 4356 | 150 | 1142.4 | 723.0 | 2700 | 3037 | 3671 |
| extrapolation-test | symbolic | 4356 | 153 | 1291.12 | 862.5 | 3006 | 3419 | 4059 |

## Reference Tokens By Requested Depth

| requested_depth | style | n | min | mean | median | p95 | max |
|---|---|---|---|---|---|---|---|
| None | natural | 9129 | 85 | 250.52 | 225 | 432 | 779 |
| None | symbolic | 9129 | 76 | 248.77 | 219 | 445 | 962 |
| 1 | natural | 2301 | 85 | 370.02 | 311 | 759 | 950 |
| 1 | symbolic | 2301 | 76 | 374.21 | 312 | 756 | 960 |
| 2 | natural | 2511 | 147 | 584.67 | 421 | 1285 | 1573 |
| 2 | symbolic | 2511 | 153 | 615.33 | 453 | 1323 | 1645 |
| 3 | natural | 2652 | 150 | 623.42 | 452.0 | 1383 | 1676 |
| 3 | symbolic | 2652 | 153 | 662.67 | 504.5 | 1435 | 1772 |
| 4 | natural | 2859 | 150 | 714.51 | 488 | 1658 | 2341 |
| 4 | symbolic | 2859 | 153 | 768.26 | 563 | 1746 | 2448 |
| 5 | natural | 3111 | 151 | 802.36 | 506 | 1911 | 2435 |
| 5 | symbolic | 3111 | 153 | 873.7 | 613 | 2036 | 2695 |
| 6 | natural | 3267 | 150 | 914.2 | 557 | 2295 | 2743 |
| 6 | symbolic | 3267 | 153 | 1008.61 | 682 | 2465 | 2978 |
| 7 | natural | 3549 | 150 | 998.45 | 614 | 2474 | 2793 |
| 7 | symbolic | 3549 | 153 | 1111.44 | 727 | 2695 | 3085 |
| 8 | natural | 3999 | 150 | 1076.83 | 675 | 2598 | 2879 |
| 8 | symbolic | 3999 | 153 | 1208.07 | 797 | 2859 | 3175 |
| 9 | natural | 2172 | 150 | 1130.42 | 732.0 | 2659 | 2948 |
| 9 | symbolic | 2172 | 153 | 1273.58 | 865.5 | 2953 | 3298 |
| 10 | natural | 2184 | 150 | 1154.31 | 653.5 | 2724 | 3037 |
| 10 | symbolic | 2184 | 153 | 1308.58 | 833.5 | 3034 | 3419 |

## Named Objects Per Base

| split | n | min | mean | median | p95 | max |
|---|---|---|---|---|---|---|
| train | 6000 | 5 | 15.3 | 15.0 | 26 | 26 |
| dev | 1200 | 5 | 14.63 | 14.0 | 26 | 26 |
| in-range-test | 2400 | 5 | 14.5 | 14.0 | 26 | 26 |
| extrapolation-test | 1200 | 15 | 24.84 | 26.0 | 26 | 26 |

## Limits

These are stored verified-reference token counts, not model-generated lengths or training measurements. Premise counts and full prose are not retained in this compact metadata, so they are not reconstructed or guessed here. Final prose review and artifact export remain separate.
