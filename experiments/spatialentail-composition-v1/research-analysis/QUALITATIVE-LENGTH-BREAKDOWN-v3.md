# Qualitative Explanation Length Breakdown

## Population and Definitions

Source: `depth-feasibility-qualitative-v3`, verified by saved output hashes. 94 distinct stories, 282 base questions, 987 paired menu/contract views: 987 Natural and 987 Symbolic responses. Each base question can have more than one menu and three contracts. Counts of bases in different contract groups overlap; they must not be summed across contracts.

Target tokens include final answer; exclude prompt, chat end marker, and audit metadata. Means weight contract/menu views equally. Token lengths use the pinned Qwen3.5-4B tokenizer. This is the feasibility pilot, not the final training or benchmark distribution. No data was generated or changed for this analysis.

Depth is the measured supported premise-path depth attached to the base question, not the number of sentences, minimum human reasoning steps, or necessarily the evidence length required for each menu-specific answer. A missing-menu F view can use one possibility even when its base has a deep invariant proof. Unassigned cases are not depth zero. Their ambiguity, inconsistency or other unsupported depth conditions must remain a separate cohort.

## Overall

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 282 | 987 | 289.8 | 73 | 868 | 244.9 | 61 | 744 |

Natural median: 282; p95: 568. Symbolic median: 233; p95: 474. Giving each base equal weight after averaging its views yields means 290.23 Natural / 245.03 Symbolic.

## Measured Depth

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37 | 126 | 191.0 | 73 | 401 | 156.5 | 61 | 334 |
| 2 | 27 | 93 | 246.2 | 79 | 473 | 203.6 | 71 | 388 |
| 3 | 19 | 69 | 252.6 | 74 | 484 | 218.9 | 61 | 435 |
| 4 | 18 | 75 | 280.4 | 76 | 568 | 234.4 | 61 | 448 |
| 5 | 18 | 57 | 346.6 | 74 | 564 | 265.6 | 78 | 441 |
| 6 | 18 | 75 | 345.8 | 78 | 618 | 283.8 | 64 | 470 |
| 7 | 18 | 69 | 318.8 | 73 | 576 | 292.7 | 61 | 500 |
| 8 | 18 | 60 | 427.9 | 77 | 669 | 317.4 | 61 | 534 |
| 9 | 18 | 66 | 372.7 | 80 | 638 | 333.3 | 67 | 576 |
| 10 | 18 | 66 | 374.1 | 81 | 658 | 344.9 | 67 | 534 |
| Unassigned | 73 | 231 | 251.1 | 73 | 868 | 216.2 | 61 | 744 |

## Question Type

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direction | 94 | 333 | 176.1 | 73 | 405 | 144.1 | 61 | 269 |
| Selection | 94 | 315 | 308.3 | 74 | 686 | 260.4 | 69 | 696 |
| Count | 94 | 339 | 384.5 | 74 | 868 | 329.5 | 69 | 744 |

## Answer Contract

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unique | 282 | 329 | 301.3 | 82 | 686 | 254.9 | 69 | 696 |
| Listed | 282 | 329 | 300.8 | 82 | 865 | 253.5 | 69 | 731 |
| Exhaustive | 282 | 329 | 267.4 | 73 | 868 | 226.3 | 61 | 744 |

Unique asks for an invariant single answer; Listed returns displayed possible values; Exhaustive requires complete coverage of possible values. Each contract has 329 views over the same 282 bases. Style-paired rows share questions and gold.

## Type and Contract

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direction / Unique | 94 | 111 | 179.7 | 82 | 404 | 146.7 | 69 | 269 |
| Direction / Listed | 94 | 111 | 191.3 | 82 | 405 | 155.4 | 69 | 264 |
| Direction / Exhaustive | 94 | 111 | 157.3 | 73 | 404 | 130.0 | 61 | 262 |
| Selection / Unique | 94 | 105 | 351.0 | 82 | 686 | 295.7 | 69 | 696 |
| Selection / Listed | 94 | 105 | 293.5 | 82 | 546 | 248.6 | 69 | 455 |
| Selection / Exhaustive | 94 | 105 | 280.3 | 74 | 584 | 236.9 | 69 | 541 |
| Count / Unique | 94 | 113 | 374.6 | 82 | 669 | 323.2 | 69 | 576 |
| Count / Listed | 94 | 113 | 415.2 | 82 | 865 | 354.4 | 69 | 731 |
| Count / Exhaustive | 94 | 113 | 363.6 | 74 | 868 | 311.1 | 69 | 744 |

## Depth and Type

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / Direction | 17 | 57 | 92.1 | 73 | 100 | 74.6 | 61 | 82 |
| 1 / Selection | 10 | 33 | 216.8 | 75 | 284 | 175.2 | 79 | 228 |
| 1 / Count | 10 | 36 | 324.1 | 79 | 401 | 269.1 | 72 | 334 |
| 2 / Direction | 9 | 33 | 138.8 | 86 | 164 | 109.3 | 71 | 119 |
| 2 / Selection | 9 | 27 | 244.8 | 81 | 347 | 203.4 | 91 | 271 |
| 2 / Count | 9 | 33 | 354.6 | 79 | 473 | 298.0 | 72 | 388 |
| 3 / Direction | 7 | 30 | 144.9 | 74 | 184 | 123.7 | 61 | 136 |
| 3 / Selection | 6 | 18 | 266.2 | 84 | 364 | 228.9 | 98 | 310 |
| 3 / Count | 6 | 21 | 394.7 | 85 | 484 | 346.2 | 88 | 435 |
| 4 / Direction | 6 | 27 | 170.1 | 76 | 224 | 140.3 | 61 | 161 |
| 4 / Selection | 6 | 21 | 273.7 | 78 | 428 | 227.6 | 82 | 317 |
| 4 / Count | 6 | 27 | 395.9 | 79 | 568 | 333.8 | 82 | 448 |
| 5 / Direction | 6 | 18 | 232.3 | 158 | 256 | 165.3 | 159 | 172 |
| 5 / Selection | 6 | 21 | 327.3 | 74 | 441 | 248.9 | 78 | 326 |
| 5 / Count | 6 | 18 | 483.5 | 392 | 564 | 385.5 | 311 | 441 |
| 6 / Direction | 6 | 24 | 227.6 | 80 | 294 | 177.2 | 64 | 196 |
| 6 / Selection | 6 | 24 | 343.8 | 78 | 499 | 278.2 | 85 | 359 |
| 6 / Count | 6 | 27 | 452.6 | 84 | 618 | 383.6 | 84 | 470 |
| 7 / Direction | 6 | 24 | 195.8 | 73 | 309 | 191.3 | 61 | 211 |
| 7 / Selection | 6 | 24 | 308.4 | 76 | 459 | 274.9 | 80 | 378 |
| 7 / Count | 6 | 21 | 471.4 | 84 | 576 | 428.9 | 84 | 500 |
| 8 / Direction | 6 | 21 | 308.5 | 77 | 346 | 213.4 | 61 | 225 |
| 8 / Selection | 6 | 18 | 422.3 | 87 | 550 | 309.3 | 101 | 406 |
| 8 / Count | 6 | 21 | 552.1 | 85 | 669 | 428.5 | 85 | 534 |
| 9 / Direction | 6 | 21 | 271.0 | 83 | 383 | 233.2 | 67 | 253 |
| 9 / Selection | 6 | 21 | 345.7 | 80 | 511 | 309.1 | 87 | 434 |
| 9 / Count | 6 | 24 | 485.4 | 83 | 638 | 442.2 | 83 | 576 |
| 10 / Direction | 6 | 21 | 273.9 | 83 | 405 | 251.1 | 67 | 269 |
| 10 / Selection | 6 | 21 | 363.2 | 81 | 537 | 334.8 | 85 | 417 |
| 10 / Count | 6 | 24 | 471.4 | 81 | 658 | 435.8 | 88 | 534 |
| Unassigned / Direction | 19 | 57 | 133.4 | 73 | 261 | 107.7 | 61 | 208 |
| Unassigned / Selection | 27 | 87 | 319.4 | 77 | 686 | 278.9 | 69 | 696 |
| Unassigned / Count | 27 | 87 | 259.9 | 74 | 868 | 224.6 | 69 | 744 |

## Depth and Contract

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / Unique | 37 | 42 | 199.6 | 92 | 401 | 163.4 | 74 | 334 |
| 1 / Listed | 37 | 42 | 189.8 | 93 | 396 | 155.1 | 75 | 329 |
| 1 / Exhaustive | 37 | 42 | 183.7 | 73 | 401 | 151.1 | 61 | 334 |
| 2 / Unique | 27 | 31 | 260.9 | 118 | 473 | 215.2 | 110 | 388 |
| 2 / Listed | 27 | 31 | 246.7 | 119 | 468 | 203.3 | 111 | 383 |
| 2 / Exhaustive | 27 | 31 | 230.8 | 79 | 473 | 192.3 | 71 | 388 |
| 3 / Unique | 19 | 23 | 269.3 | 130 | 484 | 233.4 | 126 | 435 |
| 3 / Listed | 19 | 23 | 254.8 | 131 | 479 | 220.9 | 127 | 430 |
| 3 / Exhaustive | 19 | 23 | 233.5 | 74 | 484 | 202.3 | 61 | 435 |
| 4 / Unique | 18 | 25 | 315.0 | 144 | 568 | 262.9 | 144 | 448 |
| 4 / Listed | 18 | 25 | 291.4 | 145 | 555 | 240.6 | 142 | 435 |
| 4 / Exhaustive | 18 | 25 | 234.9 | 76 | 560 | 199.8 | 61 | 440 |
| 5 / Unique | 18 | 19 | 361.3 | 158 | 564 | 275.7 | 159 | 441 |
| 5 / Listed | 18 | 19 | 333.2 | 159 | 559 | 255.7 | 138 | 436 |
| 5 / Exhaustive | 18 | 19 | 345.4 | 74 | 564 | 265.5 | 78 | 441 |
| 6 / Unique | 18 | 25 | 390.1 | 168 | 618 | 316.9 | 176 | 470 |
| 6 / Listed | 18 | 25 | 351.8 | 169 | 613 | 287.5 | 153 | 465 |
| 6 / Exhaustive | 18 | 25 | 295.6 | 78 | 618 | 247.0 | 64 | 470 |
| 7 / Unique | 18 | 23 | 353.9 | 186 | 576 | 323.9 | 198 | 500 |
| 7 / Listed | 18 | 23 | 316.7 | 176 | 571 | 291.0 | 145 | 495 |
| 7 / Exhaustive | 18 | 23 | 285.9 | 73 | 576 | 263.2 | 61 | 500 |
| 8 / Unique | 18 | 20 | 452.4 | 204 | 669 | 336.1 | 217 | 534 |
| 8 / Listed | 18 | 20 | 435.5 | 205 | 664 | 321.8 | 209 | 529 |
| 8 / Exhaustive | 18 | 20 | 395.9 | 77 | 669 | 294.4 | 61 | 534 |
| 9 / Unique | 18 | 22 | 410.0 | 214 | 638 | 367.5 | 234 | 576 |
| 9 / Listed | 18 | 22 | 382.1 | 182 | 633 | 340.0 | 150 | 563 |
| 9 / Exhaustive | 18 | 22 | 326.1 | 80 | 638 | 292.6 | 67 | 568 |
| 10 / Unique | 18 | 22 | 403.6 | 232 | 658 | 372.7 | 256 | 534 |
| 10 / Listed | 18 | 22 | 376.5 | 192 | 653 | 345.9 | 157 | 529 |
| 10 / Exhaustive | 18 | 22 | 342.3 | 81 | 658 | 316.1 | 67 | 534 |
| Unassigned / Unique | 73 | 77 | 219.3 | 82 | 686 | 191.7 | 69 | 696 |
| Unassigned / Listed | 73 | 77 | 290.9 | 82 | 865 | 249.7 | 69 | 731 |
| Unassigned / Exhaustive | 73 | 77 | 243.0 | 73 | 868 | 207.2 | 61 | 744 |

## Depth, Type and Contract

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / Direction / Unique | 17 | 19 | 92.8 | 92 | 100 | 75.2 | 74 | 82 |
| 1 / Direction / Listed | 17 | 19 | 93.4 | 93 | 97 | 75.7 | 75 | 78 |
| 1 / Direction / Exhaustive | 17 | 19 | 90.1 | 73 | 92 | 73.0 | 61 | 75 |
| 1 / Selection / Unique | 10 | 11 | 229.1 | 173 | 278 | 184.2 | 141 | 222 |
| 1 / Selection / Listed | 10 | 11 | 204.9 | 171 | 246 | 165.7 | 138 | 199 |
| 1 / Selection / Exhaustive | 10 | 11 | 216.4 | 75 | 284 | 175.5 | 79 | 228 |
| 1 / Count / Unique | 10 | 12 | 341.8 | 279 | 401 | 283.9 | 228 | 334 |
| 1 / Count / Listed | 10 | 12 | 328.5 | 230 | 396 | 271.2 | 186 | 329 |
| 1 / Count / Exhaustive | 10 | 12 | 301.9 | 79 | 401 | 252.2 | 72 | 334 |
| 2 / Direction / Unique | 9 | 11 | 142.9 | 118 | 164 | 112.3 | 110 | 119 |
| 2 / Direction / Listed | 9 | 11 | 143.2 | 119 | 161 | 112.2 | 111 | 114 |
| 2 / Direction / Exhaustive | 9 | 11 | 130.5 | 86 | 156 | 103.5 | 71 | 111 |
| 2 / Selection / Unique | 9 | 9 | 265.2 | 197 | 347 | 218.3 | 159 | 271 |
| 2 / Selection / Listed | 9 | 9 | 233.3 | 195 | 298 | 194.0 | 156 | 235 |
| 2 / Selection / Exhaustive | 9 | 9 | 235.8 | 81 | 336 | 197.8 | 91 | 264 |
| 2 / Count / Unique | 9 | 11 | 375.5 | 303 | 473 | 315.5 | 246 | 388 |
| 2 / Count / Listed | 9 | 11 | 361.3 | 262 | 468 | 302.0 | 204 | 383 |
| 2 / Count / Exhaustive | 9 | 11 | 327.0 | 79 | 473 | 276.5 | 72 | 388 |
| 3 / Direction / Unique | 7 | 10 | 152.8 | 130 | 183 | 131.3 | 126 | 135 |
| 3 / Direction / Listed | 7 | 10 | 152.6 | 131 | 184 | 130.6 | 127 | 136 |
| 3 / Direction / Exhaustive | 7 | 10 | 129.4 | 74 | 183 | 109.2 | 61 | 135 |
| 3 / Selection / Unique | 6 | 6 | 294.0 | 237 | 357 | 250.7 | 197 | 310 |
| 3 / Selection / Listed | 6 | 6 | 254.7 | 197 | 326 | 220.2 | 164 | 272 |
| 3 / Selection / Exhaustive | 6 | 6 | 249.8 | 84 | 364 | 215.8 | 98 | 289 |
| 3 / Count / Unique | 6 | 7 | 414.7 | 347 | 484 | 364.6 | 298 | 435 |
| 3 / Count / Listed | 6 | 7 | 401.0 | 342 | 479 | 350.4 | 293 | 430 |
| 3 / Count / Exhaustive | 6 | 7 | 368.3 | 85 | 484 | 323.7 | 88 | 435 |
| 4 / Direction / Unique | 6 | 9 | 183.1 | 144 | 224 | 151.0 | 144 | 161 |
| 4 / Direction / Listed | 6 | 9 | 182.8 | 145 | 223 | 150.1 | 145 | 157 |
| 4 / Direction / Exhaustive | 6 | 9 | 144.6 | 76 | 222 | 119.8 | 61 | 153 |
| 4 / Selection / Unique | 6 | 7 | 317.6 | 251 | 428 | 262.4 | 215 | 317 |
| 4 / Selection / Listed | 6 | 7 | 266.4 | 174 | 381 | 217.6 | 142 | 283 |
| 4 / Selection / Exhaustive | 6 | 7 | 237.1 | 78 | 368 | 202.9 | 82 | 266 |
| 4 / Count / Unique | 6 | 9 | 444.8 | 365 | 568 | 375.2 | 320 | 448 |
| 4 / Count / Listed | 6 | 9 | 419.4 | 313 | 555 | 348.9 | 265 | 435 |
| 4 / Count / Exhaustive | 6 | 9 | 323.6 | 79 | 560 | 277.3 | 82 | 440 |
| 5 / Direction / Unique | 6 | 6 | 232.0 | 158 | 255 | 165.0 | 159 | 171 |
| 5 / Direction / Listed | 6 | 6 | 233.0 | 159 | 256 | 166.0 | 160 | 172 |
| 5 / Direction / Exhaustive | 6 | 6 | 232.0 | 158 | 255 | 165.0 | 159 | 171 |
| 5 / Selection / Unique | 6 | 7 | 365.9 | 287 | 437 | 275.0 | 215 | 322 |
| 5 / Selection / Listed | 6 | 7 | 293.1 | 171 | 400 | 224.3 | 138 | 294 |
| 5 / Selection / Exhaustive | 6 | 7 | 322.9 | 74 | 441 | 247.4 | 78 | 326 |
| 5 / Count / Unique | 6 | 6 | 485.2 | 397 | 564 | 387.2 | 316 | 441 |
| 5 / Count / Listed | 6 | 6 | 480.2 | 392 | 559 | 382.2 | 311 | 436 |
| 5 / Count / Exhaustive | 6 | 6 | 485.2 | 397 | 564 | 387.2 | 316 | 441 |
| 6 / Direction / Unique | 6 | 8 | 245.1 | 168 | 293 | 188.2 | 176 | 196 |
| 6 / Direction / Listed | 6 | 8 | 245.1 | 169 | 294 | 187.8 | 177 | 193 |
| 6 / Direction / Exhaustive | 6 | 8 | 192.6 | 80 | 293 | 155.5 | 64 | 192 |
| 6 / Selection / Unique | 6 | 8 | 404.4 | 306 | 493 | 318.2 | 290 | 353 |
| 6 / Selection / Listed | 6 | 8 | 313.6 | 184 | 458 | 256.2 | 153 | 327 |
| 6 / Selection / Exhaustive | 6 | 8 | 313.5 | 78 | 499 | 260.2 | 85 | 359 |
| 6 / Count / Unique | 6 | 9 | 506.2 | 429 | 618 | 430.1 | 402 | 470 |
| 6 / Count / Listed | 6 | 9 | 480.4 | 372 | 613 | 404.0 | 345 | 465 |
| 6 / Count / Exhaustive | 6 | 9 | 371.1 | 84 | 618 | 316.6 | 84 | 470 |
| 7 / Direction / Unique | 6 | 8 | 206.0 | 186 | 308 | 203.8 | 198 | 211 |
| 7 / Direction / Listed | 6 | 8 | 206.0 | 187 | 309 | 203.5 | 199 | 207 |
| 7 / Direction / Exhaustive | 6 | 8 | 175.2 | 73 | 308 | 166.8 | 61 | 203 |
| 7 / Selection / Unique | 6 | 8 | 379.8 | 321 | 459 | 333.8 | 309 | 378 |
| 7 / Selection / Listed | 6 | 8 | 284.6 | 176 | 411 | 251.5 | 145 | 332 |
| 7 / Selection / Exhaustive | 6 | 8 | 260.9 | 76 | 451 | 239.5 | 80 | 364 |
| 7 / Count / Unique | 6 | 7 | 493.4 | 444 | 576 | 449.9 | 421 | 500 |
| 7 / Count / Listed | 6 | 7 | 479.9 | 387 | 571 | 436.3 | 364 | 495 |
| 7 / Count / Exhaustive | 6 | 7 | 440.9 | 84 | 576 | 400.6 | 84 | 500 |
| 8 / Direction / Unique | 6 | 7 | 321.1 | 204 | 346 | 221.1 | 217 | 225 |
| 8 / Direction / Listed | 6 | 7 | 321.6 | 205 | 344 | 221.3 | 218 | 225 |
| 8 / Direction / Exhaustive | 6 | 7 | 282.7 | 77 | 343 | 197.7 | 61 | 224 |
| 8 / Selection / Unique | 6 | 6 | 454.7 | 336 | 544 | 336.8 | 243 | 406 |
| 8 / Selection / Listed | 6 | 6 | 414.0 | 295 | 510 | 304.7 | 209 | 362 |
| 8 / Selection / Exhaustive | 6 | 6 | 398.3 | 87 | 550 | 286.3 | 101 | 392 |
| 8 / Count / Unique | 6 | 7 | 581.6 | 446 | 669 | 450.6 | 344 | 534 |
| 8 / Count / Listed | 6 | 7 | 567.9 | 441 | 664 | 436.9 | 339 | 529 |
| 8 / Count / Exhaustive | 6 | 7 | 506.9 | 85 | 669 | 398.0 | 85 | 534 |
| 9 / Direction / Unique | 6 | 7 | 285.1 | 214 | 383 | 242.0 | 234 | 253 |
| 9 / Direction / Listed | 6 | 7 | 285.6 | 215 | 380 | 242.1 | 235 | 248 |
| 9 / Direction / Exhaustive | 6 | 7 | 242.3 | 83 | 375 | 215.4 | 67 | 245 |
| 9 / Selection / Unique | 6 | 7 | 397.9 | 347 | 511 | 358.6 | 276 | 434 |
| 9 / Selection / Listed | 6 | 7 | 334.6 | 182 | 471 | 298.3 | 150 | 392 |
| 9 / Selection / Exhaustive | 6 | 7 | 304.6 | 80 | 511 | 270.6 | 87 | 366 |
| 9 / Count / Unique | 6 | 8 | 529.8 | 468 | 638 | 485.0 | 388 | 576 |
| 9 / Count / Listed | 6 | 8 | 508.1 | 412 | 633 | 462.1 | 383 | 563 |
| 9 / Count / Exhaustive | 6 | 8 | 418.2 | 83 | 638 | 379.4 | 83 | 568 |
| 10 / Direction / Unique | 6 | 7 | 285.7 | 232 | 404 | 260.7 | 256 | 269 |
| 10 / Direction / Listed | 6 | 7 | 286.1 | 233 | 405 | 260.9 | 257 | 264 |
| 10 / Direction / Exhaustive | 6 | 7 | 249.9 | 83 | 404 | 231.9 | 67 | 262 |
| 10 / Selection / Unique | 6 | 7 | 399.0 | 305 | 537 | 369.6 | 282 | 415 |
| 10 / Selection / Listed | 6 | 7 | 336.3 | 192 | 500 | 309.1 | 157 | 385 |
| 10 / Selection / Exhaustive | 6 | 7 | 354.3 | 81 | 537 | 325.6 | 85 | 417 |
| 10 / Count / Unique | 6 | 8 | 510.8 | 422 | 658 | 473.4 | 394 | 534 |
| 10 / Count / Listed | 6 | 8 | 490.6 | 368 | 653 | 452.4 | 333 | 529 |
| 10 / Count / Exhaustive | 6 | 8 | 412.8 | 81 | 658 | 381.5 | 88 | 534 |
| Unassigned / Direction / Unique | 19 | 19 | 115.3 | 82 | 144 | 92.5 | 69 | 113 |
| Unassigned / Direction / Listed | 19 | 19 | 181.8 | 82 | 261 | 143.5 | 69 | 208 |
| Unassigned / Direction / Exhaustive | 19 | 19 | 103.1 | 73 | 251 | 87.0 | 61 | 205 |
| Unassigned / Selection / Unique | 27 | 29 | 373.1 | 82 | 686 | 326.1 | 69 | 696 |
| Unassigned / Selection / Listed | 27 | 29 | 312.3 | 82 | 546 | 275.1 | 69 | 455 |
| Unassigned / Selection / Exhaustive | 27 | 29 | 272.9 | 77 | 584 | 235.5 | 69 | 541 |
| Unassigned / Count / Unique | 27 | 29 | 133.7 | 82 | 170 | 122.3 | 69 | 158 |
| Unassigned / Count / Listed | 27 | 29 | 341.0 | 82 | 865 | 293.8 | 69 | 731 |
| Unassigned / Count / Exhaustive | 27 | 29 | 304.9 | 74 | 868 | 257.7 | 69 | 744 |

## Semantic Cases

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inconsistent | 18 | 54 | 89.5 | 82 | 96 | 78.2 | 69 | 108 |
| multiple-certain-match | 30 | 99 | 337.6 | 81 | 584 | 284.7 | 85 | 447 |
| no-match | 10 | 30 | 246.6 | 171 | 368 | 201.6 | 137 | 282 |
| underdetermined | 51 | 162 | 289.2 | 73 | 868 | 250.6 | 61 | 744 |
| unique | 173 | 642 | 301.5 | 73 | 669 | 253.4 | 61 | 576 |

## Gold Distribution

| Contract | Views | Substantive A-D sets | E | F | E/F rate |
|---|---:|---:|---:|---:|---:|
| Unique | 329 | 173 | 115 | 41 | 47.4% |
| Exhaustive | 329 | 229 | 18 | 82 | 30.4% |
| Listed | 329 | 260 | 18 | 51 | 21.0% |

These counts describe the existing pilot, not approved production quotas. E/F are standalone escapes and are not substantive one-element gold sets.

| Contract | Answer | Views |
|---|---|---:|
| Unique | A | 35 |
| Exhaustive | A | 38 |
| Listed | A | 38 |
| Unique | B | 41 |
| Exhaustive | B | 41 |
| Listed | B | 41 |
| Unique | D | 46 |
| Exhaustive | D | 47 |
| Listed | D | 49 |
| Unique | E | 115 |
| Exhaustive | AC | 5 |
| Listed | AC | 5 |
| Unique | C | 51 |
| Exhaustive | C | 53 |
| Listed | C | 53 |
| Unique | F | 41 |
| Exhaustive | F | 82 |
| Listed | F | 51 |
| Exhaustive | ACD | 3 |
| Listed | ACD | 3 |
| Exhaustive | ABCD | 15 |
| Listed | ABCD | 40 |
| Exhaustive | ABD | 6 |
| Listed | ABD | 6 |
| Exhaustive | AD | 4 |
| Listed | AD | 4 |
| Exhaustive | BCD | 4 |
| Listed | BCD | 7 |
| Listed | CD | 6 |
| Exhaustive | CD | 5 |
| Exhaustive | BD | 3 |
| Listed | BD | 3 |
| Exhaustive | ABC | 3 |
| Listed | ABC | 3 |
| Exhaustive | BC | 1 |
| Listed | BC | 1 |
| Exhaustive | E | 18 |
| Listed | E | 18 |
| Exhaustive | AB | 1 |
| Listed | AB | 1 |

## Source Recipe

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 3 | 12 | 219.6 | 79 | 319 | 172.5 | 72 | 254 |
| coincidence | 3 | 9 | 95.0 | 95 | 95 | 108.0 | 108 | 108 |
| contingent | 3 | 12 | 195.8 | 74 | 289 | 169.8 | 71 | 248 |
| controlled-depth | 180 | 654 | 320.8 | 73 | 669 | 270.5 | 61 | 576 |
| cycle | 6 | 18 | 156.9 | 84 | 354 | 128.3 | 69 | 293 |
| empty | 3 | 15 | 166.3 | 74 | 287 | 135.7 | 61 | 236 |
| fork | 3 | 9 | 217.0 | 131 | 303 | 216.4 | 113 | 300 |
| joint | 3 | 9 | 370.9 | 74 | 868 | 318.9 | 61 | 733 |
| multiple | 3 | 12 | 194.9 | 73 | 354 | 159.2 | 61 | 293 |
| random-atoms | 36 | 111 | 201.4 | 74 | 681 | 169.3 | 61 | 696 |
| random-world | 36 | 117 | 285.3 | 73 | 825 | 243.4 | 61 | 744 |
| singleton | 3 | 9 | 193.3 | 92 | 304 | 154.3 | 74 | 241 |

## Split

| Group | Bases | Views | Natural mean | Min | Max | Symbolic mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev-pilot | 69 | 246 | 282.1 | 73 | 608 | 242.1 | 61 | 534 |
| extrapolation-pilot | 36 | 132 | 373.4 | 80 | 658 | 339.1 | 67 | 576 |
| in-range-test-pilot | 78 | 279 | 263.0 | 74 | 618 | 210.4 | 61 | 506 |
| train-pilot | 99 | 330 | 284.9 | 73 | 868 | 238.5 | 61 | 744 |

## Global Extremes

### natural min: 73 tokens

ID: `depth-7-sample-1-direction-m1-all-possible`; Direction; Exhaustive; depth: 7; semantic case: unique; answer: F. All tied IDs are retained in the JSON.

#### Natural

Case 1: One possibility consistent with all premises is the following. From left to right, the order is Object8 and Object5. Object5 and Object8 share the same height. So Object5 is east of Object8.

The witnessed value East is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F

#### Symbolic

```text
Case 1 (possible together under all premises): x(Object8) < x(Object5). y(Object5) = y(Object8). => E(Object5,Object8).

The witnessed value East is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F
```

### natural max: 868 tokens

ID: `joint-0-base-count-m0-all-possible`; Count; Exhaustive; depth: Unassigned; semantic case: underdetermined; answer: AD. All tied IDs are retained in the JSON.

#### Natural

Suppose Object1 and Object2 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object1 has a smaller y-coordinate than Object2; Object2 shares its y-coordinate with Object0; Object0 shares its y-coordinate with Object1 (premise sources: 2, temporary assumption 2 and temporary assumption 1).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object1 and Object2 cannot qualify together.

Suppose Object1 and Object3 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object2 has a smaller y-coordinate than Object4; Object4 has a smaller y-coordinate than Object3; Object3 shares its y-coordinate with Object0; Object0 shares its y-coordinate with Object1; Object1 has a smaller y-coordinate than Object2 (premise sources: 1, 3, temporary assumption 2, temporary assumption 1 and 2).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object1 and Object3 cannot qualify together.

Suppose Object1 and Object4 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object2 has a smaller y-coordinate than Object4; Object4 shares its y-coordinate with Object0; Object0 shares its y-coordinate with Object1; Object1 has a smaller y-coordinate than Object2 (premise sources: 1, temporary assumption 2, temporary assumption 1 and 2).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object1 and Object4 cannot qualify together.

Suppose Object2 and Object3 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object2 has a smaller y-coordinate than Object4; Object4 has a smaller y-coordinate than Object3; Object3 shares its y-coordinate with Object0; Object0 shares its y-coordinate with Object2 (premise sources: 1, 3, temporary assumption 2 and temporary assumption 1).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object2 and Object3 cannot qualify together.

Suppose Object2 and Object4 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object2 has a smaller y-coordinate than Object4; Object4 shares its y-coordinate with Object0; Object0 shares its y-coordinate with Object2 (premise sources: 1, temporary assumption 2 and temporary assumption 1).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object2 and Object4 cannot qualify together.

Suppose Object3 and Object4 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object4 has a smaller y-coordinate than Object3; Object3 shares its y-coordinate with Object0; Object0 shares its y-coordinate with Object4 (premise sources: 3, temporary assumption 1 and temporary assumption 2).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object3 and Object4 cannot qualify together.

No object is required to qualify. No object is individually ruled out. No qualifying group can contain both members of any forbidden pair: Object1 and Object2; Object1 and Object3; Object1 and Object4; Object2 and Object3; Object2 and Object4; Object3 and Object4. Checking the eligible-object subsets against these constraints leaves only sizes 0 and 1. Counts 2, 3 and 4 are therefore impossible.

Case 1: One possibility consistent with all premises is the following. Object1, Object2, Object3 and Object4 are south of Object0. No eligible object qualifies in this arrangement. This gives a count of 0.

Case 2: One possibility consistent with all premises is the following. Object1 is east of Object0. Object2, Object3 and Object4 are northeast of Object0. Only Object1 qualifies in this arrangement. This gives a count of 1.

The complete possible values are 0 and 1. Select AD.
Answer: AD

#### Symbolic

```text
Assume Object1 and Object2 both qualify (temporary assumptions 1 and 2).

[p2, assumption 2, assumption 1] y(Object1) < y(Object2) = y(Object0) = y(Object1) => y(Object1) < y(Object1).

y(Object1) < y(Object1) is impossible.

Therefore, Object1 and Object2 cannot qualify together.

Assume Object1 and Object3 both qualify (temporary assumptions 1 and 2).

[p1, p3, assumption 2, assumption 1, p2] y(Object2) < y(Object4) < y(Object3) = y(Object0) = y(Object1) < y(Object2) => y(Object2) < y(Object2).

y(Object2) < y(Object2) is impossible.

Therefore, Object1 and Object3 cannot qualify together.

Assume Object1 and Object4 both qualify (temporary assumptions 1 and 2).

[p1, assumption 2, assumption 1, p2] y(Object2) < y(Object4) = y(Object0) = y(Object1) < y(Object2) => y(Object2) < y(Object2).

y(Object2) < y(Object2) is impossible.

Therefore, Object1 and Object4 cannot qualify together.

Assume Object2 and Object3 both qualify (temporary assumptions 1 and 2).

[p1, p3, assumption 2, assumption 1] y(Object2) < y(Object4) < y(Object3) = y(Object0) = y(Object2) => y(Object2) < y(Object2).

y(Object2) < y(Object2) is impossible.

Therefore, Object2 and Object3 cannot qualify together.

Assume Object2 and Object4 both qualify (temporary assumptions 1 and 2).

[p1, assumption 2, assumption 1] y(Object2) < y(Object4) = y(Object0) = y(Object2) => y(Object2) < y(Object2).

y(Object2) < y(Object2) is impossible.

Therefore, Object2 and Object4 cannot qualify together.

Assume Object3 and Object4 both qualify (temporary assumptions 1 and 2).

[p3, assumption 1, assumption 2] y(Object4) < y(Object3) = y(Object0) = y(Object4) => y(Object4) < y(Object4).

y(Object4) < y(Object4) is impossible.

Therefore, Object3 and Object4 cannot qualify together.

Required={}; excluded={}; forbidden pairs={{Object1, Object2}, {Object1, Object3}, {Object1, Object4}, {Object2, Object3}, {Object2, Object4}, {Object3, Object4}}.
Joint membership subsets satisfying these constraints have sizes {0, 1}; excluded counts={2, 3, 4}.

Case 1 (possible together under all premises): S(Object1,Object0) & S(Object2,Object0) & S(Object3,Object0) & S(Object4,Object0). Qualifying objects = {}. count=0.

Case 2 (possible together under all premises): E(Object1,Object0). NE(Object2,Object0) & NE(Object3,Object0) & NE(Object4,Object0). Qualifying objects = {Object1}. count=1.

Complete possible set = {0, 1}. Select AD.
Answer: AD
```

### symbolic min: 61 tokens

ID: `depth-3-sample-2-direction-m1-all-possible`; Direction; Exhaustive; depth: 3; semantic case: unique; answer: F. All tied IDs are retained in the JSON.

#### Natural

Case 1: One possibility consistent with all premises is the following. From left to right, the order is Object9 and Object6. From bottom to top, the order is Object9 and Object6. So Object6 is northeast of Object9.

The witnessed value Northeast is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F

#### Symbolic

```text
Case 1 (possible together under all premises): x(Object9) < x(Object6). y(Object9) < y(Object6). => NE(Object6,Object9).

The witnessed value Northeast is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F
```

### symbolic max: 744 tokens

ID: `random-world-8-count-m0-all-possible`; Count; Exhaustive; depth: Unassigned; semantic case: underdetermined; answer: ABCD. All tied IDs are retained in the JSON.

#### Natural

Object4 has a greater x-coordinate than Object0 (premise sources: 3).

This comparison is incompatible with Object0 being north of Object4, so it cannot qualify.

The x-coordinates increase strictly along Object5, Object0 and Object4 (premise sources: 8 and 3). So Object4 has a greater x-coordinate than Object5.

This comparison is incompatible with Object5 being north of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object2 (premise sources: 5).

This comparison is incompatible with Object2 being north of Object4, so it cannot qualify.

Suppose Object7 and Object3 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object3 has a smaller x-coordinate than Object7; Object7 shares its x-coordinate with Object4; Object4 shares its x-coordinate with Object3 (premise sources: 7, temporary assumption 1 and temporary assumption 2).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object7 and Object3 cannot qualify together.

Suppose Object6 and Object3 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object3 has a smaller x-coordinate than Object6; Object6 shares its x-coordinate with Object4; Object4 shares its x-coordinate with Object3 (premise sources: 2, temporary assumption 1 and temporary assumption 2).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object6 and Object3 cannot qualify together.

Suppose Object1 and Object3 both qualify (temporary assumptions 1 and 2).

The cited relations would require that Object3 has a smaller x-coordinate than Object1; Object1 shares its x-coordinate with Object4; Object4 shares its x-coordinate with Object3 (premise sources: 6, temporary assumption 1 and temporary assumption 2).

This would make a coordinate smaller than itself, which is impossible.

Therefore, Object1 and Object3 cannot qualify together.

No object is required to qualify. Object0, Object5 and Object2 cannot qualify. No qualifying group can contain both members of any forbidden pair: Object7 and Object3; Object6 and Object3; Object1 and Object3. Checking the eligible-object subsets against these constraints leaves only sizes 0, 1, 2 and 3. Counts 4, 5, 6 and 7 are therefore impossible.

Case 1: One possibility consistent with all premises is the following. Object0, Object5 and Object2 are southwest of Object4. Object7, Object6 and Object1 are northeast of Object4. Object3 is northwest of Object4. No eligible object qualifies in this arrangement. This gives a count of 0.

Case 2: One possibility consistent with all premises is the following. Object0, Object5, Object3 and Object2 are southwest of Object4. Object7 and Object1 are south of Object4. Object6 is north of Object4. Only Object6 qualifies in this arrangement. This gives a count of 1.

Case 3: One possibility consistent with all premises is the following. Object0, Object5, Object3 and Object2 are southwest of Object4. Object7 and Object6 are north of Object4. Object1 is south of Object4. Only Object6 and Object7 qualify in this arrangement. This gives a count of 2.

Case 4: One possibility consistent with all premises is the following. Object0, Object5 and Object2 are southwest of Object4. Object7, Object6 and Object1 are north of Object4. Object3 is northwest of Object4. Only Object1, Object6 and Object7 qualify in this arrangement. This gives a count of 3.

The complete possible values are 0, 1, 2 and 3. Select ABCD.
Answer: ABCD

#### Symbolic

```text
[p3] x(Object0) < x(Object4).

Axis comparison contradicts N(Object0,Object4) => Object0 cannot qualify.

[p8, p3] x(Object5) < x(Object0) < x(Object4) => x(Object5) < x(Object4).

Axis comparison contradicts N(Object5,Object4) => Object5 cannot qualify.

[p5] x(Object2) < x(Object4).

Axis comparison contradicts N(Object2,Object4) => Object2 cannot qualify.

Assume Object7 and Object3 both qualify (temporary assumptions 1 and 2).

[p7, assumption 1, assumption 2] x(Object3) < x(Object7) = x(Object4) = x(Object3) => x(Object3) < x(Object3).

x(Object3) < x(Object3) is impossible.

Therefore, Object7 and Object3 cannot qualify together.

Assume Object6 and Object3 both qualify (temporary assumptions 1 and 2).

[p2, assumption 1, assumption 2] x(Object3) < x(Object6) = x(Object4) = x(Object3) => x(Object3) < x(Object3).

x(Object3) < x(Object3) is impossible.

Therefore, Object6 and Object3 cannot qualify together.

Assume Object1 and Object3 both qualify (temporary assumptions 1 and 2).

[p6, assumption 1, assumption 2] x(Object3) < x(Object1) = x(Object4) = x(Object3) => x(Object3) < x(Object3).

x(Object3) < x(Object3) is impossible.

Therefore, Object1 and Object3 cannot qualify together.

Required={}; excluded={Object0, Object5, Object2}; forbidden pairs={{Object7, Object3}, {Object6, Object3}, {Object1, Object3}}.
Joint membership subsets satisfying these constraints have sizes {0, 1, 2, 3}; excluded counts={4, 5, 6, 7}.

Case 1 (possible together under all premises): SW(Object0,Object4) & SW(Object5,Object4) & SW(Object2,Object4). NE(Object7,Object4) & NE(Object6,Object4) & NE(Object1,Object4). NW(Object3,Object4). Qualifying objects = {}. count=0.

Case 2 (possible together under all premises): SW(Object0,Object4) & SW(Object5,Object4) & SW(Object3,Object4) & SW(Object2,Object4). S(Object7,Object4) & S(Object1,Object4). N(Object6,Object4). Qualifying objects = {Object6}. count=1.

Case 3 (possible together under all premises): SW(Object0,Object4) & SW(Object5,Object4) & SW(Object3,Object4) & SW(Object2,Object4). N(Object7,Object4) & N(Object6,Object4). S(Object1,Object4). Qualifying objects = {Object6, Object7}. count=2.

Case 4 (possible together under all premises): SW(Object0,Object4) & SW(Object5,Object4) & SW(Object2,Object4). N(Object7,Object4) & N(Object6,Object4) & N(Object1,Object4). NW(Object3,Object4). Qualifying objects = {Object1, Object6, Object7}. count=3.

Complete possible set = {0, 1, 2, 3}. Select ABCD.
Answer: ABCD
```

## Interpretation and Limits

The inspected examples are now qualitative mathematical explanations, not numerical solver-model listings. Regular Direction chains are compact. Selection can require several different candidate scenarios; Count can require several pair exclusions and joint cases, which explains their longer tails. The longest outputs are not necessarily the deepest ones. The maximum Natural output is a joint Count case with no numeric depth assignment.

The all-pair automated evidence audit is not a human fluency study. Prior direct inspection covered eleven preview fixtures, a deep Count example, the formerly longest Selection answer, and both current maxima; this analysis also reads the minimum examples. It does not claim manual reading of all 987 pairs. Long exclusion lists and repetitive sentence templates can still be improved. Neither shortness nor solver correctness alone establishes human difficulty.

Full statistics: `QUALITATIVE-LENGTH-BREAKDOWN-v3.json`; one paired view per row: `QUALITATIVE-LENGTH-BREAKDOWN-v3-rows.csv`; all grouped statistics: `QUALITATIVE-LENGTH-BREAKDOWN-v3-groups.csv`.
