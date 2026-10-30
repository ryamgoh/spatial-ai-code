# Question-Led Paired Explanations

Saved development questions, not production data. Full evidence and gold are preserved.

## direction

`balance-depth-1-sample-0-direction-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object3, Object1, Object4, Object2, Object6, Object0, Object7
Query: direction Object6 Object7
Consider a map with multiple objects:
Object7 is to the Southeast of Object3.
Object7 is to the Northwest of Object6.
Object5 is to the South of Object6.
Object0 is to the Northwest of Object7.
Object2 is to the North of Object6.
Object7 is to the Southeast of Object1.
Object4 is to the Northwest of Object7.
Question: In which direction is Object6 relative to Object7 in every valid map?
A. Southwest
B. Northeast
C. West
D. Southeast
E. Cannot be determined
F. None of the above
```

### Natural

Determine the direction of Object6 relative to Object7, checking both horizontal and vertical relations.

Object6 is to the right of Object7 (premise sources: 2).

Object7 is above Object6 (premise sources: 2).

Together, these comparisons place Object6 exactly southeast of Object7. The x-comparison rules out north, south, southwest, west and northwest. The y-comparison rules out northeast and east.

The only possible value is Southeast. Select D.
Answer: D

### Symbolic

```text
Determine the direction of Object6 relative to Object7, checking both horizontal and vertical relations.

[p2] x(Object7) < x(Object6).

[p2] y(Object6) < y(Object7).

=> SE(Object6,Object7).
x-comparison excludes {N, S, SW, W, NW}.
y-comparison excludes {NE, E}.

Complete possible set = {Southeast}. Select D.
Answer: D
```

## which

`balance-depth-1-sample-0-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object3, Object1, Object4, Object2, Object6, Object0, Object7
Query: which Southeast Object7
Eligible: Object3, Object0, Object1, Object6, Object4
Consider a map with multiple objects:
Object7 is to the Southeast of Object3.
Object7 is to the Northwest of Object6.
Object5 is to the South of Object6.
Object0 is to the Northwest of Object7.
Object2 is to the North of Object6.
Object7 is to the Southeast of Object1.
Object4 is to the Northwest of Object7.
Question: Which eligible object is the only one southeast of Object7 in every valid map?
A. Object1
B. Object3
C. Object0
D. Object6
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be southeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object6 is to the right of Object7 (premise sources: 2).

Object7 is above Object6 (premise sources: 2).

So Object6 is southeast of Object7 in every valid map and always qualifies.

Object7 is to the right of Object3 (premise sources: 1).

Object7 is to the right of Object0 (premise sources: 4).

Object7 is to the right of Object1 (premise sources: 6).

Object7 is to the right of Object4 (premise sources: 7).

Object3, Object0, Object1 and Object4 are all to the left of Object7, so none can be southeast of Object7.

The only possible value is Object6. Select D.
Answer: D

### Symbolic

```text
We need exactly one eligible object to be southeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p2] x(Object7) < x(Object6).

[p2] y(Object6) < y(Object7).

SE(Object6,Object7) in every valid map => Object6 always qualifies.

[p1] x(Object3) < x(Object7).

[p4] x(Object0) < x(Object7).

[p6] x(Object1) < x(Object7).

[p7] x(Object4) < x(Object7).

For each c in {Object3, Object0, Object1, Object4}: x(c) < x(Object7) => not SE(c,Object7).

Complete possible set = {Object6}. Select D.
Answer: D
```

## count

`balance-depth-1-sample-0-count-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object3, Object1, Object4, Object2, Object6, Object0, Object7
Query: count Southeast Object7
Eligible: Object3, Object0, Object1, Object6, Object4
Consider a map with multiple objects:
Object7 is to the Southeast of Object3.
Object7 is to the Northwest of Object6.
Object5 is to the South of Object6.
Object0 is to the Northwest of Object7.
Object2 is to the North of Object6.
Object7 is to the Southeast of Object1.
Object4 is to the Northwest of Object7.
Question: What exact number of eligible objects are southeast of Object7 in every valid map?
A. 1
B. 3
C. 4
D. 2
E. Cannot be determined
F. None of the above
```

### Natural

Determine how many eligible objects can be southeast of Object7 together. Individual possibilities alone do not establish a joint count.

Object6 is to the right of Object7 (premise sources: 2).

Object7 is above Object6 (premise sources: 2).

So Object6 is southeast of Object7 in every valid map and always qualifies.

Object7 is to the right of Object3 (premise sources: 1).

Object7 is to the right of Object0 (premise sources: 4).

Object7 is to the right of Object1 (premise sources: 6).

Object7 is to the right of Object4 (premise sources: 7).

Object3, Object0, Object1 and Object4 are all to the left of Object7, so none can be southeast of Object7.

Every qualifying group must include Object6. Object3, Object0, Object1 and Object4 cannot qualify. There are no additional forbidden pairs. Checking the eligible-object subsets against these constraints leaves only size 1. Counts 0, 2, 3, 4 and 5 are therefore impossible.

Case 1: One possibility consistent with all premises is the following. Object3, Object0, Object1 and Object4 are northwest of Object7. Object6 is southeast of Object7. Only Object6 qualifies in this arrangement. This gives a count of 1.

The only possible value is 1. Select A.
Answer: A

### Symbolic

```text
Determine how many eligible objects can be southeast of Object7 together. Individual possibilities alone do not establish a joint count.

[p2] x(Object7) < x(Object6).

[p2] y(Object6) < y(Object7).

SE(Object6,Object7) in every valid map => Object6 always qualifies.

[p1] x(Object3) < x(Object7).

[p4] x(Object0) < x(Object7).

[p6] x(Object1) < x(Object7).

[p7] x(Object4) < x(Object7).

For each c in {Object3, Object0, Object1, Object4}: x(c) < x(Object7) => not SE(c,Object7).

Required={Object6}; excluded={Object3, Object0, Object1, Object4}; forbidden pairs={}.
Joint membership subsets satisfying these constraints have sizes {1}; excluded counts={0, 2, 3, 4, 5}.

Case 1 (possible together under all premises): NW(Object3,Object7) & NW(Object0,Object7) & NW(Object1,Object7) & NW(Object4,Object7). SE(Object6,Object7). Qualifying objects = {Object6}. count=1.

Complete possible set = {1}. Select A.
Answer: A
```

## match-chain/depth-3

`unique-expansion-match-chain-d3-s0-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object0, Object5, Object2, Object8, Object1, Object3, Object7, Object4
Query: which East Object6
Eligible: Object1, Object7, Object8, Object2, Object5, Object4
Consider a map with multiple objects:
Object6 is to the East of Object8.
Object6 is to the West of Object0.
Object5 is to the West of Object6.
Object1 is to the East of Object3.
Object6 is to the East of Object4.
Object6 is to the South of Object2.
Object0 is to the West of Object3.
Object6 is to the East of Object7.
Question: Which eligible object is the only one east of Object6 in every valid map?
A. Object8
B. Object1
C. Object4
D. Object5
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be east of Object6 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

From left to right, the order is Object6, Object0, Object3 and Object1 (premise sources: 2, 7 and 4). So Object1 is to the right of Object6.

Object1, Object3, Object0 and Object6 share the same height (premise sources: 4, 7 and 2).

So Object1 is east of Object6 in every valid map and always qualifies.

Object6 is to the right of Object7 (premise sources: 8).

Object6 is to the right of Object8 (premise sources: 1).

Object6 is to the right of Object5 (premise sources: 3).

Object6 is to the right of Object4 (premise sources: 5).

Object7, Object8, Object5 and Object4 are all to the left of Object6, so none can be east of Object6.

Object2 and Object6 share the same horizontal position (premise sources: 6).

This comparison is incompatible with Object2 being east of Object6, so it cannot qualify.

The only possible value is Object1. Select B.
Answer: B

### Symbolic

```text
We need exactly one eligible object to be east of Object6 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p2, p7, p4] x(Object6) < x(Object0) < x(Object3) < x(Object1) => x(Object6) < x(Object1).

[p4, p7, p2] y(Object1) = y(Object3) = y(Object0) = y(Object6) => y(Object1) = y(Object6).

E(Object1,Object6) in every valid map => Object1 always qualifies.

[p8] x(Object7) < x(Object6).

[p1] x(Object8) < x(Object6).

[p3] x(Object5) < x(Object6).

[p5] x(Object4) < x(Object6).

For each c in {Object7, Object8, Object5, Object4}: x(c) < x(Object6) => not E(c,Object6).

[p6] x(Object2) = x(Object6).

Axis comparison contradicts E(Object2,Object6) => Object2 cannot qualify.

Complete possible set = {Object1}. Select B.
Answer: B
```

## match-chain/depth-10

`unique-expansion-match-chain-d10-s0-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object10, Object14, Object0, Object3, Object9, Object5, Object12, Object7, Object11, Object8, Object1, Object2, Object13, Object4, Object6, Object15
Query: which North Object0
Eligible: Object14, Object13, Object1, Object9, Object8, Object5
Consider a map with multiple objects:
Object0 is to the West of Object1.
Object15 is to the North of Object2.
Object8 is to the East of Object0.
Object4 is to the South of Object10.
Object6 is to the North of Object3.
Object5 is to the South of Object0.
Object11 is to the North of Object6.
Object12 is to the North of Object10.
Object12 is to the South of Object3.
Object7 is to the South of Object4.
Object9 is to the South of Object0.
Object13 is to the South of Object0.
Object0 is to the South of Object2.
Object11 is to the South of Object14.
Object7 is to the North of Object15.
Question: Which eligible object is the only one north of Object0 in every valid map?
A. Object13
B. Object1
C. Object14
D. Object8
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be north of Object0 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object0, Object2, Object15, Object7, Object4, Object10, Object12, Object3, Object6, Object11 and Object14 share the same horizontal position (premise sources: 13, 2, 15, 10, 4, 8, 9, 5, 7 and 14).

From bottom to top, the order is Object0, Object2, Object15, Object7, Object4, Object10, Object12, Object3, Object6, Object11 and Object14 (premise sources: 13, 2, 15, 10, 4, 8, 9, 5, 7 and 14). So Object14 is above Object0.

So Object14 is north of Object0 in every valid map and always qualifies.

Object0 is above Object13 (premise sources: 12).

Object0 is above Object9 (premise sources: 11).

Object0 is above Object5 (premise sources: 6).

Object13, Object9 and Object5 are all below Object0, so none can be north of Object0.

Object1 is to the right of Object0 (premise sources: 1).

Object8 is to the right of Object0 (premise sources: 3).

Object1 and Object8 are all to the right of Object0, so none can be north of Object0.

The only possible value is Object14. Select C.
Answer: C

### Symbolic

```text
We need exactly one eligible object to be north of Object0 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p13, p2, p15, p10, p4, p8, p9, p5, p7, p14] x(Object0) = x(Object2) = x(Object15) = x(Object7) = x(Object4) = x(Object10) = x(Object12) = x(Object3) = x(Object6) = x(Object11) = x(Object14) => x(Object0) = x(Object14).

[p13, p2, p15, p10, p4, p8, p9, p5, p7, p14] y(Object0) < y(Object2) < y(Object15) < y(Object7) < y(Object4) < y(Object10) < y(Object12) < y(Object3) < y(Object6) < y(Object11) < y(Object14) => y(Object0) < y(Object14).

N(Object14,Object0) in every valid map => Object14 always qualifies.

[p12] y(Object13) < y(Object0).

[p11] y(Object9) < y(Object0).

[p6] y(Object5) < y(Object0).

For each c in {Object13, Object9, Object5}: y(c) < y(Object0) => not N(c,Object0).

[p1] x(Object0) < x(Object1).

[p3] x(Object0) < x(Object8).

For each c in {Object1, Object8}: x(c) > x(Object0) => not N(c,Object0).

Complete possible set = {Object14}. Select C.
Answer: C
```

## rival-chain/depth-3

`unique-expansion-rival-chain-d3-s0-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object8, Object5, Object4, Object7, Object2, Object3, Object0, Object1, Object6
Query: which East Object1
Eligible: Object2, Object8, Object0, Object3, Object4, Object7
Consider a map with multiple objects:
Object1 is to the North of Object0.
Object2 is to the West of Object1.
Object6 is to the East of Object5.
Object1 is to the West of Object8.
Object3 is to the West of Object5.
Object7 is to the West of Object1.
Object6 is to the West of Object1.
Object4 is to the South of Object1.
Question: Which eligible object is the only one east of Object1 in every valid map?
A. Object4
B. Object0
C. Object8
D. Object2
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be east of Object1 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object8 is to the right of Object1 (premise sources: 4).

Object8 and Object1 share the same height (premise sources: 4).

So Object8 is east of Object1 in every valid map and always qualifies.

Object1 is to the right of Object2 (premise sources: 2).

Object1 is to the right of Object7 (premise sources: 6).

From left to right, the order is Object3, Object5, Object6 and Object1 (premise sources: 5, 3 and 7). So Object1 is to the right of Object3.

Object2, Object7 and Object3 are all to the left of Object1, so none can be east of Object1.

Object0 and Object1 share the same horizontal position (premise sources: 1).

Object4 and Object1 share the same horizontal position (premise sources: 8).

Object0 and Object4 share the same horizontal position as Object1, so none can be east of Object1.

The only possible value is Object8. Select C.
Answer: C

### Symbolic

```text
We need exactly one eligible object to be east of Object1 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p4] x(Object1) < x(Object8).

[p4] y(Object8) = y(Object1).

E(Object8,Object1) in every valid map => Object8 always qualifies.

[p2] x(Object2) < x(Object1).

[p6] x(Object7) < x(Object1).

[p5, p3, p7] x(Object3) < x(Object5) < x(Object6) < x(Object1) => x(Object3) < x(Object1).

For each c in {Object2, Object7, Object3}: x(c) < x(Object1) => not E(c,Object1).

[p1] x(Object0) = x(Object1).

[p8] x(Object4) = x(Object1).

For each c in {Object0, Object4}: x(c) = x(Object1) => not E(c,Object1).

Complete possible set = {Object8}. Select C.
Answer: C
```

## rival-chain/depth-10

`unique-expansion-rival-chain-d10-s0-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object9, Object15, Object3, Object13, Object0, Object14, Object6, Object11, Object2, Object8, Object10, Object1, Object7, Object5, Object12, Object4
Query: which East Object15
Eligible: Object12, Object9, Object6, Object11, Object0, Object8
Consider a map with multiple objects:
Object2 is to the West of Object10.
Object1 is to the West of Object4.
Object8 is to the North of Object15.
Object14 is to the West of Object13.
Object6 is to the East of Object15.
Object14 is to the East of Object7.
Object2 is to the East of Object11.
Object15 is to the East of Object5.
Object5 is to the East of Object4.
Object3 is to the East of Object13.
Object7 is to the East of Object10.
Object1 is to the East of Object3.
Object15 is to the South of Object0.
Object9 is to the South of Object15.
Object12 is to the West of Object15.
Question: Which eligible object is the only one east of Object15 in every valid map?
A. Object6
B. Object9
C. Object8
D. Object12
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be east of Object15 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object6 is to the right of Object15 (premise sources: 5).

Object6 and Object15 share the same height (premise sources: 5).

So Object6 is east of Object15 in every valid map and always qualifies.

Object15 is to the right of Object12 (premise sources: 15).

From left to right, the order is Object11, Object2, Object10, Object7, Object14, Object13, Object3, Object1, Object4, Object5 and Object15 (premise sources: 7, 1, 11, 6, 4, 10, 12, 2, 9 and 8). So Object15 is to the right of Object11.

Object12 and Object11 are all to the left of Object15, so none can be east of Object15.

Object9 and Object15 share the same horizontal position (premise sources: 14).

Object0 and Object15 share the same horizontal position (premise sources: 13).

Object8 and Object15 share the same horizontal position (premise sources: 3).

Object9, Object0 and Object8 share the same horizontal position as Object15, so none can be east of Object15.

The only possible value is Object6. Select A.
Answer: A

### Symbolic

```text
We need exactly one eligible object to be east of Object15 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p5] x(Object15) < x(Object6).

[p5] y(Object6) = y(Object15).

E(Object6,Object15) in every valid map => Object6 always qualifies.

[p15] x(Object12) < x(Object15).

[p7, p1, p11, p6, p4, p10, p12, p2, p9, p8] x(Object11) < x(Object2) < x(Object10) < x(Object7) < x(Object14) < x(Object13) < x(Object3) < x(Object1) < x(Object4) < x(Object5) < x(Object15) => x(Object11) < x(Object15).

For each c in {Object12, Object11}: x(c) < x(Object15) => not E(c,Object15).

[p14] x(Object9) = x(Object15).

[p13] x(Object0) = x(Object15).

[p3] x(Object8) = x(Object15).

For each c in {Object9, Object0, Object8}: x(c) = x(Object15) => not E(c,Object15).

Complete possible set = {Object6}. Select A.
Answer: A
```

## shared-exclusion/depth-3

`unique-expansion-shared-exclusion-d3-s0-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object7, Object1, Object4, Object6, Object8, Object3, Object2, Object5, Object9, Object0
Query: which West Object9
Eligible: Object1, Object6, Object5, Object4, Object2, Object7, Object0
Consider a map with multiple objects:
Object6 is to the North of Object3.
Object8 is to the East of Object9.
Object3 is to the West of Object7.
Object4 is to the South of Object3.
Object2 is to the West of Object9.
Object3 is to the East of Object8.
Object0 is to the South of Object3.
Object3 is to the North of Object5.
Object1 is to the North of Object3.
Question: Which eligible object is the only one west of Object9 in every valid map?
A. Object6
B. Object0
C. Object2
D. Object7
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be west of Object9 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object9 is to the right of Object2 (premise sources: 5).

Object2 and Object9 share the same height (premise sources: 5).

So Object2 is west of Object9 in every valid map and always qualifies.

Comparison 1: From left to right, the order is Object9, Object8 and Object3 (premise sources: 2 and 6). So Object3 is to the right of Object9.

Object1, Object6, Object5, Object4 and Object0 are each at the same horizontal position as Object3 (respective premise sources: 9, 1, 8, 4 and 7). By comparison 1, Object3 is to the right of Object9.

From left to right, the order is Object9, Object3 and Object7 (sources: comparison 1 and 3). So Object7 is to the right of Object9.

Object1, Object6, Object5, Object4, Object0 and Object7 are all to the right of Object9, so none can be west of Object9.

The only possible value is Object2. Select C.
Answer: C

### Symbolic

```text
We need exactly one eligible object to be west of Object9 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p5] x(Object2) < x(Object9).

[p5] y(Object2) = y(Object9).

W(Object2,Object9) in every valid map => Object2 always qualifies.

Comparison 1: [p2, p6] x(Object9) < x(Object8) < x(Object3) => x(Object9) < x(Object3).

Branches {Object1, Object6, Object5, Object4, Object0} with respective premises {p9, p1, p8, p4, p7}: x(c) = x(Object3); [comparison 1] x(Object3) > x(Object9).

[comparison 1, p3] x(Object9) < x(Object3) < x(Object7) => x(Object9) < x(Object7).

For each c in {Object1, Object6, Object5, Object4, Object0, Object7}: x(c) > x(Object9) => not W(c,Object9).

Complete possible set = {Object2}. Select C.
Answer: C
```

## shared-exclusion/depth-10

`unique-expansion-shared-exclusion-d10-s0-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object11, Object0, Object9, Object5, Object7, Object1, Object8, Object2, Object15, Object16, Object4, Object10, Object14, Object12, Object3, Object13
Query: which South Object16
Eligible: Object10, Object11, Object3, Object1, Object2, Object7, Object12
Consider a map with multiple objects:
Object5 is to the South of Object6.
Object15 is to the South of Object8.
Object0 is to the North of Object16.
Object0 is to the South of Object13.
Object7 is to the North of Object4.
Object4 is to the North of Object9.
Object10 is to the West of Object4.
Object14 is to the South of Object5.
Object4 is to the South of Object3.
Object4 is to the South of Object11.
Object9 is to the North of Object6.
Object14 is to the North of Object8.
Object4 is to the East of Object12.
Object2 is to the South of Object16.
Object15 is to the North of Object13.
Object1 is to the East of Object4.
Question: Which eligible object is the only one south of Object16 in every valid map?
A. Object10
B. Object12
C. Object2
D. Object11
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be south of Object16 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object2 and Object16 share the same horizontal position (premise sources: 14).

Object16 is above Object2 (premise sources: 14).

So Object2 is south of Object16 in every valid map and always qualifies.

Comparison 1: Object16, Object0, Object13, Object15, Object8, Object14, Object5, Object6, Object9 and Object4 share the same horizontal position (premise sources: 3, 4, 15, 2, 12, 8, 1, 11 and 6).

Object10 and Object12 are each to the left of Object4 (respective premise sources: 7 and 13). By comparison 1, Object4 is at the same horizontal position as Object16.

Object10 and Object12 are all to the left of Object16, so none can be south of Object16.

Comparison 2: From bottom to top, the order is Object16, Object0, Object13, Object15, Object8, Object14, Object5, Object6, Object9 and Object4 (premise sources: 3, 4, 15, 2, 12, 8, 1, 11 and 6). So Object4 is above Object16.

Object11, Object3 and Object7 are each above Object4 (respective premise sources: 10, 9 and 5). By comparison 2, Object4 is above Object16.

Object11, Object3 and Object7 are all above Object16, so none can be south of Object16.

Horizontally, Object16 has the same horizontal position as Object4; Object4 is to the left of Object1 (sources: comparison 1 and 16). Therefore, Object1 is to the right of Object16.

This comparison is incompatible with Object1 being south of Object16, so it cannot qualify.

The only possible value is Object2. Select C.
Answer: C

### Symbolic

```text
We need exactly one eligible object to be south of Object16 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p14] x(Object2) = x(Object16).

[p14] y(Object2) < y(Object16).

S(Object2,Object16) in every valid map => Object2 always qualifies.

Comparison 1: [p3, p4, p15, p2, p12, p8, p1, p11, p6] x(Object16) = x(Object0) = x(Object13) = x(Object15) = x(Object8) = x(Object14) = x(Object5) = x(Object6) = x(Object9) = x(Object4) => x(Object16) = x(Object4).

Branches {Object10, Object12} with respective premises {p7, p13}: x(c) < x(Object4); [comparison 1] x(Object4) = x(Object16).

For each c in {Object10, Object12}: x(c) < x(Object16) => not S(c,Object16).

Comparison 2: [p3, p4, p15, p2, p12, p8, p1, p11, p6] y(Object16) < y(Object0) < y(Object13) < y(Object15) < y(Object8) < y(Object14) < y(Object5) < y(Object6) < y(Object9) < y(Object4) => y(Object16) < y(Object4).

Branches {Object11, Object3, Object7} with respective premises {p10, p9, p5}: y(c) > y(Object4); [comparison 2] y(Object4) > y(Object16).

For each c in {Object11, Object3, Object7}: y(c) > y(Object16) => not S(c,Object16).

[comparison 1, p16] x(Object16) = x(Object4) < x(Object1) => x(Object16) < x(Object1).

Axis comparison contradicts S(Object1,Object16) => Object1 cannot qualify.

Complete possible set = {Object2}. Select C.
Answer: C
```
