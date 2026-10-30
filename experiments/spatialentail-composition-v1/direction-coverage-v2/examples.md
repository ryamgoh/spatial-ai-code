# Diagonal Selection Examples

Development-only; rotations are not independent construction families.

## match-chain, Northeast, depth 3

`unique-expansion-match-chain-d3-s0-Northeast-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object7, Object3, Object6, Object1, Object8, Object5, Object4, Object0, Object2
Query: which Northeast Object7
Eligible: Object1, Object8, Object4, Object2, Object6, Object0
Consider a map with multiple objects:
Object2 is to the East of Object7.
Object4 is to the West of Object7.
Object6 is to the South of Object7.
Object1 is to the Northeast of Object5.
Object7 is to the North of Object0.
Object7 is to the Southwest of Object3.
Object8 is to the East of Object7.
Object5 is to the Northeast of Object3.
Question: Which eligible object is the only one northeast of Object7 in every valid map?
A. Object8
B. Object4
C. Object1
D. Object2
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be northeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

From left to right, the order is Object7, Object3, Object5 and Object1 (premise sources: 6, 8 and 4). So Object1 is to the right of Object7.

From bottom to top, the order is Object7, Object3, Object5 and Object1 (premise sources: 6, 8 and 4). So Object1 is above Object7.

So Object1 is northeast of Object7 in every valid map and always qualifies.

Object8 and Object7 share the same height (premise sources: 7).

Object2 and Object7 share the same height (premise sources: 1).

Object8 and Object2 share the same height as Object7, so none can be northeast of Object7.

Object7 is to the right of Object4 (premise sources: 2).

This comparison is incompatible with Object4 being northeast of Object7, so it cannot qualify.

Object6 and Object7 share the same horizontal position (premise sources: 3).

Object0 and Object7 share the same horizontal position (premise sources: 5).

Object6 and Object0 share the same horizontal position as Object7, so none can be northeast of Object7.

The only possible value is Object1. Select C.
Answer: C

### Symbolic

```text
We need exactly one eligible object to be northeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p6, p8, p4] x(Object7) < x(Object3) < x(Object5) < x(Object1) => x(Object7) < x(Object1).

[p6, p8, p4] y(Object7) < y(Object3) < y(Object5) < y(Object1) => y(Object7) < y(Object1).

NE(Object1,Object7) in every valid map => Object1 always qualifies.

[p7] y(Object8) = y(Object7).

[p1] y(Object2) = y(Object7).

For each c in {Object8, Object2}: y(c) = y(Object7) => not NE(c,Object7).

[p2] x(Object4) < x(Object7).

Axis comparison contradicts NE(Object4,Object7) => Object4 cannot qualify.

[p3] x(Object6) = x(Object7).

[p5] x(Object0) = x(Object7).

For each c in {Object6, Object0}: x(c) = x(Object7) => not NE(c,Object7).

Complete possible set = {Object1}. Select C.
Answer: C
```

## match-chain, Northeast, depth 10

`unique-expansion-match-chain-d10-s0-Northeast-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object12, Object8, Object4, Object11, Object5, Object0, Object6, Object13, Object2, Object1, Object3, Object9, Object7, Object10, Object14
Query: which Northeast Object7
Eligible: Object12, Object11, Object0, Object3, Object2
Consider a map with multiple objects:
Object7 is to the Southwest of Object8.
Object7 is to the North of Object0.
Object13 is to the Southwest of Object14.
Object2 is to the West of Object7.
Object4 is to the Southwest of Object9.
Object11 is to the Northeast of Object5.
Object8 is to the Southwest of Object10.
Object12 is to the South of Object7.
Object13 is to the Northeast of Object10.
Object5 is to the Northeast of Object9.
Object4 is to the Northeast of Object1.
Object3 is to the East of Object7.
Object6 is to the Southwest of Object1.
Object6 is to the Northeast of Object14.
Question: Which eligible object is the only one northeast of Object7 in every valid map?
A. Object11
B. Object12
C. Object0
D. Object2
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be northeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

From left to right, the order is Object7, Object8, Object10, Object13, Object14, Object6, Object1, Object4, Object9, Object5 and Object11 (premise sources: 1, 7, 9, 3, 14, 13, 11, 5, 10 and 6). So Object11 is to the right of Object7.

From bottom to top, the order is Object7, Object8, Object10, Object13, Object14, Object6, Object1, Object4, Object9, Object5 and Object11 (premise sources: 1, 7, 9, 3, 14, 13, 11, 5, 10 and 6). So Object11 is above Object7.

So Object11 is northeast of Object7 in every valid map and always qualifies.

Object12 and Object7 share the same horizontal position (premise sources: 8).

Object0 and Object7 share the same horizontal position (premise sources: 2).

Object12 and Object0 share the same horizontal position as Object7, so none can be northeast of Object7.

Object3 and Object7 share the same height (premise sources: 12).

This comparison is incompatible with Object3 being northeast of Object7, so it cannot qualify.

Object7 is to the right of Object2 (premise sources: 4).

This comparison is incompatible with Object2 being northeast of Object7, so it cannot qualify.

The only possible value is Object11. Select A.
Answer: A

### Symbolic

```text
We need exactly one eligible object to be northeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p1, p7, p9, p3, p14, p13, p11, p5, p10, p6] x(Object7) < x(Object8) < x(Object10) < x(Object13) < x(Object14) < x(Object6) < x(Object1) < x(Object4) < x(Object9) < x(Object5) < x(Object11) => x(Object7) < x(Object11).

[p1, p7, p9, p3, p14, p13, p11, p5, p10, p6] y(Object7) < y(Object8) < y(Object10) < y(Object13) < y(Object14) < y(Object6) < y(Object1) < y(Object4) < y(Object9) < y(Object5) < y(Object11) => y(Object7) < y(Object11).

NE(Object11,Object7) in every valid map => Object11 always qualifies.

[p8] x(Object12) = x(Object7).

[p2] x(Object0) = x(Object7).

For each c in {Object12, Object0}: x(c) = x(Object7) => not NE(c,Object7).

[p12] y(Object3) = y(Object7).

Axis comparison contradicts NE(Object3,Object7) => Object3 cannot qualify.

[p4] x(Object2) < x(Object7).

Axis comparison contradicts NE(Object2,Object7) => Object2 cannot qualify.

Complete possible set = {Object11}. Select A.
Answer: A
```

## rival-chain, Northeast, depth 3

`unique-expansion-rival-chain-d3-s0-Northeast-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object3, Object2, Object1, Object4, Object8, Object7, Object6, Object0, Object5
Query: which Northeast Object2
Eligible: Object3, Object0, Object1, Object8, Object4, Object7
Consider a map with multiple objects:
Object7 is to the East of Object2.
Object2 is to the North of Object5.
Object2 is to the Southwest of Object3.
Object1 is to the East of Object2.
Object6 is to the South of Object5.
Object8 is to the East of Object2.
Object0 is to the South of Object2.
Object4 is to the South of Object6.
Question: Which eligible object is the only one northeast of Object2 in every valid map?
A. Object1
B. Object3
C. Object4
D. Object8
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be northeast of Object2 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object3 is to the right of Object2 (premise sources: 3).

Object3 is above Object2 (premise sources: 3).

So Object3 is northeast of Object2 in every valid map and always qualifies.

Object0 and Object2 share the same horizontal position (premise sources: 7).

Object4, Object6, Object5 and Object2 share the same horizontal position (premise sources: 8, 5 and 2).

Object0 and Object4 share the same horizontal position as Object2, so none can be northeast of Object2.

Object1 and Object2 share the same height (premise sources: 4).

Object8 and Object2 share the same height (premise sources: 6).

Object7 and Object2 share the same height (premise sources: 1).

Object1, Object8 and Object7 share the same height as Object2, so none can be northeast of Object2.

The only possible value is Object3. Select B.
Answer: B

### Symbolic

```text
We need exactly one eligible object to be northeast of Object2 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p3] x(Object2) < x(Object3).

[p3] y(Object2) < y(Object3).

NE(Object3,Object2) in every valid map => Object3 always qualifies.

[p7] x(Object0) = x(Object2).

[p8, p5, p2] x(Object4) = x(Object6) = x(Object5) = x(Object2) => x(Object4) = x(Object2).

For each c in {Object0, Object4}: x(c) = x(Object2) => not NE(c,Object2).

[p4] y(Object1) = y(Object2).

[p6] y(Object8) = y(Object2).

[p1] y(Object7) = y(Object2).

For each c in {Object1, Object8, Object7}: y(c) = y(Object2) => not NE(c,Object2).

Complete possible set = {Object3}. Select B.
Answer: B
```

## rival-chain, Northeast, depth 10

`unique-expansion-rival-chain-d10-s0-Northeast-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object14, Object0, Object4, Object7, Object3, Object1, Object2, Object12, Object13, Object15, Object5, Object6, Object11, Object9, Object10, Object8
Query: which Northeast Object13
Eligible: Object6, Object4, Object9, Object3, Object7, Object0
Consider a map with multiple objects:
Object15 is to the South of Object10.
Object4 is to the East of Object13.
Object11 is to the South of Object15.
Object1 is to the South of Object12.
Object0 is to the Northeast of Object13.
Object13 is to the North of Object2.
Object13 is to the East of Object6.
Object7 is to the East of Object13.
Object2 is to the North of Object5.
Object9 is to the South of Object14.
Object8 is to the South of Object11.
Object10 is to the South of Object5.
Object14 is to the South of Object1.
Object13 is to the North of Object3.
Object8 is to the North of Object12.
Question: Which eligible object is the only one northeast of Object13 in every valid map?
A. Object9
B. Object3
C. Object0
D. Object7
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be northeast of Object13 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object0 is to the right of Object13 (premise sources: 5).

Object0 is above Object13 (premise sources: 5).

So Object0 is northeast of Object13 in every valid map and always qualifies.

Object13 is to the right of Object6 (premise sources: 7).

This comparison is incompatible with Object6 being northeast of Object13, so it cannot qualify.

Object4 and Object13 share the same height (premise sources: 2).

Object7 and Object13 share the same height (premise sources: 8).

Object4 and Object7 share the same height as Object13, so none can be northeast of Object13.

Object9, Object14, Object1, Object12, Object8, Object11, Object15, Object10, Object5, Object2 and Object13 share the same horizontal position (premise sources: 10, 13, 4, 15, 11, 3, 1, 12, 9 and 6).

Object3 and Object13 share the same horizontal position (premise sources: 14).

Object9 and Object3 share the same horizontal position as Object13, so none can be northeast of Object13.

The only possible value is Object0. Select C.
Answer: C

### Symbolic

```text
We need exactly one eligible object to be northeast of Object13 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p5] x(Object13) < x(Object0).

[p5] y(Object13) < y(Object0).

NE(Object0,Object13) in every valid map => Object0 always qualifies.

[p7] x(Object6) < x(Object13).

Axis comparison contradicts NE(Object6,Object13) => Object6 cannot qualify.

[p2] y(Object4) = y(Object13).

[p8] y(Object7) = y(Object13).

For each c in {Object4, Object7}: y(c) = y(Object13) => not NE(c,Object13).

[p10, p13, p4, p15, p11, p3, p1, p12, p9, p6] x(Object9) = x(Object14) = x(Object1) = x(Object12) = x(Object8) = x(Object11) = x(Object15) = x(Object10) = x(Object5) = x(Object2) = x(Object13) => x(Object9) = x(Object13).

[p14] x(Object3) = x(Object13).

For each c in {Object9, Object3}: x(c) = x(Object13) => not NE(c,Object13).

Complete possible set = {Object0}. Select C.
Answer: C
```

## shared-exclusion, Northeast, depth 3

`unique-expansion-shared-exclusion-d3-s0-Northeast-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object4, Object6, Object0, Object2, Object1, Object7, Object5, Object3
Query: which Northeast Object7
Eligible: Object6, Object1, Object0, Object4, Object2
Consider a map with multiple objects:
Object3 is to the West of Object4.
Object1 is to the Northeast of Object7.
Object3 is to the North of Object0.
Object5 is to the North of Object3.
Object7 is to the North of Object5.
Object3 is to the East of Object6.
Object2 is to the West of Object3.
Question: Which eligible object is the only one northeast of Object7 in every valid map?
A. Object0
B. Object1
C. Object2
D. Object6
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be northeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object1 is to the right of Object7 (premise sources: 2).

Object1 is above Object7 (premise sources: 2).

So Object1 is northeast of Object7 in every valid map and always qualifies.

Comparison 1: Object3, Object5 and Object7 share the same horizontal position (premise sources: 4 and 5).

Object6 and Object2 are each to the left of Object3 (respective premise sources: 6 and 7). By comparison 1, Object3 is at the same horizontal position as Object7.

Object6 and Object2 are all to the left of Object7, so none can be northeast of Object7.

Object0, Object3 and Object7 share the same horizontal position (sources: 3 and comparison 1).

This comparison is incompatible with Object0 being northeast of Object7, so it cannot qualify.

Vertically, Object4 has the same height as Object3; Object3 is below Object5; Object5 is below Object7 (premise sources: 1, 4 and 5). Therefore, Object7 is above Object4.

This comparison is incompatible with Object4 being northeast of Object7, so it cannot qualify.

The only possible value is Object1. Select B.
Answer: B

### Symbolic

```text
We need exactly one eligible object to be northeast of Object7 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p2] x(Object7) < x(Object1).

[p2] y(Object7) < y(Object1).

NE(Object1,Object7) in every valid map => Object1 always qualifies.

Comparison 1: [p4, p5] x(Object3) = x(Object5) = x(Object7) => x(Object3) = x(Object7).

Branches {Object6, Object2} with respective premises {p6, p7}: x(c) < x(Object3); [comparison 1] x(Object3) = x(Object7).

For each c in {Object6, Object2}: x(c) < x(Object7) => not NE(c,Object7).

[p3, comparison 1] x(Object0) = x(Object3) = x(Object7) => x(Object0) = x(Object7).

Axis comparison contradicts NE(Object0,Object7) => Object0 cannot qualify.

[p1, p4, p5] y(Object4) = y(Object3) < y(Object5) < y(Object7) => y(Object4) < y(Object7).

Axis comparison contradicts NE(Object4,Object7) => Object4 cannot qualify.

Complete possible set = {Object1}. Select B.
Answer: B
```

## shared-exclusion, Northeast, depth 10

`unique-expansion-shared-exclusion-d10-s0-Northeast-which-m1-single`

### Prompt

```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object3, Object11, Object4, Object13, Object12, Object9, Object10, Object6, Object8, Object0, Object1, Object5, Object7, Object2, Object14
Query: which Northeast Object4
Eligible: Object12, Object11, Object1, Object2, Object10
Consider a map with multiple objects:
Object14 is to the South of Object9.
Object0 is to the South of Object7.
Object4 is to the Southwest of Object1.
Object8 is to the North of Object3.
Object11 is to the East of Object6.
Object0 is to the North of Object5.
Object12 is to the East of Object6.
Object3 is to the North of Object7.
Object5 is to the North of Object9.
Object2 is to the East of Object6.
Object4 is to the North of Object8.
Object6 is to the South of Object13.
Object6 is to the North of Object10.
Object14 is to the North of Object13.
Question: Which eligible object is the only one northeast of Object4 in every valid map?
A. Object2
B. Object1
C. Object10
D. Object11
E. Cannot be determined
F. None of the above
```

### Natural

We need exactly one eligible object to be northeast of Object4 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

Object1 is to the right of Object4 (premise sources: 3).

Object1 is above Object4 (premise sources: 3).

So Object1 is northeast of Object4 in every valid map and always qualifies.

Comparison 1: From bottom to top, the order is Object6, Object13, Object14, Object9, Object5, Object0, Object7, Object3, Object8 and Object4 (premise sources: 12, 14, 1, 9, 6, 2, 8, 4 and 11). So Object4 is above Object6.

Object12, Object11 and Object2 are each at the same height as Object6 (respective premise sources: 7, 5 and 10). By comparison 1, Object6 is below Object4.

Object12, Object11 and Object2 are all below Object4, so none can be northeast of Object4.

Object10, Object6, Object13, Object14, Object9, Object5, Object0, Object7, Object3, Object8 and Object4 share the same horizontal position (premise sources: 13, 12, 14, 1, 9, 6, 2, 8, 4 and 11).

This comparison is incompatible with Object10 being northeast of Object4, so it cannot qualify.

The only possible value is Object1. Select B.
Answer: B

### Symbolic

```text
We need exactly one eligible object to be northeast of Object4 in every valid map. Check both qualifying objects and the alternatives, including unlisted candidates.

[p3] x(Object4) < x(Object1).

[p3] y(Object4) < y(Object1).

NE(Object1,Object4) in every valid map => Object1 always qualifies.

Comparison 1: [p12, p14, p1, p9, p6, p2, p8, p4, p11] y(Object6) < y(Object13) < y(Object14) < y(Object9) < y(Object5) < y(Object0) < y(Object7) < y(Object3) < y(Object8) < y(Object4) => y(Object6) < y(Object4).

Branches {Object12, Object11, Object2} with respective premises {p7, p5, p10}: y(c) = y(Object6); [comparison 1] y(Object6) < y(Object4).

For each c in {Object12, Object11, Object2}: y(c) < y(Object4) => not NE(c,Object4).

[p13, p12, p14, p1, p9, p6, p2, p8, p4, p11] x(Object10) = x(Object6) = x(Object13) = x(Object14) = x(Object9) = x(Object5) = x(Object0) = x(Object7) = x(Object3) = x(Object8) = x(Object4) => x(Object10) = x(Object4).

Axis comparison contradicts NE(Object10,Object4) => Object10 cannot qualify.

Complete possible set = {Object1}. Select B.
Answer: B
```
