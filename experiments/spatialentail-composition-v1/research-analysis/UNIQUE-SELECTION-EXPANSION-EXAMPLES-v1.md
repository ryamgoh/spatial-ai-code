# Unique Selection Expansion Examples

Actual saved covered-menu Unique outputs, not hand-written illustrative proofs.

## match-chain, depth 3

unique-expansion-match-chain-d3-s0-which-m1-single

### Problem

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

The x-coordinates increase strictly along Object6, Object0, Object3 and Object1 (premise sources: 2, 7 and 4). So Object1 has a greater x-coordinate than Object6.

Object1, Object3, Object0 and Object6 share the same y-coordinate (premise sources: 4, 7 and 2).

So Object1 is east of Object6 in every valid map and always qualifies.

Object6 has a greater x-coordinate than Object7 (premise sources: 8).

This comparison is incompatible with Object7 being east of Object6, so it cannot qualify.

Object6 has a greater x-coordinate than Object8 (premise sources: 1).

This comparison is incompatible with Object8 being east of Object6, so it cannot qualify.

Object2 and Object6 share the same x-coordinate (premise sources: 6).

This comparison is incompatible with Object2 being east of Object6, so it cannot qualify.

Object6 has a greater x-coordinate than Object5 (premise sources: 3).

This comparison is incompatible with Object5 being east of Object6, so it cannot qualify.

Object6 has a greater x-coordinate than Object4 (premise sources: 5).

This comparison is incompatible with Object4 being east of Object6, so it cannot qualify.

The only possible value is Object1. Select B.
Answer: B

### Symbolic

```text
[p2, p7, p4] x(Object6) < x(Object0) < x(Object3) < x(Object1) => x(Object6) < x(Object1).

[p4, p7, p2] y(Object1) = y(Object3) = y(Object0) = y(Object6) => y(Object1) = y(Object6).

E(Object1,Object6) in every valid map => Object1 always qualifies.

[p8] x(Object7) < x(Object6).

Axis comparison contradicts E(Object7,Object6) => Object7 cannot qualify.

[p1] x(Object8) < x(Object6).

Axis comparison contradicts E(Object8,Object6) => Object8 cannot qualify.

[p6] x(Object2) = x(Object6).

Axis comparison contradicts E(Object2,Object6) => Object2 cannot qualify.

[p3] x(Object5) < x(Object6).

Axis comparison contradicts E(Object5,Object6) => Object5 cannot qualify.

[p5] x(Object4) < x(Object6).

Axis comparison contradicts E(Object4,Object6) => Object4 cannot qualify.

Complete possible set = {Object1}. Select B.
Answer: B
```

## match-chain, depth 10

unique-expansion-match-chain-d10-s0-which-m1-single

### Problem

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

Object0, Object2, Object15, Object7, Object4, Object10, Object12, Object3, Object6, Object11 and Object14 share the same x-coordinate (premise sources: 13, 2, 15, 10, 4, 8, 9, 5, 7 and 14).

The y-coordinates increase strictly along Object0, Object2, Object15, Object7, Object4, Object10, Object12, Object3, Object6, Object11 and Object14 (premise sources: 13, 2, 15, 10, 4, 8, 9, 5, 7 and 14). So Object14 has a greater y-coordinate than Object0.

So Object14 is north of Object0 in every valid map and always qualifies.

Object0 has a greater y-coordinate than Object13 (premise sources: 12).

This comparison is incompatible with Object13 being north of Object0, so it cannot qualify.

Object1 has a greater x-coordinate than Object0 (premise sources: 1).

This comparison is incompatible with Object1 being north of Object0, so it cannot qualify.

Object0 has a greater y-coordinate than Object9 (premise sources: 11).

This comparison is incompatible with Object9 being north of Object0, so it cannot qualify.

Object8 has a greater x-coordinate than Object0 (premise sources: 3).

This comparison is incompatible with Object8 being north of Object0, so it cannot qualify.

Object0 has a greater y-coordinate than Object5 (premise sources: 6).

This comparison is incompatible with Object5 being north of Object0, so it cannot qualify.

The only possible value is Object14. Select C.
Answer: C

### Symbolic

```text
[p13, p2, p15, p10, p4, p8, p9, p5, p7, p14] x(Object0) = x(Object2) = x(Object15) = x(Object7) = x(Object4) = x(Object10) = x(Object12) = x(Object3) = x(Object6) = x(Object11) = x(Object14) => x(Object0) = x(Object14).

[p13, p2, p15, p10, p4, p8, p9, p5, p7, p14] y(Object0) < y(Object2) < y(Object15) < y(Object7) < y(Object4) < y(Object10) < y(Object12) < y(Object3) < y(Object6) < y(Object11) < y(Object14) => y(Object0) < y(Object14).

N(Object14,Object0) in every valid map => Object14 always qualifies.

[p12] y(Object13) < y(Object0).

Axis comparison contradicts N(Object13,Object0) => Object13 cannot qualify.

[p1] x(Object0) < x(Object1).

Axis comparison contradicts N(Object1,Object0) => Object1 cannot qualify.

[p11] y(Object9) < y(Object0).

Axis comparison contradicts N(Object9,Object0) => Object9 cannot qualify.

[p3] x(Object0) < x(Object8).

Axis comparison contradicts N(Object8,Object0) => Object8 cannot qualify.

[p6] y(Object5) < y(Object0).

Axis comparison contradicts N(Object5,Object0) => Object5 cannot qualify.

Complete possible set = {Object14}. Select C.
Answer: C
```

## rival-chain, depth 3

unique-expansion-rival-chain-d3-s0-which-m1-single

### Problem

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

Object1 has a greater x-coordinate than Object2 (premise sources: 2).

This comparison is incompatible with Object2 being east of Object1, so it cannot qualify.

Object8 has a greater x-coordinate than Object1 (premise sources: 4).

Object8 and Object1 share the same y-coordinate (premise sources: 4).

So Object8 is east of Object1 in every valid map and always qualifies.

Object0 and Object1 share the same x-coordinate (premise sources: 1).

This comparison is incompatible with Object0 being east of Object1, so it cannot qualify.

The x-coordinates increase strictly along Object3, Object5, Object6 and Object1 (premise sources: 5, 3 and 7). So Object1 has a greater x-coordinate than Object3.

This comparison is incompatible with Object3 being east of Object1, so it cannot qualify.

Object4 and Object1 share the same x-coordinate (premise sources: 8).

This comparison is incompatible with Object4 being east of Object1, so it cannot qualify.

Object1 has a greater x-coordinate than Object7 (premise sources: 6).

This comparison is incompatible with Object7 being east of Object1, so it cannot qualify.

The only possible value is Object8. Select C.
Answer: C

### Symbolic

```text
[p2] x(Object2) < x(Object1).

Axis comparison contradicts E(Object2,Object1) => Object2 cannot qualify.

[p4] x(Object1) < x(Object8).

[p4] y(Object8) = y(Object1).

E(Object8,Object1) in every valid map => Object8 always qualifies.

[p1] x(Object0) = x(Object1).

Axis comparison contradicts E(Object0,Object1) => Object0 cannot qualify.

[p5, p3, p7] x(Object3) < x(Object5) < x(Object6) < x(Object1) => x(Object3) < x(Object1).

Axis comparison contradicts E(Object3,Object1) => Object3 cannot qualify.

[p8] x(Object4) = x(Object1).

Axis comparison contradicts E(Object4,Object1) => Object4 cannot qualify.

[p6] x(Object7) < x(Object1).

Axis comparison contradicts E(Object7,Object1) => Object7 cannot qualify.

Complete possible set = {Object8}. Select C.
Answer: C
```

## rival-chain, depth 10

unique-expansion-rival-chain-d10-s0-which-m1-single

### Problem

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

Object15 has a greater x-coordinate than Object12 (premise sources: 15).

This comparison is incompatible with Object12 being east of Object15, so it cannot qualify.

Object9 and Object15 share the same x-coordinate (premise sources: 14).

This comparison is incompatible with Object9 being east of Object15, so it cannot qualify.

Object6 has a greater x-coordinate than Object15 (premise sources: 5).

Object6 and Object15 share the same y-coordinate (premise sources: 5).

So Object6 is east of Object15 in every valid map and always qualifies.

The x-coordinates increase strictly along Object11, Object2, Object10, Object7, Object14, Object13, Object3, Object1, Object4, Object5 and Object15 (premise sources: 7, 1, 11, 6, 4, 10, 12, 2, 9 and 8). So Object15 has a greater x-coordinate than Object11.

This comparison is incompatible with Object11 being east of Object15, so it cannot qualify.

Object0 and Object15 share the same x-coordinate (premise sources: 13).

This comparison is incompatible with Object0 being east of Object15, so it cannot qualify.

Object8 and Object15 share the same x-coordinate (premise sources: 3).

This comparison is incompatible with Object8 being east of Object15, so it cannot qualify.

The only possible value is Object6. Select A.
Answer: A

### Symbolic

```text
[p15] x(Object12) < x(Object15).

Axis comparison contradicts E(Object12,Object15) => Object12 cannot qualify.

[p14] x(Object9) = x(Object15).

Axis comparison contradicts E(Object9,Object15) => Object9 cannot qualify.

[p5] x(Object15) < x(Object6).

[p5] y(Object6) = y(Object15).

E(Object6,Object15) in every valid map => Object6 always qualifies.

[p7, p1, p11, p6, p4, p10, p12, p2, p9, p8] x(Object11) < x(Object2) < x(Object10) < x(Object7) < x(Object14) < x(Object13) < x(Object3) < x(Object1) < x(Object4) < x(Object5) < x(Object15) => x(Object11) < x(Object15).

Axis comparison contradicts E(Object11,Object15) => Object11 cannot qualify.

[p13] x(Object0) = x(Object15).

Axis comparison contradicts E(Object0,Object15) => Object0 cannot qualify.

[p3] x(Object8) = x(Object15).

Axis comparison contradicts E(Object8,Object15) => Object8 cannot qualify.

Complete possible set = {Object6}. Select A.
Answer: A
```

## shared-exclusion, depth 3

unique-expansion-shared-exclusion-d3-s0-which-m1-single

### Problem

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

Following the x-coordinates, Object9 has a smaller x-coordinate than Object8; Object8 has a smaller x-coordinate than Object3; Object3 shares its x-coordinate with Object1 (premise sources: 2, 6 and 9). Therefore, Object1 has a greater x-coordinate than Object9.

This comparison is incompatible with Object1 being west of Object9, so it cannot qualify.

Following the x-coordinates, Object9 has a smaller x-coordinate than Object8; Object8 has a smaller x-coordinate than Object3; Object3 shares its x-coordinate with Object6 (premise sources: 2, 6 and 1). Therefore, Object6 has a greater x-coordinate than Object9.

This comparison is incompatible with Object6 being west of Object9, so it cannot qualify.

Following the x-coordinates, Object9 has a smaller x-coordinate than Object8; Object8 has a smaller x-coordinate than Object3; Object3 shares its x-coordinate with Object5 (premise sources: 2, 6 and 8). Therefore, Object5 has a greater x-coordinate than Object9.

This comparison is incompatible with Object5 being west of Object9, so it cannot qualify.

Following the x-coordinates, Object9 has a smaller x-coordinate than Object8; Object8 has a smaller x-coordinate than Object3; Object3 shares its x-coordinate with Object4 (premise sources: 2, 6 and 4). Therefore, Object4 has a greater x-coordinate than Object9.

This comparison is incompatible with Object4 being west of Object9, so it cannot qualify.

Object9 has a greater x-coordinate than Object2 (premise sources: 5).

Object2 and Object9 share the same y-coordinate (premise sources: 5).

So Object2 is west of Object9 in every valid map and always qualifies.

The x-coordinates increase strictly along Object9, Object8, Object3 and Object7 (premise sources: 2, 6 and 3). So Object7 has a greater x-coordinate than Object9.

This comparison is incompatible with Object7 being west of Object9, so it cannot qualify.

Following the x-coordinates, Object9 has a smaller x-coordinate than Object8; Object8 has a smaller x-coordinate than Object3; Object3 shares its x-coordinate with Object0 (premise sources: 2, 6 and 7). Therefore, Object0 has a greater x-coordinate than Object9.

This comparison is incompatible with Object0 being west of Object9, so it cannot qualify.

The only possible value is Object2. Select C.
Answer: C

### Symbolic

```text
[p2, p6, p9] x(Object9) < x(Object8) < x(Object3) = x(Object1) => x(Object9) < x(Object1).

Axis comparison contradicts W(Object1,Object9) => Object1 cannot qualify.

[p2, p6, p1] x(Object9) < x(Object8) < x(Object3) = x(Object6) => x(Object9) < x(Object6).

Axis comparison contradicts W(Object6,Object9) => Object6 cannot qualify.

[p2, p6, p8] x(Object9) < x(Object8) < x(Object3) = x(Object5) => x(Object9) < x(Object5).

Axis comparison contradicts W(Object5,Object9) => Object5 cannot qualify.

[p2, p6, p4] x(Object9) < x(Object8) < x(Object3) = x(Object4) => x(Object9) < x(Object4).

Axis comparison contradicts W(Object4,Object9) => Object4 cannot qualify.

[p5] x(Object2) < x(Object9).

[p5] y(Object2) = y(Object9).

W(Object2,Object9) in every valid map => Object2 always qualifies.

[p2, p6, p3] x(Object9) < x(Object8) < x(Object3) < x(Object7) => x(Object9) < x(Object7).

Axis comparison contradicts W(Object7,Object9) => Object7 cannot qualify.

[p2, p6, p7] x(Object9) < x(Object8) < x(Object3) = x(Object0) => x(Object9) < x(Object0).

Axis comparison contradicts W(Object0,Object9) => Object0 cannot qualify.

Complete possible set = {Object2}. Select C.
Answer: C
```

## shared-exclusion, depth 10

unique-expansion-shared-exclusion-d10-s0-which-m1-single

### Problem

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

Following the x-coordinates, Object10 has a smaller x-coordinate than Object4; Object4 shares its x-coordinate with Object9; Object9 shares its x-coordinate with Object6; Object6 shares its x-coordinate with Object5; Object5 shares its x-coordinate with Object14; Object14 shares its x-coordinate with Object8; Object8 shares its x-coordinate with Object15; Object15 shares its x-coordinate with Object13; Object13 shares its x-coordinate with Object0; Object0 shares its x-coordinate with Object16 (premise sources: 7, 6, 11, 1, 8, 12, 2, 15, 4 and 3). Therefore, Object16 has a greater x-coordinate than Object10.

This comparison is incompatible with Object10 being south of Object16, so it cannot qualify.

The y-coordinates increase strictly along Object16, Object0, Object13, Object15, Object8, Object14, Object5, Object6, Object9, Object4 and Object11 (premise sources: 3, 4, 15, 2, 12, 8, 1, 11, 6 and 10). So Object11 has a greater y-coordinate than Object16.

This comparison is incompatible with Object11 being south of Object16, so it cannot qualify.

The y-coordinates increase strictly along Object16, Object0, Object13, Object15, Object8, Object14, Object5, Object6, Object9, Object4 and Object3 (premise sources: 3, 4, 15, 2, 12, 8, 1, 11, 6 and 9). So Object3 has a greater y-coordinate than Object16.

This comparison is incompatible with Object3 being south of Object16, so it cannot qualify.

Following the x-coordinates, Object16 shares its x-coordinate with Object0; Object0 shares its x-coordinate with Object13; Object13 shares its x-coordinate with Object15; Object15 shares its x-coordinate with Object8; Object8 shares its x-coordinate with Object14; Object14 shares its x-coordinate with Object5; Object5 shares its x-coordinate with Object6; Object6 shares its x-coordinate with Object9; Object9 shares its x-coordinate with Object4; Object4 has a smaller x-coordinate than Object1 (premise sources: 3, 4, 15, 2, 12, 8, 1, 11, 6 and 16). Therefore, Object1 has a greater x-coordinate than Object16.

This comparison is incompatible with Object1 being south of Object16, so it cannot qualify.

Object2 and Object16 share the same x-coordinate (premise sources: 14).

Object16 has a greater y-coordinate than Object2 (premise sources: 14).

So Object2 is south of Object16 in every valid map and always qualifies.

The y-coordinates increase strictly along Object16, Object0, Object13, Object15, Object8, Object14, Object5, Object6, Object9, Object4 and Object7 (premise sources: 3, 4, 15, 2, 12, 8, 1, 11, 6 and 5). So Object7 has a greater y-coordinate than Object16.

This comparison is incompatible with Object7 being south of Object16, so it cannot qualify.

Following the x-coordinates, Object12 has a smaller x-coordinate than Object4; Object4 shares its x-coordinate with Object9; Object9 shares its x-coordinate with Object6; Object6 shares its x-coordinate with Object5; Object5 shares its x-coordinate with Object14; Object14 shares its x-coordinate with Object8; Object8 shares its x-coordinate with Object15; Object15 shares its x-coordinate with Object13; Object13 shares its x-coordinate with Object0; Object0 shares its x-coordinate with Object16 (premise sources: 13, 6, 11, 1, 8, 12, 2, 15, 4 and 3). Therefore, Object16 has a greater x-coordinate than Object12.

This comparison is incompatible with Object12 being south of Object16, so it cannot qualify.

The only possible value is Object2. Select C.
Answer: C

### Symbolic

```text
[p7, p6, p11, p1, p8, p12, p2, p15, p4, p3] x(Object10) < x(Object4) = x(Object9) = x(Object6) = x(Object5) = x(Object14) = x(Object8) = x(Object15) = x(Object13) = x(Object0) = x(Object16) => x(Object10) < x(Object16).

Axis comparison contradicts S(Object10,Object16) => Object10 cannot qualify.

[p3, p4, p15, p2, p12, p8, p1, p11, p6, p10] y(Object16) < y(Object0) < y(Object13) < y(Object15) < y(Object8) < y(Object14) < y(Object5) < y(Object6) < y(Object9) < y(Object4) < y(Object11) => y(Object16) < y(Object11).

Axis comparison contradicts S(Object11,Object16) => Object11 cannot qualify.

[p3, p4, p15, p2, p12, p8, p1, p11, p6, p9] y(Object16) < y(Object0) < y(Object13) < y(Object15) < y(Object8) < y(Object14) < y(Object5) < y(Object6) < y(Object9) < y(Object4) < y(Object3) => y(Object16) < y(Object3).

Axis comparison contradicts S(Object3,Object16) => Object3 cannot qualify.

[p3, p4, p15, p2, p12, p8, p1, p11, p6, p16] x(Object16) = x(Object0) = x(Object13) = x(Object15) = x(Object8) = x(Object14) = x(Object5) = x(Object6) = x(Object9) = x(Object4) < x(Object1) => x(Object16) < x(Object1).

Axis comparison contradicts S(Object1,Object16) => Object1 cannot qualify.

[p14] x(Object2) = x(Object16).

[p14] y(Object2) < y(Object16).

S(Object2,Object16) in every valid map => Object2 always qualifies.

[p3, p4, p15, p2, p12, p8, p1, p11, p6, p5] y(Object16) < y(Object0) < y(Object13) < y(Object15) < y(Object8) < y(Object14) < y(Object5) < y(Object6) < y(Object9) < y(Object4) < y(Object7) => y(Object16) < y(Object7).

Axis comparison contradicts S(Object7,Object16) => Object7 cannot qualify.

[p13, p6, p11, p1, p8, p12, p2, p15, p4, p3] x(Object12) < x(Object4) = x(Object9) = x(Object6) = x(Object5) = x(Object14) = x(Object8) = x(Object15) = x(Object13) = x(Object0) = x(Object16) => x(Object12) < x(Object16).

Axis comparison contradicts S(Object12,Object16) => Object12 cannot qualify.

Complete possible set = {Object2}. Select C.
Answer: C
```
