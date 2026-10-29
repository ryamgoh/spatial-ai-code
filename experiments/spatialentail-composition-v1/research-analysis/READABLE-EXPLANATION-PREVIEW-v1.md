# Readable Explanation Preview

Eleven actual renderer outputs for readability feedback. This is not a training set, benchmark, full pilot rerender, or release audit. Natural and Symbolic use the same selected explanation units. All meaningful path links and complete selected witness coordinates remain; internal proof IDs do not. Citations refer to the numbered premise sentences. Symbols N/NE/E/SE/S/SW/W/NW denote the eight exact directions.

Scope: exact-direction positive paths, Unique ambiguity witnesses, selected membership cases, supported Count constraints, two conflict types and missing-menu witnesses. Complete ambiguous-Direction set explanations remain explicitly unsupported by this preview renderer. The old renderers and datasets have not been changed.

The text-integrity checker regenerates the expected templates from checked evidence. It is not an independent natural-language theorem prover. Tests inspect specific rendered comparisons, and actual examples still require readability and semantic review.

## The same depth-ten question

ID: `depth-10-sample-4-direction-m0-all-possible`. Contract: `all-possible`. natural: 236 target tokens (238 with ending); admitted=True; symbolic: 262 target tokens (264 with ending); admitted=True.

### Question

```text
Objects: Object8, Object3, Object15, Object7, Object4, Object1, Object14, Object11, Object16, Object12, Object2, Object0, Object5, Object6, Object9, Object13, Object10
Query: direction Object16 Object14
Consider a map with multiple objects:
Object1 is to the South of Object14.
Object0 is to the South of Object2.
Object14 is to the North of Object13.
Object12 is to the South of Object16.
Object11 is to the South of Object14.
Object7 is to the South of Object12.
Object3 is to the South of Object10.
Object9 is to the South of Object5.
Object10 is to the South of Object9.
Object14 is to the North of Object4.
Object0 is to the North of Object14.
Object5 is to the South of Object7.
Object15 is to the North of Object2.
Object15 is to the South of Object3.
Object8 is to the Southwest of Object6.
Question: Select the complete set of directions Object16 could occupy relative to Object14.
A. North
B. South
C. Southeast
D. Southwest
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

Object14, Object0, Object2, Object15, Object3, Object10, Object9, Object5, Object7, Object12 and Object16 share the same x-coordinate (premise sources: 11, 2, 13, 14, 7, 9, 8, 12, 6 and 4).

The y-coordinates increase strictly along Object14, Object0, Object2, Object15, Object3, Object10, Object9, Object5, Object7, Object12 and Object16 (premise sources: 11, 2, 13, 14, 7, 9, 8, 12, 6 and 4). So Object16 has a greater y-coordinate than Object14.

Together, these comparisons place Object16 exactly north of Object14. The x-comparison rules out northeast, east, southeast, southwest, west and northwest. The y-comparison rules out south.

The only possible value is North. Select A.
Answer: A

### Symbolic

```text
[p11, p2, p13, p14, p7, p9, p8, p12, p6, p4] x(Object14) = x(Object0) = x(Object2) = x(Object15) = x(Object3) = x(Object10) = x(Object9) = x(Object5) = x(Object7) = x(Object12) = x(Object16) => x(Object14) = x(Object16).

[p11, p2, p13, p14, p7, p9, p8, p12, p6, p4] y(Object14) < y(Object0) < y(Object2) < y(Object15) < y(Object3) < y(Object10) < y(Object9) < y(Object5) < y(Object7) < y(Object12) < y(Object16) => y(Object14) < y(Object16).

=> N(Object16,Object14).
x-comparison excludes {NE, E, SE, SW, W, NW}.
y-comparison excludes {S}.

Complete possible set = {North}. Select A.
Answer: A
```

## Mixed-direction composition

ID: `D1`. Contract: `single`. natural: 131 target tokens (133 with ending); admitted=True; symbolic: 99 target tokens (101 with ending); admitted=True.

### Question

```text
Objects: X, Y, Z
Query: direction X Z
Consider a map with multiple objects:
X is to the Northeast of Y.
Z is to the South of Y.
Question: In which direction is X relative to Z in every valid map?
A. West
B. Northeast
C. South
D. Northwest
E. Cannot be determined
F. None of the above
```

### Natural

Following the x-coordinates, Z shares its x-coordinate with Y; Y has a smaller x-coordinate than X (premise sources: 2 and 1). Therefore, X has a greater x-coordinate than Z.

The y-coordinates increase strictly along Z, Y and X (premise sources: 2 and 1). So X has a greater y-coordinate than Z.

Together, these comparisons place X exactly northeast of Z. The x-comparison rules out north, south, southwest, west and northwest. The y-comparison rules out east and southeast.

The only possible value is Northeast. Select B.
Answer: B

### Symbolic

```text
[p2, p1] x(Z) = x(Y) < x(X) => x(Z) < x(X).

[p2, p1] y(Z) < y(Y) < y(X) => y(Z) < y(X).

=> NE(X,Z).
x-comparison excludes {N, S, SW, W, NW}.
y-comparison excludes {E, SE}.

Complete possible set = {Northeast}. Select B.
Answer: B
```

## Ambiguous Direction

ID: `D2`. Contract: `single`. natural: 106 target tokens (108 with ending); admitted=True; symbolic: 82 target tokens (84 with ending); admitted=True.

### Question

```text
Objects: X, Y, O
Query: direction X Y
Consider a map with multiple objects:
X is to the East of O.
Y is to the East of O.
Question: In which direction is X relative to Y in every valid map?
A. West
B. East
C. South
D. Northeast
E. Cannot be determined
F. None of the above
```

### Natural

For example, one valid arrangement consistent with all premises places X at (2, 0); Y at (1, 0); O at (0, 0). This gives the direction east.

For example, one valid arrangement consistent with all premises places X at (1, 0); Y at (2, 0); O at (0, 0). This gives the direction west.

These valid maps have different outcomes, so the answer is not uniquely determined. Select E.
Answer: E

### Symbolic

```text
Example valid map satisfying all premises: X=(2,0), Y=(1,0), O=(0,0). direction=East.

Example valid map satisfying all premises: X=(1,0), Y=(2,0), O=(0,0). direction=West.

These valid maps have different outcomes, so the answer is not uniquely determined. Select E.
Answer: E
```

## Unique Selection

ID: `S1`. Contract: `single`. natural: 187 target tokens (189 with ending); admitted=True; symbolic: 155 target tokens (157 with ending); admitted=True.

### Question

```text
Objects: O, Q, U, V, W, T
Query: which North O
Eligible: U, V, W, T
Consider a map with multiple objects:
U is to the North of Q.
Q is to the North of O.
V is to the East of O.
W is to the South of O.
T is to the West of O.
Question: Which eligible object is the only one north of O in every valid map?
A. U
B. V
C. W
D. T
E. Cannot be determined
F. None of the above
```

### Natural

O, Q and U share the same x-coordinate (premise sources: 2 and 1).

The y-coordinates increase strictly along O, Q and U (premise sources: 2 and 1). So U has a greater y-coordinate than O.

So U is north of O in every valid map and always qualifies.

V has a greater x-coordinate than O (premise sources: 3).

This comparison is incompatible with V being north of O, so it cannot qualify.

O has a greater y-coordinate than W (premise sources: 4).

This comparison is incompatible with W being north of O, so it cannot qualify.

O has a greater x-coordinate than T (premise sources: 5).

This comparison is incompatible with T being north of O, so it cannot qualify.

The only possible value is U. Select A.
Answer: A

### Symbolic

```text
[p2, p1] x(O) = x(Q) = x(U) => x(O) = x(U).

[p2, p1] y(O) < y(Q) < y(U) => y(O) < y(U).

N(U,O) in every valid map => U always qualifies.

[p3] x(O) < x(V).

Axis comparison contradicts N(V,O) => V cannot qualify.

[p4] y(W) < y(O).

Axis comparison contradicts N(W,O) => W cannot qualify.

[p5] x(T) < x(O).

Axis comparison contradicts N(T,O) => T cannot qualify.

Complete possible set = {U}. Select A.
Answer: A
```

## Contingent Selection

ID: `S4`. Contract: `single`. natural: 269 target tokens (271 with ending); admitted=True; symbolic: 204 target tokens (206 with ending); admitted=True.

### Question

```text
Objects: O, U, V, W, T
Query: which North O
Eligible: U, V, W, T
Consider a map with multiple objects:
U is to the Northeast of V.
O is to the Northeast of V.
W is to the East of O.
T is to the South of O.
Question: Which eligible object is the only one north of O in every valid map?
A. U
B. V
C. W
D. T
E. Cannot be determined
F. None of the above
```

### Natural

For example, one valid arrangement consistent with all premises places O at (1, 2); U at (1, 3); V at (0, 0); W at (2, 2); T at (1, 1). In this arrangement, only U qualifies.

For example, one valid arrangement consistent with all premises places O at (1, 3); U at (1, 2); V at (0, 0); W at (2, 3); T at (1, 1). In this arrangement, no eligible objects qualify.

U qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

O has a greater x-coordinate than V (premise sources: 2).

This comparison is incompatible with V being north of O, so it cannot qualify.

W has a greater x-coordinate than O (premise sources: 3).

This comparison is incompatible with W being north of O, so it cannot qualify.

O has a greater y-coordinate than T (premise sources: 4).

This comparison is incompatible with T being north of O, so it cannot qualify.

The evidence does not establish one object as the only match in every valid map. Select E.
Answer: E

### Symbolic

```text
Example valid map satisfying all premises: O=(1,2), U=(1,3), V=(0,0), W=(2,2), T=(1,1). Qualifying objects = {U}.

Example valid map satisfying all premises: O=(1,3), U=(1,2), V=(0,0), W=(2,3), T=(1,1). Qualifying objects = {}.

U qualifies in map 1, not map 2 => possible, not guaranteed.

[p2] x(V) < x(O).

Axis comparison contradicts N(V,O) => V cannot qualify.

[p3] x(O) < x(W).

Axis comparison contradicts N(W,O) => W cannot qualify.

[p4] y(T) < y(O).

Axis comparison contradicts N(T,O) => T cannot qualify.

The evidence does not establish one object as the only match in every valid map. Select E.
Answer: E
```

## Invariant Count

ID: `C2`. Contract: `single`. natural: 316 target tokens (318 with ending); admitted=True; symbolic: 251 target tokens (253 with ending); admitted=True.

### Question

```text
Objects: O, U, V, W, T
Query: count North O
Eligible: U, V, W, T
Consider a map with multiple objects:
U is to the North of O.
V is to the North of U.
W is to the East of O.
T is to the South of O.
Question: What exact number of eligible objects are north of O in every valid map?
A. 0
B. 1
C. 2
D. 3
E. Cannot be determined
F. None of the above
```

### Natural

O and U share the same x-coordinate (premise sources: 1).

U has a greater y-coordinate than O (premise sources: 1).

So U is north of O in every valid map and always qualifies.

O, U and V share the same x-coordinate (premise sources: 1 and 2).

The y-coordinates increase strictly along O, U and V (premise sources: 1 and 2). So V has a greater y-coordinate than O.

So V is north of O in every valid map and always qualifies.

W has a greater x-coordinate than O (premise sources: 3).

This comparison is incompatible with W being north of O, so it cannot qualify.

O has a greater y-coordinate than T (premise sources: 4).

This comparison is incompatible with T being north of O, so it cannot qualify.

Every qualifying group must include U and V. W and T cannot qualify. There are no additional forbidden pairs. Checking the eligible-object subsets against these constraints leaves only size 2. Counts 0, 1, 3 and 4 are therefore impossible.

For example, one valid arrangement consistent with all premises places O at (0, 1); U at (0, 2); V at (0, 3); W at (1, 1); T at (0, 0). This gives a count of 2.

The only possible value is 2. Select C.
Answer: C

### Symbolic

```text
[p1] x(O) = x(U).

[p1] y(O) < y(U).

N(U,O) in every valid map => U always qualifies.

[p1, p2] x(O) = x(U) = x(V) => x(O) = x(V).

[p1, p2] y(O) < y(U) < y(V) => y(O) < y(V).

N(V,O) in every valid map => V always qualifies.

[p3] x(O) < x(W).

Axis comparison contradicts N(W,O) => W cannot qualify.

[p4] y(T) < y(O).

Axis comparison contradicts N(T,O) => T cannot qualify.

Required={U, V}; excluded={W, T}; forbidden pairs={}.
Joint membership subsets satisfying these constraints have sizes {2}; excluded counts={0, 1, 3, 4}.

Example valid map satisfying all premises: O=(0,1), U=(0,2), V=(0,3), W=(1,1), T=(0,0). count=2.

Complete possible set = {2}. Select C.
Answer: C
```

## Zero Count

ID: `C3`. Contract: `single`. natural: 266 target tokens (268 with ending); admitted=True; symbolic: 200 target tokens (202 with ending); admitted=True.

### Question

```text
Objects: O, U, V, W, T
Query: count North O
Eligible: U, V, W, T
Consider a map with multiple objects:
U is to the East of O.
V is to the West of O.
W is to the South of O.
T is to the Southeast of O.
Question: What exact number of eligible objects are north of O in every valid map?
A. 0
B. 1
C. 2
D. 3
E. Cannot be determined
F. None of the above
```

### Natural

U has a greater x-coordinate than O (premise sources: 1).

This comparison is incompatible with U being north of O, so it cannot qualify.

O has a greater x-coordinate than V (premise sources: 2).

This comparison is incompatible with V being north of O, so it cannot qualify.

O has a greater y-coordinate than W (premise sources: 3).

This comparison is incompatible with W being north of O, so it cannot qualify.

T has a greater x-coordinate than O (premise sources: 4).

This comparison is incompatible with T being north of O, so it cannot qualify.

No object is required to qualify. U, V, W and T cannot qualify. There are no additional forbidden pairs. Checking the eligible-object subsets against these constraints leaves only size 0. Counts 1, 2, 3 and 4 are therefore impossible.

For example, one valid arrangement consistent with all premises places O at (1, 2); U at (2, 2); V at (0, 2); W at (1, 0); T at (2, 1). This gives a count of 0.

The only possible value is 0. Select A.
Answer: A

### Symbolic

```text
[p1] x(O) < x(U).

Axis comparison contradicts N(U,O) => U cannot qualify.

[p2] x(V) < x(O).

Axis comparison contradicts N(V,O) => V cannot qualify.

[p3] y(W) < y(O).

Axis comparison contradicts N(W,O) => W cannot qualify.

[p4] x(O) < x(T).

Axis comparison contradicts N(T,O) => T cannot qualify.

Required={}; excluded={U, V, W, T}; forbidden pairs={}.
Joint membership subsets satisfying these constraints have sizes {0}; excluded counts={1, 2, 3, 4}.

Example valid map satisfying all premises: O=(1,2), U=(2,2), V=(0,2), W=(1,0), T=(2,1). count=0.

Complete possible set = {0}. Select A.
Answer: A
```

## Joint-variable Count

ID: `C1`. Contract: `all-possible`. natural: 373 target tokens (375 with ending); admitted=True; symbolic: 283 target tokens (285 with ending); admitted=True.

### Question

```text
Objects: O, U, V, W, T
Query: count North O
Eligible: U, V, W, T
Consider a map with multiple objects:
U is to the East of V.
W is to the South of O.
T is to the South of W.
Question: Select the complete set of possible counts of eligible objects north of O.
A. 0
B. 1
C. 2
D. 3
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

O has a greater y-coordinate than W (premise sources: 2).

This comparison is incompatible with W being north of O, so it cannot qualify.

The y-coordinates increase strictly along T, W and O (premise sources: 3 and 2). So O has a greater y-coordinate than T.

This comparison is incompatible with T being north of O, so it cannot qualify.

Suppose U and V both qualify (temporary assumptions 1 and 2).

The cited relations would require that V has a smaller x-coordinate than U; U shares its x-coordinate with O; O shares its x-coordinate with V (premise sources: 1, temporary assumption 1 and temporary assumption 2).

This would make a coordinate smaller than itself, which is impossible.

Therefore, U and V cannot qualify together.

No object is required to qualify. W and T cannot qualify. No qualifying group can contain both members of any forbidden pair: U and V. Checking the eligible-object subsets against these constraints leaves only sizes 0 and 1. Counts 2, 3 and 4 are therefore impossible.

For example, one valid arrangement consistent with all premises places O at (1, 3); U at (1, 2); V at (0, 2); W at (1, 1); T at (1, 0). This gives a count of 0.

For example, one valid arrangement consistent with all premises places O at (1, 2); U at (1, 3); V at (0, 3); W at (1, 1); T at (1, 0). This gives a count of 1.

The complete possible values are 0 and 1. Select AB.
Answer: AB

### Symbolic

```text
[p2] y(W) < y(O).

Axis comparison contradicts N(W,O) => W cannot qualify.

[p3, p2] y(T) < y(W) < y(O) => y(T) < y(O).

Axis comparison contradicts N(T,O) => T cannot qualify.

Assume U and V both qualify (temporary assumptions 1 and 2).

[p1, assumption 1, assumption 2] x(V) < x(U) = x(O) = x(V) => x(V) < x(V).

x(V) < x(V) is impossible.

Therefore, U and V cannot qualify together.

Required={}; excluded={W, T}; forbidden pairs={{U, V}}.
Joint membership subsets satisfying these constraints have sizes {0, 1}; excluded counts={2, 3, 4}.

Example valid map satisfying all premises: O=(1,3), U=(1,2), V=(0,2), W=(1,1), T=(1,0). count=0.

Example valid map satisfying all premises: O=(1,2), U=(1,3), V=(0,3), W=(1,1), T=(1,0). count=1.

Complete possible set = {0, 1}. Select AB.
Answer: AB
```

## Strict-cycle contradiction

ID: `I1`. Contract: `single`. natural: 78 target tokens (80 with ending); admitted=True; symbolic: 61 target tokens (63 with ending); admitted=True.

### Question

```text
Objects: X, Y, Z, U, V
Query: direction U V
Consider a map with multiple objects:
X is to the East of Y.
Y is to the East of Z.
Z is to the East of X.
U is to the North of V.
Question: In which direction is U relative to V in every valid map?
A. North
B. East
C. South
D. West
E. Cannot be determined
F. None of the above
```

### Natural

The cited relations would require that Y has a smaller x-coordinate than X; X has a smaller x-coordinate than Z; Z has a smaller x-coordinate than Y (premise sources: 1, 3 and 2).

This would make a coordinate smaller than itself, which is impossible.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E

### Symbolic

```text
[p1, p3, p2] x(Y) < x(X) < x(Z) < x(Y) => x(Y) < x(Y).

x(Y) < x(Y) is impossible.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E
```

## Forced-coincidence contradiction

ID: `I3`. Contract: `single`. natural: 87 target tokens (89 with ending); admitted=True; symbolic: 92 target tokens (94 with ending); admitted=True.

### Question

```text
Objects: X, Y, O, T
Query: direction X O
Consider a map with multiple objects:
X is to the North of O.
Y is to the North of O.
X is to the East of T.
Y is to the East of T.
Question: In which direction is X relative to O in every valid map?
A. North
B. East
C. South
D. West
E. Cannot be determined
F. None of the above
```

### Natural

X, O and Y share the same x-coordinate (premise sources: 1 and 2).

X, T and Y share the same y-coordinate (premise sources: 3 and 4).

X and Y would have both coordinates equal. They are distinct objects, so the no-coincidence rule makes this impossible.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E

### Symbolic

```text
[p1, p2] x(X) = x(O) = x(Y) => x(X) = x(Y).

[p3, p4] y(X) = y(T) = y(Y) => y(X) = y(Y).

x(X)=x(Y) and y(X)=y(Y), but X!=Y: distinct-point contradiction.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E
```

## Unlisted possible answer

ID: `S6`. Contract: `all-possible`. natural: 93 target tokens (95 with ending); admitted=True; symbolic: 76 target tokens (78 with ending); admitted=True.

### Question

```text
Objects: O, U, V, W, T, Y
Query: which North O
Eligible: U, V, W, T, Y
Consider a map with multiple objects:
U is to the East of O.
V is to the West of O.
W is to the South of O.
T is to the Southeast of O.
Y is to the North of O.
Question: Select the complete set of eligible objects that could be north of O.
A. U
B. V
C. W
D. T
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

For example, one valid arrangement consistent with all premises places O at (1, 2); U at (2, 2); V at (0, 2); W at (1, 0); T at (2, 1); Y at (1, 3). In this arrangement, only Y qualifies.

The witnessed value Y is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F

### Symbolic

```text
Example valid map satisfying all premises: O=(1,2), U=(2,2), V=(0,2), W=(1,0), T=(2,1), Y=(1,3). Qualifying objects = {Y}.

The witnessed value Y is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F
```
