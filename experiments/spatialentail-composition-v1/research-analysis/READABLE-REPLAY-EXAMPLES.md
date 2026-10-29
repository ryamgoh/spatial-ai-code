# Readable Replay Examples

23 distinct saved pairs selected for inspection from the 987-view replay. Selection includes both output maxima, long depth-9/10 cases by query kind, ambiguous Direction contracts, joint Count, and escape/conflict cases. No examples were newly sampled. Full prompts, checked outlines, certificates, and token measurements remain in depth-feasibility-readable-v2/attempts.jsonl.

Numerical coordinates are example maps, not additional distance assumptions. Natural and Symbolic use the same selected evidence units. This is a template-integrity-checked export, not an independent human evaluation.

## Longest natural output; Longest symbolic output

ID: `random-atoms-4-which-m0-single`. Kind: `which`. Contract: `single`. Case: `underdetermined`. Depth: None. Target tokens: Natural 1514; Symbolic 1182.

### Question

```text
Objects: Object1, Object2, Object7, Object5, Object6, Object3, Object0, Object4
Query: which North Object2
Eligible: Object5, Object3, Object1, Object0, Object6, Object4, Object7
Consider a map with multiple objects:
Object7 is to the North of Object3.
Object6 is to the Northeast of Object5.
Question: Which eligible object is the only one north of Object2 in every valid map?
A. Object3
B. Object5
C. Object1
D. Object7
E. Cannot be determined
F. None of the above
```

### Natural

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 4); Object7 at (0, 3); Object5 at (0, 5); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 and Object5 qualify.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 5); Object7 at (0, 4); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 qualifies.

Object5 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 3); Object7 at (0, 5); Object5 at (0, 2); Object6 at (1, 7); Object3 at (0, 4); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1, Object3 and Object7 qualify.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 5); Object7 at (0, 4); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 qualifies.

Object3 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 5); Object7 at (0, 4); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 qualifies.

For example, one valid arrangement consistent with all premises places Object1 at (1, 6); Object2 at (0, 0); Object7 at (1, 5); Object5 at (1, 4); Object6 at (2, 7); Object3 at (1, 3); Object0 at (1, 2); Object4 at (1, 1). In this arrangement, no eligible objects qualify.

Object1 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 4); Object7 at (0, 3); Object5 at (0, 2); Object6 at (1, 7); Object3 at (0, 1); Object0 at (0, 5); Object4 at (0, 0). In this arrangement, only Object0 and Object1 qualify.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 5); Object7 at (0, 4); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 qualifies.

Object0 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

For example, one valid arrangement consistent with all premises places Object1 at (1, 7); Object2 at (1, 5); Object7 at (1, 4); Object5 at (0, 0); Object6 at (1, 6); Object3 at (1, 3); Object0 at (1, 2); Object4 at (1, 1). In this arrangement, only Object1 and Object6 qualify.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 5); Object7 at (0, 4); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 qualifies.

Object6 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 4); Object7 at (0, 3); Object5 at (0, 2); Object6 at (1, 7); Object3 at (0, 1); Object0 at (0, 0); Object4 at (0, 5). In this arrangement, only Object1 and Object4 qualify.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 5); Object7 at (0, 4); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 qualifies.

Object4 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 4); Object7 at (0, 5); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 and Object7 qualify.

For example, one valid arrangement consistent with all premises places Object1 at (0, 6); Object2 at (0, 5); Object7 at (0, 4); Object5 at (0, 3); Object6 at (1, 7); Object3 at (0, 2); Object0 at (0, 1); Object4 at (0, 0). In this arrangement, only Object1 qualifies.

Object7 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

The evidence does not establish one object as the only match in every valid map. Select E.
Answer: E

### Symbolic

```text
Example valid map satisfying all premises: Object1=(0,6), Object2=(0,4), Object7=(0,3), Object5=(0,5), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1, Object5}.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,5), Object7=(0,4), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1}.

Object5 qualifies in map 1, not map 2 => possible, not guaranteed.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,3), Object7=(0,5), Object5=(0,2), Object6=(1,7), Object3=(0,4), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1, Object3, Object7}.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,5), Object7=(0,4), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1}.

Object3 qualifies in map 1, not map 2 => possible, not guaranteed.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,5), Object7=(0,4), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1}.

Example valid map satisfying all premises: Object1=(1,6), Object2=(0,0), Object7=(1,5), Object5=(1,4), Object6=(2,7), Object3=(1,3), Object0=(1,2), Object4=(1,1). Qualifying objects = {}.

Object1 qualifies in map 1, not map 2 => possible, not guaranteed.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,4), Object7=(0,3), Object5=(0,2), Object6=(1,7), Object3=(0,1), Object0=(0,5), Object4=(0,0). Qualifying objects = {Object0, Object1}.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,5), Object7=(0,4), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1}.

Object0 qualifies in map 1, not map 2 => possible, not guaranteed.

Example valid map satisfying all premises: Object1=(1,7), Object2=(1,5), Object7=(1,4), Object5=(0,0), Object6=(1,6), Object3=(1,3), Object0=(1,2), Object4=(1,1). Qualifying objects = {Object1, Object6}.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,5), Object7=(0,4), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1}.

Object6 qualifies in map 1, not map 2 => possible, not guaranteed.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,4), Object7=(0,3), Object5=(0,2), Object6=(1,7), Object3=(0,1), Object0=(0,0), Object4=(0,5). Qualifying objects = {Object1, Object4}.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,5), Object7=(0,4), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1}.

Object4 qualifies in map 1, not map 2 => possible, not guaranteed.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,4), Object7=(0,5), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1, Object7}.

Example valid map satisfying all premises: Object1=(0,6), Object2=(0,5), Object7=(0,4), Object5=(0,3), Object6=(1,7), Object3=(0,2), Object0=(0,1), Object4=(0,0). Qualifying objects = {Object1}.

Object7 qualifies in map 1, not map 2 => possible, not guaranteed.

The evidence does not establish one object as the only match in every valid map. Select E.
Answer: E
```

## Same saved depth-ten Direction

ID: `depth-10-sample-4-direction-m0-all-possible`. Kind: `direction`. Contract: `all-possible`. Case: `unique`. Depth: 10. Target tokens: Natural 236; Symbolic 262.

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

## Longest Natural at depth 9, direction

ID: `depth-9-sample-3-direction-m1-single`. Kind: `direction`. Contract: `single`. Case: `unique`. Depth: 9. Target tokens: Natural 383; Symbolic 253.

### Question

```text
Objects: Object12, Object9, Object19, Object10, Object17, Object15, Object3, Object14, Object7, Object6, Object18, Object4, Object0, Object16, Object8, Object20, Object2, Object5, Object13, Object11, Object1
Query: direction Object17 Object19
Consider a map with multiple objects:
Object7 is to the North of Object10.
Object8 is to the Northwest of Object12.
Object15 is to the East of Object6.
Object15 is to the North of Object14.
Object18 is to the South of Object6.
Object15 is to the South of Object0.
Object1 is to the North of Object17.
Object4 is to the West of Object18.
Object9 is to the South of Object3.
Object19 is to the Northeast of Object13.
Object7 is to the North of Object20.
Object19 is to the Northeast of Object2.
Object5 is to the Southwest of Object19.
Object4 is to the North of Object1.
Object19 is to the Northeast of Object7.
Object0 is to the Southwest of Object10.
Object16 is to the Southwest of Object19.
Object11 is to the East of Object10.
Question: In which direction is Object17 relative to Object19 in every valid map?
A. West
B. Northeast
C. South
D. Northwest
E. Cannot be determined
F. None of the above
```

### Natural

Following the x-coordinates, Object17 shares its x-coordinate with Object1; Object1 shares its x-coordinate with Object4; Object4 has a smaller x-coordinate than Object18; Object18 shares its x-coordinate with Object6; Object6 has a smaller x-coordinate than Object15; Object15 shares its x-coordinate with Object0; Object0 has a smaller x-coordinate than Object10; Object10 shares its x-coordinate with Object7; Object7 has a smaller x-coordinate than Object19 (premise sources: 7, 14, 8, 5, 3, 6, 16, 1 and 15). Therefore, Object19 has a greater x-coordinate than Object17.

Following the y-coordinates, Object17 has a smaller y-coordinate than Object1; Object1 has a smaller y-coordinate than Object4; Object4 shares its y-coordinate with Object18; Object18 has a smaller y-coordinate than Object6; Object6 shares its y-coordinate with Object15; Object15 has a smaller y-coordinate than Object0; Object0 has a smaller y-coordinate than Object10; Object10 has a smaller y-coordinate than Object7; Object7 has a smaller y-coordinate than Object19 (premise sources: 7, 14, 8, 5, 3, 6, 16, 1 and 15). Therefore, Object19 has a greater y-coordinate than Object17.

Together, these comparisons place Object17 exactly southwest of Object19. The x-comparison rules out north, northeast, east, southeast and south. The y-comparison rules out west and northwest.

The only possible value is Southwest. The unique value is unlisted, so select F.
Answer: F

### Symbolic

```text
[p7, p14, p8, p5, p3, p6, p16, p1, p15] x(Object17) = x(Object1) = x(Object4) < x(Object18) = x(Object6) < x(Object15) = x(Object0) < x(Object10) = x(Object7) < x(Object19) => x(Object17) < x(Object19).

[p7, p14, p8, p5, p3, p6, p16, p1, p15] y(Object17) < y(Object1) < y(Object4) = y(Object18) < y(Object6) = y(Object15) < y(Object0) < y(Object10) < y(Object7) < y(Object19) => y(Object17) < y(Object19).

=> SW(Object17,Object19).
x-comparison excludes {N, NE, E, SE, S}.
y-comparison excludes {W, NW}.

Complete possible set = {Southwest}. The unique value is unlisted, so select F.
Answer: F
```

## Longest Natural at depth 9, which

ID: `depth-9-sample-2-which-m0-single`. Kind: `which`. Contract: `single`. Case: `unique`. Depth: 9. Target tokens: Natural 511; Symbolic 360.

### Question

```text
Objects: Object1, Object0, Object4, Object2, Object6, Object5, Object8, Object16, Object7, Object12, Object3, Object9, Object14, Object11, Object13, Object15, Object10
Query: which Northeast Object14
Eligible: Object15, Object0, Object13, Object16, Object5
Consider a map with multiple objects:
Object13 is to the Southwest of Object14.
Object8 is to the South of Object3.
Object2 is to the West of Object16.
Object1 is to the Northeast of Object14.
Object6 is to the East of Object7.
Object4 is to the Northeast of Object11.
Object7 is to the North of Object1.
Object14 is to the Northeast of Object0.
Object2 is to the Northeast of Object4.
Object5 is to the Southwest of Object14.
Object10 is to the Northwest of Object7.
Object2 is to the Southwest of Object9.
Object14 is to the Northeast of Object15.
Object8 is to the East of Object6.
Object11 is to the East of Object3.
Object14 is to the South of Object12.
Question: Which eligible object is the only one northeast of Object14 in every valid map?
A. Object13
B. Object16
C. Object15
D. Object0
E. Cannot be determined
F. None of the above
```

### Natural

Object14 has a greater x-coordinate than Object15 (premise sources: 13).

This comparison is incompatible with Object15 being northeast of Object14, so it cannot qualify.

Object14 has a greater x-coordinate than Object0 (premise sources: 8).

This comparison is incompatible with Object0 being northeast of Object14, so it cannot qualify.

Object14 has a greater x-coordinate than Object13 (premise sources: 1).

This comparison is incompatible with Object13 being northeast of Object14, so it cannot qualify.

Following the x-coordinates, Object14 has a smaller x-coordinate than Object1; Object1 shares its x-coordinate with Object7; Object7 has a smaller x-coordinate than Object6; Object6 has a smaller x-coordinate than Object8; Object8 shares its x-coordinate with Object3; Object3 has a smaller x-coordinate than Object11; Object11 has a smaller x-coordinate than Object4; Object4 has a smaller x-coordinate than Object2; Object2 has a smaller x-coordinate than Object16 (premise sources: 4, 7, 5, 14, 2, 15, 6, 9 and 3). Therefore, Object16 has a greater x-coordinate than Object14.

Following the y-coordinates, Object14 has a smaller y-coordinate than Object1; Object1 has a smaller y-coordinate than Object7; Object7 shares its y-coordinate with Object6; Object6 shares its y-coordinate with Object8; Object8 has a smaller y-coordinate than Object3; Object3 shares its y-coordinate with Object11; Object11 has a smaller y-coordinate than Object4; Object4 has a smaller y-coordinate than Object2; Object2 shares its y-coordinate with Object16 (premise sources: 4, 7, 5, 14, 2, 15, 6, 9 and 3). Therefore, Object16 has a greater y-coordinate than Object14.

So Object16 is northeast of Object14 in every valid map and always qualifies.

Object14 has a greater x-coordinate than Object5 (premise sources: 10).

This comparison is incompatible with Object5 being northeast of Object14, so it cannot qualify.

The only possible value is Object16. Select B.
Answer: B

### Symbolic

```text
[p13] x(Object15) < x(Object14).

Axis comparison contradicts NE(Object15,Object14) => Object15 cannot qualify.

[p8] x(Object0) < x(Object14).

Axis comparison contradicts NE(Object0,Object14) => Object0 cannot qualify.

[p1] x(Object13) < x(Object14).

Axis comparison contradicts NE(Object13,Object14) => Object13 cannot qualify.

[p4, p7, p5, p14, p2, p15, p6, p9, p3] x(Object14) < x(Object1) = x(Object7) < x(Object6) < x(Object8) = x(Object3) < x(Object11) < x(Object4) < x(Object2) < x(Object16) => x(Object14) < x(Object16).

[p4, p7, p5, p14, p2, p15, p6, p9, p3] y(Object14) < y(Object1) < y(Object7) = y(Object6) = y(Object8) < y(Object3) = y(Object11) < y(Object4) < y(Object2) = y(Object16) => y(Object14) < y(Object16).

NE(Object16,Object14) in every valid map => Object16 always qualifies.

[p10] x(Object5) < x(Object14).

Axis comparison contradicts NE(Object5,Object14) => Object5 cannot qualify.

Complete possible set = {Object16}. Select B.
Answer: B
```

## Longest Natural at depth 9, count

ID: `depth-9-sample-2-count-m0-single`. Kind: `count`. Contract: `single`. Case: `unique`. Depth: 9. Target tokens: Natural 761; Symbolic 557.

### Question

```text
Objects: Object1, Object0, Object4, Object2, Object6, Object5, Object8, Object16, Object7, Object12, Object3, Object9, Object14, Object11, Object13, Object15, Object10
Query: count Northeast Object14
Eligible: Object15, Object0, Object13, Object16, Object5
Consider a map with multiple objects:
Object13 is to the Southwest of Object14.
Object8 is to the South of Object3.
Object2 is to the West of Object16.
Object1 is to the Northeast of Object14.
Object6 is to the East of Object7.
Object4 is to the Northeast of Object11.
Object7 is to the North of Object1.
Object14 is to the Northeast of Object0.
Object2 is to the Northeast of Object4.
Object5 is to the Southwest of Object14.
Object10 is to the Northwest of Object7.
Object2 is to the Southwest of Object9.
Object14 is to the Northeast of Object15.
Object8 is to the East of Object6.
Object11 is to the East of Object3.
Object14 is to the South of Object12.
Question: What exact number of eligible objects are northeast of Object14 in every valid map?
A. 2
B. 5
C. 0
D. 1
E. Cannot be determined
F. None of the above
```

### Natural

Object14 has a greater x-coordinate than Object15 (premise sources: 13).

This comparison is incompatible with Object15 being northeast of Object14, so it cannot qualify.

Object14 has a greater x-coordinate than Object0 (premise sources: 8).

This comparison is incompatible with Object0 being northeast of Object14, so it cannot qualify.

Object14 has a greater x-coordinate than Object13 (premise sources: 1).

This comparison is incompatible with Object13 being northeast of Object14, so it cannot qualify.

Following the x-coordinates, Object14 has a smaller x-coordinate than Object1; Object1 shares its x-coordinate with Object7; Object7 has a smaller x-coordinate than Object6; Object6 has a smaller x-coordinate than Object8; Object8 shares its x-coordinate with Object3; Object3 has a smaller x-coordinate than Object11; Object11 has a smaller x-coordinate than Object4; Object4 has a smaller x-coordinate than Object2; Object2 has a smaller x-coordinate than Object16 (premise sources: 4, 7, 5, 14, 2, 15, 6, 9 and 3). Therefore, Object16 has a greater x-coordinate than Object14.

Following the y-coordinates, Object14 has a smaller y-coordinate than Object1; Object1 has a smaller y-coordinate than Object7; Object7 shares its y-coordinate with Object6; Object6 shares its y-coordinate with Object8; Object8 has a smaller y-coordinate than Object3; Object3 shares its y-coordinate with Object11; Object11 has a smaller y-coordinate than Object4; Object4 has a smaller y-coordinate than Object2; Object2 shares its y-coordinate with Object16 (premise sources: 4, 7, 5, 14, 2, 15, 6, 9 and 3). Therefore, Object16 has a greater y-coordinate than Object14.

So Object16 is northeast of Object14 in every valid map and always qualifies.

Object14 has a greater x-coordinate than Object5 (premise sources: 10).

This comparison is incompatible with Object5 being northeast of Object14, so it cannot qualify.

Every qualifying group must include Object16. Object15, Object0, Object13 and Object5 cannot qualify. There are no additional forbidden pairs. Checking the eligible-object subsets against these constraints leaves only size 1. Counts 0, 2, 3, 4 and 5 are therefore impossible.

For example, one valid arrangement consistent with all premises places Object1 at (3, 6); Object0 at (1, 3); Object4 at (7, 10); Object2 at (8, 11); Object6 at (4, 7); Object5 at (1, 2); Object8 at (5, 7); Object16 at (10, 11); Object7 at (3, 7); Object12 at (2, 5); Object3 at (5, 9); Object9 at (9, 12); Object14 at (2, 4); Object11 at (6, 9); Object13 at (1, 1); Object15 at (1, 0); Object10 at (0, 8). This gives a count of 1.

The only possible value is 1. Select D.
Answer: D

### Symbolic

```text
[p13] x(Object15) < x(Object14).

Axis comparison contradicts NE(Object15,Object14) => Object15 cannot qualify.

[p8] x(Object0) < x(Object14).

Axis comparison contradicts NE(Object0,Object14) => Object0 cannot qualify.

[p1] x(Object13) < x(Object14).

Axis comparison contradicts NE(Object13,Object14) => Object13 cannot qualify.

[p4, p7, p5, p14, p2, p15, p6, p9, p3] x(Object14) < x(Object1) = x(Object7) < x(Object6) < x(Object8) = x(Object3) < x(Object11) < x(Object4) < x(Object2) < x(Object16) => x(Object14) < x(Object16).

[p4, p7, p5, p14, p2, p15, p6, p9, p3] y(Object14) < y(Object1) < y(Object7) = y(Object6) = y(Object8) < y(Object3) = y(Object11) < y(Object4) < y(Object2) = y(Object16) => y(Object14) < y(Object16).

NE(Object16,Object14) in every valid map => Object16 always qualifies.

[p10] x(Object5) < x(Object14).

Axis comparison contradicts NE(Object5,Object14) => Object5 cannot qualify.

Required={Object16}; excluded={Object15, Object0, Object13, Object5}; forbidden pairs={}.
Joint membership subsets satisfying these constraints have sizes {1}; excluded counts={0, 2, 3, 4, 5}.

Example valid map satisfying all premises: Object1=(3,6), Object0=(1,3), Object4=(7,10), Object2=(8,11), Object6=(4,7), Object5=(1,2), Object8=(5,7), Object16=(10,11), Object7=(3,7), Object12=(2,5), Object3=(5,9), Object9=(9,12), Object14=(2,4), Object11=(6,9), Object13=(1,1), Object15=(1,0), Object10=(0,8). count=1.

Complete possible set = {1}. Select D.
Answer: D
```

## Longest Natural at depth 10, direction

ID: `depth-10-sample-2-direction-m0-visible-possible`. Kind: `direction`. Contract: `visible-possible`. Case: `unique`. Depth: 10. Target tokens: Natural 405; Symbolic 260.

### Question

```text
Objects: Object15, Object9, Object12, Object6, Object5, Object1, Object2, Object7, Object11, Object16, Object0, Object14, Object10, Object13, Object3, Object4, Object8
Query: direction Object7 Object1
Consider a map with multiple objects:
Object11 is to the North of Object6.
Object8 is to the Southeast of Object1.
Object14 is to the Northwest of Object4.
Object3 is to the Southeast of Object1.
Object4 is to the East of Object12.
Object1 is to the Northwest of Object0.
Object13 is to the West of Object11.
Object1 is to the Southeast of Object15.
Object13 is to the South of Object4.
Object5 is to the West of Object12.
Object7 is to the Northwest of Object10.
Object6 is to the West of Object15.
Object9 is to the East of Object5.
Object2 is to the West of Object5.
Object1 is to the Northwest of Object16.
Object2 is to the East of Object10.
Question: Which listed directions could Object7 occupy relative to Object1? Select all that apply.
A. Southwest
B. West
C. Northwest
D. South
E. Cannot be determined
F. None of the above
```

### Natural

Following the x-coordinates, Object7 has a smaller x-coordinate than Object10; Object10 has a smaller x-coordinate than Object2; Object2 has a smaller x-coordinate than Object5; Object5 has a smaller x-coordinate than Object12; Object12 has a smaller x-coordinate than Object4; Object4 shares its x-coordinate with Object13; Object13 has a smaller x-coordinate than Object11; Object11 shares its x-coordinate with Object6; Object6 has a smaller x-coordinate than Object15; Object15 has a smaller x-coordinate than Object1 (premise sources: 11, 16, 14, 10, 5, 9, 7, 1, 12 and 8). Therefore, Object1 has a greater x-coordinate than Object7.

Following the y-coordinates, Object1 has a smaller y-coordinate than Object15; Object15 shares its y-coordinate with Object6; Object6 has a smaller y-coordinate than Object11; Object11 shares its y-coordinate with Object13; Object13 has a smaller y-coordinate than Object4; Object4 shares its y-coordinate with Object12; Object12 shares its y-coordinate with Object5; Object5 shares its y-coordinate with Object2; Object2 shares its y-coordinate with Object10; Object10 has a smaller y-coordinate than Object7 (premise sources: 8, 12, 1, 7, 9, 5, 10, 14, 16 and 11). Therefore, Object7 has a greater y-coordinate than Object1.

Together, these comparisons place Object7 exactly northwest of Object1. The x-comparison rules out north, northeast, east, southeast and south. The y-comparison rules out southwest and west.

The only listed possible value is Northwest. Select C.
Answer: C

### Symbolic

```text
[p11, p16, p14, p10, p5, p9, p7, p1, p12, p8] x(Object7) < x(Object10) < x(Object2) < x(Object5) < x(Object12) < x(Object4) = x(Object13) < x(Object11) = x(Object6) < x(Object15) < x(Object1) => x(Object7) < x(Object1).

[p8, p12, p1, p7, p9, p5, p10, p14, p16, p11] y(Object1) < y(Object15) = y(Object6) < y(Object11) = y(Object13) < y(Object4) = y(Object12) = y(Object5) = y(Object2) = y(Object10) < y(Object7) => y(Object1) < y(Object7).

=> NW(Object7,Object1).
x-comparison excludes {N, NE, E, SE, S}.
y-comparison excludes {SW, W}.

Listed possible set = {Northwest}. Select C.
Answer: C
```

## Longest Natural at depth 10, which

ID: `depth-10-sample-2-which-m0-single`. Kind: `which`. Contract: `single`. Case: `unique`. Depth: 10. Target tokens: Natural 537; Symbolic 366.

### Question

```text
Objects: Object15, Object9, Object12, Object6, Object5, Object1, Object2, Object7, Object11, Object16, Object0, Object14, Object10, Object13, Object3, Object4, Object8
Query: which Northwest Object1
Eligible: Object3, Object7, Object8, Object16, Object0
Consider a map with multiple objects:
Object11 is to the North of Object6.
Object8 is to the Southeast of Object1.
Object14 is to the Northwest of Object4.
Object3 is to the Southeast of Object1.
Object4 is to the East of Object12.
Object1 is to the Northwest of Object0.
Object13 is to the West of Object11.
Object1 is to the Southeast of Object15.
Object13 is to the South of Object4.
Object5 is to the West of Object12.
Object7 is to the Northwest of Object10.
Object6 is to the West of Object15.
Object9 is to the East of Object5.
Object2 is to the West of Object5.
Object1 is to the Northwest of Object16.
Object2 is to the East of Object10.
Question: Which eligible object is the only one northwest of Object1 in every valid map?
A. Object7
B. Object16
C. Object3
D. Object8
E. Cannot be determined
F. None of the above
```

### Natural

Object3 has a greater x-coordinate than Object1 (premise sources: 4).

This comparison is incompatible with Object3 being northwest of Object1, so it cannot qualify.

Following the x-coordinates, Object7 has a smaller x-coordinate than Object10; Object10 has a smaller x-coordinate than Object2; Object2 has a smaller x-coordinate than Object5; Object5 has a smaller x-coordinate than Object12; Object12 has a smaller x-coordinate than Object4; Object4 shares its x-coordinate with Object13; Object13 has a smaller x-coordinate than Object11; Object11 shares its x-coordinate with Object6; Object6 has a smaller x-coordinate than Object15; Object15 has a smaller x-coordinate than Object1 (premise sources: 11, 16, 14, 10, 5, 9, 7, 1, 12 and 8). Therefore, Object1 has a greater x-coordinate than Object7.

Following the y-coordinates, Object1 has a smaller y-coordinate than Object15; Object15 shares its y-coordinate with Object6; Object6 has a smaller y-coordinate than Object11; Object11 shares its y-coordinate with Object13; Object13 has a smaller y-coordinate than Object4; Object4 shares its y-coordinate with Object12; Object12 shares its y-coordinate with Object5; Object5 shares its y-coordinate with Object2; Object2 shares its y-coordinate with Object10; Object10 has a smaller y-coordinate than Object7 (premise sources: 8, 12, 1, 7, 9, 5, 10, 14, 16 and 11). Therefore, Object7 has a greater y-coordinate than Object1.

So Object7 is northwest of Object1 in every valid map and always qualifies.

Object8 has a greater x-coordinate than Object1 (premise sources: 2).

This comparison is incompatible with Object8 being northwest of Object1, so it cannot qualify.

Object16 has a greater x-coordinate than Object1 (premise sources: 15).

This comparison is incompatible with Object16 being northwest of Object1, so it cannot qualify.

Object0 has a greater x-coordinate than Object1 (premise sources: 6).

This comparison is incompatible with Object0 being northwest of Object1, so it cannot qualify.

The only possible value is Object7. Select A.
Answer: A

### Symbolic

```text
[p4] x(Object1) < x(Object3).

Axis comparison contradicts NW(Object3,Object1) => Object3 cannot qualify.

[p11, p16, p14, p10, p5, p9, p7, p1, p12, p8] x(Object7) < x(Object10) < x(Object2) < x(Object5) < x(Object12) < x(Object4) = x(Object13) < x(Object11) = x(Object6) < x(Object15) < x(Object1) => x(Object7) < x(Object1).

[p8, p12, p1, p7, p9, p5, p10, p14, p16, p11] y(Object1) < y(Object15) = y(Object6) < y(Object11) = y(Object13) < y(Object4) = y(Object12) = y(Object5) = y(Object2) = y(Object10) < y(Object7) => y(Object1) < y(Object7).

NW(Object7,Object1) in every valid map => Object7 always qualifies.

[p2] x(Object1) < x(Object8).

Axis comparison contradicts NW(Object8,Object1) => Object8 cannot qualify.

[p15] x(Object1) < x(Object16).

Axis comparison contradicts NW(Object16,Object1) => Object16 cannot qualify.

[p6] x(Object1) < x(Object0).

Axis comparison contradicts NW(Object0,Object1) => Object0 cannot qualify.

Complete possible set = {Object7}. Select A.
Answer: A
```

## Longest Natural at depth 10, count

ID: `depth-10-sample-2-count-m0-single`. Kind: `count`. Contract: `single`. Case: `unique`. Depth: 10. Target tokens: Natural 781; Symbolic 557.

### Question

```text
Objects: Object15, Object9, Object12, Object6, Object5, Object1, Object2, Object7, Object11, Object16, Object0, Object14, Object10, Object13, Object3, Object4, Object8
Query: count Northwest Object1
Eligible: Object3, Object7, Object8, Object16, Object0
Consider a map with multiple objects:
Object11 is to the North of Object6.
Object8 is to the Southeast of Object1.
Object14 is to the Northwest of Object4.
Object3 is to the Southeast of Object1.
Object4 is to the East of Object12.
Object1 is to the Northwest of Object0.
Object13 is to the West of Object11.
Object1 is to the Southeast of Object15.
Object13 is to the South of Object4.
Object5 is to the West of Object12.
Object7 is to the Northwest of Object10.
Object6 is to the West of Object15.
Object9 is to the East of Object5.
Object2 is to the West of Object5.
Object1 is to the Northwest of Object16.
Object2 is to the East of Object10.
Question: What exact number of eligible objects are northwest of Object1 in every valid map?
A. 1
B. 0
C. 2
D. 4
E. Cannot be determined
F. None of the above
```

### Natural

Object3 has a greater x-coordinate than Object1 (premise sources: 4).

This comparison is incompatible with Object3 being northwest of Object1, so it cannot qualify.

Following the x-coordinates, Object7 has a smaller x-coordinate than Object10; Object10 has a smaller x-coordinate than Object2; Object2 has a smaller x-coordinate than Object5; Object5 has a smaller x-coordinate than Object12; Object12 has a smaller x-coordinate than Object4; Object4 shares its x-coordinate with Object13; Object13 has a smaller x-coordinate than Object11; Object11 shares its x-coordinate with Object6; Object6 has a smaller x-coordinate than Object15; Object15 has a smaller x-coordinate than Object1 (premise sources: 11, 16, 14, 10, 5, 9, 7, 1, 12 and 8). Therefore, Object1 has a greater x-coordinate than Object7.

Following the y-coordinates, Object1 has a smaller y-coordinate than Object15; Object15 shares its y-coordinate with Object6; Object6 has a smaller y-coordinate than Object11; Object11 shares its y-coordinate with Object13; Object13 has a smaller y-coordinate than Object4; Object4 shares its y-coordinate with Object12; Object12 shares its y-coordinate with Object5; Object5 shares its y-coordinate with Object2; Object2 shares its y-coordinate with Object10; Object10 has a smaller y-coordinate than Object7 (premise sources: 8, 12, 1, 7, 9, 5, 10, 14, 16 and 11). Therefore, Object7 has a greater y-coordinate than Object1.

So Object7 is northwest of Object1 in every valid map and always qualifies.

Object8 has a greater x-coordinate than Object1 (premise sources: 2).

This comparison is incompatible with Object8 being northwest of Object1, so it cannot qualify.

Object16 has a greater x-coordinate than Object1 (premise sources: 15).

This comparison is incompatible with Object16 being northwest of Object1, so it cannot qualify.

Object0 has a greater x-coordinate than Object1 (premise sources: 6).

This comparison is incompatible with Object0 being northwest of Object1, so it cannot qualify.

Every qualifying group must include Object7. Object3, Object8, Object16 and Object0 cannot qualify. There are no additional forbidden pairs. Checking the eligible-object subsets against these constraints leaves only size 1. Counts 0, 2, 3, 4 and 5 are therefore impossible.

For example, one valid arrangement consistent with all premises places Object15 at (7, 5); Object9 at (6, 7); Object12 at (4, 7); Object6 at (6, 5); Object5 at (3, 7); Object1 at (8, 4); Object2 at (2, 7); Object7 at (0, 9); Object11 at (6, 6); Object16 at (9, 3); Object0 at (9, 2); Object14 at (0, 8); Object10 at (1, 7); Object13 at (5, 6); Object3 at (9, 1); Object4 at (5, 7); Object8 at (9, 0). This gives a count of 1.

The only possible value is 1. Select A.
Answer: A

### Symbolic

```text
[p4] x(Object1) < x(Object3).

Axis comparison contradicts NW(Object3,Object1) => Object3 cannot qualify.

[p11, p16, p14, p10, p5, p9, p7, p1, p12, p8] x(Object7) < x(Object10) < x(Object2) < x(Object5) < x(Object12) < x(Object4) = x(Object13) < x(Object11) = x(Object6) < x(Object15) < x(Object1) => x(Object7) < x(Object1).

[p8, p12, p1, p7, p9, p5, p10, p14, p16, p11] y(Object1) < y(Object15) = y(Object6) < y(Object11) = y(Object13) < y(Object4) = y(Object12) = y(Object5) = y(Object2) = y(Object10) < y(Object7) => y(Object1) < y(Object7).

NW(Object7,Object1) in every valid map => Object7 always qualifies.

[p2] x(Object1) < x(Object8).

Axis comparison contradicts NW(Object8,Object1) => Object8 cannot qualify.

[p15] x(Object1) < x(Object16).

Axis comparison contradicts NW(Object16,Object1) => Object16 cannot qualify.

[p6] x(Object1) < x(Object0).

Axis comparison contradicts NW(Object0,Object1) => Object0 cannot qualify.

Required={Object7}; excluded={Object3, Object8, Object16, Object0}; forbidden pairs={}.
Joint membership subsets satisfying these constraints have sizes {1}; excluded counts={0, 2, 3, 4, 5}.

Example valid map satisfying all premises: Object15=(7,5), Object9=(6,7), Object12=(4,7), Object6=(6,5), Object5=(3,7), Object1=(8,4), Object2=(2,7), Object7=(0,9), Object11=(6,6), Object16=(9,3), Object0=(9,2), Object14=(0,8), Object10=(1,7), Object13=(5,6), Object3=(9,1), Object4=(5,7), Object8=(9,0). count=1.

Complete possible set = {1}. Select A.
Answer: A
```

## Ambiguous Direction, single

ID: `joint-0-base-direction-m0-single`. Kind: `direction`. Contract: `single`. Case: `underdetermined`. Depth: None. Target tokens: Natural 148; Symbolic 117.

### Question

```text
Objects: Object0, Object1, Object3, Object2, Object4
Query: direction Object1 Object0
Consider a map with multiple objects:
Object2 is to the South of Object4.
Object1 is to the South of Object2.
Object3 is to the North of Object4.
Question: In which direction is Object1 relative to Object0 in every valid map?
A. Southwest
B. East
C. Northwest
D. West
E. Cannot be determined
F. None of the above
```

### Natural

For example, one valid arrangement consistent with all premises places Object0 at (0, 0); Object1 at (0, 1); Object3 at (0, 4); Object2 at (0, 2); Object4 at (0, 3). This gives the direction north.

For example, one valid arrangement consistent with all premises places Object0 at (0, 0); Object1 at (1, 1); Object3 at (1, 4); Object2 at (1, 2); Object4 at (1, 3). This gives the direction northeast.

These valid maps have different outcomes, so the answer is not uniquely determined. Select E.
Answer: E

### Symbolic

```text
Example valid map satisfying all premises: Object0=(0,0), Object1=(0,1), Object3=(0,4), Object2=(0,2), Object4=(0,3). direction=North.

Example valid map satisfying all premises: Object0=(0,0), Object1=(1,1), Object3=(1,4), Object2=(1,2), Object4=(1,3). direction=Northeast.

These valid maps have different outcomes, so the answer is not uniquely determined. Select E.
Answer: E
```

## Ambiguous Direction, visible-possible

ID: `fork-0-base-direction-m0-visible-possible`. Kind: `direction`. Contract: `visible-possible`. Case: `underdetermined`. Depth: None. Target tokens: Natural 194; Symbolic 165.

### Question

```text
Objects: Object5, Object4, Object3, Object2, Object0, Object1
Query: direction Object4 Object2
Consider a map with multiple objects:
Object1 is to the South of Object3.
Object3 is to the North of Object2.
Object3 is to the North of Object5.
Object4 is to the South of Object3.
Object3 is to the North of Object0.
Question: Which listed directions could Object4 occupy relative to Object2? Select all that apply.
A. South
B. East
C. North
D. Northwest
E. Cannot be determined
F. None of the above
```

### Natural

For example, one valid arrangement consistent with all premises places Object5 at (0, 4); Object4 at (0, 2); Object3 at (0, 5); Object2 at (0, 3); Object0 at (0, 1); Object1 at (0, 0). This gives the direction south.

For example, one valid arrangement consistent with all premises places Object5 at (0, 4); Object4 at (0, 3); Object3 at (0, 5); Object2 at (0, 2); Object0 at (0, 1); Object1 at (0, 0). This gives the direction north.

Object4, Object3 and Object2 share the same x-coordinate (premise sources: 4 and 2).

This comparison rules out east and northwest.

The listed possible values are South and North. Select AC.
Answer: AC

### Symbolic

```text
Example valid map satisfying all premises: Object5=(0,4), Object4=(0,2), Object3=(0,5), Object2=(0,3), Object0=(0,1), Object1=(0,0). direction=South.

Example valid map satisfying all premises: Object5=(0,4), Object4=(0,3), Object3=(0,5), Object2=(0,2), Object0=(0,1), Object1=(0,0). direction=North.

[p4, p2] x(Object4) = x(Object3) = x(Object2) => x(Object4) = x(Object2).

This comparison excludes {E, NW}.

Listed possible set = {South, North}. Select AC.
Answer: AC
```

## Ambiguous Direction, all-possible

ID: `fork-0-base-direction-m0-all-possible`. Kind: `direction`. Contract: `all-possible`. Case: `underdetermined`. Depth: None. Target tokens: Natural 202; Symbolic 172.

### Question

```text
Objects: Object5, Object4, Object3, Object2, Object0, Object1
Query: direction Object4 Object2
Consider a map with multiple objects:
Object1 is to the South of Object3.
Object3 is to the North of Object2.
Object3 is to the North of Object5.
Object4 is to the South of Object3.
Object3 is to the North of Object0.
Question: Select the complete set of directions Object4 could occupy relative to Object2.
A. South
B. East
C. North
D. Northwest
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

For example, one valid arrangement consistent with all premises places Object5 at (0, 4); Object4 at (0, 3); Object3 at (0, 5); Object2 at (0, 2); Object0 at (0, 1); Object1 at (0, 0). This gives the direction north.

For example, one valid arrangement consistent with all premises places Object5 at (0, 4); Object4 at (0, 2); Object3 at (0, 5); Object2 at (0, 3); Object0 at (0, 1); Object1 at (0, 0). This gives the direction south.

Object4, Object3 and Object2 share the same x-coordinate (premise sources: 4 and 2).

This comparison rules out northeast, east, southeast, southwest, west and northwest.

The complete possible values are North and South. Select AC.
Answer: AC

### Symbolic

```text
Example valid map satisfying all premises: Object5=(0,4), Object4=(0,3), Object3=(0,5), Object2=(0,2), Object0=(0,1), Object1=(0,0). direction=North.

Example valid map satisfying all premises: Object5=(0,4), Object4=(0,2), Object3=(0,5), Object2=(0,3), Object0=(0,1), Object1=(0,0). direction=South.

[p4, p2] x(Object4) = x(Object3) = x(Object2) => x(Object4) = x(Object2).

This comparison excludes {NE, E, SE, SW, W, NW}.

Complete possible set = {North, South}. Select AC.
Answer: AC
```

## Joint Count exclusion

ID: `random-world-8-count-m1-visible-possible`. Kind: `count`. Contract: `visible-possible`. Case: `underdetermined`. Depth: None. Target tokens: Natural 812; Symbolic 655.

### Question

```text
Objects: Object6, Object1, Object3, Object7, Object5, Object4, Object2, Object0
Query: count North Object4
Eligible: Object0, Object7, Object5, Object6, Object1, Object3, Object2
Consider a map with multiple objects:
Object7 is to the Northeast of Object0.
Object6 is to the Northeast of Object3.
Object0 is to the Southwest of Object4.
Object5 is to the Southwest of Object7.
Object2 is to the Southwest of Object4.
Object1 is to the East of Object3.
Object7 is to the Northeast of Object3.
Object5 is to the Southwest of Object0.
Question: Which listed counts are possible for the number of eligible objects north of Object4? Select all that apply.
A. 7
B. 2
C. 0
D. 3
E. Cannot be determined
F. None of the above
```

### Natural

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

No object is required to qualify. Object0, Object5 and Object2 cannot qualify. No qualifying group can contain both members of any forbidden pair: Object7 and Object3; Object6 and Object3; Object1 and Object3. Checking the eligible-object subsets against these constraints leaves only sizes 0, 1, 2 and 3. Counts 7 are therefore impossible.

For example, one valid arrangement consistent with all premises places Object6 at (2, 6); Object1 at (2, 3); Object3 at (0, 3); Object7 at (2, 5); Object5 at (0, 1); Object4 at (2, 4); Object2 at (0, 0); Object0 at (1, 2). This gives a count of 2.

For example, one valid arrangement consistent with all premises places Object6 at (3, 6); Object1 at (3, 4); Object3 at (0, 4); Object7 at (3, 5); Object5 at (0, 1); Object4 at (2, 3); Object2 at (0, 0); Object0 at (1, 2). This gives a count of 0.

For example, one valid arrangement consistent with all premises places Object6 at (2, 6); Object1 at (2, 4); Object3 at (0, 4); Object7 at (2, 5); Object5 at (0, 1); Object4 at (2, 3); Object2 at (0, 0); Object0 at (1, 2). This gives a count of 3.

The listed possible values are 2, 0 and 3. Select BCD.
Answer: BCD

### Symbolic

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
Joint membership subsets satisfying these constraints have sizes {0, 1, 2, 3}; excluded counts={7}.

Example valid map satisfying all premises: Object6=(2,6), Object1=(2,3), Object3=(0,3), Object7=(2,5), Object5=(0,1), Object4=(2,4), Object2=(0,0), Object0=(1,2). count=2.

Example valid map satisfying all premises: Object6=(3,6), Object1=(3,4), Object3=(0,4), Object7=(3,5), Object5=(0,1), Object4=(2,3), Object2=(0,0), Object0=(1,2). count=0.

Example valid map satisfying all premises: Object6=(2,6), Object1=(2,4), Object3=(0,4), Object7=(2,5), Object5=(0,1), Object4=(2,3), Object2=(0,0), Object0=(1,2). count=3.

Listed possible set = {2, 0, 3}. Select BCD.
Answer: BCD
```

## Variable Count, Unique

ID: `joint-0-base-count-m0-single`. Kind: `count`. Contract: `single`. Case: `underdetermined`. Depth: None. Target tokens: Natural 152; Symbolic 116.

### Question

```text
Objects: Object0, Object1, Object3, Object2, Object4
Query: count East Object0
Eligible: Object1, Object2, Object3, Object4
Consider a map with multiple objects:
Object2 is to the South of Object4.
Object1 is to the South of Object2.
Object3 is to the North of Object4.
Question: What exact number of eligible objects are east of Object0 in every valid map?
A. 0
B. 2
C. 3
D. 1
E. Cannot be determined
F. None of the above
```

### Natural

For example, one valid arrangement consistent with all premises places Object0 at (0, 4); Object1 at (0, 0); Object3 at (0, 3); Object2 at (0, 1); Object4 at (0, 2). This gives a count of 0.

For example, one valid arrangement consistent with all premises places Object0 at (0, 0); Object1 at (1, 0); Object3 at (1, 3); Object2 at (1, 1); Object4 at (1, 2). This gives a count of 1.

These valid maps have different outcomes, so the answer is not uniquely determined. Select E.
Answer: E

### Symbolic

```text
Example valid map satisfying all premises: Object0=(0,4), Object1=(0,0), Object3=(0,3), Object2=(0,1), Object4=(0,2). count=0.

Example valid map satisfying all premises: Object0=(0,0), Object1=(1,0), Object3=(1,3), Object2=(1,1), Object4=(1,2). count=1.

These valid maps have different outcomes, so the answer is not uniquely determined. Select E.
Answer: E
```

## Zero Count

ID: `empty-0-base-count-m0-visible-possible`. Kind: `count`. Contract: `visible-possible`. Case: `unique`. Depth: 1. Target tokens: Natural 289; Symbolic 227.

### Question

```text
Objects: Object0, Object2, Object4, Object3, Object1
Query: count South Object4
Eligible: Object0, Object2, Object3, Object1
Consider a map with multiple objects:
Object4 is to the East of Object2.
Object4 is to the East of Object1.
Object4 is to the South of Object0.
Object4 is to the East of Object3.
Question: Which listed counts are possible for the number of eligible objects south of Object4? Select all that apply.
A. 2
B. 1
C. 0
D. 4
E. Cannot be determined
F. None of the above
```

### Natural

Object0 has a greater y-coordinate than Object4 (premise sources: 3).

This comparison is incompatible with Object0 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object2 (premise sources: 1).

This comparison is incompatible with Object2 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object3 (premise sources: 4).

This comparison is incompatible with Object3 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object1 (premise sources: 2).

This comparison is incompatible with Object1 being south of Object4, so it cannot qualify.

No object is required to qualify. Object0, Object2, Object3 and Object1 cannot qualify. There are no additional forbidden pairs. Checking the eligible-object subsets against these constraints leaves only size 0. Counts 2, 1 and 4 are therefore impossible.

For example, one valid arrangement consistent with all premises places Object0 at (3, 1); Object2 at (2, 0); Object4 at (3, 0); Object3 at (1, 0); Object1 at (0, 0). This gives a count of 0.

The only listed possible value is 0. Select C.
Answer: C

### Symbolic

```text
[p3] y(Object4) < y(Object0).

Axis comparison contradicts S(Object0,Object4) => Object0 cannot qualify.

[p1] x(Object2) < x(Object4).

Axis comparison contradicts S(Object2,Object4) => Object2 cannot qualify.

[p4] x(Object3) < x(Object4).

Axis comparison contradicts S(Object3,Object4) => Object3 cannot qualify.

[p2] x(Object1) < x(Object4).

Axis comparison contradicts S(Object1,Object4) => Object1 cannot qualify.

Required={}; excluded={Object0, Object2, Object3, Object1}; forbidden pairs={}.
Joint membership subsets satisfying these constraints have sizes {0}; excluded counts={2, 1, 4}.

Example valid map satisfying all premises: Object0=(3,1), Object2=(2,0), Object4=(3,0), Object3=(1,0), Object1=(0,0). count=0.

Listed possible set = {0}. Select C.
Answer: C
```

## Selection Unique: no-match

ID: `empty-0-base-which-m0-single`. Kind: `which`. Contract: `single`. Case: `no-match`. Depth: 1. Target tokens: Natural 173; Symbolic 141.

### Question

```text
Objects: Object0, Object2, Object4, Object3, Object1
Query: which South Object4
Eligible: Object0, Object2, Object3, Object1
Consider a map with multiple objects:
Object4 is to the East of Object2.
Object4 is to the East of Object1.
Object4 is to the South of Object0.
Object4 is to the East of Object3.
Question: Which eligible object is the only one south of Object4 in every valid map?
A. Object0
B. Object3
C. Object1
D. Object2
E. Cannot be determined
F. None of the above
```

### Natural

Object0 has a greater y-coordinate than Object4 (premise sources: 3).

This comparison is incompatible with Object0 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object2 (premise sources: 1).

This comparison is incompatible with Object2 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object3 (premise sources: 4).

This comparison is incompatible with Object3 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object1 (premise sources: 2).

This comparison is incompatible with Object1 being south of Object4, so it cannot qualify.

No eligible object can qualify, so there is no unique qualifying object. Select E.
Answer: E

### Symbolic

```text
[p3] y(Object4) < y(Object0).

Axis comparison contradicts S(Object0,Object4) => Object0 cannot qualify.

[p1] x(Object2) < x(Object4).

Axis comparison contradicts S(Object2,Object4) => Object2 cannot qualify.

[p4] x(Object3) < x(Object4).

Axis comparison contradicts S(Object3,Object4) => Object3 cannot qualify.

[p2] x(Object1) < x(Object4).

Axis comparison contradicts S(Object1,Object4) => Object1 cannot qualify.

No eligible object can qualify, so there is no unique qualifying object. Select E.
Answer: E
```

## Selection Unique: multiple-certain-match

ID: `cycle-0-base-which-m0-single`. Kind: `which`. Contract: `single`. Case: `multiple-certain-match`. Depth: 1. Target tokens: Natural 240; Symbolic 192.

### Question

```text
Objects: Object6, Object3, Object2, Object0, Object4, Object7, Object5, Object1
Query: which South Object3
Eligible: Object1, Object6, Object4, Object0
Consider a map with multiple objects:
Object7 is to the West of Object5.
Object2 is to the East of Object7.
Object1 is to the South of Object3.
Object2 is to the East of Object5.
Object3 is to the North of Object6.
Object0 is to the South of Object3.
Object3 is to the North of Object4.
Question: Which eligible object is the only one south of Object3 in every valid map?
A. Object4
B. Object0
C. Object6
D. Object1
E. Cannot be determined
F. None of the above
```

### Natural

Object1 and Object3 share the same x-coordinate (premise sources: 3).

Object3 has a greater y-coordinate than Object1 (premise sources: 3).

So Object1 is south of Object3 in every valid map and always qualifies.

Object6 and Object3 share the same x-coordinate (premise sources: 5).

Object3 has a greater y-coordinate than Object6 (premise sources: 5).

So Object6 is south of Object3 in every valid map and always qualifies.

Object4 and Object3 share the same x-coordinate (premise sources: 7).

Object3 has a greater y-coordinate than Object4 (premise sources: 7).

So Object4 is south of Object3 in every valid map and always qualifies.

Object0 and Object3 share the same x-coordinate (premise sources: 6).

Object3 has a greater y-coordinate than Object0 (premise sources: 6).

So Object0 is south of Object3 in every valid map and always qualifies.

Several objects always qualify, so there is no sole qualifying object. Select E.
Answer: E

### Symbolic

```text
[p3] x(Object1) = x(Object3).

[p3] y(Object1) < y(Object3).

S(Object1,Object3) in every valid map => Object1 always qualifies.

[p5] x(Object6) = x(Object3).

[p5] y(Object6) < y(Object3).

S(Object6,Object3) in every valid map => Object6 always qualifies.

[p7] x(Object4) = x(Object3).

[p7] y(Object4) < y(Object3).

S(Object4,Object3) in every valid map => Object4 always qualifies.

[p6] x(Object0) = x(Object3).

[p6] y(Object0) < y(Object3).

S(Object0,Object3) in every valid map => Object0 always qualifies.

Several objects always qualify, so there is no sole qualifying object. Select E.
Answer: E
```

## Selection Unique: underdetermined

ID: `contingent-0-base-which-m0-single`. Kind: `which`. Contract: `single`. Case: `underdetermined`. Depth: None. Target tokens: Natural 311; Symbolic 245.

### Question

```text
Objects: Object5, Object2, Object0, Object3, Object4, Object1
Query: which North Object4
Eligible: Object0, Object2, Object5, Object3
Consider a map with multiple objects:
Object5 is to the South of Object4.
Object0 is to the South of Object4.
Object1 is to the Southwest of Object4.
Object4 is to the North of Object2.
Object3 is to the Northeast of Object1.
Question: Which eligible object is the only one north of Object4 in every valid map?
A. Object0
B. Object5
C. Object3
D. Object2
E. Cannot be determined
F. None of the above
```

### Natural

Object4 has a greater y-coordinate than Object0 (premise sources: 2).

This comparison is incompatible with Object0 being north of Object4, so it cannot qualify.

Object4 has a greater y-coordinate than Object2 (premise sources: 4).

This comparison is incompatible with Object2 being north of Object4, so it cannot qualify.

Object4 has a greater y-coordinate than Object5 (premise sources: 1).

This comparison is incompatible with Object5 being north of Object4, so it cannot qualify.

For example, one valid arrangement consistent with all premises places Object5 at (1, 3); Object2 at (1, 2); Object0 at (1, 1); Object3 at (1, 5); Object4 at (1, 4); Object1 at (0, 0). In this arrangement, only Object3 qualifies.

For example, one valid arrangement consistent with all premises places Object5 at (1, 4); Object2 at (1, 3); Object0 at (1, 2); Object3 at (1, 1); Object4 at (1, 5); Object1 at (0, 0). In this arrangement, no eligible objects qualify.

Object3 qualifies in the first arrangement but not the second. It is a possible match, not a guaranteed one.

The evidence does not establish one object as the only match in every valid map. Select E.
Answer: E

### Symbolic

```text
[p2] y(Object0) < y(Object4).

Axis comparison contradicts N(Object0,Object4) => Object0 cannot qualify.

[p4] y(Object2) < y(Object4).

Axis comparison contradicts N(Object2,Object4) => Object2 cannot qualify.

[p1] y(Object5) < y(Object4).

Axis comparison contradicts N(Object5,Object4) => Object5 cannot qualify.

Example valid map satisfying all premises: Object5=(1,3), Object2=(1,2), Object0=(1,1), Object3=(1,5), Object4=(1,4), Object1=(0,0). Qualifying objects = {Object3}.

Example valid map satisfying all premises: Object5=(1,4), Object2=(1,3), Object0=(1,2), Object3=(1,1), Object4=(1,5), Object1=(0,0). Qualifying objects = {}.

Object3 qualifies in map 1, not map 2 => possible, not guaranteed.

The evidence does not establish one object as the only match in every valid map. Select E.
Answer: E
```

## strict-cycle

ID: `random-atoms-2-count-m0-all-possible`. Kind: `count`. Contract: `all-possible`. Case: `inconsistent`. Depth: None. Target tokens: Natural 82; Symbolic 69.

### Question

```text
Objects: Object0, Object1, Object2, Object6, Object4, Object5, Object3
Query: count North Object4
Eligible: Object1, Object6, Object3, Object0, Object5, Object2
Consider a map with multiple objects:
Object6 is to the North of Object5.
Object0 is to the West of Object4.
Object4 is to the South of Object2.
Object5 is to the North of Object2.
Object5 is to the Northeast of Object4.
Question: Select the complete set of possible counts of eligible objects north of Object4.
A. 2
B. 1
C. 4
D. 6
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

The cited relations would require that Object4 has a smaller x-coordinate than Object5; Object5 shares its x-coordinate with Object2; Object2 shares its x-coordinate with Object4 (premise sources: 5, 4 and 3).

This would make a coordinate smaller than itself, which is impossible.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E

### Symbolic

```text
[p5, p4, p3] x(Object4) < x(Object5) = x(Object2) = x(Object4) => x(Object4) < x(Object4).

x(Object4) < x(Object4) is impossible.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E
```

## forced-coincidence

ID: `coincidence-0-base-count-m0-all-possible`. Kind: `count`. Contract: `all-possible`. Case: `inconsistent`. Depth: None. Target tokens: Natural 95; Symbolic 108.

### Question

```text
Objects: Object1, Object3, Object4, Object5, Object2, Object0
Query: count South Object3
Eligible: Object0, Object2, Object1, Object4
Consider a map with multiple objects:
Object3 is to the North of Object2.
Object5 is to the East of Object4.
Object3 is to the South of Object1.
Object5 is to the East of Object2.
Object3 is to the South of Object0.
Object4 is to the South of Object3.
Question: Select the complete set of possible counts of eligible objects south of Object3.
A. 0
B. 1
C. 3
D. 4
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

Object4, Object3 and Object2 share the same x-coordinate (premise sources: 6 and 1).

Object4, Object5 and Object2 share the same y-coordinate (premise sources: 2 and 4).

Object4 and Object2 would have both coordinates equal. They are distinct objects, so the no-coincidence rule makes this impossible.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E

### Symbolic

```text
[p6, p1] x(Object4) = x(Object3) = x(Object2) => x(Object4) = x(Object2).

[p2, p4] y(Object4) = y(Object5) = y(Object2) => y(Object4) = y(Object2).

x(Object4)=x(Object2) and y(Object4)=y(Object2), but Object4!=Object2: distinct-point contradiction.

No valid map satisfies the premises. Every contract therefore requires E.
Answer: E
```

## Escape: unique-unlisted

ID: `empty-0-base-direction-m1-single`. Kind: `direction`. Contract: `single`. Case: `unique`. Depth: 1. Target tokens: Natural 100; Symbolic 82.

### Question

```text
Objects: Object0, Object2, Object4, Object3, Object1
Query: direction Object0 Object4
Consider a map with multiple objects:
Object4 is to the East of Object2.
Object4 is to the East of Object1.
Object4 is to the South of Object0.
Object4 is to the East of Object3.
Question: In which direction is Object0 relative to Object4 in every valid map?
A. Northwest
B. West
C. Northeast
D. Southeast
E. Cannot be determined
F. None of the above
```

### Natural

Object4 and Object0 share the same x-coordinate (premise sources: 3).

Object0 has a greater y-coordinate than Object4 (premise sources: 3).

Together, these comparisons place Object0 exactly north of Object4. The x-comparison rules out northeast, east, southeast, southwest, west and northwest. The y-comparison rules out south.

The only possible value is North. The unique value is unlisted, so select F.
Answer: F

### Symbolic

```text
[p3] x(Object4) = x(Object0).

[p3] y(Object4) < y(Object0).

=> N(Object0,Object4).
x-comparison excludes {NE, E, SE, SW, W, NW}.
y-comparison excludes {S}.

Complete possible set = {North}. The unique value is unlisted, so select F.
Answer: F
```

## Escape: incomplete-coverage

ID: `empty-0-base-direction-m1-all-possible`. Kind: `direction`. Contract: `all-possible`. Case: `unique`. Depth: 1. Target tokens: Natural 88; Symbolic 72.

### Question

```text
Objects: Object0, Object2, Object4, Object3, Object1
Query: direction Object0 Object4
Consider a map with multiple objects:
Object4 is to the East of Object2.
Object4 is to the East of Object1.
Object4 is to the South of Object0.
Object4 is to the East of Object3.
Question: Select the complete set of directions Object0 could occupy relative to Object4.
A. Northwest
B. West
C. Northeast
D. Southeast
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

For example, one valid arrangement consistent with all premises places Object0 at (3, 1); Object2 at (2, 0); Object4 at (3, 0); Object3 at (1, 0); Object1 at (0, 0). This gives the direction north.

The witnessed value North is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F

### Symbolic

```text
Example valid map satisfying all premises: Object0=(3,1), Object2=(2,0), Object4=(3,0), Object3=(1,0), Object1=(0,0). direction=North.

The witnessed value North is absent from A-D, so the complete answer cannot be represented. Select F.
Answer: F
```

## Escape: empty-set

ID: `empty-0-base-which-m0-all-possible`. Kind: `which`. Contract: `all-possible`. Case: `no-match`. Depth: 1. Target tokens: Natural 171; Symbolic 137.

### Question

```text
Objects: Object0, Object2, Object4, Object3, Object1
Query: which South Object4
Eligible: Object0, Object2, Object3, Object1
Consider a map with multiple objects:
Object4 is to the East of Object2.
Object4 is to the East of Object1.
Object4 is to the South of Object0.
Object4 is to the East of Object3.
Question: Select the complete set of eligible objects that could be south of Object4.
A. Object0
B. Object3
C. Object1
D. Object2
E. Cannot be determined
F. No complete answer can be formed from A-D.
```

### Natural

Object0 has a greater y-coordinate than Object4 (premise sources: 3).

This comparison is incompatible with Object0 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object2 (premise sources: 1).

This comparison is incompatible with Object2 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object3 (premise sources: 4).

This comparison is incompatible with Object3 being south of Object4, so it cannot qualify.

Object4 has a greater x-coordinate than Object1 (premise sources: 2).

This comparison is incompatible with Object1 being south of Object4, so it cannot qualify.

The complete possible set is empty. No displayed value qualifies; select F.
Answer: F

### Symbolic

```text
[p3] y(Object4) < y(Object0).

Axis comparison contradicts S(Object0,Object4) => Object0 cannot qualify.

[p1] x(Object2) < x(Object4).

Axis comparison contradicts S(Object2,Object4) => Object2 cannot qualify.

[p4] x(Object3) < x(Object4).

Axis comparison contradicts S(Object3,Object4) => Object3 cannot qualify.

[p2] x(Object1) < x(Object4).

Axis comparison contradicts S(Object1,Object4) => Object1 cannot qualify.

Complete possible set = {}. No displayed value qualifies; select F.
Answer: F
```

## Escape: no-listed-possibilities

ID: `empty-0-base-direction-m1-visible-possible`. Kind: `direction`. Contract: `visible-possible`. Case: `unique`. Depth: 1. Target tokens: Natural 97; Symbolic 78.

### Question

```text
Objects: Object0, Object2, Object4, Object3, Object1
Query: direction Object0 Object4
Consider a map with multiple objects:
Object4 is to the East of Object2.
Object4 is to the East of Object1.
Object4 is to the South of Object0.
Object4 is to the East of Object3.
Question: Which listed directions could Object0 occupy relative to Object4? Select all that apply.
A. Northwest
B. West
C. Northeast
D. Southeast
E. Cannot be determined
F. None of the above
```

### Natural

Object4 and Object0 share the same x-coordinate (premise sources: 3).

Object0 has a greater y-coordinate than Object4 (premise sources: 3).

Together, these comparisons place Object0 exactly north of Object4. The x-comparison rules out northeast, east, southeast, southwest, west and northwest. The y-comparison rules out south.

The listed possible set is empty. No displayed value qualifies; select F.
Answer: F

### Symbolic

```text
[p3] x(Object4) = x(Object0).

[p3] y(Object4) < y(Object0).

=> N(Object0,Object4).
x-comparison excludes {NE, E, SE, SW, W, NW}.
y-comparison excludes {S}.

Listed possible set = {}. No displayed value qualifies; select F.
Answer: F
```
