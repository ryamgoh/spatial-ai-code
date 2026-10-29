# Feasibility Paired Examples

Six selected pairs, not a complete category/depth release audit. The depth-ten Direction base was token-rejected because another contract view is too long; its compact Unique view is shown here.

## depth-1-sample-0-direction-m0-single

Base status: accepted; measured depth: 1.

### Prompt

Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object0, Object5, Object3, Object7, Object1, Object2, Object4
Query: direction Object6 Object3
Consider a map with multiple objects:
Object3 is to the Northeast of Object4.
Object2 is to the Southwest of Object3.
Object5 is to the Southwest of Object3.
Object0 is to the South of Object7.
Object3 is to the Northeast of Object1.
Object6 is to the Northeast of Object3.
Question: In which direction is Object6 relative to Object3 in every valid map?
A. Northeast
B. Northwest
C. Southwest
D. Southeast
E. Cannot be determined
F. None of the above

### Natural

Northeast:
P6: From p6, Object6 is exactly northeast of Object3.
Q-DIR: From P6, Object6 is exactly northeast of Object3.
The uniquely guaranteed answer is Northeast. It is displayed at A.
Answer: A

### Symbolic

Northeast:
P6: [p6] => Northeast(Object6,Object3)
Q-DIR: [P6] => Northeast(Object6,Object3)
The uniquely guaranteed answer is Northeast. It is displayed at A.
Answer: A

## depth-1-sample-0-which-m0-single

Base status: accepted; measured depth: 1.

### Prompt

Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object0, Object5, Object3, Object7, Object1, Object2, Object4
Query: which Northeast Object3
Eligible: Object1, Object5, Object4, Object6, Object2
Consider a map with multiple objects:
Object3 is to the Northeast of Object4.
Object2 is to the Southwest of Object3.
Object5 is to the Southwest of Object3.
Object0 is to the South of Object7.
Object3 is to the Northeast of Object1.
Object6 is to the Northeast of Object3.
Question: Which eligible object is the only one northeast of Object3 in every valid map?
A. Object2
B. Object6
C. Object4
D. Object5
E. Cannot be determined
F. None of the above

### Natural

Object1: impossible
P5: From p5, Object3 is exactly northeast of Object1.
A-REFUTE: Suppose Object1 is exactly northeast of Object3 for contradiction.
P5-X1: From P5, Object3's x-coordinate is greater than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object1's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P5-X1, refutation-X1, contradiction.
The assumption Object1 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object5: impossible
P3: From p3, Object5 is exactly southwest of Object3.
A-REFUTE: Suppose Object5 is exactly northeast of Object3 for contradiction.
P3-X1: From P3, Object5's x-coordinate is less than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object5's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption Object5 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object4: impossible
P1: From p1, Object3 is exactly northeast of Object4.
A-REFUTE: Suppose Object4 is exactly northeast of Object3 for contradiction.
P1-X1: From P1, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object4 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object6: entailed
P6: From p6, Object6 is exactly northeast of Object3.
Q-DIR: From P6, Object6 is exactly northeast of Object3.
Object2: impossible
P2: From p2, Object2 is exactly southwest of Object3.
A-REFUTE: Suppose Object2 is exactly northeast of Object3 for contradiction.
P2-X1: From P2, Object2's x-coordinate is less than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Object6. It is displayed at B.
Answer: B

### Symbolic

Object1: impossible
P5: [p5] => Northeast(Object3,Object1)
A-REFUTE: Suppose Northeast(Object1,Object3) for contradiction.
P5-X1: [P5] => x(Object3) > x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object1) > x(Object3)
refutation-C1: [under assumption refutation; P5-X1, refutation-X1] => contradiction
The assumption Northeast(Object1,Object3) leads to contradiction, so this direction is impossible.
Object5: impossible
P3: [p3] => Southwest(Object5,Object3)
A-REFUTE: Suppose Northeast(Object5,Object3) for contradiction.
P3-X1: [P3] => x(Object5) < x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object5) > x(Object3)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption Northeast(Object5,Object3) leads to contradiction, so this direction is impossible.
Object4: impossible
P1: [p1] => Northeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object4,Object3) for contradiction.
P1-X1: [P1] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) > x(Object3)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption Northeast(Object4,Object3) leads to contradiction, so this direction is impossible.
Object6: entailed
P6: [p6] => Northeast(Object6,Object3)
Q-DIR: [P6] => Northeast(Object6,Object3)
Object2: impossible
P2: [p2] => Southwest(Object2,Object3)
A-REFUTE: Suppose Northeast(Object2,Object3) for contradiction.
P2-X1: [P2] => x(Object2) < x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) > x(Object3)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northeast(Object2,Object3) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Object6. It is displayed at B.
Answer: B

## depth-1-sample-0-count-m0-single

Base status: accepted; measured depth: 1.

### Prompt

Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object0, Object5, Object3, Object7, Object1, Object2, Object4
Query: count Northeast Object3
Eligible: Object1, Object5, Object4, Object6, Object2
Consider a map with multiple objects:
Object3 is to the Northeast of Object4.
Object2 is to the Southwest of Object3.
Object5 is to the Southwest of Object3.
Object0 is to the South of Object7.
Object3 is to the Northeast of Object1.
Object6 is to the Northeast of Object3.
Question: What exact number of eligible objects are northeast of Object3 in every valid map?
A. 1
B. 2
C. 0
D. 3
E. Cannot be determined
F. None of the above

### Natural

P5: From p5, Object3 is exactly northeast of Object1.
A-REFUTE: Suppose Object1 is exactly northeast of Object3 for contradiction.
P5-X1: From P5, Object3's x-coordinate is greater than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object1's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P5-X1, refutation-X1, contradiction.
The assumption Object1 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Fixed membership: Object1 is impossible.
P3: From p3, Object5 is exactly southwest of Object3.
A-REFUTE: Suppose Object5 is exactly northeast of Object3 for contradiction.
P3-X1: From P3, Object5's x-coordinate is less than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object5's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption Object5 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Fixed membership: Object5 is impossible.
P1: From p1, Object3 is exactly northeast of Object4.
A-REFUTE: Suppose Object4 is exactly northeast of Object3 for contradiction.
P1-X1: From P1, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object4 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
P6: From p6, Object6 is exactly northeast of Object3.
Q-DIR: From P6, Object6 is exactly northeast of Object3.
Fixed membership: Object6 is entailed.
P2: From p2, Object2 is exactly southwest of Object3.
A-REFUTE: Suppose Object2 is exactly northeast of Object3 for contradiction.
P2-X1: From P2, Object2's x-coordinate is less than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 0 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
A valid map places Object6 at (2, 7); Object0 at (2, 5); Object5 at (0, 3); Object3 at (1, 4); Object7 at (2, 6); Object1 at (0, 2); Object2 at (0, 1); Object4 at (0, 0). In this map, count=1.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 1. It is displayed at A.
Answer: A

### Symbolic

P5: [p5] => Northeast(Object3,Object1)
A-REFUTE: Suppose Northeast(Object1,Object3) for contradiction.
P5-X1: [P5] => x(Object3) > x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object1) > x(Object3)
refutation-C1: [under assumption refutation; P5-X1, refutation-X1] => contradiction
The assumption Northeast(Object1,Object3) leads to contradiction, so this direction is impossible.
Fixed membership: Object1 is impossible.
P3: [p3] => Southwest(Object5,Object3)
A-REFUTE: Suppose Northeast(Object5,Object3) for contradiction.
P3-X1: [P3] => x(Object5) < x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object5) > x(Object3)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption Northeast(Object5,Object3) leads to contradiction, so this direction is impossible.
Fixed membership: Object5 is impossible.
P1: [p1] => Northeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object4,Object3) for contradiction.
P1-X1: [P1] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) > x(Object3)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption Northeast(Object4,Object3) leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
P6: [p6] => Northeast(Object6,Object3)
Q-DIR: [P6] => Northeast(Object6,Object3)
Fixed membership: Object6 is entailed.
P2: [p2] => Southwest(Object2,Object3)
A-REFUTE: Suppose Northeast(Object2,Object3) for contradiction.
P2-X1: [P2] => x(Object2) < x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) > x(Object3)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northeast(Object2,Object3) leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 0 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Valid map {Object6:(2,7), Object0:(2,5), Object5:(0,3), Object3:(1,4), Object7:(2,6), Object1:(0,2), Object2:(0,1), Object4:(0,0)} => count=1.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 1. It is displayed at A.
Answer: A

## depth-6-sample-0-count-m0-single

Base status: accepted; measured depth: 6.

### Prompt

Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object4, Object13, Object6, Object1, Object10, Object12, Object7, Object11, Object8, Object2, Object3, Object5, Object9
Query: count North Object1
Eligible: Object11, Object8, Object2, Object7, Object13
Consider a map with multiple objects:
Object9 is to the South of Object2.
Object0 is to the Northwest of Object4.
Object5 is to the South of Object12.
Object1 is to the South of Object5.
Object9 is to the North of Object3.
Object4 is to the South of Object10.
Object12 is to the South of Object4.
Object2 is to the Southeast of Object6.
Object3 is to the North of Object4.
Object13 is to the South of Object1.
Object7 is to the South of Object1.
Object8 is to the South of Object1.
Object11 is to the South of Object1.
Question: What exact number of eligible objects are north of Object1 in every valid map?
A. 4
B. 2
C. 0
D. 1
E. Cannot be determined
F. None of the above

### Natural

P13: From p13, Object11 is exactly south of Object1.
A-REFUTE: Suppose Object11 is exactly north of Object1 for contradiction.
P13-Y1: From P13, Object11's y-coordinate is less than Object1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object11's y-coordinate is greater than Object1's y-coordinate.
refutation-C1: From under assumption refutation; P13-Y1, refutation-Y1, contradiction.
The assumption Object11 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object11 is impossible.
P12: From p12, Object8 is exactly south of Object1.
A-REFUTE: Suppose Object8 is exactly north of Object1 for contradiction.
P12-Y1: From P12, Object8's y-coordinate is less than Object1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object8's y-coordinate is greater than Object1's y-coordinate.
refutation-C1: From under assumption refutation; P12-Y1, refutation-Y1, contradiction.
The assumption Object8 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object8 is impossible.
P1: From p1, Object9 is exactly south of Object2.
P3: From p3, Object5 is exactly south of Object12.
P4: From p4, Object1 is exactly south of Object5.
P5: From p5, Object9 is exactly north of Object3.
P7: From p7, Object12 is exactly south of Object4.
P9: From p9, Object3 is exactly north of Object4.
P1-X1: From P1, Object9's x-coordinate is equal to Object2's x-coordinate.
P1-Y1: From P1, Object9's y-coordinate is less than Object2's y-coordinate.
P3-X1: From P3, Object5's x-coordinate is equal to Object12's x-coordinate.
P3-Y1: From P3, Object5's y-coordinate is less than Object12's y-coordinate.
P4-X1: From P4, Object1's x-coordinate is equal to Object5's x-coordinate.
P4-Y1: From P4, Object1's y-coordinate is less than Object5's y-coordinate.
P5-X1: From P5, Object9's x-coordinate is equal to Object3's x-coordinate.
P5-Y1: From P5, Object9's y-coordinate is greater than Object3's y-coordinate.
P7-X1: From P7, Object12's x-coordinate is equal to Object4's x-coordinate.
P7-Y1: From P7, Object12's y-coordinate is less than Object4's y-coordinate.
P9-X1: From P9, Object3's x-coordinate is equal to Object4's x-coordinate.
P9-Y1: From P9, Object3's y-coordinate is greater than Object4's y-coordinate.
PATH-X-1-INV: From P1-X1, Object2's x-coordinate is equal to Object9's x-coordinate.
PATH-X-4-INV: From P7-X1, Object4's x-coordinate is equal to Object12's x-coordinate.
PATH-X-5-INV: From P3-X1, Object12's x-coordinate is equal to Object5's x-coordinate.
PATH-X-6-INV: From P4-X1, Object5's x-coordinate is equal to Object1's x-coordinate.
D-X-1: From PATH-X-1-INV, P5-X1, Object2's x-coordinate is equal to Object3's x-coordinate.
D-X-2: From D-X-1, P9-X1, Object2's x-coordinate is equal to Object4's x-coordinate.
D-X-3: From D-X-2, PATH-X-4-INV, Object2's x-coordinate is equal to Object12's x-coordinate.
D-X-4: From D-X-3, PATH-X-5-INV, Object2's x-coordinate is equal to Object5's x-coordinate.
D-X-5: From D-X-4, PATH-X-6-INV, Object2's x-coordinate is equal to Object1's x-coordinate.
PATH-Y-4-INV: From P9-Y1, Object4's y-coordinate is less than Object3's y-coordinate.
PATH-Y-5-INV: From P5-Y1, Object3's y-coordinate is less than Object9's y-coordinate.
D-Y-1: From P4-Y1, P3-Y1, Object1's y-coordinate is less than Object12's y-coordinate.
D-Y-2: From D-Y-1, P7-Y1, Object1's y-coordinate is less than Object4's y-coordinate.
D-Y-3: From D-Y-2, PATH-Y-4-INV, Object1's y-coordinate is less than Object3's y-coordinate.
D-Y-4: From D-Y-3, PATH-Y-5-INV, Object1's y-coordinate is less than Object9's y-coordinate.
D-Y-5: From D-Y-4, P1-Y1, Object1's y-coordinate is less than Object2's y-coordinate.
Q-DIR: From D-X-5, D-Y-5, Object2 is exactly north of Object1.
Fixed membership: Object2 is entailed.
P11: From p11, Object7 is exactly south of Object1.
A-REFUTE: Suppose Object7 is exactly north of Object1 for contradiction.
P11-Y1: From P11, Object7's y-coordinate is less than Object1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object7's y-coordinate is greater than Object1's y-coordinate.
refutation-C1: From under assumption refutation; P11-Y1, refutation-Y1, contradiction.
The assumption Object7 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object7 is impossible.
P10: From p10, Object13 is exactly south of Object1.
A-REFUTE: Suppose Object13 is exactly north of Object1 for contradiction.
P10-Y1: From P10, Object13's y-coordinate is less than Object1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object13's y-coordinate is greater than Object1's y-coordinate.
refutation-C1: From under assumption refutation; P10-Y1, refutation-Y1, contradiction.
The assumption Object13 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object13 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 0 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
A valid map places Object0 at (0, 12); Object4 at (1, 7); Object13 at (1, 3); Object6 at (0, 11); Object1 at (1, 4); Object10 at (1, 13); Object12 at (1, 6); Object7 at (1, 2); Object11 at (1, 1); Object8 at (1, 0); Object2 at (1, 10); Object3 at (1, 8); Object5 at (1, 5); Object9 at (1, 9). In this map, count=1.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 1. It is displayed at D.
Answer: D

### Symbolic

P13: [p13] => South(Object11,Object1)
A-REFUTE: Suppose North(Object11,Object1) for contradiction.
P13-Y1: [P13] => y(Object11) < y(Object1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object11) > y(Object1)
refutation-C1: [under assumption refutation; P13-Y1, refutation-Y1] => contradiction
The assumption North(Object11,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object11 is impossible.
P12: [p12] => South(Object8,Object1)
A-REFUTE: Suppose North(Object8,Object1) for contradiction.
P12-Y1: [P12] => y(Object8) < y(Object1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object8) > y(Object1)
refutation-C1: [under assumption refutation; P12-Y1, refutation-Y1] => contradiction
The assumption North(Object8,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object8 is impossible.
P1: [p1] => South(Object9,Object2)
P3: [p3] => South(Object5,Object12)
P4: [p4] => South(Object1,Object5)
P5: [p5] => North(Object9,Object3)
P7: [p7] => South(Object12,Object4)
P9: [p9] => North(Object3,Object4)
P1-X1: [P1] => x(Object9) = x(Object2)
P1-Y1: [P1] => y(Object9) < y(Object2)
P3-X1: [P3] => x(Object5) = x(Object12)
P3-Y1: [P3] => y(Object5) < y(Object12)
P4-X1: [P4] => x(Object1) = x(Object5)
P4-Y1: [P4] => y(Object1) < y(Object5)
P5-X1: [P5] => x(Object9) = x(Object3)
P5-Y1: [P5] => y(Object9) > y(Object3)
P7-X1: [P7] => x(Object12) = x(Object4)
P7-Y1: [P7] => y(Object12) < y(Object4)
P9-X1: [P9] => x(Object3) = x(Object4)
P9-Y1: [P9] => y(Object3) > y(Object4)
PATH-X-1-INV: [P1-X1] => x(Object2) = x(Object9)
PATH-X-4-INV: [P7-X1] => x(Object4) = x(Object12)
PATH-X-5-INV: [P3-X1] => x(Object12) = x(Object5)
PATH-X-6-INV: [P4-X1] => x(Object5) = x(Object1)
D-X-1: [PATH-X-1-INV, P5-X1] => x(Object2) = x(Object3)
D-X-2: [D-X-1, P9-X1] => x(Object2) = x(Object4)
D-X-3: [D-X-2, PATH-X-4-INV] => x(Object2) = x(Object12)
D-X-4: [D-X-3, PATH-X-5-INV] => x(Object2) = x(Object5)
D-X-5: [D-X-4, PATH-X-6-INV] => x(Object2) = x(Object1)
PATH-Y-4-INV: [P9-Y1] => y(Object4) < y(Object3)
PATH-Y-5-INV: [P5-Y1] => y(Object3) < y(Object9)
D-Y-1: [P4-Y1, P3-Y1] => y(Object1) < y(Object12)
D-Y-2: [D-Y-1, P7-Y1] => y(Object1) < y(Object4)
D-Y-3: [D-Y-2, PATH-Y-4-INV] => y(Object1) < y(Object3)
D-Y-4: [D-Y-3, PATH-Y-5-INV] => y(Object1) < y(Object9)
D-Y-5: [D-Y-4, P1-Y1] => y(Object1) < y(Object2)
Q-DIR: [D-X-5, D-Y-5] => North(Object2,Object1)
Fixed membership: Object2 is entailed.
P11: [p11] => South(Object7,Object1)
A-REFUTE: Suppose North(Object7,Object1) for contradiction.
P11-Y1: [P11] => y(Object7) < y(Object1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object7) > y(Object1)
refutation-C1: [under assumption refutation; P11-Y1, refutation-Y1] => contradiction
The assumption North(Object7,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object7 is impossible.
P10: [p10] => South(Object13,Object1)
A-REFUTE: Suppose North(Object13,Object1) for contradiction.
P10-Y1: [P10] => y(Object13) < y(Object1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object13) > y(Object1)
refutation-C1: [under assumption refutation; P10-Y1, refutation-Y1] => contradiction
The assumption North(Object13,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object13 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 0 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Valid map {Object0:(0,12), Object4:(1,7), Object13:(1,3), Object6:(0,11), Object1:(1,4), Object10:(1,13), Object12:(1,6), Object7:(1,2), Object11:(1,1), Object8:(1,0), Object2:(1,10), Object3:(1,8), Object5:(1,5), Object9:(1,9)} => count=1.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 1. It is displayed at D.
Answer: D

## depth-10-sample-0-direction-m0-single

Base status: token-rejected; measured depth: 10.

### Prompt

Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object8, Object1, Object13, Object14, Object9, Object7, Object10, Object12, Object5, Object0, Object11, Object4, Object2, Object3
Query: direction Object5 Object3
Consider a map with multiple objects:
Object9 is to the West of Object4.
Object0 is to the East of Object3.
Object14 is to the East of Object5.
Object3 is to the West of Object7.
Object13 is to the West of Object3.
Object8 is to the East of Object3.
Object14 is to the West of Object11.
Object1 is to the West of Object6.
Object6 is to the West of Object10.
Object3 is to the West of Object2.
Object12 is to the West of Object1.
Object10 is to the West of Object13.
Object11 is to the West of Object9.
Object4 is to the West of Object12.
Question: In which direction is Object5 relative to Object3 in every valid map?
A. Northeast
B. West
C. East
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

West:
P1: From p1, Object9 is exactly west of Object4.
P3: From p3, Object14 is exactly east of Object5.
P5: From p5, Object13 is exactly west of Object3.
P7: From p7, Object14 is exactly west of Object11.
P8: From p8, Object1 is exactly west of Object6.
P9: From p9, Object6 is exactly west of Object10.
P11: From p11, Object12 is exactly west of Object1.
P12: From p12, Object10 is exactly west of Object13.
P13: From p13, Object11 is exactly west of Object9.
P14: From p14, Object4 is exactly west of Object12.
P1-X1: From P1, Object9's x-coordinate is less than Object4's x-coordinate.
P1-Y1: From P1, Object9's y-coordinate is equal to Object4's y-coordinate.
P3-X1: From P3, Object14's x-coordinate is greater than Object5's x-coordinate.
P3-Y1: From P3, Object14's y-coordinate is equal to Object5's y-coordinate.
P5-X1: From P5, Object13's x-coordinate is less than Object3's x-coordinate.
P5-Y1: From P5, Object13's y-coordinate is equal to Object3's y-coordinate.
P7-X1: From P7, Object14's x-coordinate is less than Object11's x-coordinate.
P7-Y1: From P7, Object14's y-coordinate is equal to Object11's y-coordinate.
P8-X1: From P8, Object1's x-coordinate is less than Object6's x-coordinate.
P8-Y1: From P8, Object1's y-coordinate is equal to Object6's y-coordinate.
P9-X1: From P9, Object6's x-coordinate is less than Object10's x-coordinate.
P9-Y1: From P9, Object6's y-coordinate is equal to Object10's y-coordinate.
P11-X1: From P11, Object12's x-coordinate is less than Object1's x-coordinate.
P11-Y1: From P11, Object12's y-coordinate is equal to Object1's y-coordinate.
P12-X1: From P12, Object10's x-coordinate is less than Object13's x-coordinate.
P12-Y1: From P12, Object10's y-coordinate is equal to Object13's y-coordinate.
P13-X1: From P13, Object11's x-coordinate is less than Object9's x-coordinate.
P13-Y1: From P13, Object11's y-coordinate is equal to Object9's y-coordinate.
P14-X1: From P14, Object4's x-coordinate is less than Object12's x-coordinate.
P14-Y1: From P14, Object4's y-coordinate is equal to Object12's y-coordinate.
PATH-X-1-INV: From P3-X1, Object5's x-coordinate is less than Object14's x-coordinate.
D-X-1: From PATH-X-1-INV, P7-X1, Object5's x-coordinate is less than Object11's x-coordinate.
D-X-2: From D-X-1, P13-X1, Object5's x-coordinate is less than Object9's x-coordinate.
D-X-3: From D-X-2, P1-X1, Object5's x-coordinate is less than Object4's x-coordinate.
D-X-4: From D-X-3, P14-X1, Object5's x-coordinate is less than Object12's x-coordinate.
D-X-5: From D-X-4, P11-X1, Object5's x-coordinate is less than Object1's x-coordinate.
D-X-6: From D-X-5, P8-X1, Object5's x-coordinate is less than Object6's x-coordinate.
D-X-7: From D-X-6, P9-X1, Object5's x-coordinate is less than Object10's x-coordinate.
D-X-8: From D-X-7, P12-X1, Object5's x-coordinate is less than Object13's x-coordinate.
D-X-9: From D-X-8, P5-X1, Object5's x-coordinate is less than Object3's x-coordinate.
PATH-Y-1-INV: From P3-Y1, Object5's y-coordinate is equal to Object14's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P7-Y1, Object5's y-coordinate is equal to Object11's y-coordinate.
D-Y-2: From D-Y-1, P13-Y1, Object5's y-coordinate is equal to Object9's y-coordinate.
D-Y-3: From D-Y-2, P1-Y1, Object5's y-coordinate is equal to Object4's y-coordinate.
D-Y-4: From D-Y-3, P14-Y1, Object5's y-coordinate is equal to Object12's y-coordinate.
D-Y-5: From D-Y-4, P11-Y1, Object5's y-coordinate is equal to Object1's y-coordinate.
D-Y-6: From D-Y-5, P8-Y1, Object5's y-coordinate is equal to Object6's y-coordinate.
D-Y-7: From D-Y-6, P9-Y1, Object5's y-coordinate is equal to Object10's y-coordinate.
D-Y-8: From D-Y-7, P12-Y1, Object5's y-coordinate is equal to Object13's y-coordinate.
D-Y-9: From D-Y-8, P5-Y1, Object5's y-coordinate is equal to Object3's y-coordinate.
Q-DIR: From D-X-9, D-Y-9, Object5 is exactly west of Object3.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B

### Symbolic

West:
P1: [p1] => West(Object9,Object4)
P3: [p3] => East(Object14,Object5)
P5: [p5] => West(Object13,Object3)
P7: [p7] => West(Object14,Object11)
P8: [p8] => West(Object1,Object6)
P9: [p9] => West(Object6,Object10)
P11: [p11] => West(Object12,Object1)
P12: [p12] => West(Object10,Object13)
P13: [p13] => West(Object11,Object9)
P14: [p14] => West(Object4,Object12)
P1-X1: [P1] => x(Object9) < x(Object4)
P1-Y1: [P1] => y(Object9) = y(Object4)
P3-X1: [P3] => x(Object14) > x(Object5)
P3-Y1: [P3] => y(Object14) = y(Object5)
P5-X1: [P5] => x(Object13) < x(Object3)
P5-Y1: [P5] => y(Object13) = y(Object3)
P7-X1: [P7] => x(Object14) < x(Object11)
P7-Y1: [P7] => y(Object14) = y(Object11)
P8-X1: [P8] => x(Object1) < x(Object6)
P8-Y1: [P8] => y(Object1) = y(Object6)
P9-X1: [P9] => x(Object6) < x(Object10)
P9-Y1: [P9] => y(Object6) = y(Object10)
P11-X1: [P11] => x(Object12) < x(Object1)
P11-Y1: [P11] => y(Object12) = y(Object1)
P12-X1: [P12] => x(Object10) < x(Object13)
P12-Y1: [P12] => y(Object10) = y(Object13)
P13-X1: [P13] => x(Object11) < x(Object9)
P13-Y1: [P13] => y(Object11) = y(Object9)
P14-X1: [P14] => x(Object4) < x(Object12)
P14-Y1: [P14] => y(Object4) = y(Object12)
PATH-X-1-INV: [P3-X1] => x(Object5) < x(Object14)
D-X-1: [PATH-X-1-INV, P7-X1] => x(Object5) < x(Object11)
D-X-2: [D-X-1, P13-X1] => x(Object5) < x(Object9)
D-X-3: [D-X-2, P1-X1] => x(Object5) < x(Object4)
D-X-4: [D-X-3, P14-X1] => x(Object5) < x(Object12)
D-X-5: [D-X-4, P11-X1] => x(Object5) < x(Object1)
D-X-6: [D-X-5, P8-X1] => x(Object5) < x(Object6)
D-X-7: [D-X-6, P9-X1] => x(Object5) < x(Object10)
D-X-8: [D-X-7, P12-X1] => x(Object5) < x(Object13)
D-X-9: [D-X-8, P5-X1] => x(Object5) < x(Object3)
PATH-Y-1-INV: [P3-Y1] => y(Object5) = y(Object14)
D-Y-1: [PATH-Y-1-INV, P7-Y1] => y(Object5) = y(Object11)
D-Y-2: [D-Y-1, P13-Y1] => y(Object5) = y(Object9)
D-Y-3: [D-Y-2, P1-Y1] => y(Object5) = y(Object4)
D-Y-4: [D-Y-3, P14-Y1] => y(Object5) = y(Object12)
D-Y-5: [D-Y-4, P11-Y1] => y(Object5) = y(Object1)
D-Y-6: [D-Y-5, P8-Y1] => y(Object5) = y(Object6)
D-Y-7: [D-Y-6, P9-Y1] => y(Object5) = y(Object10)
D-Y-8: [D-Y-7, P12-Y1] => y(Object5) = y(Object13)
D-Y-9: [D-Y-8, P5-Y1] => y(Object5) = y(Object3)
Q-DIR: [D-X-9, D-Y-9] => West(Object5,Object3)
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B

## depth-10-sample-1-count-m0-single

Base status: accepted; measured depth: 10.

### Prompt

Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object18, Object21, Object16, Object20, Object0, Object7, Object8, Object9, Object11, Object5, Object1, Object15, Object4, Object14, Object17, Object3, Object19, Object2, Object12, Object10, Object13
Query: count Northwest Object20
Eligible: Object19, Object7, Object10, Object0, Object14
Consider a map with multiple objects:
Object4 is to the West of Object13.
Object0 is to the Southeast of Object20.
Object5 is to the Southeast of Object11.
Object8 is to the Southeast of Object3.
Object8 is to the West of Object4.
Object20 is to the Northwest of Object19.
Object14 is to the Southeast of Object13.
Object11 is to the East of Object6.
Object9 is to the Northwest of Object3.
Object7 is to the Southeast of Object20.
Object10 is to the Southeast of Object20.
Object4 is to the Southeast of Object12.
Object5 is to the Northwest of Object21.
Object20 is to the Northwest of Object6.
Object21 is to the West of Object9.
Object8 is to the South of Object16.
Object13 is to the Northeast of Object17.
Object2 is to the South of Object15.
Object18 is to the West of Object1.
Question: What exact number of eligible objects are northwest of Object20 in every valid map?
A. 4
B. 0
C. 1
D. 3
E. Cannot be determined
F. None of the above

### Natural

P6: From p6, Object20 is exactly northwest of Object19.
A-REFUTE: Suppose Object19 is exactly northwest of Object20 for contradiction.
P6-X1: From P6, Object20's x-coordinate is less than Object19's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object19's x-coordinate is less than Object20's x-coordinate.
refutation-C1: From under assumption refutation; P6-X1, refutation-X1, contradiction.
The assumption Object19 is exactly northwest of Object20 leads to contradiction, so this direction is impossible.
Fixed membership: Object19 is impossible.
P10: From p10, Object7 is exactly southeast of Object20.
A-REFUTE: Suppose Object7 is exactly northwest of Object20 for contradiction.
P10-X1: From P10, Object7's x-coordinate is greater than Object20's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object7's x-coordinate is less than Object20's x-coordinate.
refutation-C1: From under assumption refutation; P10-X1, refutation-X1, contradiction.
The assumption Object7 is exactly northwest of Object20 leads to contradiction, so this direction is impossible.
Fixed membership: Object7 is impossible.
P11: From p11, Object10 is exactly southeast of Object20.
A-REFUTE: Suppose Object10 is exactly northwest of Object20 for contradiction.
P11-X1: From P11, Object10's x-coordinate is greater than Object20's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object10's x-coordinate is less than Object20's x-coordinate.
refutation-C1: From under assumption refutation; P11-X1, refutation-X1, contradiction.
The assumption Object10 is exactly northwest of Object20 leads to contradiction, so this direction is impossible.
Fixed membership: Object10 is impossible.
P2: From p2, Object0 is exactly southeast of Object20.
A-REFUTE: Suppose Object0 is exactly northwest of Object20 for contradiction.
P2-X1: From P2, Object0's x-coordinate is greater than Object20's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is less than Object20's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object0 is exactly northwest of Object20 leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: From p1, Object4 is exactly west of Object13.
P3: From p3, Object5 is exactly southeast of Object11.
P4: From p4, Object8 is exactly southeast of Object3.
P5: From p5, Object8 is exactly west of Object4.
P7: From p7, Object14 is exactly southeast of Object13.
P8: From p8, Object11 is exactly east of Object6.
P9: From p9, Object9 is exactly northwest of Object3.
P13: From p13, Object5 is exactly northwest of Object21.
P14: From p14, Object20 is exactly northwest of Object6.
P15: From p15, Object21 is exactly west of Object9.
A-REFUTE: Suppose Object14 is exactly northwest of Object20 for contradiction.
P1-X1: From P1, Object4's x-coordinate is less than Object13's x-coordinate.
P3-X1: From P3, Object5's x-coordinate is greater than Object11's x-coordinate.
P4-X1: From P4, Object8's x-coordinate is greater than Object3's x-coordinate.
P5-X1: From P5, Object8's x-coordinate is less than Object4's x-coordinate.
P7-X1: From P7, Object14's x-coordinate is greater than Object13's x-coordinate.
P8-X1: From P8, Object11's x-coordinate is greater than Object6's x-coordinate.
P9-X1: From P9, Object9's x-coordinate is less than Object3's x-coordinate.
P13-X1: From P13, Object5's x-coordinate is less than Object21's x-coordinate.
P14-X1: From P14, Object20's x-coordinate is less than Object6's x-coordinate.
P15-X1: From P15, Object21's x-coordinate is less than Object9's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object14's x-coordinate is less than Object20's x-coordinate.
PATH-X-2-INV: From P8-X1, Object6's x-coordinate is less than Object11's x-coordinate.
PATH-X-3-INV: From P3-X1, Object11's x-coordinate is less than Object5's x-coordinate.
PATH-X-7-INV: From P4-X1, Object3's x-coordinate is less than Object8's x-coordinate.
PATH-X-10-INV: From P7-X1, Object13's x-coordinate is less than Object14's x-coordinate.
D-X-1: From P14-X1, PATH-X-2-INV, Object20's x-coordinate is less than Object11's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object20's x-coordinate is less than Object5's x-coordinate.
D-X-3: From D-X-2, P13-X1, Object20's x-coordinate is less than Object21's x-coordinate.
D-X-4: From D-X-3, P15-X1, Object20's x-coordinate is less than Object9's x-coordinate.
D-X-5: From D-X-4, P9-X1, Object20's x-coordinate is less than Object3's x-coordinate.
D-X-6: From D-X-5, PATH-X-7-INV, Object20's x-coordinate is less than Object8's x-coordinate.
D-X-7: From D-X-6, P5-X1, Object20's x-coordinate is less than Object4's x-coordinate.
D-X-8: From D-X-7, P1-X1, Object20's x-coordinate is less than Object13's x-coordinate.
D-X-9: From D-X-8, PATH-X-10-INV, Object20's x-coordinate is less than Object14's x-coordinate.
refutation-C1: From under assumption refutation; D-X-9, refutation-X1, contradiction.
The assumption Object14 is exactly northwest of Object20 leads to contradiction, so this direction is impossible.
Fixed membership: Object14 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places Object6 at (1, 14); Object18 at (1, 7); Object21 at (4, 12); Object16 at (7, 16); Object20 at (0, 15); Object0 at (1, 6); Object7 at (1, 5); Object8 at (7, 9); Object9 at (5, 12); Object11 at (2, 14); Object5 at (3, 13); Object1 at (4, 7); Object15 at (1, 4); Object4 at (8, 9); Object14 at (10, 8); Object17 at (1, 3); Object3 at (6, 11); Object19 at (1, 2); Object2 at (1, 1); Object12 at (1, 10); Object10 at (1, 0); Object13 at (9, 9). In this map, count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 0. It is displayed at B.
Answer: B

### Symbolic

P6: [p6] => Northwest(Object20,Object19)
A-REFUTE: Suppose Northwest(Object19,Object20) for contradiction.
P6-X1: [P6] => x(Object20) < x(Object19)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object19) < x(Object20)
refutation-C1: [under assumption refutation; P6-X1, refutation-X1] => contradiction
The assumption Northwest(Object19,Object20) leads to contradiction, so this direction is impossible.
Fixed membership: Object19 is impossible.
P10: [p10] => Southeast(Object7,Object20)
A-REFUTE: Suppose Northwest(Object7,Object20) for contradiction.
P10-X1: [P10] => x(Object7) > x(Object20)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object7) < x(Object20)
refutation-C1: [under assumption refutation; P10-X1, refutation-X1] => contradiction
The assumption Northwest(Object7,Object20) leads to contradiction, so this direction is impossible.
Fixed membership: Object7 is impossible.
P11: [p11] => Southeast(Object10,Object20)
A-REFUTE: Suppose Northwest(Object10,Object20) for contradiction.
P11-X1: [P11] => x(Object10) > x(Object20)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object10) < x(Object20)
refutation-C1: [under assumption refutation; P11-X1, refutation-X1] => contradiction
The assumption Northwest(Object10,Object20) leads to contradiction, so this direction is impossible.
Fixed membership: Object10 is impossible.
P2: [p2] => Southeast(Object0,Object20)
A-REFUTE: Suppose Northwest(Object0,Object20) for contradiction.
P2-X1: [P2] => x(Object0) > x(Object20)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) < x(Object20)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northwest(Object0,Object20) leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: [p1] => West(Object4,Object13)
P3: [p3] => Southeast(Object5,Object11)
P4: [p4] => Southeast(Object8,Object3)
P5: [p5] => West(Object8,Object4)
P7: [p7] => Southeast(Object14,Object13)
P8: [p8] => East(Object11,Object6)
P9: [p9] => Northwest(Object9,Object3)
P13: [p13] => Northwest(Object5,Object21)
P14: [p14] => Northwest(Object20,Object6)
P15: [p15] => West(Object21,Object9)
A-REFUTE: Suppose Northwest(Object14,Object20) for contradiction.
P1-X1: [P1] => x(Object4) < x(Object13)
P3-X1: [P3] => x(Object5) > x(Object11)
P4-X1: [P4] => x(Object8) > x(Object3)
P5-X1: [P5] => x(Object8) < x(Object4)
P7-X1: [P7] => x(Object14) > x(Object13)
P8-X1: [P8] => x(Object11) > x(Object6)
P9-X1: [P9] => x(Object9) < x(Object3)
P13-X1: [P13] => x(Object5) < x(Object21)
P14-X1: [P14] => x(Object20) < x(Object6)
P15-X1: [P15] => x(Object21) < x(Object9)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object14) < x(Object20)
PATH-X-2-INV: [P8-X1] => x(Object6) < x(Object11)
PATH-X-3-INV: [P3-X1] => x(Object11) < x(Object5)
PATH-X-7-INV: [P4-X1] => x(Object3) < x(Object8)
PATH-X-10-INV: [P7-X1] => x(Object13) < x(Object14)
D-X-1: [P14-X1, PATH-X-2-INV] => x(Object20) < x(Object11)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object20) < x(Object5)
D-X-3: [D-X-2, P13-X1] => x(Object20) < x(Object21)
D-X-4: [D-X-3, P15-X1] => x(Object20) < x(Object9)
D-X-5: [D-X-4, P9-X1] => x(Object20) < x(Object3)
D-X-6: [D-X-5, PATH-X-7-INV] => x(Object20) < x(Object8)
D-X-7: [D-X-6, P5-X1] => x(Object20) < x(Object4)
D-X-8: [D-X-7, P1-X1] => x(Object20) < x(Object13)
D-X-9: [D-X-8, PATH-X-10-INV] => x(Object20) < x(Object14)
refutation-C1: [under assumption refutation; D-X-9, refutation-X1] => contradiction
The assumption Northwest(Object14,Object20) leads to contradiction, so this direction is impossible.
Fixed membership: Object14 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {Object6:(1,14), Object18:(1,7), Object21:(4,12), Object16:(7,16), Object20:(0,15), Object0:(1,6), Object7:(1,5), Object8:(7,9), Object9:(5,12), Object11:(2,14), Object5:(3,13), Object1:(4,7), Object15:(1,4), Object4:(8,9), Object14:(10,8), Object17:(1,3), Object3:(6,11), Object19:(1,2), Object2:(1,1), Object12:(1,10), Object10:(1,0), Object13:(9,9)} => count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 0. It is displayed at B.
Answer: B
