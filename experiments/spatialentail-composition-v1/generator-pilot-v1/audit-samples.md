# Pilot Audit Samples

Review status: pending. Inspect both styles and record outcomes.

## chain-0-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: direction C0 O
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: In which direction is C0 relative to O in every valid map?
A. South
B. Northwest
C. Northeast
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly northeast of O.
East:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is equal to O's y-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Northeast. It is displayed at C.
Answer: C
```
### Symbolic
```text
North:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
P2-X1: [P2] => x(Link0) = x(O)
P2-Y1: [P2] => y(Link0) > y(O)
P4-X1: [P4] => x(Link0) < x(C0)
P4-Y1: [P4] => y(Link0) < y(C0)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
Q-DIR: [D-X-1, D-Y-1] => Northeast(C0,O)
East:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P2-Y1: [P2] => y(Link0) > y(O)
P4-Y1: [P4] => y(Link0) < y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) = y(O)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-Y1: [P2] => y(Link0) > y(O)
P4-Y1: [P4] => y(Link0) < y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Northeast. It is displayed at C.
Answer: C
```

## chain-0-base-direction-m0-all-possible
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: direction C0 O
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: Select the complete set of directions C0 could occupy relative to O.
A. South
B. Northwest
C. Northeast
D. Southeast
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
North:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly northeast of O.
East:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is equal to O's y-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The complete possible set is {Northeast}.
Answer: C
```
### Symbolic
```text
North:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
P2-X1: [P2] => x(Link0) = x(O)
P2-Y1: [P2] => y(Link0) > y(O)
P4-X1: [P4] => x(Link0) < x(C0)
P4-Y1: [P4] => y(Link0) < y(C0)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
Q-DIR: [D-X-1, D-Y-1] => Northeast(C0,O)
East:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P2-Y1: [P2] => y(Link0) > y(O)
P4-Y1: [P4] => y(Link0) < y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) = y(O)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-Y1: [P2] => y(Link0) > y(O)
P4-Y1: [P4] => y(Link0) < y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The complete possible set is {Northeast}.
Answer: C
```

## chain-0-base-direction-m0-visible-possible
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: direction C0 O
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: Which listed directions could C0 occupy relative to O? Select all that apply.
A. South
B. Northwest
C. Northeast
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
Northeast:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly northeast of O.
Southeast:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-Y1: From P2, Link0's y-coordinate is greater than O's y-coordinate.
P4-Y1: From P4, Link0's y-coordinate is less than C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-1-INV: From P2-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The listed possible set is {Northeast}.
Answer: C
```
### Symbolic
```text
Northeast:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
P2-X1: [P2] => x(Link0) = x(O)
P2-Y1: [P2] => y(Link0) > y(O)
P4-X1: [P4] => x(Link0) < x(C0)
P4-Y1: [P4] => y(Link0) < y(C0)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
Q-DIR: [D-X-1, D-Y-1] => Northeast(C0,O)
Southeast:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-Y1: [P2] => y(Link0) > y(O)
P4-Y1: [P4] => y(Link0) < y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-1-INV: [P2-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The listed possible set is {Northeast}.
Answer: C
```

## chain-0-base-which-m0-single
Split: train; case: no-match; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: which North O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: Which eligible object is the only one north of O in every valid map?
A. C1
B. C0
C. C3
D. C2
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: impossible
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
C1: impossible
P3: From p3, O is exactly north of C1.
A-REFUTE: Suppose C1 is exactly north of O for contradiction.
P3-Y1: From P3, O's y-coordinate is greater than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly north of O leads to contradiction, so this direction is impossible.
C2: impossible
P5: From p5, O is exactly north of C2.
A-REFUTE: Suppose C2 is exactly north of O for contradiction.
P5-Y1: From P5, O's y-coordinate is greater than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P5-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly north of O leads to contradiction, so this direction is impossible.
C3: impossible
P1: From p1, O is exactly north of C3.
A-REFUTE: Suppose C3 is exactly north of O for contradiction.
P1-Y1: From P1, O's y-coordinate is greater than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly north of O leads to contradiction, so this direction is impossible.
No eligible object can qualify, so there is no unique qualifying object.
Answer: E
```
### Symbolic
```text
C0: impossible
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
C1: impossible
P3: [p3] => North(O,C1)
A-REFUTE: Suppose North(C1,O) for contradiction.
P3-Y1: [P3] => y(O) > y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) > y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption North(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P5: [p5] => North(O,C2)
A-REFUTE: Suppose North(C2,O) for contradiction.
P5-Y1: [P5] => y(O) > y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) > y(O)
refutation-C1: [under assumption refutation; P5-Y1, refutation-Y1] => contradiction
The assumption North(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P1: [p1] => North(O,C3)
A-REFUTE: Suppose North(C3,O) for contradiction.
P1-Y1: [P1] => y(O) > y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption North(C3,O) leads to contradiction, so this direction is impossible.
No eligible object can qualify, so there is no unique qualifying object.
Answer: E
```

## chain-0-base-which-m0-all-possible
Split: train; case: no-match; answer: F

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: which North O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: Select the complete set of eligible objects that could be north of O.
A. C1
B. C0
C. C3
D. C2
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
C0: impossible
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
C1: impossible
P3: From p3, O is exactly north of C1.
A-REFUTE: Suppose C1 is exactly north of O for contradiction.
P3-Y1: From P3, O's y-coordinate is greater than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly north of O leads to contradiction, so this direction is impossible.
C2: impossible
P5: From p5, O is exactly north of C2.
A-REFUTE: Suppose C2 is exactly north of O for contradiction.
P5-Y1: From P5, O's y-coordinate is greater than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P5-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly north of O leads to contradiction, so this direction is impossible.
C3: impossible
P1: From p1, O is exactly north of C3.
A-REFUTE: Suppose C3 is exactly north of O for contradiction.
P1-Y1: From P1, O's y-coordinate is greater than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly north of O leads to contradiction, so this direction is impossible.
The complete possible set is {}. The empty set uses F.
Answer: F
```
### Symbolic
```text
C0: impossible
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
C1: impossible
P3: [p3] => North(O,C1)
A-REFUTE: Suppose North(C1,O) for contradiction.
P3-Y1: [P3] => y(O) > y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) > y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption North(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P5: [p5] => North(O,C2)
A-REFUTE: Suppose North(C2,O) for contradiction.
P5-Y1: [P5] => y(O) > y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) > y(O)
refutation-C1: [under assumption refutation; P5-Y1, refutation-Y1] => contradiction
The assumption North(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P1: [p1] => North(O,C3)
A-REFUTE: Suppose North(C3,O) for contradiction.
P1-Y1: [P1] => y(O) > y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption North(C3,O) leads to contradiction, so this direction is impossible.
The complete possible set is {}. The empty set uses F.
Answer: F
```

## chain-0-base-which-m0-visible-possible
Split: train; case: no-match; answer: F

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: which North O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: Which listed eligible objects could be north of O? Select all that apply.
A. C1
B. C0
C. C3
D. C2
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: impossible
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
C1: impossible
P3: From p3, O is exactly north of C1.
A-REFUTE: Suppose C1 is exactly north of O for contradiction.
P3-Y1: From P3, O's y-coordinate is greater than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly north of O leads to contradiction, so this direction is impossible.
C2: impossible
P5: From p5, O is exactly north of C2.
A-REFUTE: Suppose C2 is exactly north of O for contradiction.
P5-Y1: From P5, O's y-coordinate is greater than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P5-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly north of O leads to contradiction, so this direction is impossible.
C3: impossible
P1: From p1, O is exactly north of C3.
A-REFUTE: Suppose C3 is exactly north of O for contradiction.
P1-Y1: From P1, O's y-coordinate is greater than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly north of O leads to contradiction, so this direction is impossible.
The listed possible set is {}. The empty set uses F.
Answer: F
```
### Symbolic
```text
C0: impossible
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
C1: impossible
P3: [p3] => North(O,C1)
A-REFUTE: Suppose North(C1,O) for contradiction.
P3-Y1: [P3] => y(O) > y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) > y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption North(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P5: [p5] => North(O,C2)
A-REFUTE: Suppose North(C2,O) for contradiction.
P5-Y1: [P5] => y(O) > y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) > y(O)
refutation-C1: [under assumption refutation; P5-Y1, refutation-Y1] => contradiction
The assumption North(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P1: [p1] => North(O,C3)
A-REFUTE: Suppose North(C3,O) for contradiction.
P1-Y1: [P1] => y(O) > y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption North(C3,O) leads to contradiction, so this direction is impossible.
The listed possible set is {}. The empty set uses F.
Answer: F
```

## chain-0-base-count-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: count North O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: What exact number of eligible objects are north of O in every valid map?
A. 0
B. 4
C. 3
D. 2
E. Cannot be determined
F. None of the above
```
### Natural
```text
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C0 is impossible.
P3: From p3, O is exactly north of C1.
A-REFUTE: Suppose C1 is exactly north of O for contradiction.
P3-Y1: From P3, O's y-coordinate is greater than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P5: From p5, O is exactly north of C2.
A-REFUTE: Suppose C2 is exactly north of O for contradiction.
P5-Y1: From P5, O's y-coordinate is greater than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P5-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P1: From p1, O is exactly north of C3.
A-REFUTE: Suppose C3 is exactly north of O for contradiction.
P1-Y1: From P1, O's y-coordinate is greater than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places O at (0, 3); C0 at (1, 5); C1 at (0, 2); C2 at (0, 1); C3 at (0, 0); Link0 at (0, 4). In this map, count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 0. It is displayed at A.
Answer: A
```
### Symbolic
```text
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Fixed membership: C0 is impossible.
P3: [p3] => North(O,C1)
A-REFUTE: Suppose North(C1,O) for contradiction.
P3-Y1: [P3] => y(O) > y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) > y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption North(C1,O) leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P5: [p5] => North(O,C2)
A-REFUTE: Suppose North(C2,O) for contradiction.
P5-Y1: [P5] => y(O) > y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) > y(O)
refutation-C1: [under assumption refutation; P5-Y1, refutation-Y1] => contradiction
The assumption North(C2,O) leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P1: [p1] => North(O,C3)
A-REFUTE: Suppose North(C3,O) for contradiction.
P1-Y1: [P1] => y(O) > y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption North(C3,O) leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {O:(0,3), C0:(1,5), C1:(0,2), C2:(0,1), C3:(0,0), Link0:(0,4)} => count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 0. It is displayed at A.
Answer: A
```

## chain-0-base-count-m0-all-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: count North O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: Select the complete set of possible counts of eligible objects north of O.
A. 0
B. 4
C. 3
D. 2
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C0 is impossible.
P3: From p3, O is exactly north of C1.
A-REFUTE: Suppose C1 is exactly north of O for contradiction.
P3-Y1: From P3, O's y-coordinate is greater than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P5: From p5, O is exactly north of C2.
A-REFUTE: Suppose C2 is exactly north of O for contradiction.
P5-Y1: From P5, O's y-coordinate is greater than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P5-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P1: From p1, O is exactly north of C3.
A-REFUTE: Suppose C3 is exactly north of O for contradiction.
P1-Y1: From P1, O's y-coordinate is greater than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places O at (0, 3); C0 at (1, 5); C1 at (0, 2); C2 at (0, 1); C3 at (0, 0); Link0 at (0, 4). In this map, count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {0}.
Answer: A
```
### Symbolic
```text
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Fixed membership: C0 is impossible.
P3: [p3] => North(O,C1)
A-REFUTE: Suppose North(C1,O) for contradiction.
P3-Y1: [P3] => y(O) > y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) > y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption North(C1,O) leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P5: [p5] => North(O,C2)
A-REFUTE: Suppose North(C2,O) for contradiction.
P5-Y1: [P5] => y(O) > y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) > y(O)
refutation-C1: [under assumption refutation; P5-Y1, refutation-Y1] => contradiction
The assumption North(C2,O) leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P1: [p1] => North(O,C3)
A-REFUTE: Suppose North(C3,O) for contradiction.
P1-Y1: [P1] => y(O) > y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption North(C3,O) leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {O:(0,3), C0:(1,5), C1:(0,2), C2:(0,1), C3:(0,0), Link0:(0,4)} => count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {0}.
Answer: A
```

## chain-0-base-count-m0-visible-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: count North O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the North of C3.
Link0 is to the North of O.
O is to the North of C1.
Link0 is to the Southwest of C0.
O is to the North of C2.
Question: Which listed counts are possible for the number of eligible objects north of O? Select all that apply.
A. 0
B. 4
C. 3
D. 2
E. Cannot be determined
F. None of the above
```
### Natural
```text
P2: From p2, Link0 is exactly north of O.
P4: From p4, Link0 is exactly southwest of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link0's x-coordinate is equal to O's x-coordinate.
P4-X1: From P4, Link0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P2-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C0 is impossible.
P3: From p3, O is exactly north of C1.
A-REFUTE: Suppose C1 is exactly north of O for contradiction.
P3-Y1: From P3, O's y-coordinate is greater than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P5: From p5, O is exactly north of C2.
A-REFUTE: Suppose C2 is exactly north of O for contradiction.
P5-Y1: From P5, O's y-coordinate is greater than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P5-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P1: From p1, O is exactly north of C3.
A-REFUTE: Suppose C3 is exactly north of O for contradiction.
P1-Y1: From P1, O's y-coordinate is greater than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly north of O leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places O at (0, 3); C0 at (1, 5); C1 at (0, 2); C2 at (0, 1); C3 at (0, 0); Link0 at (0, 4). In this map, count=0.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The listed possible set is {0}.
Answer: A
```
### Symbolic
```text
P2: [p2] => North(Link0,O)
P4: [p4] => Southwest(Link0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link0) = x(O)
P4-X1: [P4] => x(Link0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P2-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Fixed membership: C0 is impossible.
P3: [p3] => North(O,C1)
A-REFUTE: Suppose North(C1,O) for contradiction.
P3-Y1: [P3] => y(O) > y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) > y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption North(C1,O) leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P5: [p5] => North(O,C2)
A-REFUTE: Suppose North(C2,O) for contradiction.
P5-Y1: [P5] => y(O) > y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) > y(O)
refutation-C1: [under assumption refutation; P5-Y1, refutation-Y1] => contradiction
The assumption North(C2,O) leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P1: [p1] => North(O,C3)
A-REFUTE: Suppose North(C3,O) for contradiction.
P1-Y1: [P1] => y(O) > y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption North(C3,O) leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {O:(0,3), C0:(1,5), C1:(0,2), C2:(0,1), C3:(0,0), Link0:(0,4)} => count=0.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The listed possible set is {0}.
Answer: A
```

## chain-1-base-direction-m0-single
Split: test; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Link0, Link1, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
Link1 is to the Southwest of C0.
C2 is to the South of O.
O is to the North of C4.
O is to the North of C1.
Link0 is to the South of Link1.
Extra0A is to the Southeast of Extra0B.
O is to the North of C3.
Link0 is to the North of O.
Question: In which direction is C0 relative to O in every valid map?
A. West
B. Northwest
C. Southwest
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P1-X1: From P1, Link1's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, Link0's x-coordinate is equal to Link1's x-coordinate.
P8-X1: From P8, Link0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P8-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P5-X1, O's x-coordinate is equal to Link1's x-coordinate.
D-X-2: From D-X-1, P1-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
P1-X1: From P1, Link1's x-coordinate is less than C0's x-coordinate.
P1-Y1: From P1, Link1's y-coordinate is less than C0's y-coordinate.
P5-X1: From P5, Link0's x-coordinate is equal to Link1's x-coordinate.
P5-Y1: From P5, Link0's y-coordinate is less than Link1's y-coordinate.
P8-X1: From P8, Link0's x-coordinate is equal to O's x-coordinate.
P8-Y1: From P8, Link0's y-coordinate is greater than O's y-coordinate.
PATH-X-1-INV: From P8-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P5-X1, O's x-coordinate is equal to Link1's x-coordinate.
D-X-2: From D-X-1, P1-X1, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-1-INV: From P8-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P5-Y1, O's y-coordinate is less than Link1's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, O's y-coordinate is less than C0's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, C0 is exactly northeast of O.
East:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P1-Y1: From P1, Link1's y-coordinate is less than C0's y-coordinate.
P5-Y1: From P5, Link0's y-coordinate is less than Link1's y-coordinate.
P8-Y1: From P8, Link0's y-coordinate is greater than O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is equal to O's y-coordinate.
PATH-Y-1-INV: From P8-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P5-Y1, O's y-coordinate is less than Link1's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P1-Y1: From P1, Link1's y-coordinate is less than C0's y-coordinate.
P5-Y1: From P5, Link0's y-coordinate is less than Link1's y-coordinate.
P8-Y1: From P8, Link0's y-coordinate is greater than O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-1-INV: From P8-Y1, O's y-coordinate is less than Link0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P5-Y1, O's y-coordinate is less than Link1's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P1-X1: From P1, Link1's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, Link0's x-coordinate is equal to Link1's x-coordinate.
P8-X1: From P8, Link0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P8-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P5-X1, O's x-coordinate is equal to Link1's x-coordinate.
D-X-2: From D-X-1, P1-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P1-X1: From P1, Link1's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, Link0's x-coordinate is equal to Link1's x-coordinate.
P8-X1: From P8, Link0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P8-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P5-X1, O's x-coordinate is equal to Link1's x-coordinate.
D-X-2: From D-X-1, P1-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P1-X1: From P1, Link1's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, Link0's x-coordinate is equal to Link1's x-coordinate.
P8-X1: From P8, Link0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P8-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P5-X1, O's x-coordinate is equal to Link1's x-coordinate.
D-X-2: From D-X-1, P1-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P1: From p1, Link1 is exactly southwest of C0.
P5: From p5, Link0 is exactly south of Link1.
P8: From p8, Link0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P1-X1: From P1, Link1's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, Link0's x-coordinate is equal to Link1's x-coordinate.
P8-X1: From P8, Link0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P8-X1, O's x-coordinate is equal to Link0's x-coordinate.
D-X-1: From PATH-X-1-INV, P5-X1, O's x-coordinate is equal to Link1's x-coordinate.
D-X-2: From D-X-1, P1-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Northeast. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P1-X1: [P1] => x(Link1) < x(C0)
P5-X1: [P5] => x(Link0) = x(Link1)
P8-X1: [P8] => x(Link0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P8-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P5-X1] => x(O) = x(Link1)
D-X-2: [D-X-1, P1-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
P1-X1: [P1] => x(Link1) < x(C0)
P1-Y1: [P1] => y(Link1) < y(C0)
P5-X1: [P5] => x(Link0) = x(Link1)
P5-Y1: [P5] => y(Link0) < y(Link1)
P8-X1: [P8] => x(Link0) = x(O)
P8-Y1: [P8] => y(Link0) > y(O)
PATH-X-1-INV: [P8-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P5-X1] => x(O) = x(Link1)
D-X-2: [D-X-1, P1-X1] => x(O) < x(C0)
PATH-Y-1-INV: [P8-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P5-Y1] => y(O) < y(Link1)
D-Y-2: [D-Y-1, P1-Y1] => y(O) < y(C0)
Q-DIR: [D-X-2, D-Y-2] => Northeast(C0,O)
East:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
A-REFUTE: Suppose East(C0,O) for contradiction.
P1-Y1: [P1] => y(Link1) < y(C0)
P5-Y1: [P5] => y(Link0) < y(Link1)
P8-Y1: [P8] => y(Link0) > y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) = y(O)
PATH-Y-1-INV: [P8-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P5-Y1] => y(O) < y(Link1)
D-Y-2: [D-Y-1, P1-Y1] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P1-Y1: [P1] => y(Link1) < y(C0)
P5-Y1: [P5] => y(Link0) < y(Link1)
P8-Y1: [P8] => y(Link0) > y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-1-INV: [P8-Y1] => y(O) < y(Link0)
D-Y-1: [PATH-Y-1-INV, P5-Y1] => y(O) < y(Link1)
D-Y-2: [D-Y-1, P1-Y1] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P1-X1: [P1] => x(Link1) < x(C0)
P5-X1: [P5] => x(Link0) = x(Link1)
P8-X1: [P8] => x(Link0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P8-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P5-X1] => x(O) = x(Link1)
D-X-2: [D-X-1, P1-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P1-X1: [P1] => x(Link1) < x(C0)
P5-X1: [P5] => x(Link0) = x(Link1)
P8-X1: [P8] => x(Link0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P8-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P5-X1] => x(O) = x(Link1)
D-X-2: [D-X-1, P1-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
A-REFUTE: Suppose West(C0,O) for contradiction.
P1-X1: [P1] => x(Link1) < x(C0)
P5-X1: [P5] => x(Link0) = x(Link1)
P8-X1: [P8] => x(Link0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P8-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P5-X1] => x(O) = x(Link1)
D-X-2: [D-X-1, P1-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P1: [p1] => Southwest(Link1,C0)
P5: [p5] => South(Link0,Link1)
P8: [p8] => North(Link0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P1-X1: [P1] => x(Link1) < x(C0)
P5-X1: [P5] => x(Link0) = x(Link1)
P8-X1: [P8] => x(Link0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P8-X1] => x(O) = x(Link0)
D-X-1: [PATH-X-1-INV, P5-X1] => x(O) = x(Link1)
D-X-2: [D-X-1, P1-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Northeast. It is displayed at D.
Answer: D
```

## chain-2-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Link0, Link1, Link2, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
O is to the East of C4.
Link2 is to the East of Link1.
O is to the East of C5.
C1 is to the West of O.
O is to the West of Link0.
O is to the East of C3.
Extra1A is to the East of Extra1B.
Extra0A is to the Southwest of Extra0B.
Link0 is to the West of Link1.
O is to the East of C2.
C0 is to the Southeast of Link2.
Question: In which direction is C0 relative to O in every valid map?
A. Northwest
B. Northeast
C. Southeast
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, Link2's x-coordinate is greater than Link1's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0's x-coordinate.
P9-X1: From P9, Link0's x-coordinate is less than Link1's x-coordinate.
P11-X1: From P11, C0's x-coordinate is greater than Link2's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-3-INV: From P2-X1, Link1's x-coordinate is less than Link2's x-coordinate.
PATH-X-4-INV: From P11-X1, Link2's x-coordinate is less than C0's x-coordinate.
D-X-1: From P5-X1, P9-X1, O's x-coordinate is less than Link1's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, O's x-coordinate is less than Link2's x-coordinate.
D-X-3: From D-X-2, PATH-X-4-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-3, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P2-Y1: From P2, Link2's y-coordinate is equal to Link1's y-coordinate.
P5-Y1: From P5, O's y-coordinate is equal to Link0's y-coordinate.
P9-Y1: From P9, Link0's y-coordinate is equal to Link1's y-coordinate.
P11-Y1: From P11, C0's y-coordinate is less than Link2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
PATH-Y-3-INV: From P9-Y1, Link1's y-coordinate is equal to Link0's y-coordinate.
PATH-Y-4-INV: From P5-Y1, Link0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From P11-Y1, P2-Y1, C0's y-coordinate is less than Link1's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is less than Link0's y-coordinate.
D-Y-3: From D-Y-2, PATH-Y-4-INV, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-3, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P2-Y1: From P2, Link2's y-coordinate is equal to Link1's y-coordinate.
P5-Y1: From P5, O's y-coordinate is equal to Link0's y-coordinate.
P9-Y1: From P9, Link0's y-coordinate is equal to Link1's y-coordinate.
P11-Y1: From P11, C0's y-coordinate is less than Link2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is equal to O's y-coordinate.
PATH-Y-3-INV: From P9-Y1, Link1's y-coordinate is equal to Link0's y-coordinate.
PATH-Y-4-INV: From P5-Y1, Link0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From P11-Y1, P2-Y1, C0's y-coordinate is less than Link1's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is less than Link0's y-coordinate.
D-Y-3: From D-Y-2, PATH-Y-4-INV, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-3, refutation-Y1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
P2-X1: From P2, Link2's x-coordinate is greater than Link1's x-coordinate.
P2-Y1: From P2, Link2's y-coordinate is equal to Link1's y-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0's x-coordinate.
P5-Y1: From P5, O's y-coordinate is equal to Link0's y-coordinate.
P9-X1: From P9, Link0's x-coordinate is less than Link1's x-coordinate.
P9-Y1: From P9, Link0's y-coordinate is equal to Link1's y-coordinate.
P11-X1: From P11, C0's x-coordinate is greater than Link2's x-coordinate.
P11-Y1: From P11, C0's y-coordinate is less than Link2's y-coordinate.
PATH-X-3-INV: From P2-X1, Link1's x-coordinate is less than Link2's x-coordinate.
PATH-X-4-INV: From P11-X1, Link2's x-coordinate is less than C0's x-coordinate.
D-X-1: From P5-X1, P9-X1, O's x-coordinate is less than Link1's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, O's x-coordinate is less than Link2's x-coordinate.
D-X-3: From D-X-2, PATH-X-4-INV, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-3-INV: From P9-Y1, Link1's y-coordinate is equal to Link0's y-coordinate.
PATH-Y-4-INV: From P5-Y1, Link0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From P11-Y1, P2-Y1, C0's y-coordinate is less than Link1's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is less than Link0's y-coordinate.
D-Y-3: From D-Y-2, PATH-Y-4-INV, C0's y-coordinate is less than O's y-coordinate.
Q-DIR: From D-X-3, D-Y-3, C0 is exactly southeast of O.
South:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, Link2's x-coordinate is greater than Link1's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0's x-coordinate.
P9-X1: From P9, Link0's x-coordinate is less than Link1's x-coordinate.
P11-X1: From P11, C0's x-coordinate is greater than Link2's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-3-INV: From P2-X1, Link1's x-coordinate is less than Link2's x-coordinate.
PATH-X-4-INV: From P11-X1, Link2's x-coordinate is less than C0's x-coordinate.
D-X-1: From P5-X1, P9-X1, O's x-coordinate is less than Link1's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, O's x-coordinate is less than Link2's x-coordinate.
D-X-3: From D-X-2, PATH-X-4-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-3, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-X1: From P2, Link2's x-coordinate is greater than Link1's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0's x-coordinate.
P9-X1: From P9, Link0's x-coordinate is less than Link1's x-coordinate.
P11-X1: From P11, C0's x-coordinate is greater than Link2's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-3-INV: From P2-X1, Link1's x-coordinate is less than Link2's x-coordinate.
PATH-X-4-INV: From P11-X1, Link2's x-coordinate is less than C0's x-coordinate.
D-X-1: From P5-X1, P9-X1, O's x-coordinate is less than Link1's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, O's x-coordinate is less than Link2's x-coordinate.
D-X-3: From D-X-2, PATH-X-4-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-3, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P2-X1: From P2, Link2's x-coordinate is greater than Link1's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0's x-coordinate.
P9-X1: From P9, Link0's x-coordinate is less than Link1's x-coordinate.
P11-X1: From P11, C0's x-coordinate is greater than Link2's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-3-INV: From P2-X1, Link1's x-coordinate is less than Link2's x-coordinate.
PATH-X-4-INV: From P11-X1, Link2's x-coordinate is less than C0's x-coordinate.
D-X-1: From P5-X1, P9-X1, O's x-coordinate is less than Link1's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, O's x-coordinate is less than Link2's x-coordinate.
D-X-3: From D-X-2, PATH-X-4-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-3, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, Link2 is exactly east of Link1.
P5: From p5, O is exactly west of Link0.
P9: From p9, Link0 is exactly west of Link1.
P11: From p11, C0 is exactly southeast of Link2.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, Link2's x-coordinate is greater than Link1's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0's x-coordinate.
P9-X1: From P9, Link0's x-coordinate is less than Link1's x-coordinate.
P11-X1: From P11, C0's x-coordinate is greater than Link2's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-3-INV: From P2-X1, Link1's x-coordinate is less than Link2's x-coordinate.
PATH-X-4-INV: From P11-X1, Link2's x-coordinate is less than C0's x-coordinate.
D-X-1: From P5-X1, P9-X1, O's x-coordinate is less than Link1's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, O's x-coordinate is less than Link2's x-coordinate.
D-X-3: From D-X-2, PATH-X-4-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-3, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Southeast. It is displayed at C.
Answer: C
```
### Symbolic
```text
North:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(Link2) > x(Link1)
P5-X1: [P5] => x(O) < x(Link0)
P9-X1: [P9] => x(Link0) < x(Link1)
P11-X1: [P11] => x(C0) > x(Link2)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-3-INV: [P2-X1] => x(Link1) < x(Link2)
PATH-X-4-INV: [P11-X1] => x(Link2) < x(C0)
D-X-1: [P5-X1, P9-X1] => x(O) < x(Link1)
D-X-2: [D-X-1, PATH-X-3-INV] => x(O) < x(Link2)
D-X-3: [D-X-2, PATH-X-4-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-3, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P2-Y1: [P2] => y(Link2) = y(Link1)
P5-Y1: [P5] => y(O) = y(Link0)
P9-Y1: [P9] => y(Link0) = y(Link1)
P11-Y1: [P11] => y(C0) < y(Link2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
PATH-Y-3-INV: [P9-Y1] => y(Link1) = y(Link0)
PATH-Y-4-INV: [P5-Y1] => y(Link0) = y(O)
D-Y-1: [P11-Y1, P2-Y1] => y(C0) < y(Link1)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) < y(Link0)
D-Y-3: [D-Y-2, PATH-Y-4-INV] => y(C0) < y(O)
refutation-C1: [under assumption refutation; D-Y-3, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
A-REFUTE: Suppose East(C0,O) for contradiction.
P2-Y1: [P2] => y(Link2) = y(Link1)
P5-Y1: [P5] => y(O) = y(Link0)
P9-Y1: [P9] => y(Link0) = y(Link1)
P11-Y1: [P11] => y(C0) < y(Link2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) = y(O)
PATH-Y-3-INV: [P9-Y1] => y(Link1) = y(Link0)
PATH-Y-4-INV: [P5-Y1] => y(Link0) = y(O)
D-Y-1: [P11-Y1, P2-Y1] => y(C0) < y(Link1)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) < y(Link0)
D-Y-3: [D-Y-2, PATH-Y-4-INV] => y(C0) < y(O)
refutation-C1: [under assumption refutation; D-Y-3, refutation-Y1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
P2-X1: [P2] => x(Link2) > x(Link1)
P2-Y1: [P2] => y(Link2) = y(Link1)
P5-X1: [P5] => x(O) < x(Link0)
P5-Y1: [P5] => y(O) = y(Link0)
P9-X1: [P9] => x(Link0) < x(Link1)
P9-Y1: [P9] => y(Link0) = y(Link1)
P11-X1: [P11] => x(C0) > x(Link2)
P11-Y1: [P11] => y(C0) < y(Link2)
PATH-X-3-INV: [P2-X1] => x(Link1) < x(Link2)
PATH-X-4-INV: [P11-X1] => x(Link2) < x(C0)
D-X-1: [P5-X1, P9-X1] => x(O) < x(Link1)
D-X-2: [D-X-1, PATH-X-3-INV] => x(O) < x(Link2)
D-X-3: [D-X-2, PATH-X-4-INV] => x(O) < x(C0)
PATH-Y-3-INV: [P9-Y1] => y(Link1) = y(Link0)
PATH-Y-4-INV: [P5-Y1] => y(Link0) = y(O)
D-Y-1: [P11-Y1, P2-Y1] => y(C0) < y(Link1)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) < y(Link0)
D-Y-3: [D-Y-2, PATH-Y-4-INV] => y(C0) < y(O)
Q-DIR: [D-X-3, D-Y-3] => Southeast(C0,O)
South:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(Link2) > x(Link1)
P5-X1: [P5] => x(O) < x(Link0)
P9-X1: [P9] => x(Link0) < x(Link1)
P11-X1: [P11] => x(C0) > x(Link2)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-3-INV: [P2-X1] => x(Link1) < x(Link2)
PATH-X-4-INV: [P11-X1] => x(Link2) < x(C0)
D-X-1: [P5-X1, P9-X1] => x(O) < x(Link1)
D-X-2: [D-X-1, PATH-X-3-INV] => x(O) < x(Link2)
D-X-3: [D-X-2, PATH-X-4-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-3, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-X1: [P2] => x(Link2) > x(Link1)
P5-X1: [P5] => x(O) < x(Link0)
P9-X1: [P9] => x(Link0) < x(Link1)
P11-X1: [P11] => x(C0) > x(Link2)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-3-INV: [P2-X1] => x(Link1) < x(Link2)
PATH-X-4-INV: [P11-X1] => x(Link2) < x(C0)
D-X-1: [P5-X1, P9-X1] => x(O) < x(Link1)
D-X-2: [D-X-1, PATH-X-3-INV] => x(O) < x(Link2)
D-X-3: [D-X-2, PATH-X-4-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-3, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
A-REFUTE: Suppose West(C0,O) for contradiction.
P2-X1: [P2] => x(Link2) > x(Link1)
P5-X1: [P5] => x(O) < x(Link0)
P9-X1: [P9] => x(Link0) < x(Link1)
P11-X1: [P11] => x(C0) > x(Link2)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-3-INV: [P2-X1] => x(Link1) < x(Link2)
PATH-X-4-INV: [P11-X1] => x(Link2) < x(C0)
D-X-1: [P5-X1, P9-X1] => x(O) < x(Link1)
D-X-2: [D-X-1, PATH-X-3-INV] => x(O) < x(Link2)
D-X-3: [D-X-2, PATH-X-4-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-3, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => East(Link2,Link1)
P5: [p5] => West(O,Link0)
P9: [p9] => West(Link0,Link1)
P11: [p11] => Southeast(C0,Link2)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(Link2) > x(Link1)
P5-X1: [P5] => x(O) < x(Link0)
P9-X1: [P9] => x(Link0) < x(Link1)
P11-X1: [P11] => x(C0) > x(Link2)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-3-INV: [P2-X1] => x(Link1) < x(Link2)
PATH-X-4-INV: [P11-X1] => x(Link2) < x(C0)
D-X-1: [P5-X1, P9-X1] => x(O) < x(Link1)
D-X-2: [D-X-1, PATH-X-3-INV] => x(O) < x(Link2)
D-X-3: [D-X-2, PATH-X-4-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-3, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Southeast. It is displayed at C.
Answer: C
```

## chain-3-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Link0
Query: direction C0 O
Consider a map with multiple objects:
C3 is to the West of O.
O is to the East of C2.
Link0 is to the East of O.
O is to the East of C1.
C0 is to the Southeast of Link0.
Question: In which direction is C0 relative to O in every valid map?
A. West
B. Southeast
C. Northeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P3-X1: From P3, Link0's x-coordinate is greater than O's x-coordinate.
P5-X1: From P5, C0's x-coordinate is greater than Link0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P3-X1, O's x-coordinate is less than Link0's x-coordinate.
PATH-X-2-INV: From P5-X1, Link0's x-coordinate is less than C0's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P3-Y1: From P3, Link0's y-coordinate is equal to O's y-coordinate.
P5-Y1: From P5, C0's y-coordinate is less than Link0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
D-Y-1: From P5-Y1, P3-Y1, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P3-Y1: From P3, Link0's y-coordinate is equal to O's y-coordinate.
P5-Y1: From P5, C0's y-coordinate is less than Link0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From P5-Y1, P3-Y1, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
P3-X1: From P3, Link0's x-coordinate is greater than O's x-coordinate.
P3-Y1: From P3, Link0's y-coordinate is equal to O's y-coordinate.
P5-X1: From P5, C0's x-coordinate is greater than Link0's x-coordinate.
P5-Y1: From P5, C0's y-coordinate is less than Link0's y-coordinate.
PATH-X-1-INV: From P3-X1, O's x-coordinate is less than Link0's x-coordinate.
PATH-X-2-INV: From P5-X1, Link0's x-coordinate is less than C0's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, O's x-coordinate is less than C0's x-coordinate.
D-Y-1: From P5-Y1, P3-Y1, C0's y-coordinate is less than O's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly southeast of O.
South:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P3-X1: From P3, Link0's x-coordinate is greater than O's x-coordinate.
P5-X1: From P5, C0's x-coordinate is greater than Link0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-1-INV: From P3-X1, O's x-coordinate is less than Link0's x-coordinate.
PATH-X-2-INV: From P5-X1, Link0's x-coordinate is less than C0's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P3-X1: From P3, Link0's x-coordinate is greater than O's x-coordinate.
P5-X1: From P5, C0's x-coordinate is greater than Link0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P3-X1, O's x-coordinate is less than Link0's x-coordinate.
PATH-X-2-INV: From P5-X1, Link0's x-coordinate is less than C0's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P3-X1: From P3, Link0's x-coordinate is greater than O's x-coordinate.
P5-X1: From P5, C0's x-coordinate is greater than Link0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P3-X1, O's x-coordinate is less than Link0's x-coordinate.
PATH-X-2-INV: From P5-X1, Link0's x-coordinate is less than C0's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P3: From p3, Link0 is exactly east of O.
P5: From p5, C0 is exactly southeast of Link0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P3-X1: From P3, Link0's x-coordinate is greater than O's x-coordinate.
P5-X1: From P5, C0's x-coordinate is greater than Link0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-1-INV: From P3-X1, O's x-coordinate is less than Link0's x-coordinate.
PATH-X-2-INV: From P5-X1, Link0's x-coordinate is less than C0's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Southeast. It is displayed at B.
Answer: B
```
### Symbolic
```text
North:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P3-X1: [P3] => x(Link0) > x(O)
P5-X1: [P5] => x(C0) > x(Link0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P3-X1] => x(O) < x(Link0)
PATH-X-2-INV: [P5-X1] => x(Link0) < x(C0)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P3-Y1: [P3] => y(Link0) = y(O)
P5-Y1: [P5] => y(C0) < y(Link0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
D-Y-1: [P5-Y1, P3-Y1] => y(C0) < y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P3-Y1: [P3] => y(Link0) = y(O)
P5-Y1: [P5] => y(C0) < y(Link0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) = y(O)
D-Y-1: [P5-Y1, P3-Y1] => y(C0) < y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
P3-X1: [P3] => x(Link0) > x(O)
P3-Y1: [P3] => y(Link0) = y(O)
P5-X1: [P5] => x(C0) > x(Link0)
P5-Y1: [P5] => y(C0) < y(Link0)
PATH-X-1-INV: [P3-X1] => x(O) < x(Link0)
PATH-X-2-INV: [P5-X1] => x(Link0) < x(C0)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(O) < x(C0)
D-Y-1: [P5-Y1, P3-Y1] => y(C0) < y(O)
Q-DIR: [D-X-1, D-Y-1] => Southeast(C0,O)
South:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P3-X1: [P3] => x(Link0) > x(O)
P5-X1: [P5] => x(C0) > x(Link0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-1-INV: [P3-X1] => x(O) < x(Link0)
PATH-X-2-INV: [P5-X1] => x(Link0) < x(C0)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P3-X1: [P3] => x(Link0) > x(O)
P5-X1: [P5] => x(C0) > x(Link0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P3-X1] => x(O) < x(Link0)
PATH-X-2-INV: [P5-X1] => x(Link0) < x(C0)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P3-X1: [P3] => x(Link0) > x(O)
P5-X1: [P5] => x(C0) > x(Link0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P3-X1] => x(O) < x(Link0)
PATH-X-2-INV: [P5-X1] => x(Link0) < x(C0)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P3: [p3] => East(Link0,O)
P5: [p5] => Southeast(C0,Link0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P3-X1: [P3] => x(Link0) > x(O)
P5-X1: [P5] => x(C0) > x(Link0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-1-INV: [P3-X1] => x(O) < x(Link0)
PATH-X-2-INV: [P5-X1] => x(Link0) < x(C0)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Southeast. It is displayed at B.
Answer: B
```

## chain-4-base-direction-m0-single
Split: train; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Link0, Link1, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
C4 is to the North of O.
Extra0A is to the South of Extra0B.
C3 is to the North of O.
O is to the North of Link0.
Link1 is to the South of Link0.
C2 is to the North of O.
C0 is to the Southwest of Link1.
C1 is to the North of O.
Question: In which direction is C0 relative to O in every valid map?
A. West
B. Northeast
C. Southeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P4-X1: From P4, O's x-coordinate is equal to Link0's x-coordinate.
P5-X1: From P5, Link1's x-coordinate is equal to Link0's x-coordinate.
P7-X1: From P7, C0's x-coordinate is less than Link1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-3-INV: From P4-X1, Link0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, P5-X1, C0's x-coordinate is less than Link0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P4-X1: From P4, O's x-coordinate is equal to Link0's x-coordinate.
P5-X1: From P5, Link1's x-coordinate is equal to Link0's x-coordinate.
P7-X1: From P7, C0's x-coordinate is less than Link1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-3-INV: From P4-X1, Link0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, P5-X1, C0's x-coordinate is less than Link0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P4-X1: From P4, O's x-coordinate is equal to Link0's x-coordinate.
P5-X1: From P5, Link1's x-coordinate is equal to Link0's x-coordinate.
P7-X1: From P7, C0's x-coordinate is less than Link1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-3-INV: From P4-X1, Link0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, P5-X1, C0's x-coordinate is less than Link0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P4-X1: From P4, O's x-coordinate is equal to Link0's x-coordinate.
P5-X1: From P5, Link1's x-coordinate is equal to Link0's x-coordinate.
P7-X1: From P7, C0's x-coordinate is less than Link1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-3-INV: From P4-X1, Link0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, P5-X1, C0's x-coordinate is less than Link0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P4-X1: From P4, O's x-coordinate is equal to Link0's x-coordinate.
P5-X1: From P5, Link1's x-coordinate is equal to Link0's x-coordinate.
P7-X1: From P7, C0's x-coordinate is less than Link1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-3-INV: From P4-X1, Link0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, P5-X1, C0's x-coordinate is less than Link0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
P4-X1: From P4, O's x-coordinate is equal to Link0's x-coordinate.
P4-Y1: From P4, O's y-coordinate is greater than Link0's y-coordinate.
P5-X1: From P5, Link1's x-coordinate is equal to Link0's x-coordinate.
P5-Y1: From P5, Link1's y-coordinate is less than Link0's y-coordinate.
P7-X1: From P7, C0's x-coordinate is less than Link1's x-coordinate.
P7-Y1: From P7, C0's y-coordinate is less than Link1's y-coordinate.
PATH-X-3-INV: From P4-X1, Link0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, P5-X1, C0's x-coordinate is less than Link0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is less than O's x-coordinate.
PATH-Y-3-INV: From P4-Y1, Link0's y-coordinate is less than O's y-coordinate.
D-Y-1: From P7-Y1, P5-Y1, C0's y-coordinate is less than Link0's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is less than O's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, C0 is exactly southwest of O.
West:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P4-Y1: From P4, O's y-coordinate is greater than Link0's y-coordinate.
P5-Y1: From P5, Link1's y-coordinate is less than Link0's y-coordinate.
P7-Y1: From P7, C0's y-coordinate is less than Link1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is equal to O's y-coordinate.
PATH-Y-3-INV: From P4-Y1, Link0's y-coordinate is less than O's y-coordinate.
D-Y-1: From P7-Y1, P5-Y1, C0's y-coordinate is less than Link0's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P4: From p4, O is exactly north of Link0.
P5: From p5, Link1 is exactly south of Link0.
P7: From p7, C0 is exactly southwest of Link1.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P4-Y1: From P4, O's y-coordinate is greater than Link0's y-coordinate.
P5-Y1: From P5, Link1's y-coordinate is less than Link0's y-coordinate.
P7-Y1: From P7, C0's y-coordinate is less than Link1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
PATH-Y-3-INV: From P4-Y1, Link0's y-coordinate is less than O's y-coordinate.
D-Y-1: From P7-Y1, P5-Y1, C0's y-coordinate is less than Link0's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Southwest. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
A-REFUTE: Suppose North(C0,O) for contradiction.
P4-X1: [P4] => x(O) = x(Link0)
P5-X1: [P5] => x(Link1) = x(Link0)
P7-X1: [P7] => x(C0) < x(Link1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-3-INV: [P4-X1] => x(Link0) = x(O)
D-X-1: [P7-X1, P5-X1] => x(C0) < x(Link0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P4-X1: [P4] => x(O) = x(Link0)
P5-X1: [P5] => x(Link1) = x(Link0)
P7-X1: [P7] => x(C0) < x(Link1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-3-INV: [P4-X1] => x(Link0) = x(O)
D-X-1: [P7-X1, P5-X1] => x(C0) < x(Link0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
A-REFUTE: Suppose East(C0,O) for contradiction.
P4-X1: [P4] => x(O) = x(Link0)
P5-X1: [P5] => x(Link1) = x(Link0)
P7-X1: [P7] => x(C0) < x(Link1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-3-INV: [P4-X1] => x(Link0) = x(O)
D-X-1: [P7-X1, P5-X1] => x(C0) < x(Link0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P4-X1: [P4] => x(O) = x(Link0)
P5-X1: [P5] => x(Link1) = x(Link0)
P7-X1: [P7] => x(C0) < x(Link1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-3-INV: [P4-X1] => x(Link0) = x(O)
D-X-1: [P7-X1, P5-X1] => x(C0) < x(Link0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
A-REFUTE: Suppose South(C0,O) for contradiction.
P4-X1: [P4] => x(O) = x(Link0)
P5-X1: [P5] => x(Link1) = x(Link0)
P7-X1: [P7] => x(C0) < x(Link1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-3-INV: [P4-X1] => x(Link0) = x(O)
D-X-1: [P7-X1, P5-X1] => x(C0) < x(Link0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
P4-X1: [P4] => x(O) = x(Link0)
P4-Y1: [P4] => y(O) > y(Link0)
P5-X1: [P5] => x(Link1) = x(Link0)
P5-Y1: [P5] => y(Link1) < y(Link0)
P7-X1: [P7] => x(C0) < x(Link1)
P7-Y1: [P7] => y(C0) < y(Link1)
PATH-X-3-INV: [P4-X1] => x(Link0) = x(O)
D-X-1: [P7-X1, P5-X1] => x(C0) < x(Link0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) < x(O)
PATH-Y-3-INV: [P4-Y1] => y(Link0) < y(O)
D-Y-1: [P7-Y1, P5-Y1] => y(C0) < y(Link0)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) < y(O)
Q-DIR: [D-X-2, D-Y-2] => Southwest(C0,O)
West:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
A-REFUTE: Suppose West(C0,O) for contradiction.
P4-Y1: [P4] => y(O) > y(Link0)
P5-Y1: [P5] => y(Link1) < y(Link0)
P7-Y1: [P7] => y(C0) < y(Link1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) = y(O)
PATH-Y-3-INV: [P4-Y1] => y(Link0) < y(O)
D-Y-1: [P7-Y1, P5-Y1] => y(C0) < y(Link0)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) < y(O)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P4: [p4] => North(O,Link0)
P5: [p5] => South(Link1,Link0)
P7: [p7] => Southwest(C0,Link1)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P4-Y1: [P4] => y(O) > y(Link0)
P5-Y1: [P5] => y(Link1) < y(Link0)
P7-Y1: [P7] => y(C0) < y(Link1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
PATH-Y-3-INV: [P4-Y1] => y(Link0) < y(O)
D-Y-1: [P7-Y1, P5-Y1] => y(C0) < y(Link0)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) < y(O)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Southwest. It is displayed at D.
Answer: D
```

## fork-0-base-direction-m0-single
Split: dev; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4
Query: direction C0 C1
Consider a map with multiple objects:
O is to the South of C2.
O is to the South of C1.
O is to the South of C3.
O is to the South of C4.
C0 is to the North of O.
Question: In which direction is C0 relative to C1 in every valid map?
A. Northeast
B. North
C. South
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 0); C0 at (0, 5); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); C4 at (0, 1). In this map, direction=North.
South:
A valid map places O at (0, 0); C0 at (0, 4); C1 at (0, 5); C2 at (0, 3); C3 at (0, 2); C4 at (0, 1). In this map, direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(0,0), C0:(0,5), C1:(0,4), C2:(0,3), C3:(0,2), C4:(0,1)} => direction=North.
South:
Valid map {O:(0,0), C0:(0,4), C1:(0,5), C2:(0,3), C3:(0,2), C4:(0,1)} => direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## fork-0-base-direction-m0-all-possible
Split: dev; case: underdetermined; answer: BC

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4
Query: direction C0 C1
Consider a map with multiple objects:
O is to the South of C2.
O is to the South of C1.
O is to the South of C3.
O is to the South of C4.
C0 is to the North of O.
Question: Select the complete set of directions C0 could occupy relative to C1.
A. Northeast
B. North
C. South
D. Northwest
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
North:
A valid map places O at (0, 0); C0 at (0, 5); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); C4 at (0, 1). In this map, direction=North.
Northeast:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly northeast of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of C1 leads to contradiction, so this direction is impossible.
East:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly east of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly east of C1 leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly southeast of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of C1 leads to contradiction, so this direction is impossible.
South:
A valid map places O at (0, 0); C0 at (0, 4); C1 at (0, 5); C2 at (0, 3); C3 at (0, 2); C4 at (0, 1). In this map, direction=South.
Southwest:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly southwest of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of C1 leads to contradiction, so this direction is impossible.
West:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly west of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly west of C1 leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly northwest of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of C1 leads to contradiction, so this direction is impossible.
The complete possible set is {North, South}.
Answer: BC
```
### Symbolic
```text
North:
Valid map {O:(0,0), C0:(0,5), C1:(0,4), C2:(0,3), C3:(0,2), C4:(0,1)} => direction=North.
Northeast:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose Northeast(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northeast(C0,C1) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose East(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption East(C0,C1) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose Southeast(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southeast(C0,C1) leads to contradiction, so this direction is impossible.
South:
Valid map {O:(0,0), C0:(0,4), C1:(0,5), C2:(0,3), C3:(0,2), C4:(0,1)} => direction=South.
Southwest:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose Southwest(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southwest(C0,C1) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose West(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption West(C0,C1) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose Northwest(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,C1) leads to contradiction, so this direction is impossible.
The complete possible set is {North, South}.
Answer: BC
```

## fork-0-base-direction-m0-visible-possible
Split: dev; case: underdetermined; answer: BC

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4
Query: direction C0 C1
Consider a map with multiple objects:
O is to the South of C2.
O is to the South of C1.
O is to the South of C3.
O is to the South of C4.
C0 is to the North of O.
Question: Which listed directions could C0 occupy relative to C1? Select all that apply.
A. Northeast
B. North
C. South
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 0); C0 at (0, 5); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); C4 at (0, 1). In this map, direction=North.
Northeast:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly northeast of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of C1 leads to contradiction, so this direction is impossible.
South:
A valid map places O at (0, 0); C0 at (0, 4); C1 at (0, 5); C2 at (0, 3); C3 at (0, 2); C4 at (0, 1). In this map, direction=South.
Northwest:
P2: From p2, O is exactly south of C1.
P5: From p5, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly northwest of C1 for contradiction.
P2-X1: From P2, O's x-coordinate is equal to C1's x-coordinate.
P5-X1: From P5, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than C1's x-coordinate.
D-X-1: From P5-X1, P2-X1, C0's x-coordinate is equal to C1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of C1 leads to contradiction, so this direction is impossible.
The listed possible set is {North, South}.
Answer: BC
```
### Symbolic
```text
North:
Valid map {O:(0,0), C0:(0,5), C1:(0,4), C2:(0,3), C3:(0,2), C4:(0,1)} => direction=North.
Northeast:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose Northeast(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northeast(C0,C1) leads to contradiction, so this direction is impossible.
South:
Valid map {O:(0,0), C0:(0,4), C1:(0,5), C2:(0,3), C3:(0,2), C4:(0,1)} => direction=South.
Northwest:
P2: [p2] => South(O,C1)
P5: [p5] => North(C0,O)
A-REFUTE: Suppose Northwest(C0,C1) for contradiction.
P2-X1: [P2] => x(O) = x(C1)
P5-X1: [P5] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(C1)
D-X-1: [P5-X1, P2-X1] => x(C0) = x(C1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,C1) leads to contradiction, so this direction is impossible.
The listed possible set is {North, South}.
Answer: BC
```

## fork-1-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Extra0A, Extra0B
Query: direction C0 C1
Consider a map with multiple objects:
Extra0A is to the Northwest of Extra0B.
O is to the North of C5.
C0 is to the South of O.
O is to the North of C3.
O is to the North of C1.
C2 is to the South of O.
C4 is to the South of O.
Question: In which direction is C0 relative to C1 in every valid map?
A. North
B. Northwest
C. Southwest
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 8); C0 at (0, 7); C1 at (0, 6); C2 at (0, 5); C3 at (0, 4); C4 at (0, 3); C5 at (0, 2); Extra0A at (0, 1); Extra0B at (1, 0). In this map, direction=North.
South:
A valid map places O at (0, 8); C0 at (0, 6); C1 at (0, 7); C2 at (0, 5); C3 at (0, 4); C4 at (0, 3); C5 at (0, 2); Extra0A at (0, 1); Extra0B at (1, 0). In this map, direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(0,8), C0:(0,7), C1:(0,6), C2:(0,5), C3:(0,4), C4:(0,3), C5:(0,2), Extra0A:(0,1), Extra0B:(1,0)} => direction=North.
South:
Valid map {O:(0,8), C0:(0,6), C1:(0,7), C2:(0,5), C3:(0,4), C4:(0,3), C5:(0,2), Extra0A:(0,1), Extra0B:(1,0)} => direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## fork-2-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, C6, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 C1
Consider a map with multiple objects:
C4 is to the East of O.
C5 is to the East of O.
O is to the West of C6.
C0 is to the East of O.
Extra0A is to the South of Extra0B.
Extra1A is to the East of Extra1B.
C2 is to the East of O.
O is to the West of C1.
C3 is to the East of O.
Question: In which direction is C0 relative to C1 in every valid map?
A. East
B. Southwest
C. West
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
A valid map places O at (1, 3); C0 at (8, 3); C1 at (7, 3); C2 at (6, 3); C3 at (5, 3); C4 at (4, 3); C5 at (3, 3); C6 at (2, 3); Extra0A at (1, 1); Extra0B at (1, 2); Extra1A at (1, 0); Extra1B at (0, 0). In this map, direction=East.
West:
A valid map places O at (1, 3); C0 at (7, 3); C1 at (8, 3); C2 at (6, 3); C3 at (5, 3); C4 at (4, 3); C5 at (3, 3); C6 at (2, 3); Extra0A at (1, 1); Extra0B at (1, 2); Extra1A at (1, 0); Extra1B at (0, 0). In this map, direction=West.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
East:
Valid map {O:(1,3), C0:(8,3), C1:(7,3), C2:(6,3), C3:(5,3), C4:(4,3), C5:(3,3), C6:(2,3), Extra0A:(1,1), Extra0B:(1,2), Extra1A:(1,0), Extra1B:(0,0)} => direction=East.
West:
Valid map {O:(1,3), C0:(7,3), C1:(8,3), C2:(6,3), C3:(5,3), C4:(4,3), C5:(3,3), C6:(2,3), Extra0A:(1,1), Extra0B:(1,2), Extra1A:(1,0), Extra1B:(0,0)} => direction=West.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## fork-3-base-direction-m0-single
Split: dev; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4
Query: direction C0 C1
Consider a map with multiple objects:
C1 is to the East of O.
O is to the West of C0.
C4 is to the East of O.
C2 is to the East of O.
C3 is to the East of O.
Question: In which direction is C0 relative to C1 in every valid map?
A. West
B. East
C. Southwest
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
A valid map places O at (0, 0); C0 at (5, 0); C1 at (4, 0); C2 at (3, 0); C3 at (2, 0); C4 at (1, 0). In this map, direction=East.
West:
A valid map places O at (0, 0); C0 at (4, 0); C1 at (5, 0); C2 at (3, 0); C3 at (2, 0); C4 at (1, 0). In this map, direction=West.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
East:
Valid map {O:(0,0), C0:(5,0), C1:(4,0), C2:(3,0), C3:(2,0), C4:(1,0)} => direction=East.
West:
Valid map {O:(0,0), C0:(4,0), C1:(5,0), C2:(3,0), C3:(2,0), C4:(1,0)} => direction=West.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## fork-4-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Extra0A, Extra0B
Query: direction C0 C1
Consider a map with multiple objects:
C3 is to the South of O.
O is to the North of C2.
C4 is to the South of O.
C5 is to the South of O.
Extra0A is to the Northwest of Extra0B.
O is to the North of C0.
C1 is to the South of O.
Question: In which direction is C0 relative to C1 in every valid map?
A. North
B. East
C. South
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 8); C0 at (0, 7); C1 at (0, 6); C2 at (0, 5); C3 at (0, 4); C4 at (0, 3); C5 at (0, 2); Extra0A at (0, 1); Extra0B at (1, 0). In this map, direction=North.
South:
A valid map places O at (0, 8); C0 at (0, 6); C1 at (0, 7); C2 at (0, 5); C3 at (0, 4); C4 at (0, 3); C5 at (0, 2); Extra0A at (0, 1); Extra0B at (1, 0). In this map, direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(0,8), C0:(0,7), C1:(0,6), C2:(0,5), C3:(0,4), C4:(0,3), C5:(0,2), Extra0A:(0,1), Extra0B:(1,0)} => direction=North.
South:
Valid map {O:(0,8), C0:(0,6), C1:(0,7), C2:(0,5), C3:(0,4), C4:(0,3), C5:(0,2), Extra0A:(0,1), Extra0B:(1,0)} => direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## empty-0-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
C1 is to the South of O.
O is to the West of C0.
C2 is to the North of O.
C3 is to the East of O.
Question: In which direction is C0 relative to O in every valid map?
A. Northeast
B. Northwest
C. East
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, O is exactly west of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, O is exactly west of C0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P2: From p2, O is exactly west of C0.
P2-X1: From P2, O's x-coordinate is less than C0's x-coordinate.
P2-Y1: From P2, O's y-coordinate is equal to C0's y-coordinate.
Q-DIR: From P2-X1, P2-Y1, C0 is exactly east of O.
Southeast:
P2: From p2, O is exactly west of C0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, O is exactly west of C0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, O is exactly west of C0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, O is exactly west of C0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, O is exactly west of C0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```
### Symbolic
```text
North:
P2: [p2] => West(O,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => West(O,C0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => West(O,C0)
P2-X1: [P2] => x(O) < x(C0)
P2-Y1: [P2] => y(O) = y(C0)
Q-DIR: [P2-X1, P2-Y1] => East(C0,O)
Southeast:
P2: [p2] => West(O,C0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => West(O,C0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => West(O,C0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => West(O,C0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => West(O,C0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```

## empty-1-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
Extra0A is to the Southwest of Extra0B.
C4 is to the South of O.
C1 is to the North of O.
C0 is to the East of O.
O is to the South of C3.
C2 is to the East of O.
Question: In which direction is C0 relative to O in every valid map?
A. North
B. East
C. Southwest
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P4: From p4, C0 is exactly east of O.
Q-DIR: From P4, C0 is exactly east of O.
Southeast:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at B.
Answer: B
```
### Symbolic
```text
North:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P4: [p4] => East(C0,O)
Q-DIR: [P4] => East(C0,O)
Southeast:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose West(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at B.
Answer: B
```

## empty-2-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
Extra1A is to the Southeast of Extra1B.
O is to the West of C5.
O is to the North of C1.
C0 is to the West of O.
C3 is to the East of O.
Extra0A is to the East of Extra0B.
O is to the West of C2.
O is to the West of C4.
Question: In which direction is C0 relative to O in every valid map?
A. Northwest
B. West
C. South
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P4: From p4, C0 is exactly west of O.
Q-DIR: From P4, C0 is exactly west of O.
Northwest:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
North:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose East(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P4: [p4] => West(C0,O)
Q-DIR: [P4] => West(C0,O)
Northwest:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```

## empty-3-base-direction-m0-single
Split: test; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
O is to the East of C3.
C0 is to the West of O.
C1 is to the South of O.
C2 is to the South of O.
Question: In which direction is C0 relative to O in every valid map?
A. Northwest
B. Northeast
C. Southwest
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P2-X1: From P2, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P2: From p2, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P2-X1: From P2, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-X1: From P2, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-Y1: From P2, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, C0 is exactly west of O.
Q-DIR: From P2, C0 is exactly west of O.
Northwest:
P2: From p2, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-Y1: From P2, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P2: [p2] => West(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => West(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P2-X1: [P2] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => West(C0,O)
A-REFUTE: Suppose East(C0,O) for contradiction.
P2-X1: [P2] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => West(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-X1: [P2] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => West(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => West(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-Y1: [P2] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => West(C0,O)
Q-DIR: [P2] => West(C0,O)
Northwest:
P2: [p2] => West(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-Y1: [P2] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at D.
Answer: D
```

## empty-4-base-direction-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
O is to the North of C2.
C4 is to the West of O.
Extra0A is to the Southeast of Extra0B.
C0 is to the North of O.
O is to the East of C3.
O is to the South of C1.
Question: In which direction is C0 relative to O in every valid map?
A. North
B. West
C. Northwest
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P4: From p4, C0 is exactly north of O.
Q-DIR: From P4, C0 is exactly north of O.
Northeast:
P4: From p4, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P4-X1: From P4, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P4: From p4, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P4-X1: From P4, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P4: From p4, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P4-X1: From P4, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P4: From p4, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is greater than O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P4: From p4, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P4-X1: From P4, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P4: From p4, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P4-X1: From P4, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P4: From p4, C0 is exactly north of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P4-X1: From P4, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is North. It is displayed at A.
Answer: A
```
### Symbolic
```text
North:
P4: [p4] => North(C0,O)
Q-DIR: [P4] => North(C0,O)
Northeast:
P4: [p4] => North(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P4-X1: [P4] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P4: [p4] => North(C0,O)
A-REFUTE: Suppose East(C0,O) for contradiction.
P4-X1: [P4] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P4: [p4] => North(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P4-X1: [P4] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P4: [p4] => North(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) > y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P4: [p4] => North(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P4-X1: [P4] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P4: [p4] => North(C0,O)
A-REFUTE: Suppose West(C0,O) for contradiction.
P4-X1: [P4] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P4: [p4] => North(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P4-X1: [P4] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is North. It is displayed at A.
Answer: A
```

## singleton-0-base-direction-m0-single
Split: dev; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
O is to the East of C0.
O is to the West of C1.
C3 is to the East of O.
C2 is to the North of O.
Question: In which direction is C0 relative to O in every valid map?
A. Northwest
B. West
C. Southeast
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P1: From p1, O is exactly east of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P1: From p1, O is exactly east of C0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P1: From p1, O is exactly east of C0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P1: From p1, O is exactly east of C0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P1: From p1, O is exactly east of C0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P1: From p1, O is exactly east of C0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P1-Y1: From P1, O's y-coordinate is equal to C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P1: From p1, O is exactly east of C0.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
P1-Y1: From P1, O's y-coordinate is equal to C0's y-coordinate.
Q-DIR: From P1-X1, P1-Y1, C0 is exactly west of O.
Northwest:
P1: From p1, O is exactly east of C0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P1-Y1: From P1, O's y-coordinate is equal to C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
North:
P1: [p1] => East(O,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P1-X1: [P1] => x(O) > x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P1: [p1] => East(O,C0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P1-X1: [P1] => x(O) > x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P1: [p1] => East(O,C0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P1-X1: [P1] => x(O) > x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P1: [p1] => East(O,C0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P1-X1: [P1] => x(O) > x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P1: [p1] => East(O,C0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P1-X1: [P1] => x(O) > x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P1: [p1] => East(O,C0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P1-Y1: [P1] => y(O) = y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P1: [p1] => East(O,C0)
P1-X1: [P1] => x(O) > x(C0)
P1-Y1: [P1] => y(O) = y(C0)
Q-DIR: [P1-X1, P1-Y1] => West(C0,O)
Northwest:
P1: [p1] => East(O,C0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P1-Y1: [P1] => y(O) = y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```

## singleton-0-base-which-m0-single
Split: dev; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: which West O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the East of C0.
O is to the West of C1.
C3 is to the East of O.
C2 is to the North of O.
Question: Which eligible object is the only one west of O in every valid map?
A. C2
B. C1
C. C0
D. C3
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: entailed
P1: From p1, O is exactly east of C0.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
P1-Y1: From P1, O's y-coordinate is equal to C0's y-coordinate.
Q-DIR: From P1-X1, P1-Y1, C0 is exactly west of O.
C1: impossible
P2: From p2, O is exactly west of C1.
A-REFUTE: Suppose C1 is exactly west of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C1's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C1 is exactly west of O leads to contradiction, so this direction is impossible.
C2: impossible
P4: From p4, C2 is exactly north of O.
A-REFUTE: Suppose C2 is exactly west of O for contradiction.
P4-X1: From P4, C2's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C2's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C2 is exactly west of O leads to contradiction, so this direction is impossible.
C3: impossible
P3: From p3, C3 is exactly east of O.
A-REFUTE: Suppose C3 is exactly west of O for contradiction.
P3-X1: From P3, C3's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C3's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C3 is exactly west of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is C0. It is displayed at C.
Answer: C
```
### Symbolic
```text
C0: entailed
P1: [p1] => East(O,C0)
P1-X1: [P1] => x(O) > x(C0)
P1-Y1: [P1] => y(O) = y(C0)
Q-DIR: [P1-X1, P1-Y1] => West(C0,O)
C1: impossible
P2: [p2] => West(O,C1)
A-REFUTE: Suppose West(C1,O) for contradiction.
P2-X1: [P2] => x(O) < x(C1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C1) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption West(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P4: [p4] => North(C2,O)
A-REFUTE: Suppose West(C2,O) for contradiction.
P4-X1: [P4] => x(C2) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C2) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P3: [p3] => East(C3,O)
A-REFUTE: Suppose West(C3,O) for contradiction.
P3-X1: [P3] => x(C3) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C3) < x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption West(C3,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is C0. It is displayed at C.
Answer: C
```

## singleton-0-base-which-m0-all-possible
Split: dev; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: which West O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the East of C0.
O is to the West of C1.
C3 is to the East of O.
C2 is to the North of O.
Question: Select the complete set of eligible objects that could be west of O.
A. C2
B. C1
C. C0
D. C3
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
C0: entailed
P1: From p1, O is exactly east of C0.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
P1-Y1: From P1, O's y-coordinate is equal to C0's y-coordinate.
Q-DIR: From P1-X1, P1-Y1, C0 is exactly west of O.
C1: impossible
P2: From p2, O is exactly west of C1.
A-REFUTE: Suppose C1 is exactly west of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C1's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C1 is exactly west of O leads to contradiction, so this direction is impossible.
C2: impossible
P4: From p4, C2 is exactly north of O.
A-REFUTE: Suppose C2 is exactly west of O for contradiction.
P4-X1: From P4, C2's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C2's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C2 is exactly west of O leads to contradiction, so this direction is impossible.
C3: impossible
P3: From p3, C3 is exactly east of O.
A-REFUTE: Suppose C3 is exactly west of O for contradiction.
P3-X1: From P3, C3's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C3's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C3 is exactly west of O leads to contradiction, so this direction is impossible.
The complete possible set is {C0}.
Answer: C
```
### Symbolic
```text
C0: entailed
P1: [p1] => East(O,C0)
P1-X1: [P1] => x(O) > x(C0)
P1-Y1: [P1] => y(O) = y(C0)
Q-DIR: [P1-X1, P1-Y1] => West(C0,O)
C1: impossible
P2: [p2] => West(O,C1)
A-REFUTE: Suppose West(C1,O) for contradiction.
P2-X1: [P2] => x(O) < x(C1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C1) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption West(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P4: [p4] => North(C2,O)
A-REFUTE: Suppose West(C2,O) for contradiction.
P4-X1: [P4] => x(C2) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C2) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P3: [p3] => East(C3,O)
A-REFUTE: Suppose West(C3,O) for contradiction.
P3-X1: [P3] => x(C3) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C3) < x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption West(C3,O) leads to contradiction, so this direction is impossible.
The complete possible set is {C0}.
Answer: C
```

## singleton-0-base-which-m0-visible-possible
Split: dev; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: which West O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
O is to the East of C0.
O is to the West of C1.
C3 is to the East of O.
C2 is to the North of O.
Question: Which listed eligible objects could be west of O? Select all that apply.
A. C2
B. C1
C. C0
D. C3
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: entailed
P1: From p1, O is exactly east of C0.
P1-X1: From P1, O's x-coordinate is greater than C0's x-coordinate.
P1-Y1: From P1, O's y-coordinate is equal to C0's y-coordinate.
Q-DIR: From P1-X1, P1-Y1, C0 is exactly west of O.
C1: impossible
P2: From p2, O is exactly west of C1.
A-REFUTE: Suppose C1 is exactly west of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than C1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C1's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C1 is exactly west of O leads to contradiction, so this direction is impossible.
C2: impossible
P4: From p4, C2 is exactly north of O.
A-REFUTE: Suppose C2 is exactly west of O for contradiction.
P4-X1: From P4, C2's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C2's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C2 is exactly west of O leads to contradiction, so this direction is impossible.
C3: impossible
P3: From p3, C3 is exactly east of O.
A-REFUTE: Suppose C3 is exactly west of O for contradiction.
P3-X1: From P3, C3's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C3's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C3 is exactly west of O leads to contradiction, so this direction is impossible.
The listed possible set is {C0}.
Answer: C
```
### Symbolic
```text
C0: entailed
P1: [p1] => East(O,C0)
P1-X1: [P1] => x(O) > x(C0)
P1-Y1: [P1] => y(O) = y(C0)
Q-DIR: [P1-X1, P1-Y1] => West(C0,O)
C1: impossible
P2: [p2] => West(O,C1)
A-REFUTE: Suppose West(C1,O) for contradiction.
P2-X1: [P2] => x(O) < x(C1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C1) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption West(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P4: [p4] => North(C2,O)
A-REFUTE: Suppose West(C2,O) for contradiction.
P4-X1: [P4] => x(C2) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C2) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P3: [p3] => East(C3,O)
A-REFUTE: Suppose West(C3,O) for contradiction.
P3-X1: [P3] => x(C3) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C3) < x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption West(C3,O) leads to contradiction, so this direction is impossible.
The listed possible set is {C0}.
Answer: C
```

## singleton-1-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Link0X0, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
O is to the North of Link0X0.
C0 is to the South of Link0X0.
C2 is to the West of O.
O is to the East of C4.
Extra0A is to the Southeast of Extra0B.
O is to the South of C3.
C1 is to the North of O.
Question: In which direction is C0 relative to O in every valid map?
A. North
B. West
C. South
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P1-Y1: From P1, O's y-coordinate is greater than Link0X0's y-coordinate.
P2-Y1: From P2, C0's y-coordinate is less than Link0X0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
PATH-Y-2-INV: From P1-Y1, Link0X0's y-coordinate is less than O's y-coordinate.
D-Y-1: From P2-Y1, PATH-Y-2-INV, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P1-X1: From P1, O's x-coordinate is equal to Link0X0's x-coordinate.
P2-X1: From P2, C0's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P1-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, PATH-X-2-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P1-X1: From P1, O's x-coordinate is equal to Link0X0's x-coordinate.
P2-X1: From P2, C0's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P1-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, PATH-X-2-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P1-X1: From P1, O's x-coordinate is equal to Link0X0's x-coordinate.
P2-X1: From P2, C0's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P1-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, PATH-X-2-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
P1-X1: From P1, O's x-coordinate is equal to Link0X0's x-coordinate.
P1-Y1: From P1, O's y-coordinate is greater than Link0X0's y-coordinate.
P2-X1: From P2, C0's x-coordinate is equal to Link0X0's x-coordinate.
P2-Y1: From P2, C0's y-coordinate is less than Link0X0's y-coordinate.
PATH-X-2-INV: From P1-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, PATH-X-2-INV, C0's x-coordinate is equal to O's x-coordinate.
PATH-Y-2-INV: From P1-Y1, Link0X0's y-coordinate is less than O's y-coordinate.
D-Y-1: From P2-Y1, PATH-Y-2-INV, C0's y-coordinate is less than O's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly south of O.
Southwest:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P1-X1: From P1, O's x-coordinate is equal to Link0X0's x-coordinate.
P2-X1: From P2, C0's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-2-INV: From P1-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, PATH-X-2-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P1-X1: From P1, O's x-coordinate is equal to Link0X0's x-coordinate.
P2-X1: From P2, C0's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-2-INV: From P1-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, PATH-X-2-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P1: From p1, O is exactly north of Link0X0.
P2: From p2, C0 is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P1-X1: From P1, O's x-coordinate is equal to Link0X0's x-coordinate.
P2-X1: From P2, C0's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-2-INV: From P1-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, PATH-X-2-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is South. It is displayed at C.
Answer: C
```
### Symbolic
```text
North:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P1-Y1: [P1] => y(O) > y(Link0X0)
P2-Y1: [P2] => y(C0) < y(Link0X0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
PATH-Y-2-INV: [P1-Y1] => y(Link0X0) < y(O)
D-Y-1: [P2-Y1, PATH-Y-2-INV] => y(C0) < y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P1-X1: [P1] => x(O) = x(Link0X0)
P2-X1: [P2] => x(C0) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P1-X1] => x(Link0X0) = x(O)
D-X-1: [P2-X1, PATH-X-2-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P1-X1: [P1] => x(O) = x(Link0X0)
P2-X1: [P2] => x(C0) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P1-X1] => x(Link0X0) = x(O)
D-X-1: [P2-X1, PATH-X-2-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P1-X1: [P1] => x(O) = x(Link0X0)
P2-X1: [P2] => x(C0) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P1-X1] => x(Link0X0) = x(O)
D-X-1: [P2-X1, PATH-X-2-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
P1-X1: [P1] => x(O) = x(Link0X0)
P1-Y1: [P1] => y(O) > y(Link0X0)
P2-X1: [P2] => x(C0) = x(Link0X0)
P2-Y1: [P2] => y(C0) < y(Link0X0)
PATH-X-2-INV: [P1-X1] => x(Link0X0) = x(O)
D-X-1: [P2-X1, PATH-X-2-INV] => x(C0) = x(O)
PATH-Y-2-INV: [P1-Y1] => y(Link0X0) < y(O)
D-Y-1: [P2-Y1, PATH-Y-2-INV] => y(C0) < y(O)
Q-DIR: [D-X-1, D-Y-1] => South(C0,O)
Southwest:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P1-X1: [P1] => x(O) = x(Link0X0)
P2-X1: [P2] => x(C0) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-2-INV: [P1-X1] => x(Link0X0) = x(O)
D-X-1: [P2-X1, PATH-X-2-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P1-X1: [P1] => x(O) = x(Link0X0)
P2-X1: [P2] => x(C0) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-2-INV: [P1-X1] => x(Link0X0) = x(O)
D-X-1: [P2-X1, PATH-X-2-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P1: [p1] => North(O,Link0X0)
P2: [p2] => South(C0,Link0X0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P1-X1: [P1] => x(O) = x(Link0X0)
P2-X1: [P2] => x(C0) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-2-INV: [P1-X1] => x(Link0X0) = x(O)
D-X-1: [P2-X1, PATH-X-2-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is South. It is displayed at C.
Answer: C
```

## singleton-2-base-direction-m0-single
Split: test; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Link0X0, Link0X1, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
Extra1A is to the Southeast of Extra1B.
O is to the West of Link0X0.
Extra0A is to the South of Extra0B.
O is to the North of C1.
C3 is to the North of O.
O is to the East of C5.
Link0X1 is to the West of C0.
Link0X0 is to the West of Link0X1.
O is to the East of C4.
C2 is to the West of O.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. West
C. North
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X1's x-coordinate is less than C0's x-coordinate.
P8-X1: From P8, Link0X0's x-coordinate is less than Link0X1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, P8-X1, O's x-coordinate is less than Link0X1's x-coordinate.
D-X-2: From D-X-1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P7-Y1: From P7, Link0X1's y-coordinate is equal to C0's y-coordinate.
P8-Y1: From P8, Link0X0's y-coordinate is equal to Link0X1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
PATH-Y-1-INV: From P7-Y1, C0's y-coordinate is equal to Link0X1's y-coordinate.
PATH-Y-2-INV: From P8-Y1, Link0X1's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-3-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to Link0X0's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P7-X1: From P7, Link0X1's x-coordinate is less than C0's x-coordinate.
P7-Y1: From P7, Link0X1's y-coordinate is equal to C0's y-coordinate.
P8-X1: From P8, Link0X0's x-coordinate is less than Link0X1's x-coordinate.
P8-Y1: From P8, Link0X0's y-coordinate is equal to Link0X1's y-coordinate.
D-X-1: From P2-X1, P8-X1, O's x-coordinate is less than Link0X1's x-coordinate.
D-X-2: From D-X-1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-1-INV: From P7-Y1, C0's y-coordinate is equal to Link0X1's y-coordinate.
PATH-Y-2-INV: From P8-Y1, Link0X1's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-3-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to Link0X0's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is equal to O's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, C0 is exactly east of O.
Southeast:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P7-Y1: From P7, Link0X1's y-coordinate is equal to C0's y-coordinate.
P8-Y1: From P8, Link0X0's y-coordinate is equal to Link0X1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-1-INV: From P7-Y1, C0's y-coordinate is equal to Link0X1's y-coordinate.
PATH-Y-2-INV: From P8-Y1, Link0X1's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-3-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to Link0X0's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X1's x-coordinate is less than C0's x-coordinate.
P8-X1: From P8, Link0X0's x-coordinate is less than Link0X1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, P8-X1, O's x-coordinate is less than Link0X1's x-coordinate.
D-X-2: From D-X-1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X1's x-coordinate is less than C0's x-coordinate.
P8-X1: From P8, Link0X0's x-coordinate is less than Link0X1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P2-X1, P8-X1, O's x-coordinate is less than Link0X1's x-coordinate.
D-X-2: From D-X-1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X1's x-coordinate is less than C0's x-coordinate.
P8-X1: From P8, Link0X0's x-coordinate is less than Link0X1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P2-X1, P8-X1, O's x-coordinate is less than Link0X1's x-coordinate.
D-X-2: From D-X-1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X1 is exactly west of C0.
P8: From p8, Link0X0 is exactly west of Link0X1.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X1's x-coordinate is less than C0's x-coordinate.
P8-X1: From P8, Link0X0's x-coordinate is less than Link0X1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P2-X1, P8-X1, O's x-coordinate is less than Link0X1's x-coordinate.
D-X-2: From D-X-1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X1) < x(C0)
P8-X1: [P8] => x(Link0X0) < x(Link0X1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
D-X-1: [P2-X1, P8-X1] => x(O) < x(Link0X1)
D-X-2: [D-X-1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(Link0X0)
P7-Y1: [P7] => y(Link0X1) = y(C0)
P8-Y1: [P8] => y(Link0X0) = y(Link0X1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
PATH-Y-1-INV: [P7-Y1] => y(C0) = y(Link0X1)
PATH-Y-2-INV: [P8-Y1] => y(Link0X1) = y(Link0X0)
PATH-Y-3-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(Link0X0)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
P2-X1: [P2] => x(O) < x(Link0X0)
P2-Y1: [P2] => y(O) = y(Link0X0)
P7-X1: [P7] => x(Link0X1) < x(C0)
P7-Y1: [P7] => y(Link0X1) = y(C0)
P8-X1: [P8] => x(Link0X0) < x(Link0X1)
P8-Y1: [P8] => y(Link0X0) = y(Link0X1)
D-X-1: [P2-X1, P8-X1] => x(O) < x(Link0X1)
D-X-2: [D-X-1, P7-X1] => x(O) < x(C0)
PATH-Y-1-INV: [P7-Y1] => y(C0) = y(Link0X1)
PATH-Y-2-INV: [P8-Y1] => y(Link0X1) = y(Link0X0)
PATH-Y-3-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(Link0X0)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) = y(O)
Q-DIR: [D-X-2, D-Y-2] => East(C0,O)
Southeast:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(Link0X0)
P7-Y1: [P7] => y(Link0X1) = y(C0)
P8-Y1: [P8] => y(Link0X0) = y(Link0X1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-1-INV: [P7-Y1] => y(C0) = y(Link0X1)
PATH-Y-2-INV: [P8-Y1] => y(Link0X1) = y(Link0X0)
PATH-Y-3-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(Link0X0)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X1) < x(C0)
P8-X1: [P8] => x(Link0X0) < x(Link0X1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
D-X-1: [P2-X1, P8-X1] => x(O) < x(Link0X1)
D-X-2: [D-X-1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X1) < x(C0)
P8-X1: [P8] => x(Link0X0) < x(Link0X1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P2-X1, P8-X1] => x(O) < x(Link0X1)
D-X-2: [D-X-1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
A-REFUTE: Suppose West(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X1) < x(C0)
P8-X1: [P8] => x(Link0X0) < x(Link0X1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P2-X1, P8-X1] => x(O) < x(Link0X1)
D-X-2: [D-X-1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X1,C0)
P8: [p8] => West(Link0X0,Link0X1)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X1) < x(C0)
P8-X1: [P8] => x(Link0X0) < x(Link0X1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P2-X1, P8-X1] => x(O) < x(Link0X1)
D-X-2: [D-X-1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at D.
Answer: D
```

## singleton-3-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
O is to the South of C1.
C0 is to the East of O.
O is to the East of C2.
O is to the South of C3.
Question: In which direction is C0 relative to O in every valid map?
A. Northeast
B. Northwest
C. East
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P2-Y1: From P2, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P2: From p2, C0 is exactly east of O.
Q-DIR: From P2, C0 is exactly east of O.
Southeast:
P2: From p2, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-Y1: From P2, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-X1: From P2, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P2-X1: From P2, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```
### Symbolic
```text
North:
P2: [p2] => East(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => East(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P2-Y1: [P2] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => East(C0,O)
Q-DIR: [P2] => East(C0,O)
Southeast:
P2: [p2] => East(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-Y1: [P2] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => East(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => East(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-X1: [P2] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => East(C0,O)
A-REFUTE: Suppose West(C0,O) for contradiction.
P2-X1: [P2] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => East(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```

## singleton-4-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Link0X0, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
C4 is to the South of O.
Extra0A is to the East of Extra0B.
C3 is to the West of O.
Link0X0 is to the West of C0.
O is to the West of Link0X0.
C1 is to the West of O.
C2 is to the West of O.
Question: In which direction is C0 relative to O in every valid map?
A. Southwest
B. East
C. Northwest
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P4-X1: From P4, Link0X0's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P5-X1, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P4-Y1: From P4, Link0X0's y-coordinate is equal to C0's y-coordinate.
P5-Y1: From P5, O's y-coordinate is equal to Link0X0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
PATH-Y-1-INV: From P4-Y1, C0's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-2-INV: From P5-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
P4-X1: From P4, Link0X0's x-coordinate is less than C0's x-coordinate.
P4-Y1: From P4, Link0X0's y-coordinate is equal to C0's y-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0X0's x-coordinate.
P5-Y1: From P5, O's y-coordinate is equal to Link0X0's y-coordinate.
D-X-1: From P5-X1, P4-X1, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-1-INV: From P4-Y1, C0's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-2-INV: From P5-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly east of O.
Southeast:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P4-Y1: From P4, Link0X0's y-coordinate is equal to C0's y-coordinate.
P5-Y1: From P5, O's y-coordinate is equal to Link0X0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-1-INV: From P4-Y1, C0's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-2-INV: From P5-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P4-X1: From P4, Link0X0's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P5-X1, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P4-X1: From P4, Link0X0's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P5-X1, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P4-X1: From P4, Link0X0's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P5-X1, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P4: From p4, Link0X0 is exactly west of C0.
P5: From p5, O is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P4-X1: From P4, Link0X0's x-coordinate is less than C0's x-coordinate.
P5-X1: From P5, O's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P5-X1, P4-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at B.
Answer: B
```
### Symbolic
```text
North:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P4-X1: [P4] => x(Link0X0) < x(C0)
P5-X1: [P5] => x(O) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
D-X-1: [P5-X1, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P4-Y1: [P4] => y(Link0X0) = y(C0)
P5-Y1: [P5] => y(O) = y(Link0X0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
PATH-Y-1-INV: [P4-Y1] => y(C0) = y(Link0X0)
PATH-Y-2-INV: [P5-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
P4-X1: [P4] => x(Link0X0) < x(C0)
P4-Y1: [P4] => y(Link0X0) = y(C0)
P5-X1: [P5] => x(O) < x(Link0X0)
P5-Y1: [P5] => y(O) = y(Link0X0)
D-X-1: [P5-X1, P4-X1] => x(O) < x(C0)
PATH-Y-1-INV: [P4-Y1] => y(C0) = y(Link0X0)
PATH-Y-2-INV: [P5-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(O)
Q-DIR: [D-X-1, D-Y-1] => East(C0,O)
Southeast:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P4-Y1: [P4] => y(Link0X0) = y(C0)
P5-Y1: [P5] => y(O) = y(Link0X0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-1-INV: [P4-Y1] => y(C0) = y(Link0X0)
PATH-Y-2-INV: [P5-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P4-X1: [P4] => x(Link0X0) < x(C0)
P5-X1: [P5] => x(O) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
D-X-1: [P5-X1, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P4-X1: [P4] => x(Link0X0) < x(C0)
P5-X1: [P5] => x(O) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P5-X1, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P4-X1: [P4] => x(Link0X0) < x(C0)
P5-X1: [P5] => x(O) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P5-X1, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P4: [p4] => West(Link0X0,C0)
P5: [p5] => West(O,Link0X0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P4-X1: [P4] => x(Link0X0) < x(C0)
P5-X1: [P5] => x(O) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P5-X1, P4-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at B.
Answer: B
```

## multiple-0-base-direction-m0-single
Split: dev; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
C0 is to the East of O.
C2 is to the East of O.
O is to the West of C1.
C3 is to the East of O.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. Southwest
C. East
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P1: From p1, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P1-X1: From P1, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P1: From p1, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P1-Y1: From P1, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P1: From p1, C0 is exactly east of O.
Q-DIR: From P1, C0 is exactly east of O.
Southeast:
P1: From p1, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P1-Y1: From P1, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P1-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P1: From p1, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P1-X1: From P1, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P1: From p1, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P1-X1: From P1, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P1: From p1, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P1-X1: From P1, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P1: From p1, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P1-X1: From P1, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```
### Symbolic
```text
North:
P1: [p1] => East(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P1-X1: [P1] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P1: [p1] => East(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P1-Y1: [P1] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P1: [p1] => East(C0,O)
Q-DIR: [P1] => East(C0,O)
Southeast:
P1: [p1] => East(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P1-Y1: [P1] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P1-Y1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P1: [p1] => East(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P1-X1: [P1] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P1: [p1] => East(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P1-X1: [P1] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P1: [p1] => East(C0,O)
A-REFUTE: Suppose West(C0,O) for contradiction.
P1-X1: [P1] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P1: [p1] => East(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P1-X1: [P1] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```

## multiple-0-base-which-m0-single
Split: dev; case: multiple-certain-match; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: which East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the East of O.
C2 is to the East of O.
O is to the West of C1.
C3 is to the East of O.
Question: Which eligible object is the only one east of O in every valid map?
A. C1
B. C2
C. C3
D. C0
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: entailed
P1: From p1, C0 is exactly east of O.
Q-DIR: From P1, C0 is exactly east of O.
C1: entailed
P3: From p3, O is exactly west of C1.
P3-X1: From P3, O's x-coordinate is less than C1's x-coordinate.
P3-Y1: From P3, O's y-coordinate is equal to C1's y-coordinate.
Q-DIR: From P3-X1, P3-Y1, C1 is exactly east of O.
C2: entailed
P2: From p2, C2 is exactly east of O.
Q-DIR: From P2, C2 is exactly east of O.
C3: entailed
P4: From p4, C3 is exactly east of O.
Q-DIR: From P4, C3 is exactly east of O.
Multiple objects always qualify, so there is no sole qualifying object.
Answer: E
```
### Symbolic
```text
C0: entailed
P1: [p1] => East(C0,O)
Q-DIR: [P1] => East(C0,O)
C1: entailed
P3: [p3] => West(O,C1)
P3-X1: [P3] => x(O) < x(C1)
P3-Y1: [P3] => y(O) = y(C1)
Q-DIR: [P3-X1, P3-Y1] => East(C1,O)
C2: entailed
P2: [p2] => East(C2,O)
Q-DIR: [P2] => East(C2,O)
C3: entailed
P4: [p4] => East(C3,O)
Q-DIR: [P4] => East(C3,O)
Multiple objects always qualify, so there is no sole qualifying object.
Answer: E
```

## multiple-0-base-which-m0-all-possible
Split: dev; case: multiple-certain-match; answer: ABCD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: which East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the East of O.
C2 is to the East of O.
O is to the West of C1.
C3 is to the East of O.
Question: Select the complete set of eligible objects that could be east of O.
A. C1
B. C2
C. C3
D. C0
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
C0: entailed
P1: From p1, C0 is exactly east of O.
Q-DIR: From P1, C0 is exactly east of O.
C1: entailed
P3: From p3, O is exactly west of C1.
P3-X1: From P3, O's x-coordinate is less than C1's x-coordinate.
P3-Y1: From P3, O's y-coordinate is equal to C1's y-coordinate.
Q-DIR: From P3-X1, P3-Y1, C1 is exactly east of O.
C2: entailed
P2: From p2, C2 is exactly east of O.
Q-DIR: From P2, C2 is exactly east of O.
C3: entailed
P4: From p4, C3 is exactly east of O.
Q-DIR: From P4, C3 is exactly east of O.
The complete possible set is {C0, C1, C2, C3}.
Answer: ABCD
```
### Symbolic
```text
C0: entailed
P1: [p1] => East(C0,O)
Q-DIR: [P1] => East(C0,O)
C1: entailed
P3: [p3] => West(O,C1)
P3-X1: [P3] => x(O) < x(C1)
P3-Y1: [P3] => y(O) = y(C1)
Q-DIR: [P3-X1, P3-Y1] => East(C1,O)
C2: entailed
P2: [p2] => East(C2,O)
Q-DIR: [P2] => East(C2,O)
C3: entailed
P4: [p4] => East(C3,O)
Q-DIR: [P4] => East(C3,O)
The complete possible set is {C0, C1, C2, C3}.
Answer: ABCD
```

## multiple-0-base-which-m0-visible-possible
Split: dev; case: multiple-certain-match; answer: ABCD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: which East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the East of O.
C2 is to the East of O.
O is to the West of C1.
C3 is to the East of O.
Question: Which listed eligible objects could be east of O? Select all that apply.
A. C1
B. C2
C. C3
D. C0
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: entailed
P1: From p1, C0 is exactly east of O.
Q-DIR: From P1, C0 is exactly east of O.
C1: entailed
P3: From p3, O is exactly west of C1.
P3-X1: From P3, O's x-coordinate is less than C1's x-coordinate.
P3-Y1: From P3, O's y-coordinate is equal to C1's y-coordinate.
Q-DIR: From P3-X1, P3-Y1, C1 is exactly east of O.
C2: entailed
P2: From p2, C2 is exactly east of O.
Q-DIR: From P2, C2 is exactly east of O.
C3: entailed
P4: From p4, C3 is exactly east of O.
Q-DIR: From P4, C3 is exactly east of O.
The listed possible set is {C1, C2, C3, C0}.
Answer: ABCD
```
### Symbolic
```text
C0: entailed
P1: [p1] => East(C0,O)
Q-DIR: [P1] => East(C0,O)
C1: entailed
P3: [p3] => West(O,C1)
P3-X1: [P3] => x(O) < x(C1)
P3-Y1: [P3] => y(O) = y(C1)
Q-DIR: [P3-X1, P3-Y1] => East(C1,O)
C2: entailed
P2: [p2] => East(C2,O)
Q-DIR: [P2] => East(C2,O)
C3: entailed
P4: [p4] => East(C3,O)
Q-DIR: [P4] => East(C3,O)
The listed possible set is {C1, C2, C3, C0}.
Answer: ABCD
```

## multiple-1-base-direction-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Link0X0, Link1X0, Link2X0, Link3X0, Link4X0, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
C2 is to the East of Link2X0.
O is to the West of Link0X0.
Link2X0 is to the East of O.
Link4X0 is to the East of O.
Extra0A is to the East of Extra0B.
Link3X0 is to the East of O.
Link0X0 is to the West of C0.
C3 is to the East of Link3X0.
Link1X0 is to the East of O.
Link4X0 is to the West of C4.
C1 is to the East of Link1X0.
Question: In which direction is C0 relative to O in every valid map?
A. East
B. Northeast
C. Southeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P7-Y1: From P7, Link0X0's y-coordinate is equal to C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
PATH-Y-1-INV: From P7-Y1, C0's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-2-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P7-X1: From P7, Link0X0's x-coordinate is less than C0's x-coordinate.
P7-Y1: From P7, Link0X0's y-coordinate is equal to C0's y-coordinate.
D-X-1: From P2-X1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
PATH-Y-1-INV: From P7-Y1, C0's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-2-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly east of O.
Southeast:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P7-Y1: From P7, Link0X0's y-coordinate is equal to C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-1-INV: From P7-Y1, C0's y-coordinate is equal to Link0X0's y-coordinate.
PATH-Y-2-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P2-X1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P2-X1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P2-X1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, O is exactly west of Link0X0.
P7: From p7, Link0X0 is exactly west of C0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-X1: From P2, O's x-coordinate is less than Link0X0's x-coordinate.
P7-X1: From P7, Link0X0's x-coordinate is less than C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
D-X-1: From P2-X1, P7-X1, O's x-coordinate is less than C0's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```
### Symbolic
```text
North:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
D-X-1: [P2-X1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(Link0X0)
P7-Y1: [P7] => y(Link0X0) = y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
PATH-Y-1-INV: [P7-Y1] => y(C0) = y(Link0X0)
PATH-Y-2-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
P2-X1: [P2] => x(O) < x(Link0X0)
P2-Y1: [P2] => y(O) = y(Link0X0)
P7-X1: [P7] => x(Link0X0) < x(C0)
P7-Y1: [P7] => y(Link0X0) = y(C0)
D-X-1: [P2-X1, P7-X1] => x(O) < x(C0)
PATH-Y-1-INV: [P7-Y1] => y(C0) = y(Link0X0)
PATH-Y-2-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(O)
Q-DIR: [D-X-1, D-Y-1] => East(C0,O)
Southeast:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(Link0X0)
P7-Y1: [P7] => y(Link0X0) = y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-1-INV: [P7-Y1] => y(C0) = y(Link0X0)
PATH-Y-2-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
D-X-1: [P2-X1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P2-X1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P2-X1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => West(O,Link0X0)
P7: [p7] => West(Link0X0,C0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-X1: [P2] => x(O) < x(Link0X0)
P7-X1: [P7] => x(Link0X0) < x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
D-X-1: [P2-X1, P7-X1] => x(O) < x(C0)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```

## multiple-2-base-direction-m0-single
Split: test; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Link0X0, Link0X1, Link1X0, Link1X1, Link2X0, Link2X1, Link3X0, Link3X1, Link4X0, Link4X1, Link5X0, Link5X1, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
Extra1A is to the Southeast of Extra1B.
Link5X0 is to the North of O.
Link0X0 is to the South of Link0X1.
Link2X0 is to the South of Link2X1.
Link4X1 is to the South of C4.
Extra0A is to the West of Extra0B.
C0 is to the North of Link0X1.
O is to the South of Link4X0.
Link5X0 is to the South of Link5X1.
O is to the South of Link0X0.
Link3X0 is to the North of O.
Link1X1 is to the South of C1.
O is to the South of Link2X0.
C2 is to the North of Link2X1.
Link1X0 is to the North of O.
Link1X0 is to the South of Link1X1.
Link3X0 is to the South of Link3X1.
Link3X1 is to the South of C3.
Link5X1 is to the South of C5.
Link4X1 is to the North of Link4X0.
Question: In which direction is C0 relative to O in every valid map?
A. East
B. Northwest
C. Southeast
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
P3-X1: From P3, Link0X0's x-coordinate is equal to Link0X1's x-coordinate.
P3-Y1: From P3, Link0X0's y-coordinate is less than Link0X1's y-coordinate.
P7-X1: From P7, C0's x-coordinate is equal to Link0X1's x-coordinate.
P7-Y1: From P7, C0's y-coordinate is greater than Link0X1's y-coordinate.
P10-X1: From P10, O's x-coordinate is equal to Link0X0's x-coordinate.
P10-Y1: From P10, O's y-coordinate is less than Link0X0's y-coordinate.
PATH-X-2-INV: From P3-X1, Link0X1's x-coordinate is equal to Link0X0's x-coordinate.
PATH-X-3-INV: From P10-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, C0's x-coordinate is equal to Link0X0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is equal to O's x-coordinate.
PATH-Y-3-INV: From P7-Y1, Link0X1's y-coordinate is less than C0's y-coordinate.
D-Y-1: From P10-Y1, P3-Y1, O's y-coordinate is less than Link0X1's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, O's y-coordinate is less than C0's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, C0 is exactly north of O.
Northeast:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P3-X1: From P3, Link0X0's x-coordinate is equal to Link0X1's x-coordinate.
P7-X1: From P7, C0's x-coordinate is equal to Link0X1's x-coordinate.
P10-X1: From P10, O's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P3-X1, Link0X1's x-coordinate is equal to Link0X0's x-coordinate.
PATH-X-3-INV: From P10-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, C0's x-coordinate is equal to Link0X0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P3-X1: From P3, Link0X0's x-coordinate is equal to Link0X1's x-coordinate.
P7-X1: From P7, C0's x-coordinate is equal to Link0X1's x-coordinate.
P10-X1: From P10, O's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P3-X1, Link0X1's x-coordinate is equal to Link0X0's x-coordinate.
PATH-X-3-INV: From P10-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, C0's x-coordinate is equal to Link0X0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P3-X1: From P3, Link0X0's x-coordinate is equal to Link0X1's x-coordinate.
P7-X1: From P7, C0's x-coordinate is equal to Link0X1's x-coordinate.
P10-X1: From P10, O's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P3-X1, Link0X1's x-coordinate is equal to Link0X0's x-coordinate.
PATH-X-3-INV: From P10-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, C0's x-coordinate is equal to Link0X0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P3-Y1: From P3, Link0X0's y-coordinate is less than Link0X1's y-coordinate.
P7-Y1: From P7, C0's y-coordinate is greater than Link0X1's y-coordinate.
P10-Y1: From P10, O's y-coordinate is less than Link0X0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-3-INV: From P7-Y1, Link0X1's y-coordinate is less than C0's y-coordinate.
D-Y-1: From P10-Y1, P3-Y1, O's y-coordinate is less than Link0X1's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, O's y-coordinate is less than C0's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P3-X1: From P3, Link0X0's x-coordinate is equal to Link0X1's x-coordinate.
P7-X1: From P7, C0's x-coordinate is equal to Link0X1's x-coordinate.
P10-X1: From P10, O's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-2-INV: From P3-X1, Link0X1's x-coordinate is equal to Link0X0's x-coordinate.
PATH-X-3-INV: From P10-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, C0's x-coordinate is equal to Link0X0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P3-X1: From P3, Link0X0's x-coordinate is equal to Link0X1's x-coordinate.
P7-X1: From P7, C0's x-coordinate is equal to Link0X1's x-coordinate.
P10-X1: From P10, O's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-2-INV: From P3-X1, Link0X1's x-coordinate is equal to Link0X0's x-coordinate.
PATH-X-3-INV: From P10-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, C0's x-coordinate is equal to Link0X0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P3: From p3, Link0X0 is exactly south of Link0X1.
P7: From p7, C0 is exactly north of Link0X1.
P10: From p10, O is exactly south of Link0X0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P3-X1: From P3, Link0X0's x-coordinate is equal to Link0X1's x-coordinate.
P7-X1: From P7, C0's x-coordinate is equal to Link0X1's x-coordinate.
P10-X1: From P10, O's x-coordinate is equal to Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
PATH-X-2-INV: From P3-X1, Link0X1's x-coordinate is equal to Link0X0's x-coordinate.
PATH-X-3-INV: From P10-X1, Link0X0's x-coordinate is equal to O's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, C0's x-coordinate is equal to Link0X0's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is North. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
P3-X1: [P3] => x(Link0X0) = x(Link0X1)
P3-Y1: [P3] => y(Link0X0) < y(Link0X1)
P7-X1: [P7] => x(C0) = x(Link0X1)
P7-Y1: [P7] => y(C0) > y(Link0X1)
P10-X1: [P10] => x(O) = x(Link0X0)
P10-Y1: [P10] => y(O) < y(Link0X0)
PATH-X-2-INV: [P3-X1] => x(Link0X1) = x(Link0X0)
PATH-X-3-INV: [P10-X1] => x(Link0X0) = x(O)
D-X-1: [P7-X1, PATH-X-2-INV] => x(C0) = x(Link0X0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) = x(O)
PATH-Y-3-INV: [P7-Y1] => y(Link0X1) < y(C0)
D-Y-1: [P10-Y1, P3-Y1] => y(O) < y(Link0X1)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(O) < y(C0)
Q-DIR: [D-X-2, D-Y-2] => North(C0,O)
Northeast:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P3-X1: [P3] => x(Link0X0) = x(Link0X1)
P7-X1: [P7] => x(C0) = x(Link0X1)
P10-X1: [P10] => x(O) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P3-X1] => x(Link0X1) = x(Link0X0)
PATH-X-3-INV: [P10-X1] => x(Link0X0) = x(O)
D-X-1: [P7-X1, PATH-X-2-INV] => x(C0) = x(Link0X0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P3-X1: [P3] => x(Link0X0) = x(Link0X1)
P7-X1: [P7] => x(C0) = x(Link0X1)
P10-X1: [P10] => x(O) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P3-X1] => x(Link0X1) = x(Link0X0)
PATH-X-3-INV: [P10-X1] => x(Link0X0) = x(O)
D-X-1: [P7-X1, PATH-X-2-INV] => x(C0) = x(Link0X0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P3-X1: [P3] => x(Link0X0) = x(Link0X1)
P7-X1: [P7] => x(C0) = x(Link0X1)
P10-X1: [P10] => x(O) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P3-X1] => x(Link0X1) = x(Link0X0)
PATH-X-3-INV: [P10-X1] => x(Link0X0) = x(O)
D-X-1: [P7-X1, PATH-X-2-INV] => x(C0) = x(Link0X0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P3-Y1: [P3] => y(Link0X0) < y(Link0X1)
P7-Y1: [P7] => y(C0) > y(Link0X1)
P10-Y1: [P10] => y(O) < y(Link0X0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-3-INV: [P7-Y1] => y(Link0X1) < y(C0)
D-Y-1: [P10-Y1, P3-Y1] => y(O) < y(Link0X1)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(O) < y(C0)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P3-X1: [P3] => x(Link0X0) = x(Link0X1)
P7-X1: [P7] => x(C0) = x(Link0X1)
P10-X1: [P10] => x(O) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-2-INV: [P3-X1] => x(Link0X1) = x(Link0X0)
PATH-X-3-INV: [P10-X1] => x(Link0X0) = x(O)
D-X-1: [P7-X1, PATH-X-2-INV] => x(C0) = x(Link0X0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P3-X1: [P3] => x(Link0X0) = x(Link0X1)
P7-X1: [P7] => x(C0) = x(Link0X1)
P10-X1: [P10] => x(O) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-2-INV: [P3-X1] => x(Link0X1) = x(Link0X0)
PATH-X-3-INV: [P10-X1] => x(Link0X0) = x(O)
D-X-1: [P7-X1, PATH-X-2-INV] => x(C0) = x(Link0X0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P3: [p3] => South(Link0X0,Link0X1)
P7: [p7] => North(C0,Link0X1)
P10: [p10] => South(O,Link0X0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P3-X1: [P3] => x(Link0X0) = x(Link0X1)
P7-X1: [P7] => x(C0) = x(Link0X1)
P10-X1: [P10] => x(O) = x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
PATH-X-2-INV: [P3-X1] => x(Link0X1) = x(Link0X0)
PATH-X-3-INV: [P10-X1] => x(Link0X0) = x(O)
D-X-1: [P7-X1, PATH-X-2-INV] => x(C0) = x(Link0X0)
D-X-2: [D-X-1, PATH-X-3-INV] => x(C0) = x(O)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is North. It is displayed at D.
Answer: D
```

## multiple-3-base-direction-m0-single
Split: dev; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
C1 is to the North of O.
O is to the South of C2.
O is to the South of C0.
C3 is to the North of O.
Question: In which direction is C0 relative to O in every valid map?
A. North
B. Northwest
C. South
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P3: From p3, O is exactly south of C0.
P3-X1: From P3, O's x-coordinate is equal to C0's x-coordinate.
P3-Y1: From P3, O's y-coordinate is less than C0's y-coordinate.
Q-DIR: From P3-X1, P3-Y1, C0 is exactly north of O.
Northeast:
P3: From p3, O is exactly south of C0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P3-X1: From P3, O's x-coordinate is equal to C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P3: From p3, O is exactly south of C0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P3-X1: From P3, O's x-coordinate is equal to C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P3: From p3, O is exactly south of C0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P3-X1: From P3, O's x-coordinate is equal to C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P3: From p3, O is exactly south of C0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P3-Y1: From P3, O's y-coordinate is less than C0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P3: From p3, O is exactly south of C0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P3-X1: From P3, O's x-coordinate is equal to C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P3: From p3, O is exactly south of C0.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P3-X1: From P3, O's x-coordinate is equal to C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P3: From p3, O is exactly south of C0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P3-X1: From P3, O's x-coordinate is equal to C0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P3-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is North. It is displayed at A.
Answer: A
```
### Symbolic
```text
North:
P3: [p3] => South(O,C0)
P3-X1: [P3] => x(O) = x(C0)
P3-Y1: [P3] => y(O) < y(C0)
Q-DIR: [P3-X1, P3-Y1] => North(C0,O)
Northeast:
P3: [p3] => South(O,C0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P3-X1: [P3] => x(O) = x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P3: [p3] => South(O,C0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P3-X1: [P3] => x(O) = x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P3: [p3] => South(O,C0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P3-X1: [P3] => x(O) = x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P3: [p3] => South(O,C0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P3-Y1: [P3] => y(O) < y(C0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P3: [p3] => South(O,C0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P3-X1: [P3] => x(O) = x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P3: [p3] => South(O,C0)
A-REFUTE: Suppose West(C0,O) for contradiction.
P3-X1: [P3] => x(O) = x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P3: [p3] => South(O,C0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P3-X1: [P3] => x(O) = x(C0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P3-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is North. It is displayed at A.
Answer: A
```

## multiple-4-base-direction-m0-single
Split: train; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Link0X0, Link1X0, Link2X0, Link3X0, Link4X0, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
C4 is to the West of Link4X0.
O is to the East of Link0X0.
Link4X0 is to the West of O.
Extra0A is to the East of Extra0B.
O is to the East of Link3X0.
O is to the East of Link1X0.
Link2X0 is to the West of O.
Link1X0 is to the East of C1.
C2 is to the West of Link2X0.
Link3X0 is to the East of C3.
C0 is to the West of Link0X0.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. East
C. Southwest
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P2-X1: From P2, O's x-coordinate is greater than Link0X0's x-coordinate.
P11-X1: From P11, C0's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-2-INV: From P2-X1, Link0X0's x-coordinate is less than O's x-coordinate.
D-X-1: From P11-X1, PATH-X-2-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P2-X1: From P2, O's x-coordinate is greater than Link0X0's x-coordinate.
P11-X1: From P11, C0's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P2-X1, Link0X0's x-coordinate is less than O's x-coordinate.
D-X-1: From P11-X1, PATH-X-2-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P2-X1: From P2, O's x-coordinate is greater than Link0X0's x-coordinate.
P11-X1: From P11, C0's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P2-X1, Link0X0's x-coordinate is less than O's x-coordinate.
D-X-1: From P11-X1, PATH-X-2-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P2-X1: From P2, O's x-coordinate is greater than Link0X0's x-coordinate.
P11-X1: From P11, C0's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
PATH-X-2-INV: From P2-X1, Link0X0's x-coordinate is less than O's x-coordinate.
D-X-1: From P11-X1, PATH-X-2-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P2-X1: From P2, O's x-coordinate is greater than Link0X0's x-coordinate.
P11-X1: From P11, C0's x-coordinate is less than Link0X0's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
PATH-X-2-INV: From P2-X1, Link0X0's x-coordinate is less than O's x-coordinate.
D-X-1: From P11-X1, PATH-X-2-INV, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P11-Y1: From P11, C0's y-coordinate is equal to Link0X0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
PATH-Y-2-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From P11-Y1, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
P2-X1: From P2, O's x-coordinate is greater than Link0X0's x-coordinate.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P11-X1: From P11, C0's x-coordinate is less than Link0X0's x-coordinate.
P11-Y1: From P11, C0's y-coordinate is equal to Link0X0's y-coordinate.
PATH-X-2-INV: From P2-X1, Link0X0's x-coordinate is less than O's x-coordinate.
D-X-1: From P11-X1, PATH-X-2-INV, C0's x-coordinate is less than O's x-coordinate.
PATH-Y-2-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From P11-Y1, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, C0 is exactly west of O.
Northwest:
P2: From p2, O is exactly east of Link0X0.
P11: From p11, C0 is exactly west of Link0X0.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P2-Y1: From P2, O's y-coordinate is equal to Link0X0's y-coordinate.
P11-Y1: From P11, C0's y-coordinate is equal to Link0X0's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
PATH-Y-2-INV: From P2-Y1, Link0X0's y-coordinate is equal to O's y-coordinate.
D-Y-1: From P11-Y1, PATH-Y-2-INV, C0's y-coordinate is equal to O's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
A-REFUTE: Suppose North(C0,O) for contradiction.
P2-X1: [P2] => x(O) > x(Link0X0)
P11-X1: [P11] => x(C0) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-2-INV: [P2-X1] => x(Link0X0) < x(O)
D-X-1: [P11-X1, PATH-X-2-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P2-X1: [P2] => x(O) > x(Link0X0)
P11-X1: [P11] => x(C0) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P2-X1] => x(Link0X0) < x(O)
D-X-1: [P11-X1, PATH-X-2-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
A-REFUTE: Suppose East(C0,O) for contradiction.
P2-X1: [P2] => x(O) > x(Link0X0)
P11-X1: [P11] => x(C0) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P2-X1] => x(Link0X0) < x(O)
D-X-1: [P11-X1, PATH-X-2-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P2-X1: [P2] => x(O) > x(Link0X0)
P11-X1: [P11] => x(C0) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
PATH-X-2-INV: [P2-X1] => x(Link0X0) < x(O)
D-X-1: [P11-X1, PATH-X-2-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
A-REFUTE: Suppose South(C0,O) for contradiction.
P2-X1: [P2] => x(O) > x(Link0X0)
P11-X1: [P11] => x(C0) < x(Link0X0)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
PATH-X-2-INV: [P2-X1] => x(Link0X0) < x(O)
D-X-1: [P11-X1, PATH-X-2-INV] => x(C0) < x(O)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(Link0X0)
P11-Y1: [P11] => y(C0) = y(Link0X0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
PATH-Y-2-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [P11-Y1, PATH-Y-2-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
P2-X1: [P2] => x(O) > x(Link0X0)
P2-Y1: [P2] => y(O) = y(Link0X0)
P11-X1: [P11] => x(C0) < x(Link0X0)
P11-Y1: [P11] => y(C0) = y(Link0X0)
PATH-X-2-INV: [P2-X1] => x(Link0X0) < x(O)
D-X-1: [P11-X1, PATH-X-2-INV] => x(C0) < x(O)
PATH-Y-2-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [P11-Y1, PATH-Y-2-INV] => y(C0) = y(O)
Q-DIR: [D-X-1, D-Y-1] => West(C0,O)
Northwest:
P2: [p2] => East(O,Link0X0)
P11: [p11] => West(C0,Link0X0)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P2-Y1: [P2] => y(O) = y(Link0X0)
P11-Y1: [P11] => y(C0) = y(Link0X0)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
PATH-Y-2-INV: [P2-Y1] => y(Link0X0) = y(O)
D-Y-1: [P11-Y1, PATH-Y-2-INV] => y(C0) = y(O)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at D.
Answer: D
```

## contingent-0-base-direction-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: direction C0 O
Consider a map with multiple objects:
C0 is to the Southwest of Anchor.
O is to the South of C3.
O is to the South of C1.
O is to the South of C2.
O is to the Southwest of Anchor.
Question: In which direction is C0 relative to O in every valid map?
A. North
B. Southwest
C. South
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 0); C0 at (0, 4); C1 at (0, 3); C2 at (0, 2); C3 at (0, 1); Anchor at (1, 5). In this map, direction=North.
Northeast:
A valid map places O at (0, 0); C0 at (1, 4); C1 at (0, 3); C2 at (0, 2); C3 at (0, 1); Anchor at (2, 5). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(0,0), C0:(0,4), C1:(0,3), C2:(0,2), C3:(0,1), Anchor:(1,5)} => direction=North.
Northeast:
Valid map {O:(0,0), C0:(1,4), C1:(0,3), C2:(0,2), C3:(0,1), Anchor:(2,5)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## contingent-0-base-which-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: which South O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the Southwest of Anchor.
O is to the South of C3.
O is to the South of C1.
O is to the South of C2.
O is to the Southwest of Anchor.
Question: Which eligible object is the only one south of O in every valid map?
A. C1
B. C0
C. C2
D. C3
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: contingent
A valid map places O at (0, 1); C0 at (0, 0); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); Anchor at (1, 5). In this map, qualifying objects={C0}.
A valid map places O at (1, 1); C0 at (0, 0); C1 at (1, 4); C2 at (1, 3); C3 at (1, 2); Anchor at (2, 5). In this map, qualifying objects={}.
C0 qualifies in the first map but not the second.
C1: impossible
P3: From p3, O is exactly south of C1.
A-REFUTE: Suppose C1 is exactly south of O for contradiction.
P3-Y1: From P3, O's y-coordinate is less than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly south of O leads to contradiction, so this direction is impossible.
C2: impossible
P4: From p4, O is exactly south of C2.
A-REFUTE: Suppose C2 is exactly south of O for contradiction.
P4-Y1: From P4, O's y-coordinate is less than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly south of O leads to contradiction, so this direction is impossible.
C3: impossible
P2: From p2, O is exactly south of C3.
A-REFUTE: Suppose C3 is exactly south of O for contradiction.
P2-Y1: From P2, O's y-coordinate is less than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly south of O leads to contradiction, so this direction is impossible.
The memberships do not establish one object as the only match in every valid map.
Answer: E
```
### Symbolic
```text
C0: contingent
Valid map {O:(0,1), C0:(0,0), C1:(0,4), C2:(0,3), C3:(0,2), Anchor:(1,5)} => qualifying objects={C0}.
Valid map {O:(1,1), C0:(0,0), C1:(1,4), C2:(1,3), C3:(1,2), Anchor:(2,5)} => qualifying objects={}.
C0 qualifies in the first map but not the second.
C1: impossible
P3: [p3] => South(O,C1)
A-REFUTE: Suppose South(C1,O) for contradiction.
P3-Y1: [P3] => y(O) < y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) < y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption South(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P4: [p4] => South(O,C2)
A-REFUTE: Suppose South(C2,O) for contradiction.
P4-Y1: [P4] => y(O) < y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption South(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P2: [p2] => South(O,C3)
A-REFUTE: Suppose South(C3,O) for contradiction.
P2-Y1: [P2] => y(O) < y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption South(C3,O) leads to contradiction, so this direction is impossible.
The memberships do not establish one object as the only match in every valid map.
Answer: E
```

## contingent-0-base-which-m0-all-possible
Split: test; case: underdetermined; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: which South O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the Southwest of Anchor.
O is to the South of C3.
O is to the South of C1.
O is to the South of C2.
O is to the Southwest of Anchor.
Question: Select the complete set of eligible objects that could be south of O.
A. C1
B. C0
C. C2
D. C3
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
C0: contingent
A valid map places O at (0, 1); C0 at (0, 0); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); Anchor at (1, 5). In this map, qualifying objects={C0}.
A valid map places O at (1, 1); C0 at (0, 0); C1 at (1, 4); C2 at (1, 3); C3 at (1, 2); Anchor at (2, 5). In this map, qualifying objects={}.
C0 qualifies in the first map but not the second.
C1: impossible
P3: From p3, O is exactly south of C1.
A-REFUTE: Suppose C1 is exactly south of O for contradiction.
P3-Y1: From P3, O's y-coordinate is less than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly south of O leads to contradiction, so this direction is impossible.
C2: impossible
P4: From p4, O is exactly south of C2.
A-REFUTE: Suppose C2 is exactly south of O for contradiction.
P4-Y1: From P4, O's y-coordinate is less than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly south of O leads to contradiction, so this direction is impossible.
C3: impossible
P2: From p2, O is exactly south of C3.
A-REFUTE: Suppose C3 is exactly south of O for contradiction.
P2-Y1: From P2, O's y-coordinate is less than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly south of O leads to contradiction, so this direction is impossible.
The complete possible set is {C0}.
Answer: B
```
### Symbolic
```text
C0: contingent
Valid map {O:(0,1), C0:(0,0), C1:(0,4), C2:(0,3), C3:(0,2), Anchor:(1,5)} => qualifying objects={C0}.
Valid map {O:(1,1), C0:(0,0), C1:(1,4), C2:(1,3), C3:(1,2), Anchor:(2,5)} => qualifying objects={}.
C0 qualifies in the first map but not the second.
C1: impossible
P3: [p3] => South(O,C1)
A-REFUTE: Suppose South(C1,O) for contradiction.
P3-Y1: [P3] => y(O) < y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) < y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption South(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P4: [p4] => South(O,C2)
A-REFUTE: Suppose South(C2,O) for contradiction.
P4-Y1: [P4] => y(O) < y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption South(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P2: [p2] => South(O,C3)
A-REFUTE: Suppose South(C3,O) for contradiction.
P2-Y1: [P2] => y(O) < y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption South(C3,O) leads to contradiction, so this direction is impossible.
The complete possible set is {C0}.
Answer: B
```

## contingent-0-base-which-m0-visible-possible
Split: test; case: underdetermined; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: which South O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the Southwest of Anchor.
O is to the South of C3.
O is to the South of C1.
O is to the South of C2.
O is to the Southwest of Anchor.
Question: Which listed eligible objects could be south of O? Select all that apply.
A. C1
B. C0
C. C2
D. C3
E. Cannot be determined
F. None of the above
```
### Natural
```text
C0: contingent
A valid map places O at (0, 1); C0 at (0, 0); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); Anchor at (1, 5). In this map, qualifying objects={C0}.
A valid map places O at (1, 1); C0 at (0, 0); C1 at (1, 4); C2 at (1, 3); C3 at (1, 2); Anchor at (2, 5). In this map, qualifying objects={}.
C0 qualifies in the first map but not the second.
C1: impossible
P3: From p3, O is exactly south of C1.
A-REFUTE: Suppose C1 is exactly south of O for contradiction.
P3-Y1: From P3, O's y-coordinate is less than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly south of O leads to contradiction, so this direction is impossible.
C2: impossible
P4: From p4, O is exactly south of C2.
A-REFUTE: Suppose C2 is exactly south of O for contradiction.
P4-Y1: From P4, O's y-coordinate is less than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly south of O leads to contradiction, so this direction is impossible.
C3: impossible
P2: From p2, O is exactly south of C3.
A-REFUTE: Suppose C3 is exactly south of O for contradiction.
P2-Y1: From P2, O's y-coordinate is less than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly south of O leads to contradiction, so this direction is impossible.
The listed possible set is {C0}.
Answer: B
```
### Symbolic
```text
C0: contingent
Valid map {O:(0,1), C0:(0,0), C1:(0,4), C2:(0,3), C3:(0,2), Anchor:(1,5)} => qualifying objects={C0}.
Valid map {O:(1,1), C0:(0,0), C1:(1,4), C2:(1,3), C3:(1,2), Anchor:(2,5)} => qualifying objects={}.
C0 qualifies in the first map but not the second.
C1: impossible
P3: [p3] => South(O,C1)
A-REFUTE: Suppose South(C1,O) for contradiction.
P3-Y1: [P3] => y(O) < y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) < y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption South(C1,O) leads to contradiction, so this direction is impossible.
C2: impossible
P4: [p4] => South(O,C2)
A-REFUTE: Suppose South(C2,O) for contradiction.
P4-Y1: [P4] => y(O) < y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption South(C2,O) leads to contradiction, so this direction is impossible.
C3: impossible
P2: [p2] => South(O,C3)
A-REFUTE: Suppose South(C3,O) for contradiction.
P2-Y1: [P2] => y(O) < y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption South(C3,O) leads to contradiction, so this direction is impossible.
The listed possible set is {C0}.
Answer: B
```

## contingent-0-base-count-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: count South O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the Southwest of Anchor.
O is to the South of C3.
O is to the South of C1.
O is to the South of C2.
O is to the Southwest of Anchor.
Question: What exact number of eligible objects are south of O in every valid map?
A. 3
B. 2
C. 1
D. 0
E. Cannot be determined
F. None of the above
```
### Natural
```text
A valid map places O at (1, 1); C0 at (0, 0); C1 at (1, 4); C2 at (1, 3); C3 at (1, 2); Anchor at (2, 5). In this map, count=0.
A valid map places O at (0, 1); C0 at (0, 0); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); Anchor at (1, 5). In this map, count=1.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
Valid map {O:(1,1), C0:(0,0), C1:(1,4), C2:(1,3), C3:(1,2), Anchor:(2,5)} => count=0.
Valid map {O:(0,1), C0:(0,0), C1:(0,4), C2:(0,3), C3:(0,2), Anchor:(1,5)} => count=1.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## contingent-0-base-count-m0-all-possible
Split: test; case: underdetermined; answer: CD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: count South O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the Southwest of Anchor.
O is to the South of C3.
O is to the South of C1.
O is to the South of C2.
O is to the Southwest of Anchor.
Question: Select the complete set of possible counts of eligible objects south of O.
A. 3
B. 2
C. 1
D. 0
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
P3: From p3, O is exactly south of C1.
A-REFUTE: Suppose C1 is exactly south of O for contradiction.
P3-Y1: From P3, O's y-coordinate is less than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly south of O leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P4: From p4, O is exactly south of C2.
A-REFUTE: Suppose C2 is exactly south of O for contradiction.
P4-Y1: From P4, O's y-coordinate is less than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly south of O leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P2: From p2, O is exactly south of C3.
A-REFUTE: Suppose C3 is exactly south of O for contradiction.
P2-Y1: From P2, O's y-coordinate is less than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly south of O leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places O at (1, 1); C0 at (0, 0); C1 at (1, 4); C2 at (1, 3); C3 at (1, 2); Anchor at (2, 5). In this map, count=0.
A valid map places O at (0, 1); C0 at (0, 0); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); Anchor at (1, 5). In this map, count=1.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {0, 1}.
Answer: CD
```
### Symbolic
```text
P3: [p3] => South(O,C1)
A-REFUTE: Suppose South(C1,O) for contradiction.
P3-Y1: [P3] => y(O) < y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) < y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption South(C1,O) leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P4: [p4] => South(O,C2)
A-REFUTE: Suppose South(C2,O) for contradiction.
P4-Y1: [P4] => y(O) < y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption South(C2,O) leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P2: [p2] => South(O,C3)
A-REFUTE: Suppose South(C3,O) for contradiction.
P2-Y1: [P2] => y(O) < y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption South(C3,O) leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {O:(1,1), C0:(0,0), C1:(1,4), C2:(1,3), C3:(1,2), Anchor:(2,5)} => count=0.
Valid map {O:(0,1), C0:(0,0), C1:(0,4), C2:(0,3), C3:(0,2), Anchor:(1,5)} => count=1.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {0, 1}.
Answer: CD
```

## contingent-0-base-count-m0-visible-possible
Split: test; case: underdetermined; answer: CD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: count South O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
C0 is to the Southwest of Anchor.
O is to the South of C3.
O is to the South of C1.
O is to the South of C2.
O is to the Southwest of Anchor.
Question: Which listed counts are possible for the number of eligible objects south of O? Select all that apply.
A. 3
B. 2
C. 1
D. 0
E. Cannot be determined
F. None of the above
```
### Natural
```text
P3: From p3, O is exactly south of C1.
A-REFUTE: Suppose C1 is exactly south of O for contradiction.
P3-Y1: From P3, O's y-coordinate is less than C1's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C1's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption C1 is exactly south of O leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P4: From p4, O is exactly south of C2.
A-REFUTE: Suppose C2 is exactly south of O for contradiction.
P4-Y1: From P4, O's y-coordinate is less than C2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C2's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C2 is exactly south of O leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P2: From p2, O is exactly south of C3.
A-REFUTE: Suppose C3 is exactly south of O for contradiction.
P2-Y1: From P2, O's y-coordinate is less than C3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C3's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption C3 is exactly south of O leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
A valid map places O at (0, 1); C0 at (0, 0); C1 at (0, 4); C2 at (0, 3); C3 at (0, 2); Anchor at (1, 5). In this map, count=1.
A valid map places O at (1, 1); C0 at (0, 0); C1 at (1, 4); C2 at (1, 3); C3 at (1, 2); Anchor at (2, 5). In this map, count=0.
The listed possible set is {1, 0}.
Answer: CD
```
### Symbolic
```text
P3: [p3] => South(O,C1)
A-REFUTE: Suppose South(C1,O) for contradiction.
P3-Y1: [P3] => y(O) < y(C1)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C1) < y(O)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption South(C1,O) leads to contradiction, so this direction is impossible.
Fixed membership: C1 is impossible.
P4: [p4] => South(O,C2)
A-REFUTE: Suppose South(C2,O) for contradiction.
P4-Y1: [P4] => y(O) < y(C2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C2) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption South(C2,O) leads to contradiction, so this direction is impossible.
Fixed membership: C2 is impossible.
P2: [p2] => South(O,C3)
A-REFUTE: Suppose South(C3,O) for contradiction.
P2-Y1: [P2] => y(O) < y(C3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C3) < y(O)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption South(C3,O) leads to contradiction, so this direction is impossible.
Fixed membership: C3 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Valid map {O:(0,1), C0:(0,0), C1:(0,4), C2:(0,3), C3:(0,2), Anchor:(1,5)} => count=1.
Valid map {O:(1,1), C0:(0,0), C1:(1,4), C2:(1,3), C3:(1,2), Anchor:(2,5)} => count=0.
The listed possible set is {1, 0}.
Answer: CD
```

## contingent-1-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Anchor, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
O is to the East of C4.
Extra0A is to the Northwest of Extra0B.
O is to the East of C2.
C0 is to the Southeast of Anchor.
O is to the East of C3.
C1 is to the West of O.
Anchor is to the Northwest of O.
Question: In which direction is C0 relative to O in every valid map?
A. East
B. Northwest
C. Northeast
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (5, 2); C0 at (5, 3); C1 at (4, 2); C2 at (3, 2); C3 at (2, 2); C4 at (1, 2); Anchor at (0, 4); Extra0A at (5, 1); Extra0B at (6, 0). In this map, direction=North.
Northeast:
A valid map places O at (5, 2); C0 at (6, 3); C1 at (4, 2); C2 at (3, 2); C3 at (2, 2); C4 at (1, 2); Anchor at (0, 4); Extra0A at (5, 1); Extra0B at (6, 0). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(5,2), C0:(5,3), C1:(4,2), C2:(3,2), C3:(2,2), C4:(1,2), Anchor:(0,4), Extra0A:(5,1), Extra0B:(6,0)} => direction=North.
Northeast:
Valid map {O:(5,2), C0:(6,3), C1:(4,2), C2:(3,2), C3:(2,2), C4:(1,2), Anchor:(0,4), Extra0A:(5,1), Extra0B:(6,0)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## contingent-2-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Anchor, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
C5 is to the South of O.
Extra1A is to the South of Extra1B.
C3 is to the South of O.
O is to the North of C2.
O is to the North of C4.
Extra0A is to the Northeast of Extra0B.
O is to the Northeast of Anchor.
C0 is to the Northeast of Anchor.
C1 is to the South of O.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. South
C. Southwest
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (1, 10); C0 at (1, 11); C1 at (1, 9); C2 at (1, 8); C3 at (1, 7); C4 at (1, 6); C5 at (1, 5); Anchor at (0, 1); Extra0A at (1, 4); Extra0B at (0, 0); Extra1A at (1, 2); Extra1B at (1, 3). In this map, direction=North.
Northeast:
A valid map places O at (1, 10); C0 at (2, 11); C1 at (1, 9); C2 at (1, 8); C3 at (1, 7); C4 at (1, 6); C5 at (1, 5); Anchor at (0, 1); Extra0A at (1, 4); Extra0B at (0, 0); Extra1A at (1, 2); Extra1B at (1, 3). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(1,10), C0:(1,11), C1:(1,9), C2:(1,8), C3:(1,7), C4:(1,6), C5:(1,5), Anchor:(0,1), Extra0A:(1,4), Extra0B:(0,0), Extra1A:(1,2), Extra1B:(1,3)} => direction=North.
Northeast:
Valid map {O:(1,10), C0:(2,11), C1:(1,9), C2:(1,8), C3:(1,7), C4:(1,6), C5:(1,5), Anchor:(0,1), Extra0A:(1,4), Extra0B:(0,0), Extra1A:(1,2), Extra1B:(1,3)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## contingent-3-base-direction-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: direction C0 O
Consider a map with multiple objects:
O is to the South of C2.
O is to the Southwest of Anchor.
C0 is to the Southwest of Anchor.
C3 is to the North of O.
O is to the South of C1.
Question: In which direction is C0 relative to O in every valid map?
A. East
B. Southeast
C. Northeast
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 0); C0 at (0, 4); C1 at (0, 3); C2 at (0, 2); C3 at (0, 1); Anchor at (1, 5). In this map, direction=North.
Northeast:
A valid map places O at (0, 0); C0 at (1, 4); C1 at (0, 3); C2 at (0, 2); C3 at (0, 1); Anchor at (2, 5). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(0,0), C0:(0,4), C1:(0,3), C2:(0,2), C3:(0,1), Anchor:(1,5)} => direction=North.
Northeast:
Valid map {O:(0,0), C0:(1,4), C1:(0,3), C2:(0,2), C3:(0,1), Anchor:(2,5)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## contingent-4-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Anchor, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
O is to the East of C2.
C3 is to the West of O.
C1 is to the West of O.
Extra0A is to the Northwest of Extra0B.
O is to the Southeast of Anchor.
C4 is to the West of O.
Anchor is to the Northwest of C0.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. Northwest
C. Southwest
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (5, 2); C0 at (5, 3); C1 at (4, 2); C2 at (3, 2); C3 at (2, 2); C4 at (1, 2); Anchor at (0, 4); Extra0A at (5, 1); Extra0B at (6, 0). In this map, direction=North.
Northeast:
A valid map places O at (5, 2); C0 at (6, 3); C1 at (4, 2); C2 at (3, 2); C3 at (2, 2); C4 at (1, 2); Anchor at (0, 4); Extra0A at (5, 1); Extra0B at (6, 0). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(5,2), C0:(5,3), C1:(4,2), C2:(3,2), C3:(2,2), C4:(1,2), Anchor:(0,4), Extra0A:(5,1), Extra0B:(6,0)} => direction=North.
Northeast:
Valid map {O:(5,2), C0:(6,3), C1:(4,2), C2:(3,2), C3:(2,2), C4:(1,2), Anchor:(0,4), Extra0A:(5,1), Extra0B:(6,0)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## joint-0-base-direction-m0-single
Split: dev; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
C3 is to the West of C2.
C0 is to the East of C1.
C1 is to the East of C2.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. West
C. East
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (3, 0); C0 at (3, 1); C1 at (2, 1); C2 at (1, 1); C3 at (0, 1). In this map, direction=North.
Northeast:
A valid map places O at (3, 0); C0 at (4, 1); C1 at (2, 1); C2 at (1, 1); C3 at (0, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(3,0), C0:(3,1), C1:(2,1), C2:(1,1), C3:(0,1)} => direction=North.
Northeast:
Valid map {O:(3,0), C0:(4,1), C1:(2,1), C2:(1,1), C3:(0,1)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## joint-1-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
C3 is to the North of C4.
Extra0A is to the South of Extra0B.
C2 is to the North of C3.
C1 is to the North of C2.
C1 is to the South of C0.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. Southwest
C. Northwest
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 6); C0 at (0, 7); C1 at (0, 5); C2 at (0, 4); C3 at (0, 3); C4 at (0, 2); Extra0A at (0, 0); Extra0B at (0, 1). In this map, direction=North.
Northeast:
A valid map places O at (0, 2); C0 at (1, 7); C1 at (1, 6); C2 at (1, 5); C3 at (1, 4); C4 at (1, 3); Extra0A at (0, 0); Extra0B at (0, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(0,6), C0:(0,7), C1:(0,5), C2:(0,4), C3:(0,3), C4:(0,2), Extra0A:(0,0), Extra0B:(0,1)} => direction=North.
Northeast:
Valid map {O:(0,2), C0:(1,7), C1:(1,6), C2:(1,5), C3:(1,4), C4:(1,3), Extra0A:(0,0), Extra0B:(0,1)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## joint-2-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
C4 is to the North of C5.
Extra1A is to the Southwest of Extra1B.
Extra0A is to the East of Extra0B.
C0 is to the North of C1.
C2 is to the North of C3.
C3 is to the North of C4.
C1 is to the North of C2.
Question: In which direction is C0 relative to O in every valid map?
A. East
B. Southwest
C. Northeast
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (1, 7); C0 at (1, 8); C1 at (1, 6); C2 at (1, 5); C3 at (1, 4); C4 at (1, 3); C5 at (1, 2); Extra0A at (1, 1); Extra0B at (0, 1); Extra1A at (1, 0); Extra1B at (2, 9). In this map, direction=North.
Northeast:
A valid map places O at (1, 2); C0 at (2, 9); C1 at (2, 8); C2 at (2, 7); C3 at (2, 6); C4 at (2, 5); C5 at (2, 4); Extra0A at (1, 1); Extra0B at (0, 1); Extra1A at (1, 0); Extra1B at (2, 3). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(1,7), C0:(1,8), C1:(1,6), C2:(1,5), C3:(1,4), C4:(1,3), C5:(1,2), Extra0A:(1,1), Extra0B:(0,1), Extra1A:(1,0), Extra1B:(2,9)} => direction=North.
Northeast:
Valid map {O:(1,2), C0:(2,9), C1:(2,8), C2:(2,7), C3:(2,6), C4:(2,5), C5:(2,4), Extra0A:(1,1), Extra0B:(0,1), Extra1A:(1,0), Extra1B:(2,3)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## joint-3-base-direction-m0-single
Split: dev; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3
Query: direction C0 O
Consider a map with multiple objects:
C1 is to the North of C2.
C2 is to the North of C3.
C1 is to the South of C0.
Question: In which direction is C0 relative to O in every valid map?
A. Southwest
B. West
C. Southeast
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (0, 3); C0 at (0, 4); C1 at (0, 2); C2 at (0, 1); C3 at (0, 0). In this map, direction=North.
Northeast:
A valid map places O at (0, 0); C0 at (1, 4); C1 at (1, 3); C2 at (1, 2); C3 at (1, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(0,3), C0:(0,4), C1:(0,2), C2:(0,1), C3:(0,0)} => direction=North.
Northeast:
Valid map {O:(0,0), C0:(1,4), C1:(1,3), C2:(1,2), C3:(1,1)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## joint-4-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
C3 is to the West of C2.
C1 is to the West of C0.
C3 is to the East of C4.
Extra0A is to the East of Extra0B.
C2 is to the West of C1.
Question: In which direction is C0 relative to O in every valid map?
A. West
B. Northeast
C. Southeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places O at (4, 1); C0 at (4, 2); C1 at (3, 2); C2 at (2, 2); C3 at (1, 2); C4 at (0, 2); Extra0A at (4, 0); Extra0B at (3, 0). In this map, direction=North.
Northeast:
A valid map places O at (4, 1); C0 at (5, 2); C1 at (3, 2); C2 at (2, 2); C3 at (1, 2); C4 at (0, 2); Extra0A at (4, 0); Extra0B at (3, 0). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {O:(4,1), C0:(4,2), C1:(3,2), C2:(2,2), C3:(1,2), C4:(0,2), Extra0A:(4,0), Extra0B:(3,0)} => direction=North.
Northeast:
Valid map {O:(4,1), C0:(5,2), C1:(3,2), C2:(2,2), C3:(1,2), C4:(0,2), Extra0A:(4,0), Extra0B:(3,0)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## cycle-0-base-direction-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: direction C0 O
Consider a map with multiple objects:
Cycle2 is to the North of Cycle1.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
C0 is to the East of O.
O is to the West of C1.
Cycle2 is to the North of Cycle0.
C2 is to the East of O.
Question: In which direction is C0 relative to O in every valid map?
A. East
B. Southeast
C. South
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P4: From p4, C0 is exactly east of O.
Q-DIR: From P4, C0 is exactly east of O.
Southeast:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P4: From p4, C0 is exactly east of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P4-X1: From P4, C0's x-coordinate is greater than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```
### Symbolic
```text
North:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P4: [p4] => East(C0,O)
Q-DIR: [P4] => East(C0,O)
Southeast:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose West(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P4: [p4] => East(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P4-X1: [P4] => x(C0) > x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```

## cycle-0-inconsistent-direction-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: direction C0 O
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: In which direction is C0 relative to O in every valid map?
A. Northeast
B. East
C. Southwest
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-direction-m0-all-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: direction C0 O
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: Select the complete set of directions C0 could occupy relative to O.
A. Northeast
B. East
C. Southwest
D. West
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-direction-m0-visible-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: direction C0 O
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: Which listed directions could C0 occupy relative to O? Select all that apply.
A. Northeast
B. East
C. Southwest
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-which-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: which East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: Which eligible object is the only one east of O in every valid map?
A. C3
B. C2
C. C0
D. C1
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-which-m0-all-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: which East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: Select the complete set of eligible objects that could be east of O.
A. C3
B. C2
C. C0
D. C1
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-which-m0-visible-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: which East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: Which listed eligible objects could be east of O? Select all that apply.
A. C3
B. C2
C. C0
D. C1
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-count-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: count East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: What exact number of eligible objects are east of O in every valid map?
A. 2
B. 0
C. 4
D. 3
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-count-m0-all-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: count East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: Select the complete set of possible counts of eligible objects east of O.
A. 2
B. 0
C. 4
D. 3
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-0-inconsistent-count-m0-visible-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Cycle0, Cycle1, Cycle2
Query: count East O
Eligible: C0, C1, C2, C3
Consider a map with multiple objects:
Cycle1 is to the South of Cycle2.
C1 is to the East of O.
Cycle0 is to the North of Cycle2.
O is to the West of C3.
Cycle1 is to the North of Cycle0.
O is to the West of C2.
C0 is to the East of O.
Question: Which listed counts are possible for the number of eligible objects east of O? Select all that apply.
A. 2
B. 0
C. 4
D. 3
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Cycle1's y-coordinate is less than Cycle2's y-coordinate.
From p3, Cycle2's y-coordinate is less than Cycle0's y-coordinate.
From p5, Cycle0's y-coordinate is less than Cycle1's y-coordinate.
Composing this path gives Cycle1's y-coordinate less than Cycle1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle1) < y(Cycle2)
p3: y(Cycle2) < y(Cycle0)
p5: y(Cycle0) < y(Cycle1)
Transitivity: y(Cycle1) < y(Cycle1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-1-base-direction-m0-single
Split: test; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Cycle0, Cycle1, Cycle2, Cycle3, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
Cycle3 is to the South of Cycle2.
Cycle1 is to the South of Cycle0.
Cycle1 is to the North of Cycle2.
C0 is to the West of O.
O is to the East of C3.
O is to the East of C4.
O is to the East of C1.
O is to the East of C2.
Cycle0 is to the North of Cycle3.
Extra0A is to the Southwest of Extra0B.
Question: In which direction is C0 relative to O in every valid map?
A. East
B. West
C. South
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly south of O for contradiction.
P4-X1: From P4, C0's x-coordinate is less than O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is equal to O's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption C0 is exactly south of O leads to contradiction, so this direction is impossible.
Southwest:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is less than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P4: From p4, C0 is exactly west of O.
Q-DIR: From P4, C0 is exactly west of O.
Northwest:
P4: From p4, C0 is exactly west of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P4-Y1: From P4, C0's y-coordinate is equal to O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P4-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
North:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose East(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose South(C0,O) for contradiction.
P4-X1: [P4] => x(C0) < x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) = x(O)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption South(C0,O) leads to contradiction, so this direction is impossible.
Southwest:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) < y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P4: [p4] => West(C0,O)
Q-DIR: [P4] => West(C0,O)
Northwest:
P4: [p4] => West(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P4-Y1: [P4] => y(C0) = y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P4-Y1, refutation-Y1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```

## cycle-1-inconsistent-direction-m0-single
Split: test; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Cycle0, Cycle1, Cycle2, Cycle3, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
Cycle0 is to the South of Cycle3.
O is to the East of C1.
Cycle3 is to the South of Cycle2.
C0 is to the West of O.
O is to the East of C3.
Cycle2 is to the South of Cycle1.
Extra0A is to the Northwest of Extra0B.
O is to the East of C4.
O is to the East of C2.
Cycle0 is to the North of Cycle1.
Question: In which direction is C0 relative to O in every valid map?
A. Northwest
B. East
C. Southeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Cycle0's y-coordinate is less than Cycle3's y-coordinate.
From p3, Cycle3's y-coordinate is less than Cycle2's y-coordinate.
From p6, Cycle2's y-coordinate is less than Cycle1's y-coordinate.
From p10, Cycle1's y-coordinate is less than Cycle0's y-coordinate.
Composing this path gives Cycle0's y-coordinate less than Cycle0's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p6, p10. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Cycle0) < y(Cycle3)
p3: y(Cycle3) < y(Cycle2)
p6: y(Cycle2) < y(Cycle1)
p10: y(Cycle1) < y(Cycle0)
Transitivity: y(Cycle0) < y(Cycle0)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p6, p10. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-2-base-direction-m0-single
Split: dev; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Cycle0, Cycle1, Cycle2, Cycle3, Cycle4, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
C4 is to the South of O.
Extra1A is to the North of Extra1B.
Cycle4 is to the East of Cycle0.
Extra0A is to the Southeast of Extra0B.
O is to the North of C5.
O is to the North of C3.
Cycle2 is to the East of Cycle1.
C0 is to the South of O.
Cycle0 is to the West of Cycle1.
Cycle2 is to the West of Cycle3.
C1 is to the South of O.
Cycle3 is to the West of Cycle4.
O is to the North of C2.
Question: In which direction is C0 relative to O in every valid map?
A. Southeast
B. East
C. West
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P8: From p8, C0 is exactly south of O.
A-REFUTE: Suppose C0 is exactly north of O for contradiction.
P8-Y1: From P8, C0's y-coordinate is less than O's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, C0's y-coordinate is greater than O's y-coordinate.
refutation-C1: From under assumption refutation; P8-Y1, refutation-Y1, contradiction.
The assumption C0 is exactly north of O leads to contradiction, so this direction is impossible.
Northeast:
P8: From p8, C0 is exactly south of O.
A-REFUTE: Suppose C0 is exactly northeast of O for contradiction.
P8-X1: From P8, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P8-X1, refutation-X1, contradiction.
The assumption C0 is exactly northeast of O leads to contradiction, so this direction is impossible.
East:
P8: From p8, C0 is exactly south of O.
A-REFUTE: Suppose C0 is exactly east of O for contradiction.
P8-X1: From P8, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P8-X1, refutation-X1, contradiction.
The assumption C0 is exactly east of O leads to contradiction, so this direction is impossible.
Southeast:
P8: From p8, C0 is exactly south of O.
A-REFUTE: Suppose C0 is exactly southeast of O for contradiction.
P8-X1: From P8, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is greater than O's x-coordinate.
refutation-C1: From under assumption refutation; P8-X1, refutation-X1, contradiction.
The assumption C0 is exactly southeast of O leads to contradiction, so this direction is impossible.
South:
P8: From p8, C0 is exactly south of O.
Q-DIR: From P8, C0 is exactly south of O.
Southwest:
P8: From p8, C0 is exactly south of O.
A-REFUTE: Suppose C0 is exactly southwest of O for contradiction.
P8-X1: From P8, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P8-X1, refutation-X1, contradiction.
The assumption C0 is exactly southwest of O leads to contradiction, so this direction is impossible.
West:
P8: From p8, C0 is exactly south of O.
A-REFUTE: Suppose C0 is exactly west of O for contradiction.
P8-X1: From P8, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P8-X1, refutation-X1, contradiction.
The assumption C0 is exactly west of O leads to contradiction, so this direction is impossible.
Northwest:
P8: From p8, C0 is exactly south of O.
A-REFUTE: Suppose C0 is exactly northwest of O for contradiction.
P8-X1: From P8, C0's x-coordinate is equal to O's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, C0's x-coordinate is less than O's x-coordinate.
refutation-C1: From under assumption refutation; P8-X1, refutation-X1, contradiction.
The assumption C0 is exactly northwest of O leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is South. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P8: [p8] => South(C0,O)
A-REFUTE: Suppose North(C0,O) for contradiction.
P8-Y1: [P8] => y(C0) < y(O)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(C0) > y(O)
refutation-C1: [under assumption refutation; P8-Y1, refutation-Y1] => contradiction
The assumption North(C0,O) leads to contradiction, so this direction is impossible.
Northeast:
P8: [p8] => South(C0,O)
A-REFUTE: Suppose Northeast(C0,O) for contradiction.
P8-X1: [P8] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P8-X1, refutation-X1] => contradiction
The assumption Northeast(C0,O) leads to contradiction, so this direction is impossible.
East:
P8: [p8] => South(C0,O)
A-REFUTE: Suppose East(C0,O) for contradiction.
P8-X1: [P8] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P8-X1, refutation-X1] => contradiction
The assumption East(C0,O) leads to contradiction, so this direction is impossible.
Southeast:
P8: [p8] => South(C0,O)
A-REFUTE: Suppose Southeast(C0,O) for contradiction.
P8-X1: [P8] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) > x(O)
refutation-C1: [under assumption refutation; P8-X1, refutation-X1] => contradiction
The assumption Southeast(C0,O) leads to contradiction, so this direction is impossible.
South:
P8: [p8] => South(C0,O)
Q-DIR: [P8] => South(C0,O)
Southwest:
P8: [p8] => South(C0,O)
A-REFUTE: Suppose Southwest(C0,O) for contradiction.
P8-X1: [P8] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P8-X1, refutation-X1] => contradiction
The assumption Southwest(C0,O) leads to contradiction, so this direction is impossible.
West:
P8: [p8] => South(C0,O)
A-REFUTE: Suppose West(C0,O) for contradiction.
P8-X1: [P8] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P8-X1, refutation-X1] => contradiction
The assumption West(C0,O) leads to contradiction, so this direction is impossible.
Northwest:
P8: [p8] => South(C0,O)
A-REFUTE: Suppose Northwest(C0,O) for contradiction.
P8-X1: [P8] => x(C0) = x(O)
refutation-X1: [under assumption refutation; A-REFUTE] => x(C0) < x(O)
refutation-C1: [under assumption refutation; P8-X1, refutation-X1] => contradiction
The assumption Northwest(C0,O) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is South. It is displayed at D.
Answer: D
```

## coincidence-0-base-direction-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: direction C0 O
Consider a map with multiple objects:
C0 is to the North of O.
O is to the South of C1.
O is to the North of C2.
Anchor is to the West of C0.
C1 is to the East of Anchor.
O is to the North of C3.
Question: In which direction is C0 relative to O in every valid map?
A. South
B. West
C. East
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, C0's x-coordinate is equal to O's x-coordinate.
From p2, O's x-coordinate is equal to C1's x-coordinate.
Composing this path gives C0's x-coordinate equal to C1's x-coordinate.
From p4, C0's y-coordinate is equal to Anchor's y-coordinate.
From p5, Anchor's y-coordinate is equal to C1's y-coordinate.
Composing this path gives C0's y-coordinate equal to C1's y-coordinate.
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p2, p4, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: x(C0) = x(O)
p2: x(O) = x(C1)
Transitivity: x(C0) = x(C1)
p4: y(C0) = y(Anchor)
p5: y(Anchor) = y(C1)
Transitivity: y(C0) = y(C1)
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p2, p4, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## coincidence-1-base-direction-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Anchor, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
O is to the South of C3.
O is to the North of C0.
C0 is to the West of Anchor.
C4 is to the North of O.
C1 is to the South of O.
Extra0A is to the Southwest of Extra0B.
O is to the South of C2.
C1 is to the West of Anchor.
Question: In which direction is C0 relative to O in every valid map?
A. West
B. Southeast
C. Southwest
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p2, C0's x-coordinate is equal to O's x-coordinate.
From p5, O's x-coordinate is equal to C1's x-coordinate.
Composing this path gives C0's x-coordinate equal to C1's x-coordinate.
From p3, C0's y-coordinate is equal to Anchor's y-coordinate.
From p8, Anchor's y-coordinate is equal to C1's y-coordinate.
Composing this path gives C0's y-coordinate equal to C1's y-coordinate.
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p5, p8. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p2: x(C0) = x(O)
p5: x(O) = x(C1)
Transitivity: x(C0) = x(C1)
p3: y(C0) = y(Anchor)
p8: y(Anchor) = y(C1)
Transitivity: y(C0) = y(C1)
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p5, p8. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## coincidence-2-base-direction-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, C5, Anchor, Extra0A, Extra0B, Extra1A, Extra1B
Query: direction C0 O
Consider a map with multiple objects:
Extra1A is to the West of Extra1B.
C1 is to the South of O.
O is to the North of C0.
C0 is to the West of Anchor.
O is to the South of C4.
C5 is to the North of O.
O is to the South of C3.
Extra0A is to the Southeast of Extra0B.
Anchor is to the East of C1.
C2 is to the North of O.
Question: In which direction is C0 relative to O in every valid map?
A. Southwest
B. Northwest
C. Northeast
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p3, C0's x-coordinate is equal to O's x-coordinate.
From p2, O's x-coordinate is equal to C1's x-coordinate.
Composing this path gives C0's x-coordinate equal to C1's x-coordinate.
From p4, C0's y-coordinate is equal to Anchor's y-coordinate.
From p9, Anchor's y-coordinate is equal to C1's y-coordinate.
Composing this path gives C0's y-coordinate equal to C1's y-coordinate.
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p4, p9. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p3: x(C0) = x(O)
p2: x(O) = x(C1)
Transitivity: x(C0) = x(C1)
p4: y(C0) = y(Anchor)
p9: y(Anchor) = y(C1)
Transitivity: y(C0) = y(C1)
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p4, p9. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## coincidence-3-base-direction-m0-single
Split: test; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, Anchor
Query: direction C0 O
Consider a map with multiple objects:
O is to the West of C2.
C3 is to the East of O.
O is to the East of C0.
O is to the East of C1.
Anchor is to the South of C0.
C1 is to the North of Anchor.
Question: In which direction is C0 relative to O in every valid map?
A. West
B. Northwest
C. Northeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p5, C0's x-coordinate is equal to Anchor's x-coordinate.
From p6, Anchor's x-coordinate is equal to C1's x-coordinate.
Composing this path gives C0's x-coordinate equal to C1's x-coordinate.
From p3, C0's y-coordinate is equal to O's y-coordinate.
From p4, O's y-coordinate is equal to C1's y-coordinate.
Composing this path gives C0's y-coordinate equal to C1's y-coordinate.
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p3, p4, p5, p6. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p5: x(C0) = x(Anchor)
p6: x(Anchor) = x(C1)
Transitivity: x(C0) = x(C1)
p3: y(C0) = y(O)
p4: y(O) = y(C1)
Transitivity: y(C0) = y(C1)
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p3, p4, p5, p6. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## coincidence-4-base-direction-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: O, C0, C1, C2, C3, C4, Anchor, Extra0A, Extra0B
Query: direction C0 O
Consider a map with multiple objects:
Anchor is to the South of C0.
C4 is to the East of O.
C1 is to the West of O.
C0 is to the West of O.
C3 is to the East of O.
C1 is to the North of Anchor.
Extra0A is to the Southeast of Extra0B.
C2 is to the East of O.
Question: In which direction is C0 relative to O in every valid map?
A. North
B. East
C. Southwest
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, C0's x-coordinate is equal to Anchor's x-coordinate.
From p6, Anchor's x-coordinate is equal to C1's x-coordinate.
Composing this path gives C0's x-coordinate equal to C1's x-coordinate.
From p4, C0's y-coordinate is equal to O's y-coordinate.
From p3, O's y-coordinate is equal to C1's y-coordinate.
Composing this path gives C0's y-coordinate equal to C1's y-coordinate.
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p3, p4, p6. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: x(C0) = x(Anchor)
p6: x(Anchor) = x(C1)
Transitivity: x(C0) = x(C1)
p4: y(C0) = y(O)
p3: y(O) = y(C1)
Transitivity: y(C0) = y(C1)
C0 and C1 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p3, p4, p6. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
