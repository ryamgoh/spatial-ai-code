# Pilot Audit Samples

Review status: pending. Inspect both styles and record outcomes.

## random-world-0-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: direction Object3 Object7
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: In which direction is Object3 relative to Object7 in every valid map?
A. Northeast
B. Northwest
C. North
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object0 at (1, 4); Object1 at (1, 3); Object2 at (2, 5); Object3 at (1, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, direction=North.
Northeast:
A valid map places Object0 at (2, 4); Object1 at (2, 3); Object2 at (3, 5); Object3 at (2, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object0:(1,4), Object1:(1,3), Object2:(2,5), Object3:(1,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => direction=North.
Northeast:
Valid map {Object0:(2,4), Object1:(2,3), Object2:(3,5), Object3:(2,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## random-world-0-direction-m0-all-possible
Split: train; case: underdetermined; answer: F

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: direction Object3 Object7
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: Select the complete set of directions Object3 could occupy relative to Object7.
A. Northeast
B. Northwest
C. North
D. Southwest
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
A valid map places Object0 at (2, 3); Object1 at (2, 2); Object2 at (3, 4); Object3 at (2, 0); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, direction=East.
The possible value East is unlisted, so A-D cannot form a complete answer.
Answer: F
```
### Symbolic
```text
Valid map {Object0:(2,3), Object1:(2,2), Object2:(3,4), Object3:(2,0), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => direction=East.
The possible value East is unlisted, so A-D cannot form a complete answer.
Answer: F
```

## random-world-0-direction-m0-visible-possible
Split: train; case: underdetermined; answer: ABCD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: direction Object3 Object7
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: Which listed directions could Object3 occupy relative to Object7? Select all that apply.
A. Northeast
B. Northwest
C. North
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object0 at (1, 4); Object1 at (1, 3); Object2 at (2, 5); Object3 at (1, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, direction=North.
Northeast:
A valid map places Object0 at (2, 4); Object1 at (2, 3); Object2 at (3, 5); Object3 at (2, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, direction=Northeast.
Southwest:
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 0); Object4 at (0, 2); Object5 at (0, 1); Object6 at (1, 2); Object7 at (1, 1). In this map, direction=Southwest.
Northwest:
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, direction=Northwest.
The listed possible set is {Northeast, Northwest, North, Southwest}.
Answer: ABCD
```
### Symbolic
```text
North:
Valid map {Object0:(1,4), Object1:(1,3), Object2:(2,5), Object3:(1,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => direction=North.
Northeast:
Valid map {Object0:(2,4), Object1:(2,3), Object2:(3,5), Object3:(2,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => direction=Northeast.
Southwest:
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,0), Object4:(0,2), Object5:(0,1), Object6:(1,2), Object7:(1,1)} => direction=Southwest.
Northwest:
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => direction=Northwest.
The listed possible set is {Northeast, Northwest, North, Southwest}.
Answer: ABCD
```

## random-world-0-which-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: which West Object7
Eligible: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: Which eligible object is the only one west of Object7 in every valid map?
A. Object0
B. Object5
C. Object3
D. Object1
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: contingent
A valid map places Object0 at (1, 2); Object1 at (1, 1); Object2 at (2, 4); Object3 at (1, 0); Object4 at (0, 3); Object5 at (0, 2); Object6 at (2, 3); Object7 at (2, 2). In this map, qualifying objects={Object0, Object5}.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, qualifying objects={Object5}.
Object0 qualifies in the first map but not the second.
Object1: contingent
A valid map places Object0 at (1, 3); Object1 at (1, 1); Object2 at (2, 4); Object3 at (1, 0); Object4 at (1, 2); Object5 at (0, 1); Object6 at (2, 2); Object7 at (2, 1). In this map, qualifying objects={Object1, Object5}.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, qualifying objects={Object5}.
Object1 qualifies in the first map but not the second.
Object2: contingent
A valid map places Object0 at (1, 2); Object1 at (1, 1); Object2 at (2, 3); Object3 at (1, 0); Object4 at (0, 4); Object5 at (0, 3); Object6 at (3, 4); Object7 at (3, 3). In this map, qualifying objects={Object2, Object5}.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, qualifying objects={Object5}.
Object2 qualifies in the first map but not the second.
Object3: contingent
A valid map places Object0 at (1, 3); Object1 at (1, 2); Object2 at (2, 4); Object3 at (1, 0); Object4 at (1, 1); Object5 at (0, 0); Object6 at (2, 1); Object7 at (2, 0). In this map, qualifying objects={Object3, Object5}.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, qualifying objects={Object5}.
Object3 qualifies in the first map but not the second.
Object4: impossible
P2: From p2, Object4 is exactly west of Object6.
P4: From p4, Object6 is exactly north of Object7.
A-REFUTE: Suppose Object4 is exactly west of Object7 for contradiction.
P2-Y1: From P2, Object4's y-coordinate is equal to Object6's y-coordinate.
P4-Y1: From P4, Object6's y-coordinate is greater than Object7's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object4's y-coordinate is equal to Object7's y-coordinate.
PATH-Y-1-INV: From P4-Y1, Object7's y-coordinate is less than Object6's y-coordinate.
PATH-Y-2-INV: From P2-Y1, Object6's y-coordinate is equal to Object4's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object7's y-coordinate is less than Object4's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption Object4 is exactly west of Object7 leads to contradiction, so this direction is impossible.
Object5: entailed
P3: From p3, Object5 is exactly west of Object7.
Q-DIR: From P3, Object5 is exactly west of Object7.
Object6: impossible
P4: From p4, Object6 is exactly north of Object7.
A-REFUTE: Suppose Object6 is exactly west of Object7 for contradiction.
P4-X1: From P4, Object6's x-coordinate is equal to Object7's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object6's x-coordinate is less than Object7's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption Object6 is exactly west of Object7 leads to contradiction, so this direction is impossible.
The memberships do not establish one object as the only match in every valid map.
Answer: E
```
### Symbolic
```text
Object0: contingent
Valid map {Object0:(1,2), Object1:(1,1), Object2:(2,4), Object3:(1,0), Object4:(0,3), Object5:(0,2), Object6:(2,3), Object7:(2,2)} => qualifying objects={Object0, Object5}.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => qualifying objects={Object5}.
Object0 qualifies in the first map but not the second.
Object1: contingent
Valid map {Object0:(1,3), Object1:(1,1), Object2:(2,4), Object3:(1,0), Object4:(1,2), Object5:(0,1), Object6:(2,2), Object7:(2,1)} => qualifying objects={Object1, Object5}.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => qualifying objects={Object5}.
Object1 qualifies in the first map but not the second.
Object2: contingent
Valid map {Object0:(1,2), Object1:(1,1), Object2:(2,3), Object3:(1,0), Object4:(0,4), Object5:(0,3), Object6:(3,4), Object7:(3,3)} => qualifying objects={Object2, Object5}.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => qualifying objects={Object5}.
Object2 qualifies in the first map but not the second.
Object3: contingent
Valid map {Object0:(1,3), Object1:(1,2), Object2:(2,4), Object3:(1,0), Object4:(1,1), Object5:(0,0), Object6:(2,1), Object7:(2,0)} => qualifying objects={Object3, Object5}.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => qualifying objects={Object5}.
Object3 qualifies in the first map but not the second.
Object4: impossible
P2: [p2] => West(Object4,Object6)
P4: [p4] => North(Object6,Object7)
A-REFUTE: Suppose West(Object4,Object7) for contradiction.
P2-Y1: [P2] => y(Object4) = y(Object6)
P4-Y1: [P4] => y(Object6) > y(Object7)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object4) = y(Object7)
PATH-Y-1-INV: [P4-Y1] => y(Object7) < y(Object6)
PATH-Y-2-INV: [P2-Y1] => y(Object6) = y(Object4)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object7) < y(Object4)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption West(Object4,Object7) leads to contradiction, so this direction is impossible.
Object5: entailed
P3: [p3] => West(Object5,Object7)
Q-DIR: [P3] => West(Object5,Object7)
Object6: impossible
P4: [p4] => North(Object6,Object7)
A-REFUTE: Suppose West(Object6,Object7) for contradiction.
P4-X1: [P4] => x(Object6) = x(Object7)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object6) < x(Object7)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(Object6,Object7) leads to contradiction, so this direction is impossible.
The memberships do not establish one object as the only match in every valid map.
Answer: E
```

## random-world-0-which-m0-all-possible
Split: train; case: underdetermined; answer: F

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: which West Object7
Eligible: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: Select the complete set of eligible objects that could be west of Object7.
A. Object0
B. Object5
C. Object3
D. Object1
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
A valid map places Object0 at (1, 2); Object1 at (1, 1); Object2 at (2, 3); Object3 at (1, 0); Object4 at (0, 4); Object5 at (0, 3); Object6 at (3, 4); Object7 at (3, 3). In this map, qualifying objects={Object2, Object5}.
The possible value Object2 is unlisted, so A-D cannot form a complete answer.
Answer: F
```
### Symbolic
```text
Valid map {Object0:(1,2), Object1:(1,1), Object2:(2,3), Object3:(1,0), Object4:(0,4), Object5:(0,3), Object6:(3,4), Object7:(3,3)} => qualifying objects={Object2, Object5}.
The possible value Object2 is unlisted, so A-D cannot form a complete answer.
Answer: F
```

## random-world-0-which-m0-visible-possible
Split: train; case: underdetermined; answer: ABCD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: which West Object7
Eligible: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: Which listed eligible objects could be west of Object7? Select all that apply.
A. Object0
B. Object5
C. Object3
D. Object1
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: contingent
A valid map places Object0 at (1, 2); Object1 at (1, 1); Object2 at (2, 4); Object3 at (1, 0); Object4 at (0, 3); Object5 at (0, 2); Object6 at (2, 3); Object7 at (2, 2). In this map, qualifying objects={Object0, Object5}.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, qualifying objects={Object5}.
Object0 qualifies in the first map but not the second.
Object1: contingent
A valid map places Object0 at (1, 3); Object1 at (1, 1); Object2 at (2, 4); Object3 at (1, 0); Object4 at (1, 2); Object5 at (0, 1); Object6 at (2, 2); Object7 at (2, 1). In this map, qualifying objects={Object1, Object5}.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, qualifying objects={Object5}.
Object1 qualifies in the first map but not the second.
Object3: contingent
A valid map places Object0 at (1, 3); Object1 at (1, 2); Object2 at (2, 4); Object3 at (1, 0); Object4 at (1, 1); Object5 at (0, 0); Object6 at (2, 1); Object7 at (2, 0). In this map, qualifying objects={Object3, Object5}.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, qualifying objects={Object5}.
Object3 qualifies in the first map but not the second.
Object5: entailed
P3: From p3, Object5 is exactly west of Object7.
Q-DIR: From P3, Object5 is exactly west of Object7.
The listed possible set is {Object0, Object5, Object3, Object1}.
Answer: ABCD
```
### Symbolic
```text
Object0: contingent
Valid map {Object0:(1,2), Object1:(1,1), Object2:(2,4), Object3:(1,0), Object4:(0,3), Object5:(0,2), Object6:(2,3), Object7:(2,2)} => qualifying objects={Object0, Object5}.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => qualifying objects={Object5}.
Object0 qualifies in the first map but not the second.
Object1: contingent
Valid map {Object0:(1,3), Object1:(1,1), Object2:(2,4), Object3:(1,0), Object4:(1,2), Object5:(0,1), Object6:(2,2), Object7:(2,1)} => qualifying objects={Object1, Object5}.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => qualifying objects={Object5}.
Object1 qualifies in the first map but not the second.
Object3: contingent
Valid map {Object0:(1,3), Object1:(1,2), Object2:(2,4), Object3:(1,0), Object4:(1,1), Object5:(0,0), Object6:(2,1), Object7:(2,0)} => qualifying objects={Object3, Object5}.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => qualifying objects={Object5}.
Object3 qualifies in the first map but not the second.
Object5: entailed
P3: [p3] => West(Object5,Object7)
Q-DIR: [P3] => West(Object5,Object7)
The listed possible set is {Object0, Object5, Object3, Object1}.
Answer: ABCD
```

## random-world-0-count-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: count West Object7
Eligible: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: What exact number of eligible objects are west of Object7 in every valid map?
A. 1
B. 2
C. 3
D. 4
E. Cannot be determined
F. None of the above
```
### Natural
```text
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, count=1.
A valid map places Object0 at (1, 3); Object1 at (1, 2); Object2 at (2, 4); Object3 at (1, 0); Object4 at (1, 1); Object5 at (0, 0); Object6 at (2, 1); Object7 at (2, 0). In this map, count=2.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => count=1.
Valid map {Object0:(1,3), Object1:(1,2), Object2:(2,4), Object3:(1,0), Object4:(1,1), Object5:(0,0), Object6:(2,1), Object7:(2,0)} => count=2.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## random-world-0-count-m0-all-possible
Split: train; case: underdetermined; answer: ABCD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: count West Object7
Eligible: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: Select the complete set of possible counts of eligible objects west of Object7.
A. 1
B. 2
C. 3
D. 4
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
P2: From p2, Object4 is exactly west of Object6.
P4: From p4, Object6 is exactly north of Object7.
A-REFUTE: Suppose Object4 is exactly west of Object7 for contradiction.
P2-Y1: From P2, Object4's y-coordinate is equal to Object6's y-coordinate.
P4-Y1: From P4, Object6's y-coordinate is greater than Object7's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object4's y-coordinate is equal to Object7's y-coordinate.
PATH-Y-1-INV: From P4-Y1, Object7's y-coordinate is less than Object6's y-coordinate.
PATH-Y-2-INV: From P2-Y1, Object6's y-coordinate is equal to Object4's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object7's y-coordinate is less than Object4's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption Object4 is exactly west of Object7 leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
P3: From p3, Object5 is exactly west of Object7.
Q-DIR: From P3, Object5 is exactly west of Object7.
Fixed membership: Object5 is entailed.
P4: From p4, Object6 is exactly north of Object7.
A-REFUTE: Suppose Object6 is exactly west of Object7 for contradiction.
P4-X1: From P4, Object6's x-coordinate is equal to Object7's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object6's x-coordinate is less than Object7's x-coordinate.
refutation-C1: From under assumption refutation; P4-X1, refutation-X1, contradiction.
The assumption Object6 is exactly west of Object7 leads to contradiction, so this direction is impossible.
Fixed membership: Object6 is impossible.
Suppose both qualify: p6: Object0 is exactly west of Object7; p7: Object2 is exactly west of Object7.
From p5, Object0's y-coordinate is less than Object2's y-coordinate.
From p7, Object2's y-coordinate is equal to Object7's y-coordinate.
From p6, Object7's y-coordinate is equal to Object0's y-coordinate.
Composing this path gives Object0's y-coordinate less than Object0's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p5, p6, p7. No minimality claim is made.
Thus Object0 and Object2 cannot qualify together.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 0 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, count=1.
A valid map places Object0 at (1, 3); Object1 at (1, 2); Object2 at (2, 4); Object3 at (1, 0); Object4 at (1, 1); Object5 at (0, 0); Object6 at (2, 1); Object7 at (2, 0). In this map, count=2.
A valid map places Object0 at (2, 2); Object1 at (2, 0); Object2 at (3, 3); Object3 at (1, 0); Object4 at (2, 1); Object5 at (0, 0); Object6 at (3, 1); Object7 at (3, 0). In this map, count=3.
A valid map places Object0 at (3, 0); Object1 at (2, 0); Object2 at (4, 2); Object3 at (1, 0); Object4 at (0, 1); Object5 at (0, 0); Object6 at (4, 1); Object7 at (4, 0). In this map, count=4.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 6 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 7 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {1, 2, 3, 4}.
Answer: ABCD
```
### Symbolic
```text
P2: [p2] => West(Object4,Object6)
P4: [p4] => North(Object6,Object7)
A-REFUTE: Suppose West(Object4,Object7) for contradiction.
P2-Y1: [P2] => y(Object4) = y(Object6)
P4-Y1: [P4] => y(Object6) > y(Object7)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object4) = y(Object7)
PATH-Y-1-INV: [P4-Y1] => y(Object7) < y(Object6)
PATH-Y-2-INV: [P2-Y1] => y(Object6) = y(Object4)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object7) < y(Object4)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption West(Object4,Object7) leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
P3: [p3] => West(Object5,Object7)
Q-DIR: [P3] => West(Object5,Object7)
Fixed membership: Object5 is entailed.
P4: [p4] => North(Object6,Object7)
A-REFUTE: Suppose West(Object6,Object7) for contradiction.
P4-X1: [P4] => x(Object6) = x(Object7)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object6) < x(Object7)
refutation-C1: [under assumption refutation; P4-X1, refutation-X1] => contradiction
The assumption West(Object6,Object7) leads to contradiction, so this direction is impossible.
Fixed membership: Object6 is impossible.
Suppose both qualify: p6: West(Object0,Object7); p7: West(Object2,Object7).
p5: y(Object0) < y(Object2)
p7: y(Object2) = y(Object7)
p6: y(Object7) = y(Object0)
Transitivity: y(Object0) < y(Object0)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p5, p6, p7. No minimality claim is made.
Thus Object0 and Object2 cannot qualify together.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Count 0 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => count=1.
Valid map {Object0:(1,3), Object1:(1,2), Object2:(2,4), Object3:(1,0), Object4:(1,1), Object5:(0,0), Object6:(2,1), Object7:(2,0)} => count=2.
Valid map {Object0:(2,2), Object1:(2,0), Object2:(3,3), Object3:(1,0), Object4:(2,1), Object5:(0,0), Object6:(3,1), Object7:(3,0)} => count=3.
Valid map {Object0:(3,0), Object1:(2,0), Object2:(4,2), Object3:(1,0), Object4:(0,1), Object5:(0,0), Object6:(4,1), Object7:(4,0)} => count=4.
Count 5 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 6 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 7 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {1, 2, 3, 4}.
Answer: ABCD
```

## random-world-0-count-m0-visible-possible
Split: train; case: underdetermined; answer: ABCD

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: count West Object7
Eligible: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Consider a map with multiple objects:
Object6 is to the Northeast of Object5.
Object4 is to the West of Object6.
Object5 is to the West of Object7.
Object6 is to the North of Object7.
Object0 is to the Southwest of Object2.
Question: Which listed counts are possible for the number of eligible objects west of Object7? Select all that apply.
A. 1
B. 2
C. 3
D. 4
E. Cannot be determined
F. None of the above
```
### Natural
```text
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (1, 5); Object3 at (0, 2); Object4 at (0, 1); Object5 at (0, 0); Object6 at (1, 1); Object7 at (1, 0). In this map, count=1.
A valid map places Object0 at (1, 3); Object1 at (1, 2); Object2 at (2, 4); Object3 at (1, 0); Object4 at (1, 1); Object5 at (0, 0); Object6 at (2, 1); Object7 at (2, 0). In this map, count=2.
A valid map places Object0 at (2, 2); Object1 at (2, 0); Object2 at (3, 3); Object3 at (1, 0); Object4 at (2, 1); Object5 at (0, 0); Object6 at (3, 1); Object7 at (3, 0). In this map, count=3.
A valid map places Object0 at (3, 0); Object1 at (2, 0); Object2 at (4, 2); Object3 at (1, 0); Object4 at (0, 1); Object5 at (0, 0); Object6 at (4, 1); Object7 at (4, 0). In this map, count=4.
The listed possible set is {1, 2, 3, 4}.
Answer: ABCD
```
### Symbolic
```text
Valid map {Object0:(0,4), Object1:(0,3), Object2:(1,5), Object3:(0,2), Object4:(0,1), Object5:(0,0), Object6:(1,1), Object7:(1,0)} => count=1.
Valid map {Object0:(1,3), Object1:(1,2), Object2:(2,4), Object3:(1,0), Object4:(1,1), Object5:(0,0), Object6:(2,1), Object7:(2,0)} => count=2.
Valid map {Object0:(2,2), Object1:(2,0), Object2:(3,3), Object3:(1,0), Object4:(2,1), Object5:(0,0), Object6:(3,1), Object7:(3,0)} => count=3.
Valid map {Object0:(3,0), Object1:(2,0), Object2:(4,2), Object3:(1,0), Object4:(0,1), Object5:(0,0), Object6:(4,1), Object7:(4,0)} => count=4.
The listed possible set is {1, 2, 3, 4}.
Answer: ABCD
```

## random-world-1-direction-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: direction Object2 Object4
Consider a map with multiple objects:
Object3 is to the West of Object0.
Object2 is to the Southeast of Object4.
Object2 is to the Southeast of Object0.
Object3 is to the South of Object4.
Object0 is to the Southeast of Object4.
Question: In which direction is Object2 relative to Object4 in every valid map?
A. Southeast
B. Southwest
C. Northeast
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
Southeast:
P2: From p2, Object2 is exactly southeast of Object4.
Q-DIR: From P2, Object2 is exactly southeast of Object4.
The uniquely guaranteed answer is Southeast. It is displayed at A.
Answer: A
```
### Symbolic
```text
Southeast:
P2: [p2] => Southeast(Object2,Object4)
Q-DIR: [P2] => Southeast(Object2,Object4)
The uniquely guaranteed answer is Southeast. It is displayed at A.
Answer: A
```

## random-world-1-direction-m0-all-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: direction Object2 Object4
Consider a map with multiple objects:
Object3 is to the West of Object0.
Object2 is to the Southeast of Object4.
Object2 is to the Southeast of Object0.
Object3 is to the South of Object4.
Object0 is to the Southeast of Object4.
Question: Select the complete set of directions Object2 could occupy relative to Object4.
A. Southeast
B. Southwest
C. Northeast
D. Northwest
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
North:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly north of Object4 for contradiction.
P2-X1: From P2, Object2's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is equal to Object4's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly north of Object4 leads to contradiction, so this direction is impossible.
Northeast:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly northeast of Object4 for contradiction.
P2-Y1: From P2, Object2's y-coordinate is less than Object4's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object4's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption Object2 is exactly northeast of Object4 leads to contradiction, so this direction is impossible.
East:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly east of Object4 for contradiction.
P2-Y1: From P2, Object2's y-coordinate is less than Object4's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is equal to Object4's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption Object2 is exactly east of Object4 leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, Object2 is exactly southeast of Object4.
Q-DIR: From P2, Object2 is exactly southeast of Object4.
South:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly south of Object4 for contradiction.
P2-X1: From P2, Object2's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is equal to Object4's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly south of Object4 leads to contradiction, so this direction is impossible.
Southwest:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly southwest of Object4 for contradiction.
P2-X1: From P2, Object2's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is less than Object4's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
West:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly west of Object4 for contradiction.
P2-X1: From P2, Object2's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is less than Object4's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly west of Object4 leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly northwest of Object4 for contradiction.
P2-X1: From P2, Object2's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is less than Object4's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly northwest of Object4 leads to contradiction, so this direction is impossible.
The complete possible set is {Southeast}.
Answer: A
```
### Symbolic
```text
North:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose North(Object2,Object4) for contradiction.
P2-X1: [P2] => x(Object2) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) = x(Object4)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption North(Object2,Object4) leads to contradiction, so this direction is impossible.
Northeast:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose Northeast(Object2,Object4) for contradiction.
P2-Y1: [P2] => y(Object2) < y(Object4)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object4)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Northeast(Object2,Object4) leads to contradiction, so this direction is impossible.
East:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose East(Object2,Object4) for contradiction.
P2-Y1: [P2] => y(Object2) < y(Object4)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) = y(Object4)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption East(Object2,Object4) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => Southeast(Object2,Object4)
Q-DIR: [P2] => Southeast(Object2,Object4)
South:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose South(Object2,Object4) for contradiction.
P2-X1: [P2] => x(Object2) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) = x(Object4)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption South(Object2,Object4) leads to contradiction, so this direction is impossible.
Southwest:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose Southwest(Object2,Object4) for contradiction.
P2-X1: [P2] => x(Object2) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) < x(Object4)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Southwest(Object2,Object4) leads to contradiction, so this direction is impossible.
West:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose West(Object2,Object4) for contradiction.
P2-X1: [P2] => x(Object2) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) < x(Object4)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption West(Object2,Object4) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose Northwest(Object2,Object4) for contradiction.
P2-X1: [P2] => x(Object2) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) < x(Object4)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northwest(Object2,Object4) leads to contradiction, so this direction is impossible.
The complete possible set is {Southeast}.
Answer: A
```

## random-world-1-direction-m0-visible-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: direction Object2 Object4
Consider a map with multiple objects:
Object3 is to the West of Object0.
Object2 is to the Southeast of Object4.
Object2 is to the Southeast of Object0.
Object3 is to the South of Object4.
Object0 is to the Southeast of Object4.
Question: Which listed directions could Object2 occupy relative to Object4? Select all that apply.
A. Southeast
B. Southwest
C. Northeast
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
Northeast:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly northeast of Object4 for contradiction.
P2-Y1: From P2, Object2's y-coordinate is less than Object4's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object4's y-coordinate.
refutation-C1: From under assumption refutation; P2-Y1, refutation-Y1, contradiction.
The assumption Object2 is exactly northeast of Object4 leads to contradiction, so this direction is impossible.
Southeast:
P2: From p2, Object2 is exactly southeast of Object4.
Q-DIR: From P2, Object2 is exactly southeast of Object4.
Southwest:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly southwest of Object4 for contradiction.
P2-X1: From P2, Object2's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is less than Object4's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
Northwest:
P2: From p2, Object2 is exactly southeast of Object4.
A-REFUTE: Suppose Object2 is exactly northwest of Object4 for contradiction.
P2-X1: From P2, Object2's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object2's x-coordinate is less than Object4's x-coordinate.
refutation-C1: From under assumption refutation; P2-X1, refutation-X1, contradiction.
The assumption Object2 is exactly northwest of Object4 leads to contradiction, so this direction is impossible.
The listed possible set is {Southeast}.
Answer: A
```
### Symbolic
```text
Northeast:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose Northeast(Object2,Object4) for contradiction.
P2-Y1: [P2] => y(Object2) < y(Object4)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object4)
refutation-C1: [under assumption refutation; P2-Y1, refutation-Y1] => contradiction
The assumption Northeast(Object2,Object4) leads to contradiction, so this direction is impossible.
Southeast:
P2: [p2] => Southeast(Object2,Object4)
Q-DIR: [P2] => Southeast(Object2,Object4)
Southwest:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose Southwest(Object2,Object4) for contradiction.
P2-X1: [P2] => x(Object2) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) < x(Object4)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Southwest(Object2,Object4) leads to contradiction, so this direction is impossible.
Northwest:
P2: [p2] => Southeast(Object2,Object4)
A-REFUTE: Suppose Northwest(Object2,Object4) for contradiction.
P2-X1: [P2] => x(Object2) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object2) < x(Object4)
refutation-C1: [under assumption refutation; P2-X1, refutation-X1] => contradiction
The assumption Northwest(Object2,Object4) leads to contradiction, so this direction is impossible.
The listed possible set is {Southeast}.
Answer: A
```

## random-world-2-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: direction Object0 Object4
Consider a map with multiple objects:
Object1 is to the South of Object0.
Object2 is to the Northwest of Object1.
Object2 is to the North of Object4.
Question: In which direction is Object0 relative to Object4 in every valid map?
A. West
B. Southeast
C. Northeast
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
Northeast:
A valid map places Object0 at (1, 4); Object1 at (1, 2); Object2 at (0, 3); Object3 at (1, 1); Object4 at (0, 0). In this map, direction=Northeast.
East:
A valid map places Object0 at (1, 2); Object1 at (1, 1); Object2 at (0, 3); Object3 at (1, 0); Object4 at (0, 2). In this map, direction=East.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
Northeast:
Valid map {Object0:(1,4), Object1:(1,2), Object2:(0,3), Object3:(1,1), Object4:(0,0)} => direction=Northeast.
East:
Valid map {Object0:(1,2), Object1:(1,1), Object2:(0,3), Object3:(1,0), Object4:(0,2)} => direction=East.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## random-world-3-direction-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Query: direction Object2 Object6
Consider a map with multiple objects:
Object6 is to the Northeast of Object1.
Object4 is to the South of Object6.
Object4 is to the Northeast of Object5.
Object5 is to the East of Object3.
Object0 is to the North of Object4.
Object1 is to the West of Object3.
Object2 is to the Northeast of Object3.
Question: In which direction is Object2 relative to Object6 in every valid map?
A. West
B. North
C. Northeast
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object0 at (3, 4); Object1 at (0, 0); Object2 at (3, 3); Object3 at (1, 0); Object4 at (3, 1); Object5 at (2, 0); Object6 at (3, 2). In this map, direction=North.
Northeast:
A valid map places Object0 at (3, 3); Object1 at (0, 0); Object2 at (4, 4); Object3 at (1, 0); Object4 at (3, 1); Object5 at (2, 0); Object6 at (3, 2). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object0:(3,4), Object1:(0,0), Object2:(3,3), Object3:(1,0), Object4:(3,1), Object5:(2,0), Object6:(3,2)} => direction=North.
Northeast:
Valid map {Object0:(3,3), Object1:(0,0), Object2:(4,4), Object3:(1,0), Object4:(3,1), Object5:(2,0), Object6:(3,2)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## random-world-4-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: direction Object3 Object4
Consider a map with multiple objects:
Object3 is to the Northwest of Object2.
Object1 is to the Southeast of Object2.
Question: In which direction is Object3 relative to Object4 in every valid map?
A. South
B. West
C. East
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object0 at (2, 4); Object1 at (2, 1); Object2 at (1, 2); Object3 at (0, 3); Object4 at (0, 0). In this map, direction=North.
Northeast:
A valid map places Object0 at (3, 4); Object1 at (3, 1); Object2 at (2, 2); Object3 at (1, 3); Object4 at (0, 0). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object0:(2,4), Object1:(2,1), Object2:(1,2), Object3:(0,3), Object4:(0,0)} => direction=North.
Northeast:
Valid map {Object0:(3,4), Object1:(3,1), Object2:(2,2), Object3:(1,3), Object4:(0,0)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## random-world-9-which-m0-single
Split: train; case: no-match; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: which North Object1
Eligible: Object0, Object2, Object3, Object4
Consider a map with multiple objects:
Object0 is to the Southwest of Object1.
Object4 is to the Northwest of Object2.
Object0 is to the Northeast of Object3.
Object3 is to the Southwest of Object4.
Object3 is to the West of Object2.
Object0 is to the Southeast of Object4.
Question: Which eligible object is the only one north of Object1 in every valid map?
A. Object0
B. Object2
C. Object4
D. Object3
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: impossible
P1: From p1, Object0 is exactly southwest of Object1.
A-REFUTE: Suppose Object0 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is equal to Object1's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object0 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object2: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
P5: From p5, Object3 is exactly west of Object2.
A-REFUTE: Suppose Object2 is exactly north of Object1 for contradiction.
P1-Y1: From P1, Object0's y-coordinate is less than Object1's y-coordinate.
P3-Y1: From P3, Object0's y-coordinate is greater than Object3's y-coordinate.
P5-Y1: From P5, Object3's y-coordinate is equal to Object2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object1's y-coordinate.
PATH-Y-1-INV: From P5-Y1, Object2's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P3-Y1, Object3's y-coordinate is less than Object0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object2's y-coordinate is less than Object0's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, Object2's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption Object2 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object3: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
A-REFUTE: Suppose Object3 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P3-X1: From P3, Object0's x-coordinate is greater than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object3's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P3-X1, Object3's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object3's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object3 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object4: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P6: From p6, Object0 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P6-X1: From P6, Object0's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P6-X1, Object4's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object4's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object4 is exactly north of Object1 leads to contradiction, so this direction is impossible.
No eligible object can qualify, so there is no unique qualifying object.
Answer: E
```
### Symbolic
```text
Object0: impossible
P1: [p1] => Southwest(Object0,Object1)
A-REFUTE: Suppose North(Object0,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) = x(Object1)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(Object0,Object1) leads to contradiction, so this direction is impossible.
Object2: impossible
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
P5: [p5] => West(Object3,Object2)
A-REFUTE: Suppose North(Object2,Object1) for contradiction.
P1-Y1: [P1] => y(Object0) < y(Object1)
P3-Y1: [P3] => y(Object0) > y(Object3)
P5-Y1: [P5] => y(Object3) = y(Object2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object1)
PATH-Y-1-INV: [P5-Y1] => y(Object2) = y(Object3)
PATH-Y-2-INV: [P3-Y1] => y(Object3) < y(Object0)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object2) < y(Object0)
D-Y-2: [D-Y-1, P1-Y1] => y(Object2) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption North(Object2,Object1) leads to contradiction, so this direction is impossible.
Object3: impossible
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
A-REFUTE: Suppose North(Object3,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P3-X1: [P3] => x(Object0) > x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object3) = x(Object1)
PATH-X-1-INV: [P3-X1] => x(Object3) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object3) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object3,Object1) leads to contradiction, so this direction is impossible.
Object4: impossible
P1: [p1] => Southwest(Object0,Object1)
P6: [p6] => Southeast(Object0,Object4)
A-REFUTE: Suppose North(Object4,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P6-X1: [P6] => x(Object0) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) = x(Object1)
PATH-X-1-INV: [P6-X1] => x(Object4) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object4) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object4,Object1) leads to contradiction, so this direction is impossible.
No eligible object can qualify, so there is no unique qualifying object.
Answer: E
```

## random-world-9-which-m0-all-possible
Split: train; case: no-match; answer: F

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: which North Object1
Eligible: Object0, Object2, Object3, Object4
Consider a map with multiple objects:
Object0 is to the Southwest of Object1.
Object4 is to the Northwest of Object2.
Object0 is to the Northeast of Object3.
Object3 is to the Southwest of Object4.
Object3 is to the West of Object2.
Object0 is to the Southeast of Object4.
Question: Select the complete set of eligible objects that could be north of Object1.
A. Object0
B. Object2
C. Object4
D. Object3
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
Object0: impossible
P1: From p1, Object0 is exactly southwest of Object1.
A-REFUTE: Suppose Object0 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is equal to Object1's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object0 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object2: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
P5: From p5, Object3 is exactly west of Object2.
A-REFUTE: Suppose Object2 is exactly north of Object1 for contradiction.
P1-Y1: From P1, Object0's y-coordinate is less than Object1's y-coordinate.
P3-Y1: From P3, Object0's y-coordinate is greater than Object3's y-coordinate.
P5-Y1: From P5, Object3's y-coordinate is equal to Object2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object1's y-coordinate.
PATH-Y-1-INV: From P5-Y1, Object2's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P3-Y1, Object3's y-coordinate is less than Object0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object2's y-coordinate is less than Object0's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, Object2's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption Object2 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object3: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
A-REFUTE: Suppose Object3 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P3-X1: From P3, Object0's x-coordinate is greater than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object3's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P3-X1, Object3's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object3's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object3 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object4: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P6: From p6, Object0 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P6-X1: From P6, Object0's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P6-X1, Object4's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object4's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object4 is exactly north of Object1 leads to contradiction, so this direction is impossible.
The complete possible set is {}. The empty set uses F.
Answer: F
```
### Symbolic
```text
Object0: impossible
P1: [p1] => Southwest(Object0,Object1)
A-REFUTE: Suppose North(Object0,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) = x(Object1)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(Object0,Object1) leads to contradiction, so this direction is impossible.
Object2: impossible
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
P5: [p5] => West(Object3,Object2)
A-REFUTE: Suppose North(Object2,Object1) for contradiction.
P1-Y1: [P1] => y(Object0) < y(Object1)
P3-Y1: [P3] => y(Object0) > y(Object3)
P5-Y1: [P5] => y(Object3) = y(Object2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object1)
PATH-Y-1-INV: [P5-Y1] => y(Object2) = y(Object3)
PATH-Y-2-INV: [P3-Y1] => y(Object3) < y(Object0)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object2) < y(Object0)
D-Y-2: [D-Y-1, P1-Y1] => y(Object2) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption North(Object2,Object1) leads to contradiction, so this direction is impossible.
Object3: impossible
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
A-REFUTE: Suppose North(Object3,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P3-X1: [P3] => x(Object0) > x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object3) = x(Object1)
PATH-X-1-INV: [P3-X1] => x(Object3) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object3) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object3,Object1) leads to contradiction, so this direction is impossible.
Object4: impossible
P1: [p1] => Southwest(Object0,Object1)
P6: [p6] => Southeast(Object0,Object4)
A-REFUTE: Suppose North(Object4,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P6-X1: [P6] => x(Object0) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) = x(Object1)
PATH-X-1-INV: [P6-X1] => x(Object4) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object4) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object4,Object1) leads to contradiction, so this direction is impossible.
The complete possible set is {}. The empty set uses F.
Answer: F
```

## random-world-9-which-m0-visible-possible
Split: train; case: no-match; answer: F

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: which North Object1
Eligible: Object0, Object2, Object3, Object4
Consider a map with multiple objects:
Object0 is to the Southwest of Object1.
Object4 is to the Northwest of Object2.
Object0 is to the Northeast of Object3.
Object3 is to the Southwest of Object4.
Object3 is to the West of Object2.
Object0 is to the Southeast of Object4.
Question: Which listed eligible objects could be north of Object1? Select all that apply.
A. Object0
B. Object2
C. Object4
D. Object3
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: impossible
P1: From p1, Object0 is exactly southwest of Object1.
A-REFUTE: Suppose Object0 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is equal to Object1's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object0 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object2: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
P5: From p5, Object3 is exactly west of Object2.
A-REFUTE: Suppose Object2 is exactly north of Object1 for contradiction.
P1-Y1: From P1, Object0's y-coordinate is less than Object1's y-coordinate.
P3-Y1: From P3, Object0's y-coordinate is greater than Object3's y-coordinate.
P5-Y1: From P5, Object3's y-coordinate is equal to Object2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object1's y-coordinate.
PATH-Y-1-INV: From P5-Y1, Object2's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P3-Y1, Object3's y-coordinate is less than Object0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object2's y-coordinate is less than Object0's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, Object2's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption Object2 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object3: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
A-REFUTE: Suppose Object3 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P3-X1: From P3, Object0's x-coordinate is greater than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object3's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P3-X1, Object3's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object3's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object3 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Object4: impossible
P1: From p1, Object0 is exactly southwest of Object1.
P6: From p6, Object0 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P6-X1: From P6, Object0's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P6-X1, Object4's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object4's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object4 is exactly north of Object1 leads to contradiction, so this direction is impossible.
The listed possible set is {}. The empty set uses F.
Answer: F
```
### Symbolic
```text
Object0: impossible
P1: [p1] => Southwest(Object0,Object1)
A-REFUTE: Suppose North(Object0,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) = x(Object1)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(Object0,Object1) leads to contradiction, so this direction is impossible.
Object2: impossible
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
P5: [p5] => West(Object3,Object2)
A-REFUTE: Suppose North(Object2,Object1) for contradiction.
P1-Y1: [P1] => y(Object0) < y(Object1)
P3-Y1: [P3] => y(Object0) > y(Object3)
P5-Y1: [P5] => y(Object3) = y(Object2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object1)
PATH-Y-1-INV: [P5-Y1] => y(Object2) = y(Object3)
PATH-Y-2-INV: [P3-Y1] => y(Object3) < y(Object0)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object2) < y(Object0)
D-Y-2: [D-Y-1, P1-Y1] => y(Object2) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption North(Object2,Object1) leads to contradiction, so this direction is impossible.
Object3: impossible
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
A-REFUTE: Suppose North(Object3,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P3-X1: [P3] => x(Object0) > x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object3) = x(Object1)
PATH-X-1-INV: [P3-X1] => x(Object3) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object3) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object3,Object1) leads to contradiction, so this direction is impossible.
Object4: impossible
P1: [p1] => Southwest(Object0,Object1)
P6: [p6] => Southeast(Object0,Object4)
A-REFUTE: Suppose North(Object4,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P6-X1: [P6] => x(Object0) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) = x(Object1)
PATH-X-1-INV: [P6-X1] => x(Object4) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object4) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object4,Object1) leads to contradiction, so this direction is impossible.
The listed possible set is {}. The empty set uses F.
Answer: F
```

## random-world-9-count-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: count North Object1
Eligible: Object0, Object2, Object3, Object4
Consider a map with multiple objects:
Object0 is to the Southwest of Object1.
Object4 is to the Northwest of Object2.
Object0 is to the Northeast of Object3.
Object3 is to the Southwest of Object4.
Object3 is to the West of Object2.
Object0 is to the Southeast of Object4.
Question: What exact number of eligible objects are north of Object1 in every valid map?
A. 0
B. 4
C. 2
D. 3
E. Cannot be determined
F. None of the above
```
### Natural
```text
P1: From p1, Object0 is exactly southwest of Object1.
A-REFUTE: Suppose Object0 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is equal to Object1's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object0 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
P5: From p5, Object3 is exactly west of Object2.
A-REFUTE: Suppose Object2 is exactly north of Object1 for contradiction.
P1-Y1: From P1, Object0's y-coordinate is less than Object1's y-coordinate.
P3-Y1: From P3, Object0's y-coordinate is greater than Object3's y-coordinate.
P5-Y1: From P5, Object3's y-coordinate is equal to Object2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object1's y-coordinate.
PATH-Y-1-INV: From P5-Y1, Object2's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P3-Y1, Object3's y-coordinate is less than Object0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object2's y-coordinate is less than Object0's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, Object2's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption Object2 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
A-REFUTE: Suppose Object3 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P3-X1: From P3, Object0's x-coordinate is greater than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object3's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P3-X1, Object3's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object3's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object3 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object3 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P6: From p6, Object0 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P6-X1: From P6, Object0's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P6-X1, Object4's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object4's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object4 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places Object0 at (2, 1); Object1 at (3, 3); Object2 at (2, 0); Object3 at (0, 0); Object4 at (1, 2). In this map, count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 0. It is displayed at A.
Answer: A
```
### Symbolic
```text
P1: [p1] => Southwest(Object0,Object1)
A-REFUTE: Suppose North(Object0,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) = x(Object1)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(Object0,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
P5: [p5] => West(Object3,Object2)
A-REFUTE: Suppose North(Object2,Object1) for contradiction.
P1-Y1: [P1] => y(Object0) < y(Object1)
P3-Y1: [P3] => y(Object0) > y(Object3)
P5-Y1: [P5] => y(Object3) = y(Object2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object1)
PATH-Y-1-INV: [P5-Y1] => y(Object2) = y(Object3)
PATH-Y-2-INV: [P3-Y1] => y(Object3) < y(Object0)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object2) < y(Object0)
D-Y-2: [D-Y-1, P1-Y1] => y(Object2) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption North(Object2,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
A-REFUTE: Suppose North(Object3,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P3-X1: [P3] => x(Object0) > x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object3) = x(Object1)
PATH-X-1-INV: [P3-X1] => x(Object3) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object3) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object3,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object3 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P6: [p6] => Southeast(Object0,Object4)
A-REFUTE: Suppose North(Object4,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P6-X1: [P6] => x(Object0) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) = x(Object1)
PATH-X-1-INV: [P6-X1] => x(Object4) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object4) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object4,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {Object0:(2,1), Object1:(3,3), Object2:(2,0), Object3:(0,0), Object4:(1,2)} => count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The uniquely guaranteed answer is 0. It is displayed at A.
Answer: A
```

## random-world-9-count-m0-all-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: count North Object1
Eligible: Object0, Object2, Object3, Object4
Consider a map with multiple objects:
Object0 is to the Southwest of Object1.
Object4 is to the Northwest of Object2.
Object0 is to the Northeast of Object3.
Object3 is to the Southwest of Object4.
Object3 is to the West of Object2.
Object0 is to the Southeast of Object4.
Question: Select the complete set of possible counts of eligible objects north of Object1.
A. 0
B. 4
C. 2
D. 3
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
P1: From p1, Object0 is exactly southwest of Object1.
A-REFUTE: Suppose Object0 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is equal to Object1's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object0 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
P5: From p5, Object3 is exactly west of Object2.
A-REFUTE: Suppose Object2 is exactly north of Object1 for contradiction.
P1-Y1: From P1, Object0's y-coordinate is less than Object1's y-coordinate.
P3-Y1: From P3, Object0's y-coordinate is greater than Object3's y-coordinate.
P5-Y1: From P5, Object3's y-coordinate is equal to Object2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object1's y-coordinate.
PATH-Y-1-INV: From P5-Y1, Object2's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P3-Y1, Object3's y-coordinate is less than Object0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object2's y-coordinate is less than Object0's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, Object2's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption Object2 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
A-REFUTE: Suppose Object3 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P3-X1: From P3, Object0's x-coordinate is greater than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object3's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P3-X1, Object3's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object3's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object3 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object3 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P6: From p6, Object0 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P6-X1: From P6, Object0's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P6-X1, Object4's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object4's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object4 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places Object0 at (2, 1); Object1 at (3, 3); Object2 at (2, 0); Object3 at (0, 0); Object4 at (1, 2). In this map, count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {0}.
Answer: A
```
### Symbolic
```text
P1: [p1] => Southwest(Object0,Object1)
A-REFUTE: Suppose North(Object0,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) = x(Object1)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(Object0,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
P5: [p5] => West(Object3,Object2)
A-REFUTE: Suppose North(Object2,Object1) for contradiction.
P1-Y1: [P1] => y(Object0) < y(Object1)
P3-Y1: [P3] => y(Object0) > y(Object3)
P5-Y1: [P5] => y(Object3) = y(Object2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object1)
PATH-Y-1-INV: [P5-Y1] => y(Object2) = y(Object3)
PATH-Y-2-INV: [P3-Y1] => y(Object3) < y(Object0)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object2) < y(Object0)
D-Y-2: [D-Y-1, P1-Y1] => y(Object2) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption North(Object2,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
A-REFUTE: Suppose North(Object3,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P3-X1: [P3] => x(Object0) > x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object3) = x(Object1)
PATH-X-1-INV: [P3-X1] => x(Object3) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object3) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object3,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object3 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P6: [p6] => Southeast(Object0,Object4)
A-REFUTE: Suppose North(Object4,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P6-X1: [P6] => x(Object0) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) = x(Object1)
PATH-X-1-INV: [P6-X1] => x(Object4) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object4) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object4,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {Object0:(2,1), Object1:(3,3), Object2:(2,0), Object3:(0,0), Object4:(1,2)} => count=0.
Count 1 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The complete possible set is {0}.
Answer: A
```

## random-world-9-count-m0-visible-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: count North Object1
Eligible: Object0, Object2, Object3, Object4
Consider a map with multiple objects:
Object0 is to the Southwest of Object1.
Object4 is to the Northwest of Object2.
Object0 is to the Northeast of Object3.
Object3 is to the Southwest of Object4.
Object3 is to the West of Object2.
Object0 is to the Southeast of Object4.
Question: Which listed counts are possible for the number of eligible objects north of Object1? Select all that apply.
A. 0
B. 4
C. 2
D. 3
E. Cannot be determined
F. None of the above
```
### Natural
```text
P1: From p1, Object0 is exactly southwest of Object1.
A-REFUTE: Suppose Object0 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is equal to Object1's x-coordinate.
refutation-C1: From under assumption refutation; P1-X1, refutation-X1, contradiction.
The assumption Object0 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
P5: From p5, Object3 is exactly west of Object2.
A-REFUTE: Suppose Object2 is exactly north of Object1 for contradiction.
P1-Y1: From P1, Object0's y-coordinate is less than Object1's y-coordinate.
P3-Y1: From P3, Object0's y-coordinate is greater than Object3's y-coordinate.
P5-Y1: From P5, Object3's y-coordinate is equal to Object2's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object2's y-coordinate is greater than Object1's y-coordinate.
PATH-Y-1-INV: From P5-Y1, Object2's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P3-Y1, Object3's y-coordinate is less than Object0's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object2's y-coordinate is less than Object0's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, Object2's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-2, refutation-Y1, contradiction.
The assumption Object2 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P3: From p3, Object0 is exactly northeast of Object3.
A-REFUTE: Suppose Object3 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P3-X1: From P3, Object0's x-coordinate is greater than Object3's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object3's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P3-X1, Object3's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object3's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object3 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object3 is impossible.
P1: From p1, Object0 is exactly southwest of Object1.
P6: From p6, Object0 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly north of Object1 for contradiction.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P6-X1: From P6, Object0's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is equal to Object1's x-coordinate.
PATH-X-1-INV: From P6-X1, Object4's x-coordinate is less than Object0's x-coordinate.
D-X-1: From PATH-X-1-INV, P1-X1, Object4's x-coordinate is less than Object1's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object4 is exactly north of Object1 leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
A valid map places Object0 at (2, 1); Object1 at (3, 3); Object2 at (2, 0); Object3 at (0, 0); Object4 at (1, 2). In this map, count=0.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The listed possible set is {0}.
Answer: A
```
### Symbolic
```text
P1: [p1] => Southwest(Object0,Object1)
A-REFUTE: Suppose North(Object0,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) = x(Object1)
refutation-C1: [under assumption refutation; P1-X1, refutation-X1] => contradiction
The assumption North(Object0,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object0 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
P5: [p5] => West(Object3,Object2)
A-REFUTE: Suppose North(Object2,Object1) for contradiction.
P1-Y1: [P1] => y(Object0) < y(Object1)
P3-Y1: [P3] => y(Object0) > y(Object3)
P5-Y1: [P5] => y(Object3) = y(Object2)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object2) > y(Object1)
PATH-Y-1-INV: [P5-Y1] => y(Object2) = y(Object3)
PATH-Y-2-INV: [P3-Y1] => y(Object3) < y(Object0)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object2) < y(Object0)
D-Y-2: [D-Y-1, P1-Y1] => y(Object2) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-2, refutation-Y1] => contradiction
The assumption North(Object2,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object2 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P3: [p3] => Northeast(Object0,Object3)
A-REFUTE: Suppose North(Object3,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P3-X1: [P3] => x(Object0) > x(Object3)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object3) = x(Object1)
PATH-X-1-INV: [P3-X1] => x(Object3) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object3) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object3,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object3 is impossible.
P1: [p1] => Southwest(Object0,Object1)
P6: [p6] => Southeast(Object0,Object4)
A-REFUTE: Suppose North(Object4,Object1) for contradiction.
P1-X1: [P1] => x(Object0) < x(Object1)
P6-X1: [P6] => x(Object0) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) = x(Object1)
PATH-X-1-INV: [P6-X1] => x(Object4) < x(Object0)
D-X-1: [PATH-X-1-INV, P1-X1] => x(Object4) < x(Object1)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption North(Object4,Object1) leads to contradiction, so this direction is impossible.
Fixed membership: Object4 is impossible.
Every joint membership assignment must include all fixed matches, exclude all fixed nonmatches, and contain no incompatible pair.
Valid map {Object0:(2,1), Object1:(3,3), Object2:(2,0), Object3:(0,0), Object4:(1,2)} => count=0.
Count 4 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 2 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
Count 3 is impossible: every eligible-object subset of that size violates at least one of the established membership constraints.
The listed possible set is {0}.
Answer: A
```

## random-world-19-which-m0-single
Split: test; case: multiple-certain-match; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: which Southwest Object4
Eligible: Object0, Object1, Object2, Object3
Consider a map with multiple objects:
Object2 is to the South of Object3.
Object4 is to the Northeast of Object0.
Object3 is to the West of Object4.
Object2 is to the Southwest of Object4.
Object0 is to the Southwest of Object1.
Object1 is to the Northeast of Object3.
Question: Which eligible object is the only one southwest of Object4 in every valid map?
A. Object0
B. Object2
C. Object1
D. Object3
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: entailed
P2: From p2, Object4 is exactly northeast of Object0.
P2-X1: From P2, Object4's x-coordinate is greater than Object0's x-coordinate.
P2-Y1: From P2, Object4's y-coordinate is greater than Object0's y-coordinate.
Q-DIR: From P2-X1, P2-Y1, Object0 is exactly southwest of Object4.
Object1: impossible
P3: From p3, Object3 is exactly west of Object4.
P6: From p6, Object1 is exactly northeast of Object3.
A-REFUTE: Suppose Object1 is exactly southwest of Object4 for contradiction.
P3-Y1: From P3, Object3's y-coordinate is equal to Object4's y-coordinate.
P6-Y1: From P6, Object1's y-coordinate is greater than Object3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object1's y-coordinate is less than Object4's y-coordinate.
PATH-Y-1-INV: From P3-Y1, Object4's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P6-Y1, Object3's y-coordinate is less than Object1's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object4's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption Object1 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
Object2: entailed
P4: From p4, Object2 is exactly southwest of Object4.
Q-DIR: From P4, Object2 is exactly southwest of Object4.
Object3: impossible
P3: From p3, Object3 is exactly west of Object4.
A-REFUTE: Suppose Object3 is exactly southwest of Object4 for contradiction.
P3-Y1: From P3, Object3's y-coordinate is equal to Object4's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object3's y-coordinate is less than Object4's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption Object3 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
Multiple objects always qualify, so there is no sole qualifying object.
Answer: E
```
### Symbolic
```text
Object0: entailed
P2: [p2] => Northeast(Object4,Object0)
P2-X1: [P2] => x(Object4) > x(Object0)
P2-Y1: [P2] => y(Object4) > y(Object0)
Q-DIR: [P2-X1, P2-Y1] => Southwest(Object0,Object4)
Object1: impossible
P3: [p3] => West(Object3,Object4)
P6: [p6] => Northeast(Object1,Object3)
A-REFUTE: Suppose Southwest(Object1,Object4) for contradiction.
P3-Y1: [P3] => y(Object3) = y(Object4)
P6-Y1: [P6] => y(Object1) > y(Object3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object1) < y(Object4)
PATH-Y-1-INV: [P3-Y1] => y(Object4) = y(Object3)
PATH-Y-2-INV: [P6-Y1] => y(Object3) < y(Object1)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object4) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southwest(Object1,Object4) leads to contradiction, so this direction is impossible.
Object2: entailed
P4: [p4] => Southwest(Object2,Object4)
Q-DIR: [P4] => Southwest(Object2,Object4)
Object3: impossible
P3: [p3] => West(Object3,Object4)
A-REFUTE: Suppose Southwest(Object3,Object4) for contradiction.
P3-Y1: [P3] => y(Object3) = y(Object4)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object3) < y(Object4)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption Southwest(Object3,Object4) leads to contradiction, so this direction is impossible.
Multiple objects always qualify, so there is no sole qualifying object.
Answer: E
```

## random-world-19-which-m0-all-possible
Split: test; case: multiple-certain-match; answer: AB

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: which Southwest Object4
Eligible: Object0, Object1, Object2, Object3
Consider a map with multiple objects:
Object2 is to the South of Object3.
Object4 is to the Northeast of Object0.
Object3 is to the West of Object4.
Object2 is to the Southwest of Object4.
Object0 is to the Southwest of Object1.
Object1 is to the Northeast of Object3.
Question: Select the complete set of eligible objects that could be southwest of Object4.
A. Object0
B. Object2
C. Object1
D. Object3
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
Object0: entailed
P2: From p2, Object4 is exactly northeast of Object0.
P2-X1: From P2, Object4's x-coordinate is greater than Object0's x-coordinate.
P2-Y1: From P2, Object4's y-coordinate is greater than Object0's y-coordinate.
Q-DIR: From P2-X1, P2-Y1, Object0 is exactly southwest of Object4.
Object1: impossible
P3: From p3, Object3 is exactly west of Object4.
P6: From p6, Object1 is exactly northeast of Object3.
A-REFUTE: Suppose Object1 is exactly southwest of Object4 for contradiction.
P3-Y1: From P3, Object3's y-coordinate is equal to Object4's y-coordinate.
P6-Y1: From P6, Object1's y-coordinate is greater than Object3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object1's y-coordinate is less than Object4's y-coordinate.
PATH-Y-1-INV: From P3-Y1, Object4's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P6-Y1, Object3's y-coordinate is less than Object1's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object4's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption Object1 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
Object2: entailed
P4: From p4, Object2 is exactly southwest of Object4.
Q-DIR: From P4, Object2 is exactly southwest of Object4.
Object3: impossible
P3: From p3, Object3 is exactly west of Object4.
A-REFUTE: Suppose Object3 is exactly southwest of Object4 for contradiction.
P3-Y1: From P3, Object3's y-coordinate is equal to Object4's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object3's y-coordinate is less than Object4's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption Object3 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
The complete possible set is {Object0, Object2}.
Answer: AB
```
### Symbolic
```text
Object0: entailed
P2: [p2] => Northeast(Object4,Object0)
P2-X1: [P2] => x(Object4) > x(Object0)
P2-Y1: [P2] => y(Object4) > y(Object0)
Q-DIR: [P2-X1, P2-Y1] => Southwest(Object0,Object4)
Object1: impossible
P3: [p3] => West(Object3,Object4)
P6: [p6] => Northeast(Object1,Object3)
A-REFUTE: Suppose Southwest(Object1,Object4) for contradiction.
P3-Y1: [P3] => y(Object3) = y(Object4)
P6-Y1: [P6] => y(Object1) > y(Object3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object1) < y(Object4)
PATH-Y-1-INV: [P3-Y1] => y(Object4) = y(Object3)
PATH-Y-2-INV: [P6-Y1] => y(Object3) < y(Object1)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object4) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southwest(Object1,Object4) leads to contradiction, so this direction is impossible.
Object2: entailed
P4: [p4] => Southwest(Object2,Object4)
Q-DIR: [P4] => Southwest(Object2,Object4)
Object3: impossible
P3: [p3] => West(Object3,Object4)
A-REFUTE: Suppose Southwest(Object3,Object4) for contradiction.
P3-Y1: [P3] => y(Object3) = y(Object4)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object3) < y(Object4)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption Southwest(Object3,Object4) leads to contradiction, so this direction is impossible.
The complete possible set is {Object0, Object2}.
Answer: AB
```

## random-world-19-which-m0-visible-possible
Split: test; case: multiple-certain-match; answer: AB

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4
Query: which Southwest Object4
Eligible: Object0, Object1, Object2, Object3
Consider a map with multiple objects:
Object2 is to the South of Object3.
Object4 is to the Northeast of Object0.
Object3 is to the West of Object4.
Object2 is to the Southwest of Object4.
Object0 is to the Southwest of Object1.
Object1 is to the Northeast of Object3.
Question: Which listed eligible objects could be southwest of Object4? Select all that apply.
A. Object0
B. Object2
C. Object1
D. Object3
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: entailed
P2: From p2, Object4 is exactly northeast of Object0.
P2-X1: From P2, Object4's x-coordinate is greater than Object0's x-coordinate.
P2-Y1: From P2, Object4's y-coordinate is greater than Object0's y-coordinate.
Q-DIR: From P2-X1, P2-Y1, Object0 is exactly southwest of Object4.
Object1: impossible
P3: From p3, Object3 is exactly west of Object4.
P6: From p6, Object1 is exactly northeast of Object3.
A-REFUTE: Suppose Object1 is exactly southwest of Object4 for contradiction.
P3-Y1: From P3, Object3's y-coordinate is equal to Object4's y-coordinate.
P6-Y1: From P6, Object1's y-coordinate is greater than Object3's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object1's y-coordinate is less than Object4's y-coordinate.
PATH-Y-1-INV: From P3-Y1, Object4's y-coordinate is equal to Object3's y-coordinate.
PATH-Y-2-INV: From P6-Y1, Object3's y-coordinate is less than Object1's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object4's y-coordinate is less than Object1's y-coordinate.
refutation-C1: From under assumption refutation; D-Y-1, refutation-Y1, contradiction.
The assumption Object1 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
Object2: entailed
P4: From p4, Object2 is exactly southwest of Object4.
Q-DIR: From P4, Object2 is exactly southwest of Object4.
Object3: impossible
P3: From p3, Object3 is exactly west of Object4.
A-REFUTE: Suppose Object3 is exactly southwest of Object4 for contradiction.
P3-Y1: From P3, Object3's y-coordinate is equal to Object4's y-coordinate.
refutation-Y1: From under assumption refutation; A-REFUTE, Object3's y-coordinate is less than Object4's y-coordinate.
refutation-C1: From under assumption refutation; P3-Y1, refutation-Y1, contradiction.
The assumption Object3 is exactly southwest of Object4 leads to contradiction, so this direction is impossible.
The listed possible set is {Object0, Object2}.
Answer: AB
```
### Symbolic
```text
Object0: entailed
P2: [p2] => Northeast(Object4,Object0)
P2-X1: [P2] => x(Object4) > x(Object0)
P2-Y1: [P2] => y(Object4) > y(Object0)
Q-DIR: [P2-X1, P2-Y1] => Southwest(Object0,Object4)
Object1: impossible
P3: [p3] => West(Object3,Object4)
P6: [p6] => Northeast(Object1,Object3)
A-REFUTE: Suppose Southwest(Object1,Object4) for contradiction.
P3-Y1: [P3] => y(Object3) = y(Object4)
P6-Y1: [P6] => y(Object1) > y(Object3)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object1) < y(Object4)
PATH-Y-1-INV: [P3-Y1] => y(Object4) = y(Object3)
PATH-Y-2-INV: [P6-Y1] => y(Object3) < y(Object1)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object4) < y(Object1)
refutation-C1: [under assumption refutation; D-Y-1, refutation-Y1] => contradiction
The assumption Southwest(Object1,Object4) leads to contradiction, so this direction is impossible.
Object2: entailed
P4: [p4] => Southwest(Object2,Object4)
Q-DIR: [P4] => Southwest(Object2,Object4)
Object3: impossible
P3: [p3] => West(Object3,Object4)
A-REFUTE: Suppose Southwest(Object3,Object4) for contradiction.
P3-Y1: [P3] => y(Object3) = y(Object4)
refutation-Y1: [under assumption refutation; A-REFUTE] => y(Object3) < y(Object4)
refutation-C1: [under assumption refutation; P3-Y1, refutation-Y1] => contradiction
The assumption Southwest(Object3,Object4) leads to contradiction, so this direction is impossible.
The listed possible set is {Object0, Object2}.
Answer: AB
```

## random-world-57-which-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: which Northeast Object3
Eligible: Object0, Object1, Object2, Object4, Object5
Consider a map with multiple objects:
Object0 is to the South of Object1.
Object3 is to the Southwest of Object2.
Object4 is to the South of Object1.
Object5 is to the South of Object1.
Object1 is to the Northwest of Object2.
Object3 is to the Southeast of Object4.
Question: Which eligible object is the only one northeast of Object3 in every valid map?
A. Object2
B. Object0
C. Object4
D. Object1
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: impossible
P1: From p1, Object0 is exactly south of Object1.
P3: From p3, Object4 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object0 is exactly northeast of Object3 for contradiction.
P1-X1: From P1, Object0's x-coordinate is equal to Object1's x-coordinate.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is greater than Object3's x-coordinate.
PATH-X-2-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-3-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From P1-X1, PATH-X-2-INV, Object0's x-coordinate is equal to Object4's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object0's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption Object0 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object1: impossible
P3: From p3, Object4 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object1 is exactly northeast of Object3 for contradiction.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object1's x-coordinate is greater than Object3's x-coordinate.
PATH-X-1-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-2-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, Object1's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object1 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object2: entailed
P2: From p2, Object3 is exactly southwest of Object2.
P2-X1: From P2, Object3's x-coordinate is less than Object2's x-coordinate.
P2-Y1: From P2, Object3's y-coordinate is less than Object2's y-coordinate.
Q-DIR: From P2-X1, P2-Y1, Object2 is exactly northeast of Object3.
Object4: impossible
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly northeast of Object3 for contradiction.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P6-X1, refutation-X1, contradiction.
The assumption Object4 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object5: impossible
P3: From p3, Object4 is exactly south of Object1.
P4: From p4, Object5 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object5 is exactly northeast of Object3 for contradiction.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P4-X1: From P4, Object5's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object5's x-coordinate is greater than Object3's x-coordinate.
PATH-X-2-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-3-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From P4-X1, PATH-X-2-INV, Object5's x-coordinate is equal to Object4's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object5's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption Object5 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Object2. It is displayed at A.
Answer: A
```
### Symbolic
```text
Object0: impossible
P1: [p1] => South(Object0,Object1)
P3: [p3] => South(Object4,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object0,Object3) for contradiction.
P1-X1: [P1] => x(Object0) = x(Object1)
P3-X1: [P3] => x(Object4) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) > x(Object3)
PATH-X-2-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-3-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [P1-X1, PATH-X-2-INV] => x(Object0) = x(Object4)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object0) < x(Object3)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northeast(Object0,Object3) leads to contradiction, so this direction is impossible.
Object1: impossible
P3: [p3] => South(Object4,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object1,Object3) for contradiction.
P3-X1: [P3] => x(Object4) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object1) > x(Object3)
PATH-X-1-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-2-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(Object1) < x(Object3)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northeast(Object1,Object3) leads to contradiction, so this direction is impossible.
Object2: entailed
P2: [p2] => Southwest(Object3,Object2)
P2-X1: [P2] => x(Object3) < x(Object2)
P2-Y1: [P2] => y(Object3) < y(Object2)
Q-DIR: [P2-X1, P2-Y1] => Northeast(Object2,Object3)
Object4: impossible
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object4,Object3) for contradiction.
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) > x(Object3)
refutation-C1: [under assumption refutation; P6-X1, refutation-X1] => contradiction
The assumption Northeast(Object4,Object3) leads to contradiction, so this direction is impossible.
Object5: impossible
P3: [p3] => South(Object4,Object1)
P4: [p4] => South(Object5,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object5,Object3) for contradiction.
P3-X1: [P3] => x(Object4) = x(Object1)
P4-X1: [P4] => x(Object5) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object5) > x(Object3)
PATH-X-2-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-3-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [P4-X1, PATH-X-2-INV] => x(Object5) = x(Object4)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object5) < x(Object3)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northeast(Object5,Object3) leads to contradiction, so this direction is impossible.
The uniquely guaranteed answer is Object2. It is displayed at A.
Answer: A
```

## random-world-57-which-m0-all-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: which Northeast Object3
Eligible: Object0, Object1, Object2, Object4, Object5
Consider a map with multiple objects:
Object0 is to the South of Object1.
Object3 is to the Southwest of Object2.
Object4 is to the South of Object1.
Object5 is to the South of Object1.
Object1 is to the Northwest of Object2.
Object3 is to the Southeast of Object4.
Question: Select the complete set of eligible objects that could be northeast of Object3.
A. Object2
B. Object0
C. Object4
D. Object1
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
Object0: impossible
P1: From p1, Object0 is exactly south of Object1.
P3: From p3, Object4 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object0 is exactly northeast of Object3 for contradiction.
P1-X1: From P1, Object0's x-coordinate is equal to Object1's x-coordinate.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is greater than Object3's x-coordinate.
PATH-X-2-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-3-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From P1-X1, PATH-X-2-INV, Object0's x-coordinate is equal to Object4's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object0's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption Object0 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object1: impossible
P3: From p3, Object4 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object1 is exactly northeast of Object3 for contradiction.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object1's x-coordinate is greater than Object3's x-coordinate.
PATH-X-1-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-2-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, Object1's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object1 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object2: entailed
P2: From p2, Object3 is exactly southwest of Object2.
P2-X1: From P2, Object3's x-coordinate is less than Object2's x-coordinate.
P2-Y1: From P2, Object3's y-coordinate is less than Object2's y-coordinate.
Q-DIR: From P2-X1, P2-Y1, Object2 is exactly northeast of Object3.
Object4: impossible
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly northeast of Object3 for contradiction.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P6-X1, refutation-X1, contradiction.
The assumption Object4 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object5: impossible
P3: From p3, Object4 is exactly south of Object1.
P4: From p4, Object5 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object5 is exactly northeast of Object3 for contradiction.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P4-X1: From P4, Object5's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object5's x-coordinate is greater than Object3's x-coordinate.
PATH-X-2-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-3-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From P4-X1, PATH-X-2-INV, Object5's x-coordinate is equal to Object4's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object5's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption Object5 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
The complete possible set is {Object2}.
Answer: A
```
### Symbolic
```text
Object0: impossible
P1: [p1] => South(Object0,Object1)
P3: [p3] => South(Object4,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object0,Object3) for contradiction.
P1-X1: [P1] => x(Object0) = x(Object1)
P3-X1: [P3] => x(Object4) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) > x(Object3)
PATH-X-2-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-3-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [P1-X1, PATH-X-2-INV] => x(Object0) = x(Object4)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object0) < x(Object3)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northeast(Object0,Object3) leads to contradiction, so this direction is impossible.
Object1: impossible
P3: [p3] => South(Object4,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object1,Object3) for contradiction.
P3-X1: [P3] => x(Object4) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object1) > x(Object3)
PATH-X-1-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-2-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(Object1) < x(Object3)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northeast(Object1,Object3) leads to contradiction, so this direction is impossible.
Object2: entailed
P2: [p2] => Southwest(Object3,Object2)
P2-X1: [P2] => x(Object3) < x(Object2)
P2-Y1: [P2] => y(Object3) < y(Object2)
Q-DIR: [P2-X1, P2-Y1] => Northeast(Object2,Object3)
Object4: impossible
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object4,Object3) for contradiction.
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) > x(Object3)
refutation-C1: [under assumption refutation; P6-X1, refutation-X1] => contradiction
The assumption Northeast(Object4,Object3) leads to contradiction, so this direction is impossible.
Object5: impossible
P3: [p3] => South(Object4,Object1)
P4: [p4] => South(Object5,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object5,Object3) for contradiction.
P3-X1: [P3] => x(Object4) = x(Object1)
P4-X1: [P4] => x(Object5) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object5) > x(Object3)
PATH-X-2-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-3-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [P4-X1, PATH-X-2-INV] => x(Object5) = x(Object4)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object5) < x(Object3)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northeast(Object5,Object3) leads to contradiction, so this direction is impossible.
The complete possible set is {Object2}.
Answer: A
```

## random-world-57-which-m0-visible-possible
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: which Northeast Object3
Eligible: Object0, Object1, Object2, Object4, Object5
Consider a map with multiple objects:
Object0 is to the South of Object1.
Object3 is to the Southwest of Object2.
Object4 is to the South of Object1.
Object5 is to the South of Object1.
Object1 is to the Northwest of Object2.
Object3 is to the Southeast of Object4.
Question: Which listed eligible objects could be northeast of Object3? Select all that apply.
A. Object2
B. Object0
C. Object4
D. Object1
E. Cannot be determined
F. None of the above
```
### Natural
```text
Object0: impossible
P1: From p1, Object0 is exactly south of Object1.
P3: From p3, Object4 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object0 is exactly northeast of Object3 for contradiction.
P1-X1: From P1, Object0's x-coordinate is equal to Object1's x-coordinate.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object0's x-coordinate is greater than Object3's x-coordinate.
PATH-X-2-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-3-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From P1-X1, PATH-X-2-INV, Object0's x-coordinate is equal to Object4's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object0's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-2, refutation-X1, contradiction.
The assumption Object0 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object1: impossible
P3: From p3, Object4 is exactly south of Object1.
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object1 is exactly northeast of Object3 for contradiction.
P3-X1: From P3, Object4's x-coordinate is equal to Object1's x-coordinate.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object1's x-coordinate is greater than Object3's x-coordinate.
PATH-X-1-INV: From P3-X1, Object1's x-coordinate is equal to Object4's x-coordinate.
PATH-X-2-INV: From P6-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, Object1's x-coordinate is less than Object3's x-coordinate.
refutation-C1: From under assumption refutation; D-X-1, refutation-X1, contradiction.
The assumption Object1 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
Object2: entailed
P2: From p2, Object3 is exactly southwest of Object2.
P2-X1: From P2, Object3's x-coordinate is less than Object2's x-coordinate.
P2-Y1: From P2, Object3's y-coordinate is less than Object2's y-coordinate.
Q-DIR: From P2-X1, P2-Y1, Object2 is exactly northeast of Object3.
Object4: impossible
P6: From p6, Object3 is exactly southeast of Object4.
A-REFUTE: Suppose Object4 is exactly northeast of Object3 for contradiction.
P6-X1: From P6, Object3's x-coordinate is greater than Object4's x-coordinate.
refutation-X1: From under assumption refutation; A-REFUTE, Object4's x-coordinate is greater than Object3's x-coordinate.
refutation-C1: From under assumption refutation; P6-X1, refutation-X1, contradiction.
The assumption Object4 is exactly northeast of Object3 leads to contradiction, so this direction is impossible.
The listed possible set is {Object2}.
Answer: A
```
### Symbolic
```text
Object0: impossible
P1: [p1] => South(Object0,Object1)
P3: [p3] => South(Object4,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object0,Object3) for contradiction.
P1-X1: [P1] => x(Object0) = x(Object1)
P3-X1: [P3] => x(Object4) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object0) > x(Object3)
PATH-X-2-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-3-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [P1-X1, PATH-X-2-INV] => x(Object0) = x(Object4)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object0) < x(Object3)
refutation-C1: [under assumption refutation; D-X-2, refutation-X1] => contradiction
The assumption Northeast(Object0,Object3) leads to contradiction, so this direction is impossible.
Object1: impossible
P3: [p3] => South(Object4,Object1)
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object1,Object3) for contradiction.
P3-X1: [P3] => x(Object4) = x(Object1)
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object1) > x(Object3)
PATH-X-1-INV: [P3-X1] => x(Object1) = x(Object4)
PATH-X-2-INV: [P6-X1] => x(Object4) < x(Object3)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(Object1) < x(Object3)
refutation-C1: [under assumption refutation; D-X-1, refutation-X1] => contradiction
The assumption Northeast(Object1,Object3) leads to contradiction, so this direction is impossible.
Object2: entailed
P2: [p2] => Southwest(Object3,Object2)
P2-X1: [P2] => x(Object3) < x(Object2)
P2-Y1: [P2] => y(Object3) < y(Object2)
Q-DIR: [P2-X1, P2-Y1] => Northeast(Object2,Object3)
Object4: impossible
P6: [p6] => Southeast(Object3,Object4)
A-REFUTE: Suppose Northeast(Object4,Object3) for contradiction.
P6-X1: [P6] => x(Object3) > x(Object4)
refutation-X1: [under assumption refutation; A-REFUTE] => x(Object4) > x(Object3)
refutation-C1: [under assumption refutation; P6-X1, refutation-X1] => contradiction
The assumption Northeast(Object4,Object3) leads to contradiction, so this direction is impossible.
The listed possible set is {Object2}.
Answer: A
```

## random-atoms-0-direction-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Query: direction Object2 Object3
Consider a map with multiple objects:
Object5 is to the South of Object1.
Object2 is to the North of Object4.
Object3 is to the Northwest of Object4.
Question: In which direction is Object2 relative to Object3 in every valid map?
A. Southeast
B. East
C. Northeast
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
Northeast:
A valid map places Object0 at (1, 6); Object1 at (1, 5); Object2 at (1, 4); Object3 at (0, 1); Object4 at (1, 0); Object5 at (1, 3); Object6 at (1, 2). In this map, direction=Northeast.
East:
A valid map places Object0 at (1, 5); Object1 at (1, 4); Object2 at (1, 3); Object3 at (0, 3); Object4 at (1, 2); Object5 at (1, 1); Object6 at (1, 0). In this map, direction=East.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
Northeast:
Valid map {Object0:(1,6), Object1:(1,5), Object2:(1,4), Object3:(0,1), Object4:(1,0), Object5:(1,3), Object6:(1,2)} => direction=Northeast.
East:
Valid map {Object0:(1,5), Object1:(1,4), Object2:(1,3), Object3:(0,3), Object4:(1,2), Object5:(1,1), Object6:(1,0)} => direction=East.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## random-atoms-1-direction-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Query: direction Object4 Object0
Consider a map with multiple objects:
Object0 is to the West of Object2.
Object5 is to the West of Object0.
Object4 is to the South of Object2.
Object1 is to the Northeast of Object3.
Object2 is to the Southeast of Object6.
Question: In which direction is Object4 relative to Object0 in every valid map?
A. Southeast
B. Southwest
C. North
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
Southeast:
P1: From p1, Object0 is exactly west of Object2.
P3: From p3, Object4 is exactly south of Object2.
P1-X1: From P1, Object0's x-coordinate is less than Object2's x-coordinate.
P1-Y1: From P1, Object0's y-coordinate is equal to Object2's y-coordinate.
P3-X1: From P3, Object4's x-coordinate is equal to Object2's x-coordinate.
P3-Y1: From P3, Object4's y-coordinate is less than Object2's y-coordinate.
PATH-X-2-INV: From P3-X1, Object2's x-coordinate is equal to Object4's x-coordinate.
D-X-1: From P1-X1, PATH-X-2-INV, Object0's x-coordinate is less than Object4's x-coordinate.
PATH-Y-2-INV: From P1-Y1, Object2's y-coordinate is equal to Object0's y-coordinate.
D-Y-1: From P3-Y1, PATH-Y-2-INV, Object4's y-coordinate is less than Object0's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, Object4 is exactly southeast of Object0.
The uniquely guaranteed answer is Southeast. It is displayed at A.
Answer: A
```
### Symbolic
```text
Southeast:
P1: [p1] => West(Object0,Object2)
P3: [p3] => South(Object4,Object2)
P1-X1: [P1] => x(Object0) < x(Object2)
P1-Y1: [P1] => y(Object0) = y(Object2)
P3-X1: [P3] => x(Object4) = x(Object2)
P3-Y1: [P3] => y(Object4) < y(Object2)
PATH-X-2-INV: [P3-X1] => x(Object2) = x(Object4)
D-X-1: [P1-X1, PATH-X-2-INV] => x(Object0) < x(Object4)
PATH-Y-2-INV: [P1-Y1] => y(Object2) = y(Object0)
D-Y-1: [P3-Y1, PATH-Y-2-INV] => y(Object4) < y(Object0)
Q-DIR: [D-X-1, D-Y-1] => Southeast(Object4,Object0)
The uniquely guaranteed answer is Southeast. It is displayed at A.
Answer: A
```

## random-atoms-2-direction-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: direction Object2 Object5
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: In which direction is Object2 relative to Object5 in every valid map?
A. East
B. Southwest
C. West
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-direction-m0-all-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: direction Object2 Object5
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: Select the complete set of directions Object2 could occupy relative to Object5.
A. East
B. Southwest
C. West
D. Northeast
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-direction-m0-visible-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: direction Object2 Object5
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: Which listed directions could Object2 occupy relative to Object5? Select all that apply.
A. East
B. Southwest
C. West
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-which-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: which Southeast Object5
Eligible: Object0, Object1, Object2, Object3, Object4
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: Which eligible object is the only one southeast of Object5 in every valid map?
A. Object3
B. Object1
C. Object2
D. Object4
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-which-m0-all-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: which Southeast Object5
Eligible: Object0, Object1, Object2, Object3, Object4
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: Select the complete set of eligible objects that could be southeast of Object5.
A. Object3
B. Object1
C. Object2
D. Object4
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-which-m0-visible-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: which Southeast Object5
Eligible: Object0, Object1, Object2, Object3, Object4
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: Which listed eligible objects could be southeast of Object5? Select all that apply.
A. Object3
B. Object1
C. Object2
D. Object4
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-count-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: count Southeast Object5
Eligible: Object0, Object1, Object2, Object3, Object4
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: What exact number of eligible objects are southeast of Object5 in every valid map?
A. 4
B. 2
C. 5
D. 0
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-count-m0-all-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: all-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values only if they represent the complete nonempty possible set over the declared domain. Select F if any possible value is missing or the complete set is empty. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: count Southeast Object5
Eligible: Object0, Object1, Object2, Object3, Object4
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: Select the complete set of possible counts of eligible objects southeast of Object5.
A. 4
B. 2
C. 5
D. 0
E. Cannot be determined
F. No complete answer can be formed from A-D.
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-2-count-m0-visible-possible
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: visible-possible
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select all A-D values possible in at least one valid map. Select F if no listed value is possible. Multiple possibilities do not mean E.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: count Southeast Object5
Eligible: Object0, Object1, Object2, Object3, Object4
Consider a map with multiple objects:
Object5 is to the Southeast of Object2.
Object0 is to the Southwest of Object2.
Object1 is to the West of Object5.
Object2 is to the West of Object1.
Object0 is to the Southwest of Object1.
Object4 is to the Southeast of Object1.
Object3 is to the South of Object5.
Question: Which listed counts are possible for the number of eligible objects southeast of Object5? Select all that apply.
A. 4
B. 2
C. 5
D. 0
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Object5's y-coordinate is less than Object2's y-coordinate.
From p4, Object2's y-coordinate is equal to Object1's y-coordinate.
From p3, Object1's y-coordinate is equal to Object5's y-coordinate.
Composing this path gives Object5's y-coordinate less than Object5's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: y(Object5) < y(Object2)
p4: y(Object2) = y(Object1)
p3: y(Object1) = y(Object5)
Transitivity: y(Object5) < y(Object5)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-3-direction-m0-single
Split: test; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: direction Object5 Object2
Consider a map with multiple objects:
Object0 is to the Northwest of Object2.
Object1 is to the East of Object3.
Object3 is to the Northeast of Object2.
Object0 is to the Southeast of Object4.
Object3 is to the North of Object4.
Object4 is to the West of Object5.
Question: In which direction is Object5 relative to Object2 in every valid map?
A. West
B. Southwest
C. South
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p1, Object0's x-coordinate is less than Object2's x-coordinate.
From p3, Object2's x-coordinate is less than Object3's x-coordinate.
From p5, Object3's x-coordinate is equal to Object4's x-coordinate.
From p4, Object4's x-coordinate is less than Object0's x-coordinate.
Composing this path gives Object0's x-coordinate less than Object0's x-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p1: x(Object0) < x(Object2)
p3: x(Object2) < x(Object3)
p5: x(Object3) = x(Object4)
p4: x(Object4) < x(Object0)
Transitivity: x(Object0) < x(Object0)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p3, p4, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## random-atoms-4-direction-m0-single
Split: dev; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: direction Object2 Object1
Consider a map with multiple objects:
Object0 is to the Southeast of Object1.
Object0 is to the South of Object3.
Object4 is to the South of Object2.
Question: In which direction is Object2 relative to Object1 in every valid map?
A. North
B. West
C. Southwest
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object0 at (1, 2); Object1 at (0, 3); Object2 at (0, 4); Object3 at (1, 5); Object4 at (0, 0); Object5 at (1, 1). In this map, direction=North.
Northeast:
A valid map places Object0 at (2, 2); Object1 at (0, 3); Object2 at (1, 4); Object3 at (2, 5); Object4 at (1, 0); Object5 at (2, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object0:(1,2), Object1:(0,3), Object2:(0,4), Object3:(1,5), Object4:(0,0), Object5:(1,1)} => direction=North.
Northeast:
Valid map {Object0:(2,2), Object1:(0,3), Object2:(1,4), Object3:(2,5), Object4:(1,0), Object5:(2,1)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

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
Objects: Object3, Object0, Object2, Object5, Object1, Object4
Query: direction Object0 Object3
Consider a map with multiple objects:
Object1 is to the East of Object3.
Object3 is to the West of Object5.
Object0 is to the Northwest of Object4.
Object3 is to the East of Object4.
Object3 is to the West of Object2.
Question: In which direction is Object0 relative to Object3 in every valid map?
A. Northeast
B. Southeast
C. Northwest
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
Northwest:
P3: From p3, Object0 is exactly northwest of Object4.
P4: From p4, Object3 is exactly east of Object4.
P3-X1: From P3, Object0's x-coordinate is less than Object4's x-coordinate.
P3-Y1: From P3, Object0's y-coordinate is greater than Object4's y-coordinate.
P4-X1: From P4, Object3's x-coordinate is greater than Object4's x-coordinate.
P4-Y1: From P4, Object3's y-coordinate is equal to Object4's y-coordinate.
PATH-X-2-INV: From P4-X1, Object4's x-coordinate is less than Object3's x-coordinate.
D-X-1: From P3-X1, PATH-X-2-INV, Object0's x-coordinate is less than Object3's x-coordinate.
PATH-Y-2-INV: From P3-Y1, Object4's y-coordinate is less than Object0's y-coordinate.
D-Y-1: From P4-Y1, PATH-Y-2-INV, Object3's y-coordinate is less than Object0's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, Object0 is exactly northwest of Object3.
The uniquely guaranteed answer is Northwest. It is displayed at C.
Answer: C
```
### Symbolic
```text
Northwest:
P3: [p3] => Northwest(Object0,Object4)
P4: [p4] => East(Object3,Object4)
P3-X1: [P3] => x(Object0) < x(Object4)
P3-Y1: [P3] => y(Object0) > y(Object4)
P4-X1: [P4] => x(Object3) > x(Object4)
P4-Y1: [P4] => y(Object3) = y(Object4)
PATH-X-2-INV: [P4-X1] => x(Object4) < x(Object3)
D-X-1: [P3-X1, PATH-X-2-INV] => x(Object0) < x(Object3)
PATH-Y-2-INV: [P3-Y1] => y(Object4) < y(Object0)
D-Y-1: [P4-Y1, PATH-Y-2-INV] => y(Object3) < y(Object0)
Q-DIR: [D-X-1, D-Y-1] => Northwest(Object0,Object3)
The uniquely guaranteed answer is Northwest. It is displayed at C.
Answer: C
```

## chain-1-base-direction-m0-single
Split: test; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object6, Object9, Object7, Object0, Object3, Object4, Object1, Object2, Object8
Query: direction Object9 Object3
Consider a map with multiple objects:
Object0 is to the West of Object3.
Object3 is to the East of Object7.
Object2 is to the Northeast of Object6.
Object8 is to the East of Object1.
Object5 is to the West of Object3.
Object8 is to the Northwest of Object9.
Object3 is to the West of Object1.
Object4 is to the West of Object3.
Question: In which direction is Object9 relative to Object3 in every valid map?
A. West
B. Southeast
C. Northeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
Southeast:
P4: From p4, Object8 is exactly east of Object1.
P6: From p6, Object8 is exactly northwest of Object9.
P7: From p7, Object3 is exactly west of Object1.
P4-X1: From P4, Object8's x-coordinate is greater than Object1's x-coordinate.
P4-Y1: From P4, Object8's y-coordinate is equal to Object1's y-coordinate.
P6-X1: From P6, Object8's x-coordinate is less than Object9's x-coordinate.
P6-Y1: From P6, Object8's y-coordinate is greater than Object9's y-coordinate.
P7-X1: From P7, Object3's x-coordinate is less than Object1's x-coordinate.
P7-Y1: From P7, Object3's y-coordinate is equal to Object1's y-coordinate.
PATH-X-2-INV: From P4-X1, Object1's x-coordinate is less than Object8's x-coordinate.
D-X-1: From P7-X1, PATH-X-2-INV, Object3's x-coordinate is less than Object8's x-coordinate.
D-X-2: From D-X-1, P6-X1, Object3's x-coordinate is less than Object9's x-coordinate.
PATH-Y-1-INV: From P6-Y1, Object9's y-coordinate is less than Object8's y-coordinate.
PATH-Y-3-INV: From P7-Y1, Object1's y-coordinate is equal to Object3's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P4-Y1, Object9's y-coordinate is less than Object1's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, Object9's y-coordinate is less than Object3's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, Object9 is exactly southeast of Object3.
The uniquely guaranteed answer is Southeast. It is displayed at B.
Answer: B
```
### Symbolic
```text
Southeast:
P4: [p4] => East(Object8,Object1)
P6: [p6] => Northwest(Object8,Object9)
P7: [p7] => West(Object3,Object1)
P4-X1: [P4] => x(Object8) > x(Object1)
P4-Y1: [P4] => y(Object8) = y(Object1)
P6-X1: [P6] => x(Object8) < x(Object9)
P6-Y1: [P6] => y(Object8) > y(Object9)
P7-X1: [P7] => x(Object3) < x(Object1)
P7-Y1: [P7] => y(Object3) = y(Object1)
PATH-X-2-INV: [P4-X1] => x(Object1) < x(Object8)
D-X-1: [P7-X1, PATH-X-2-INV] => x(Object3) < x(Object8)
D-X-2: [D-X-1, P6-X1] => x(Object3) < x(Object9)
PATH-Y-1-INV: [P6-Y1] => y(Object9) < y(Object8)
PATH-Y-3-INV: [P7-Y1] => y(Object1) = y(Object3)
D-Y-1: [PATH-Y-1-INV, P4-Y1] => y(Object9) < y(Object1)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(Object9) < y(Object3)
Q-DIR: [D-X-2, D-Y-2] => Southeast(Object9,Object3)
The uniquely guaranteed answer is Southeast. It is displayed at B.
Answer: B
```

## chain-2-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object3, Object8, Object11, Object13, Object4, Object6, Object0, Object2, Object5, Object10, Object7, Object1, Object9, Object12
Query: direction Object5 Object13
Consider a map with multiple objects:
Object13 is to the East of Object7.
Object2 is to the East of Object12.
Object9 is to the East of Object2.
Object1 is to the West of Object13.
Object9 is to the Northwest of Object5.
Object13 is to the West of Object12.
Object10 is to the Northwest of Object4.
Object13 is to the East of Object0.
Object13 is to the East of Object11.
Object13 is to the East of Object3.
Object8 is to the West of Object6.
Question: In which direction is Object5 relative to Object13 in every valid map?
A. Northwest
B. Southeast
C. West
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
Southeast:
P2: From p2, Object2 is exactly east of Object12.
P3: From p3, Object9 is exactly east of Object2.
P5: From p5, Object9 is exactly northwest of Object5.
P6: From p6, Object13 is exactly west of Object12.
P2-X1: From P2, Object2's x-coordinate is greater than Object12's x-coordinate.
P2-Y1: From P2, Object2's y-coordinate is equal to Object12's y-coordinate.
P3-X1: From P3, Object9's x-coordinate is greater than Object2's x-coordinate.
P3-Y1: From P3, Object9's y-coordinate is equal to Object2's y-coordinate.
P5-X1: From P5, Object9's x-coordinate is less than Object5's x-coordinate.
P5-Y1: From P5, Object9's y-coordinate is greater than Object5's y-coordinate.
P6-X1: From P6, Object13's x-coordinate is less than Object12's x-coordinate.
P6-Y1: From P6, Object13's y-coordinate is equal to Object12's y-coordinate.
PATH-X-2-INV: From P2-X1, Object12's x-coordinate is less than Object2's x-coordinate.
PATH-X-3-INV: From P3-X1, Object2's x-coordinate is less than Object9's x-coordinate.
D-X-1: From P6-X1, PATH-X-2-INV, Object13's x-coordinate is less than Object2's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object13's x-coordinate is less than Object9's x-coordinate.
D-X-3: From D-X-2, P5-X1, Object13's x-coordinate is less than Object5's x-coordinate.
PATH-Y-1-INV: From P5-Y1, Object5's y-coordinate is less than Object9's y-coordinate.
PATH-Y-4-INV: From P6-Y1, Object12's y-coordinate is equal to Object13's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P3-Y1, Object5's y-coordinate is less than Object2's y-coordinate.
D-Y-2: From D-Y-1, P2-Y1, Object5's y-coordinate is less than Object12's y-coordinate.
D-Y-3: From D-Y-2, PATH-Y-4-INV, Object5's y-coordinate is less than Object13's y-coordinate.
Q-DIR: From D-X-3, D-Y-3, Object5 is exactly southeast of Object13.
The uniquely guaranteed answer is Southeast. It is displayed at B.
Answer: B
```
### Symbolic
```text
Southeast:
P2: [p2] => East(Object2,Object12)
P3: [p3] => East(Object9,Object2)
P5: [p5] => Northwest(Object9,Object5)
P6: [p6] => West(Object13,Object12)
P2-X1: [P2] => x(Object2) > x(Object12)
P2-Y1: [P2] => y(Object2) = y(Object12)
P3-X1: [P3] => x(Object9) > x(Object2)
P3-Y1: [P3] => y(Object9) = y(Object2)
P5-X1: [P5] => x(Object9) < x(Object5)
P5-Y1: [P5] => y(Object9) > y(Object5)
P6-X1: [P6] => x(Object13) < x(Object12)
P6-Y1: [P6] => y(Object13) = y(Object12)
PATH-X-2-INV: [P2-X1] => x(Object12) < x(Object2)
PATH-X-3-INV: [P3-X1] => x(Object2) < x(Object9)
D-X-1: [P6-X1, PATH-X-2-INV] => x(Object13) < x(Object2)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object13) < x(Object9)
D-X-3: [D-X-2, P5-X1] => x(Object13) < x(Object5)
PATH-Y-1-INV: [P5-Y1] => y(Object5) < y(Object9)
PATH-Y-4-INV: [P6-Y1] => y(Object12) = y(Object13)
D-Y-1: [PATH-Y-1-INV, P3-Y1] => y(Object5) < y(Object2)
D-Y-2: [D-Y-1, P2-Y1] => y(Object5) < y(Object12)
D-Y-3: [D-Y-2, PATH-Y-4-INV] => y(Object5) < y(Object13)
Q-DIR: [D-X-3, D-Y-3] => Southeast(Object5,Object13)
The uniquely guaranteed answer is Southeast. It is displayed at B.
Answer: B
```

## chain-3-base-direction-m0-single
Split: train; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object2, Object3, Object1, Object0, Object4
Query: direction Object1 Object4
Consider a map with multiple objects:
Object4 is to the West of Object3.
Object5 is to the East of Object4.
Object0 is to the Southeast of Object1.
Object4 is to the East of Object0.
Object4 is to the West of Object2.
Question: In which direction is Object1 relative to Object4 in every valid map?
A. East
B. Southwest
C. South
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
Northwest:
P3: From p3, Object0 is exactly southeast of Object1.
P4: From p4, Object4 is exactly east of Object0.
P3-X1: From P3, Object0's x-coordinate is greater than Object1's x-coordinate.
P3-Y1: From P3, Object0's y-coordinate is less than Object1's y-coordinate.
P4-X1: From P4, Object4's x-coordinate is greater than Object0's x-coordinate.
P4-Y1: From P4, Object4's y-coordinate is equal to Object0's y-coordinate.
PATH-X-1-INV: From P3-X1, Object1's x-coordinate is less than Object0's x-coordinate.
PATH-X-2-INV: From P4-X1, Object0's x-coordinate is less than Object4's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, Object1's x-coordinate is less than Object4's x-coordinate.
D-Y-1: From P4-Y1, P3-Y1, Object4's y-coordinate is less than Object1's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, Object1 is exactly northwest of Object4.
The uniquely guaranteed answer is Northwest. It is displayed at D.
Answer: D
```
### Symbolic
```text
Northwest:
P3: [p3] => Southeast(Object0,Object1)
P4: [p4] => East(Object4,Object0)
P3-X1: [P3] => x(Object0) > x(Object1)
P3-Y1: [P3] => y(Object0) < y(Object1)
P4-X1: [P4] => x(Object4) > x(Object0)
P4-Y1: [P4] => y(Object4) = y(Object0)
PATH-X-1-INV: [P3-X1] => x(Object1) < x(Object0)
PATH-X-2-INV: [P4-X1] => x(Object0) < x(Object4)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(Object1) < x(Object4)
D-Y-1: [P4-Y1, P3-Y1] => y(Object4) < y(Object1)
Q-DIR: [D-X-1, D-Y-1] => Northwest(Object1,Object4)
The uniquely guaranteed answer is Northwest. It is displayed at D.
Answer: D
```

## chain-4-base-direction-m0-single
Split: test; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object8, Object0, Object9, Object4, Object5, Object1, Object7, Object3, Object2, Object6
Query: direction Object2 Object3
Consider a map with multiple objects:
Object3 is to the North of Object1.
Object3 is to the South of Object5.
Object9 is to the North of Object4.
Object2 is to the Southwest of Object8.
Object3 is to the South of Object6.
Object0 is to the North of Object3.
Object1 is to the North of Object8.
Object3 is to the South of Object7.
Question: In which direction is Object2 relative to Object3 in every valid map?
A. Northeast
B. West
C. Southwest
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
Southwest:
P1: From p1, Object3 is exactly north of Object1.
P4: From p4, Object2 is exactly southwest of Object8.
P7: From p7, Object1 is exactly north of Object8.
P1-X1: From P1, Object3's x-coordinate is equal to Object1's x-coordinate.
P1-Y1: From P1, Object3's y-coordinate is greater than Object1's y-coordinate.
P4-X1: From P4, Object2's x-coordinate is less than Object8's x-coordinate.
P4-Y1: From P4, Object2's y-coordinate is less than Object8's y-coordinate.
P7-X1: From P7, Object1's x-coordinate is equal to Object8's x-coordinate.
P7-Y1: From P7, Object1's y-coordinate is greater than Object8's y-coordinate.
PATH-X-2-INV: From P7-X1, Object8's x-coordinate is equal to Object1's x-coordinate.
PATH-X-3-INV: From P1-X1, Object1's x-coordinate is equal to Object3's x-coordinate.
D-X-1: From P4-X1, PATH-X-2-INV, Object2's x-coordinate is less than Object1's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object2's x-coordinate is less than Object3's x-coordinate.
PATH-Y-2-INV: From P7-Y1, Object8's y-coordinate is less than Object1's y-coordinate.
PATH-Y-3-INV: From P1-Y1, Object1's y-coordinate is less than Object3's y-coordinate.
D-Y-1: From P4-Y1, PATH-Y-2-INV, Object2's y-coordinate is less than Object1's y-coordinate.
D-Y-2: From D-Y-1, PATH-Y-3-INV, Object2's y-coordinate is less than Object3's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, Object2 is exactly southwest of Object3.
The uniquely guaranteed answer is Southwest. It is displayed at C.
Answer: C
```
### Symbolic
```text
Southwest:
P1: [p1] => North(Object3,Object1)
P4: [p4] => Southwest(Object2,Object8)
P7: [p7] => North(Object1,Object8)
P1-X1: [P1] => x(Object3) = x(Object1)
P1-Y1: [P1] => y(Object3) > y(Object1)
P4-X1: [P4] => x(Object2) < x(Object8)
P4-Y1: [P4] => y(Object2) < y(Object8)
P7-X1: [P7] => x(Object1) = x(Object8)
P7-Y1: [P7] => y(Object1) > y(Object8)
PATH-X-2-INV: [P7-X1] => x(Object8) = x(Object1)
PATH-X-3-INV: [P1-X1] => x(Object1) = x(Object3)
D-X-1: [P4-X1, PATH-X-2-INV] => x(Object2) < x(Object1)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object2) < x(Object3)
PATH-Y-2-INV: [P7-Y1] => y(Object8) < y(Object1)
PATH-Y-3-INV: [P1-Y1] => y(Object1) < y(Object3)
D-Y-1: [P4-Y1, PATH-Y-2-INV] => y(Object2) < y(Object1)
D-Y-2: [D-Y-1, PATH-Y-3-INV] => y(Object2) < y(Object3)
Q-DIR: [D-X-2, D-Y-2] => Southwest(Object2,Object3)
The uniquely guaranteed answer is Southwest. It is displayed at C.
Answer: C
```

## fork-0-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object1, Object4, Object3, Object2, Object5, Object0
Query: direction Object1 Object5
Consider a map with multiple objects:
Object0 is to the North of Object4.
Object3 is to the South of Object0.
Object0 is to the North of Object1.
Object5 is to the South of Object0.
Object2 is to the South of Object0.
Question: In which direction is Object1 relative to Object5 in every valid map?
A. North
B. South
C. East
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object1 at (0, 4); Object4 at (0, 3); Object3 at (0, 2); Object2 at (0, 1); Object5 at (0, 0); Object0 at (0, 5). In this map, direction=North.
South:
A valid map places Object1 at (0, 3); Object4 at (0, 2); Object3 at (0, 1); Object2 at (0, 0); Object5 at (0, 4); Object0 at (0, 5). In this map, direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object1:(0,4), Object4:(0,3), Object3:(0,2), Object2:(0,1), Object5:(0,0), Object0:(0,5)} => direction=North.
South:
Valid map {Object1:(0,3), Object4:(0,2), Object3:(0,1), Object2:(0,0), Object5:(0,4), Object0:(0,5)} => direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
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
Objects: Object8, Object3, Object1, Object4, Object7, Object6, Object0, Object2, Object5
Query: direction Object6 Object4
Consider a map with multiple objects:
Object5 is to the West of Object8.
Object0 is to the West of Object8.
Object7 is to the West of Object8.
Object2 is to the West of Object8.
Object4 is to the West of Object8.
Object3 is to the East of Object1.
Object6 is to the West of Object8.
Question: In which direction is Object6 relative to Object4 in every valid map?
A. Northeast
B. Northwest
C. East
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
A valid map places Object8 at (7, 1); Object3 at (7, 0); Object1 at (6, 0); Object4 at (4, 1); Object7 at (3, 1); Object6 at (5, 1); Object0 at (2, 1); Object2 at (1, 1); Object5 at (0, 1). In this map, direction=East.
West:
A valid map places Object8 at (7, 1); Object3 at (7, 0); Object1 at (6, 0); Object4 at (5, 1); Object7 at (4, 1); Object6 at (3, 1); Object0 at (2, 1); Object2 at (1, 1); Object5 at (0, 1). In this map, direction=West.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
East:
Valid map {Object8:(7,1), Object3:(7,0), Object1:(6,0), Object4:(4,1), Object7:(3,1), Object6:(5,1), Object0:(2,1), Object2:(1,1), Object5:(0,1)} => direction=East.
West:
Valid map {Object8:(7,1), Object3:(7,0), Object1:(6,0), Object4:(5,1), Object7:(4,1), Object6:(3,1), Object0:(2,1), Object2:(1,1), Object5:(0,1)} => direction=West.
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
Objects: Object0, Object2, Object11, Object3, Object5, Object6, Object4, Object8, Object1, Object9, Object7, Object10
Query: direction Object7 Object10
Consider a map with multiple objects:
Object10 is to the East of Object9.
Object9 is to the West of Object6.
Object1 is to the East of Object9.
Object9 is to the West of Object4.
Object9 is to the West of Object2.
Object11 is to the West of Object8.
Object7 is to the East of Object9.
Object9 is to the West of Object3.
Object0 is to the South of Object5.
Question: In which direction is Object7 relative to Object10 in every valid map?
A. West
B. Northeast
C. East
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
A valid map places Object0 at (7, 2); Object2 at (7, 1); Object11 at (7, 0); Object3 at (6, 1); Object5 at (7, 3); Object6 at (5, 1); Object4 at (4, 1); Object8 at (8, 0); Object1 at (3, 1); Object9 at (0, 1); Object7 at (2, 1); Object10 at (1, 1). In this map, direction=East.
West:
A valid map places Object0 at (7, 2); Object2 at (7, 1); Object11 at (7, 0); Object3 at (6, 1); Object5 at (7, 3); Object6 at (5, 1); Object4 at (4, 1); Object8 at (8, 0); Object1 at (3, 1); Object9 at (0, 1); Object7 at (1, 1); Object10 at (2, 1). In this map, direction=West.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
East:
Valid map {Object0:(7,2), Object2:(7,1), Object11:(7,0), Object3:(6,1), Object5:(7,3), Object6:(5,1), Object4:(4,1), Object8:(8,0), Object1:(3,1), Object9:(0,1), Object7:(2,1), Object10:(1,1)} => direction=East.
West:
Valid map {Object0:(7,2), Object2:(7,1), Object11:(7,0), Object3:(6,1), Object5:(7,3), Object6:(5,1), Object4:(4,1), Object8:(8,0), Object1:(3,1), Object9:(0,1), Object7:(1,1), Object10:(2,1)} => direction=West.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## fork-3-base-direction-m0-single
Split: train; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object5, Object4
Query: direction Object0 Object3
Consider a map with multiple objects:
Object5 is to the North of Object2.
Object3 is to the North of Object2.
Object1 is to the North of Object2.
Object2 is to the South of Object4.
Object2 is to the South of Object0.
Question: In which direction is Object0 relative to Object3 in every valid map?
A. Northeast
B. South
C. North
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object0 at (0, 5); Object1 at (0, 4); Object2 at (0, 0); Object3 at (0, 3); Object5 at (0, 2); Object4 at (0, 1). In this map, direction=North.
South:
A valid map places Object0 at (0, 4); Object1 at (0, 3); Object2 at (0, 0); Object3 at (0, 5); Object5 at (0, 2); Object4 at (0, 1). In this map, direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object0:(0,5), Object1:(0,4), Object2:(0,0), Object3:(0,3), Object5:(0,2), Object4:(0,1)} => direction=North.
South:
Valid map {Object0:(0,4), Object1:(0,3), Object2:(0,0), Object3:(0,5), Object5:(0,2), Object4:(0,1)} => direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## fork-4-base-direction-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object8, Object4, Object1, Object7, Object3, Object2, Object0, Object6
Query: direction Object3 Object7
Consider a map with multiple objects:
Object8 is to the South of Object4.
Object3 is to the South of Object4.
Object4 is to the North of Object6.
Object2 is to the Northwest of Object0.
Object4 is to the North of Object5.
Object4 is to the North of Object1.
Object7 is to the South of Object4.
Question: In which direction is Object3 relative to Object7 in every valid map?
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
A valid map places Object5 at (0, 7); Object8 at (0, 6); Object4 at (0, 8); Object1 at (0, 5); Object7 at (0, 3); Object3 at (0, 4); Object2 at (0, 2); Object0 at (1, 1); Object6 at (0, 0). In this map, direction=North.
South:
A valid map places Object5 at (0, 7); Object8 at (0, 6); Object4 at (0, 8); Object1 at (0, 5); Object7 at (0, 4); Object3 at (0, 3); Object2 at (0, 2); Object0 at (1, 1); Object6 at (0, 0). In this map, direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object5:(0,7), Object8:(0,6), Object4:(0,8), Object1:(0,5), Object7:(0,3), Object3:(0,4), Object2:(0,2), Object0:(1,1), Object6:(0,0)} => direction=North.
South:
Valid map {Object5:(0,7), Object8:(0,6), Object4:(0,8), Object1:(0,5), Object7:(0,4), Object3:(0,3), Object2:(0,2), Object0:(1,1), Object6:(0,0)} => direction=South.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## empty-0-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object1, Object3, Object4, Object0, Object2
Query: direction Object3 Object4
Consider a map with multiple objects:
Object1 is to the West of Object4.
Object3 is to the West of Object4.
Object0 is to the North of Object4.
Object4 is to the East of Object2.
Question: In which direction is Object3 relative to Object4 in every valid map?
A. South
B. West
C. Southwest
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
West:
P2: From p2, Object3 is exactly west of Object4.
Q-DIR: From P2, Object3 is exactly west of Object4.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
West:
P2: [p2] => West(Object3,Object4)
Q-DIR: [P2] => West(Object3,Object4)
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
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
Objects: Object6, Object5, Object3, Object0, Object7, Object4, Object1, Object2
Query: direction Object3 Object2
Consider a map with multiple objects:
Object6 is to the South of Object2.
Object5 is to the North of Object2.
Object3 is to the North of Object2.
Object2 is to the North of Object0.
Object4 is to the West of Object2.
Object7 is to the West of Object1.
Question: In which direction is Object3 relative to Object2 in every valid map?
A. Southeast
B. North
C. Southwest
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P3: From p3, Object3 is exactly north of Object2.
Q-DIR: From P3, Object3 is exactly north of Object2.
The uniquely guaranteed answer is North. It is displayed at B.
Answer: B
```
### Symbolic
```text
North:
P3: [p3] => North(Object3,Object2)
Q-DIR: [P3] => North(Object3,Object2)
The uniquely guaranteed answer is North. It is displayed at B.
Answer: B
```

## empty-2-base-direction-m0-single
Split: train; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object8, Object6, Object2, Object7, Object0, Object10, Object1, Object4, Object9, Object3, Object5
Query: direction Object5 Object0
Consider a map with multiple objects:
Object8 is to the North of Object7.
Object0 is to the South of Object1.
Object3 is to the East of Object0.
Object2 is to the East of Object0.
Object4 is to the North of Object0.
Object6 is to the West of Object0.
Object10 is to the Northeast of Object9.
Object0 is to the South of Object5.
Question: In which direction is Object5 relative to Object0 in every valid map?
A. West
B. South
C. Southeast
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P8: From p8, Object0 is exactly south of Object5.
P8-X1: From P8, Object0's x-coordinate is equal to Object5's x-coordinate.
P8-Y1: From P8, Object0's y-coordinate is less than Object5's y-coordinate.
Q-DIR: From P8-X1, P8-Y1, Object5 is exactly north of Object0.
The uniquely guaranteed answer is North. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P8: [p8] => South(Object0,Object5)
P8-X1: [P8] => x(Object0) = x(Object5)
P8-Y1: [P8] => y(Object0) < y(Object5)
Q-DIR: [P8-X1, P8-Y1] => North(Object5,Object0)
The uniquely guaranteed answer is North. It is displayed at D.
Answer: D
```

## empty-3-base-direction-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object2, Object0, Object4, Object3, Object1
Query: direction Object4 Object0
Consider a map with multiple objects:
Object0 is to the West of Object1.
Object3 is to the South of Object0.
Object2 is to the West of Object0.
Object4 is to the East of Object0.
Question: In which direction is Object4 relative to Object0 in every valid map?
A. East
B. South
C. Northeast
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
P4: From p4, Object4 is exactly east of Object0.
Q-DIR: From P4, Object4 is exactly east of Object0.
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```
### Symbolic
```text
East:
P4: [p4] => East(Object4,Object0)
Q-DIR: [P4] => East(Object4,Object0)
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```

## empty-4-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object0, Object7, Object5, Object2, Object1, Object3, Object4
Query: direction Object7 Object4
Consider a map with multiple objects:
Object4 is to the East of Object0.
Object2 is to the East of Object4.
Object7 is to the East of Object4.
Object4 is to the East of Object6.
Object1 is to the West of Object3.
Object4 is to the North of Object5.
Question: In which direction is Object7 relative to Object4 in every valid map?
A. South
B. West
C. East
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
P3: From p3, Object7 is exactly east of Object4.
Q-DIR: From P3, Object7 is exactly east of Object4.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```
### Symbolic
```text
East:
P3: [p3] => East(Object7,Object4)
Q-DIR: [P3] => East(Object7,Object4)
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```

## singleton-0-base-direction-m0-single
Split: train; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object4, Object2, Object1, Object3, Object0
Query: direction Object3 Object2
Consider a map with multiple objects:
Object0 is to the West of Object2.
Object3 is to the North of Object2.
Object4 is to the West of Object2.
Object2 is to the North of Object1.
Question: In which direction is Object3 relative to Object2 in every valid map?
A. Southeast
B. West
C. Northeast
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P2: From p2, Object3 is exactly north of Object2.
Q-DIR: From P2, Object3 is exactly north of Object2.
The uniquely guaranteed answer is North. It is displayed at D.
Answer: D
```
### Symbolic
```text
North:
P2: [p2] => North(Object3,Object2)
Q-DIR: [P2] => North(Object3,Object2)
The uniquely guaranteed answer is North. It is displayed at D.
Answer: D
```

## singleton-1-base-direction-m0-single
Split: train; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object8, Object3, Object2, Object1, Object0, Object6, Object7, Object4
Query: direction Object6 Object1
Consider a map with multiple objects:
Object1 is to the West of Object2.
Object6 is to the East of Object2.
Object5 is to the West of Object1.
Object1 is to the South of Object3.
Object1 is to the East of Object8.
Object0 is to the West of Object7.
Object1 is to the South of Object4.
Question: In which direction is Object6 relative to Object1 in every valid map?
A. East
B. West
C. Southeast
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
P1: From p1, Object1 is exactly west of Object2.
P2: From p2, Object6 is exactly east of Object2.
P1-X1: From P1, Object1's x-coordinate is less than Object2's x-coordinate.
P1-Y1: From P1, Object1's y-coordinate is equal to Object2's y-coordinate.
P2-X1: From P2, Object6's x-coordinate is greater than Object2's x-coordinate.
P2-Y1: From P2, Object6's y-coordinate is equal to Object2's y-coordinate.
PATH-X-2-INV: From P2-X1, Object2's x-coordinate is less than Object6's x-coordinate.
D-X-1: From P1-X1, PATH-X-2-INV, Object1's x-coordinate is less than Object6's x-coordinate.
PATH-Y-2-INV: From P1-Y1, Object2's y-coordinate is equal to Object1's y-coordinate.
D-Y-1: From P2-Y1, PATH-Y-2-INV, Object6's y-coordinate is equal to Object1's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, Object6 is exactly east of Object1.
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```
### Symbolic
```text
East:
P1: [p1] => West(Object1,Object2)
P2: [p2] => East(Object6,Object2)
P1-X1: [P1] => x(Object1) < x(Object2)
P1-Y1: [P1] => y(Object1) = y(Object2)
P2-X1: [P2] => x(Object6) > x(Object2)
P2-Y1: [P2] => y(Object6) = y(Object2)
PATH-X-2-INV: [P2-X1] => x(Object2) < x(Object6)
D-X-1: [P1-X1, PATH-X-2-INV] => x(Object1) < x(Object6)
PATH-Y-2-INV: [P1-Y1] => y(Object2) = y(Object1)
D-Y-1: [P2-Y1, PATH-Y-2-INV] => y(Object6) = y(Object1)
Q-DIR: [D-X-1, D-Y-1] => East(Object6,Object1)
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```

## singleton-2-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object8, Object12, Object6, Object7, Object9, Object1, Object5, Object3, Object10, Object2, Object0, Object11, Object4
Query: direction Object6 Object1
Consider a map with multiple objects:
Object0 is to the West of Object1.
Object12 is to the North of Object1.
Object1 is to the South of Object5.
Object2 is to the East of Object1.
Object6 is to the West of Object7.
Object1 is to the South of Object11.
Object1 is to the North of Object9.
Object0 is to the East of Object7.
Object3 is to the Southwest of Object10.
Object4 is to the North of Object8.
Question: In which direction is Object6 relative to Object1 in every valid map?
A. North
B. West
C. Northeast
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
West:
P1: From p1, Object0 is exactly west of Object1.
P5: From p5, Object6 is exactly west of Object7.
P8: From p8, Object0 is exactly east of Object7.
P1-X1: From P1, Object0's x-coordinate is less than Object1's x-coordinate.
P1-Y1: From P1, Object0's y-coordinate is equal to Object1's y-coordinate.
P5-X1: From P5, Object6's x-coordinate is less than Object7's x-coordinate.
P5-Y1: From P5, Object6's y-coordinate is equal to Object7's y-coordinate.
P8-X1: From P8, Object0's x-coordinate is greater than Object7's x-coordinate.
P8-Y1: From P8, Object0's y-coordinate is equal to Object7's y-coordinate.
PATH-X-2-INV: From P8-X1, Object7's x-coordinate is less than Object0's x-coordinate.
D-X-1: From P5-X1, PATH-X-2-INV, Object6's x-coordinate is less than Object0's x-coordinate.
D-X-2: From D-X-1, P1-X1, Object6's x-coordinate is less than Object1's x-coordinate.
PATH-Y-2-INV: From P8-Y1, Object7's y-coordinate is equal to Object0's y-coordinate.
D-Y-1: From P5-Y1, PATH-Y-2-INV, Object6's y-coordinate is equal to Object0's y-coordinate.
D-Y-2: From D-Y-1, P1-Y1, Object6's y-coordinate is equal to Object1's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, Object6 is exactly west of Object1.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
West:
P1: [p1] => West(Object0,Object1)
P5: [p5] => West(Object6,Object7)
P8: [p8] => East(Object0,Object7)
P1-X1: [P1] => x(Object0) < x(Object1)
P1-Y1: [P1] => y(Object0) = y(Object1)
P5-X1: [P5] => x(Object6) < x(Object7)
P5-Y1: [P5] => y(Object6) = y(Object7)
P8-X1: [P8] => x(Object0) > x(Object7)
P8-Y1: [P8] => y(Object0) = y(Object7)
PATH-X-2-INV: [P8-X1] => x(Object7) < x(Object0)
D-X-1: [P5-X1, PATH-X-2-INV] => x(Object6) < x(Object0)
D-X-2: [D-X-1, P1-X1] => x(Object6) < x(Object1)
PATH-Y-2-INV: [P8-Y1] => y(Object7) = y(Object0)
D-Y-1: [P5-Y1, PATH-Y-2-INV] => y(Object6) = y(Object0)
D-Y-2: [D-Y-1, P1-Y1] => y(Object6) = y(Object1)
Q-DIR: [D-X-2, D-Y-2] => West(Object6,Object1)
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
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
Objects: Object0, Object1, Object2, Object4, Object3
Query: direction Object4 Object2
Consider a map with multiple objects:
Object2 is to the North of Object1.
Object4 is to the East of Object2.
Object2 is to the North of Object3.
Object0 is to the West of Object2.
Question: In which direction is Object4 relative to Object2 in every valid map?
A. Northwest
B. South
C. East
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
P2: From p2, Object4 is exactly east of Object2.
Q-DIR: From P2, Object4 is exactly east of Object2.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```
### Symbolic
```text
East:
P2: [p2] => East(Object4,Object2)
Q-DIR: [P2] => East(Object4,Object2)
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
Objects: Object0, Object4, Object3, Object7, Object1, Object5, Object8, Object6, Object2
Query: direction Object0 Object6
Consider a map with multiple objects:
Object1 is to the East of Object6.
Object3 is to the West of Object6.
Object7 is to the North of Object6.
Object2 is to the East of Object5.
Object3 is to the East of Object0.
Object4 is to the North of Object6.
Object6 is to the West of Object8.
Question: In which direction is Object0 relative to Object6 in every valid map?
A. Northwest
B. West
C. Northeast
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
West:
P2: From p2, Object3 is exactly west of Object6.
P5: From p5, Object3 is exactly east of Object0.
P2-X1: From P2, Object3's x-coordinate is less than Object6's x-coordinate.
P2-Y1: From P2, Object3's y-coordinate is equal to Object6's y-coordinate.
P5-X1: From P5, Object3's x-coordinate is greater than Object0's x-coordinate.
P5-Y1: From P5, Object3's y-coordinate is equal to Object0's y-coordinate.
PATH-X-1-INV: From P5-X1, Object0's x-coordinate is less than Object3's x-coordinate.
D-X-1: From PATH-X-1-INV, P2-X1, Object0's x-coordinate is less than Object6's x-coordinate.
PATH-Y-1-INV: From P5-Y1, Object0's y-coordinate is equal to Object3's y-coordinate.
D-Y-1: From PATH-Y-1-INV, P2-Y1, Object0's y-coordinate is equal to Object6's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, Object0 is exactly west of Object6.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
West:
P2: [p2] => West(Object3,Object6)
P5: [p5] => East(Object3,Object0)
P2-X1: [P2] => x(Object3) < x(Object6)
P2-Y1: [P2] => y(Object3) = y(Object6)
P5-X1: [P5] => x(Object3) > x(Object0)
P5-Y1: [P5] => y(Object3) = y(Object0)
PATH-X-1-INV: [P5-X1] => x(Object0) < x(Object3)
D-X-1: [PATH-X-1-INV, P2-X1] => x(Object0) < x(Object6)
PATH-Y-1-INV: [P5-Y1] => y(Object0) = y(Object3)
D-Y-1: [PATH-Y-1-INV, P2-Y1] => y(Object0) = y(Object6)
Q-DIR: [D-X-1, D-Y-1] => West(Object0,Object6)
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```

## multiple-0-base-direction-m0-single
Split: dev; case: unique; answer: D

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object1, Object4, Object2, Object0, Object3
Query: direction Object0 Object4
Consider a map with multiple objects:
Object3 is to the South of Object4.
Object0 is to the South of Object4.
Object4 is to the North of Object2.
Object4 is to the North of Object1.
Question: In which direction is Object0 relative to Object4 in every valid map?
A. West
B. Southeast
C. Northeast
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
South:
P2: From p2, Object0 is exactly south of Object4.
Q-DIR: From P2, Object0 is exactly south of Object4.
The uniquely guaranteed answer is South. It is displayed at D.
Answer: D
```
### Symbolic
```text
South:
P2: [p2] => South(Object0,Object4)
Q-DIR: [P2] => South(Object0,Object4)
The uniquely guaranteed answer is South. It is displayed at D.
Answer: D
```

## multiple-1-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object1, Object6, Object0, Object11, Object3, Object2, Object10, Object9, Object12, Object5, Object4, Object7, Object8
Query: direction Object10 Object4
Consider a map with multiple objects:
Object4 is to the West of Object1.
Object12 is to the East of Object4.
Object7 is to the East of Object4.
Object9 is to the East of Object4.
Object4 is to the West of Object5.
Object10 is to the East of Object7.
Object2 is to the East of Object9.
Object12 is to the West of Object8.
Object0 is to the East of Object1.
Object3 is to the West of Object11.
Object6 is to the East of Object5.
Question: In which direction is Object10 relative to Object4 in every valid map?
A. North
B. Northeast
C. East
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
P3: From p3, Object7 is exactly east of Object4.
P6: From p6, Object10 is exactly east of Object7.
P3-X1: From P3, Object7's x-coordinate is greater than Object4's x-coordinate.
P3-Y1: From P3, Object7's y-coordinate is equal to Object4's y-coordinate.
P6-X1: From P6, Object10's x-coordinate is greater than Object7's x-coordinate.
P6-Y1: From P6, Object10's y-coordinate is equal to Object7's y-coordinate.
PATH-X-1-INV: From P3-X1, Object4's x-coordinate is less than Object7's x-coordinate.
PATH-X-2-INV: From P6-X1, Object7's x-coordinate is less than Object10's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, Object4's x-coordinate is less than Object10's x-coordinate.
D-Y-1: From P6-Y1, P3-Y1, Object10's y-coordinate is equal to Object4's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, Object10 is exactly east of Object4.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```
### Symbolic
```text
East:
P3: [p3] => East(Object7,Object4)
P6: [p6] => East(Object10,Object7)
P3-X1: [P3] => x(Object7) > x(Object4)
P3-Y1: [P3] => y(Object7) = y(Object4)
P6-X1: [P6] => x(Object10) > x(Object7)
P6-Y1: [P6] => y(Object10) = y(Object7)
PATH-X-1-INV: [P3-X1] => x(Object4) < x(Object7)
PATH-X-2-INV: [P6-X1] => x(Object7) < x(Object10)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(Object4) < x(Object10)
D-Y-1: [P6-Y1, P3-Y1] => y(Object10) = y(Object4)
Q-DIR: [D-X-1, D-Y-1] => East(Object10,Object4)
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```

## multiple-2-base-direction-m0-single
Split: train; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object8, Object15, Object13, Object0, Object9, Object10, Object7, Object20, Object1, Object11, Object14, Object19, Object22, Object12, Object17, Object21, Object6, Object18, Object16, Object4, Object3, Object5, Object2
Query: direction Object2 Object19
Consider a map with multiple objects:
Object12 is to the South of Object6.
Object6 is to the South of Object2.
Object0 is to the Northeast of Object14.
Object19 is to the South of Object12.
Object19 is to the South of Object13.
Object10 is to the North of Object20.
Object8 is to the North of Object19.
Object11 is to the North of Object8.
Object11 is to the South of Object18.
Object21 is to the North of Object10.
Object20 is to the North of Object19.
Object17 is to the North of Object22.
Object5 is to the North of Object4.
Object13 is to the South of Object16.
Object1 is to the South of Object15.
Object22 is to the North of Object7.
Object3 is to the South of Object4.
Object7 is to the North of Object19.
Object16 is to the South of Object9.
Object19 is to the South of Object3.
Question: In which direction is Object2 relative to Object19 in every valid map?
A. Northwest
B. East
C. North
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
P1: From p1, Object12 is exactly south of Object6.
P2: From p2, Object6 is exactly south of Object2.
P4: From p4, Object19 is exactly south of Object12.
P1-X1: From P1, Object12's x-coordinate is equal to Object6's x-coordinate.
P1-Y1: From P1, Object12's y-coordinate is less than Object6's y-coordinate.
P2-X1: From P2, Object6's x-coordinate is equal to Object2's x-coordinate.
P2-Y1: From P2, Object6's y-coordinate is less than Object2's y-coordinate.
P4-X1: From P4, Object19's x-coordinate is equal to Object12's x-coordinate.
P4-Y1: From P4, Object19's y-coordinate is less than Object12's y-coordinate.
PATH-X-1-INV: From P2-X1, Object2's x-coordinate is equal to Object6's x-coordinate.
PATH-X-2-INV: From P1-X1, Object6's x-coordinate is equal to Object12's x-coordinate.
PATH-X-3-INV: From P4-X1, Object12's x-coordinate is equal to Object19's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, Object2's x-coordinate is equal to Object12's x-coordinate.
D-X-2: From D-X-1, PATH-X-3-INV, Object2's x-coordinate is equal to Object19's x-coordinate.
D-Y-1: From P4-Y1, P1-Y1, Object19's y-coordinate is less than Object6's y-coordinate.
D-Y-2: From D-Y-1, P2-Y1, Object19's y-coordinate is less than Object2's y-coordinate.
Q-DIR: From D-X-2, D-Y-2, Object2 is exactly north of Object19.
The uniquely guaranteed answer is North. It is displayed at C.
Answer: C
```
### Symbolic
```text
North:
P1: [p1] => South(Object12,Object6)
P2: [p2] => South(Object6,Object2)
P4: [p4] => South(Object19,Object12)
P1-X1: [P1] => x(Object12) = x(Object6)
P1-Y1: [P1] => y(Object12) < y(Object6)
P2-X1: [P2] => x(Object6) = x(Object2)
P2-Y1: [P2] => y(Object6) < y(Object2)
P4-X1: [P4] => x(Object19) = x(Object12)
P4-Y1: [P4] => y(Object19) < y(Object12)
PATH-X-1-INV: [P2-X1] => x(Object2) = x(Object6)
PATH-X-2-INV: [P1-X1] => x(Object6) = x(Object12)
PATH-X-3-INV: [P4-X1] => x(Object12) = x(Object19)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(Object2) = x(Object12)
D-X-2: [D-X-1, PATH-X-3-INV] => x(Object2) = x(Object19)
D-Y-1: [P4-Y1, P1-Y1] => y(Object19) < y(Object6)
D-Y-2: [D-Y-1, P2-Y1] => y(Object19) < y(Object2)
Q-DIR: [D-X-2, D-Y-2] => North(Object2,Object19)
The uniquely guaranteed answer is North. It is displayed at C.
Answer: C
```

## multiple-3-base-direction-m0-single
Split: dev; case: unique; answer: C

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object4, Object3, Object2, Object1, Object0
Query: direction Object3 Object0
Consider a map with multiple objects:
Object4 is to the East of Object0.
Object1 is to the East of Object0.
Object0 is to the West of Object2.
Object3 is to the East of Object0.
Question: In which direction is Object3 relative to Object0 in every valid map?
A. South
B. West
C. East
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
P4: From p4, Object3 is exactly east of Object0.
Q-DIR: From P4, Object3 is exactly east of Object0.
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
```
### Symbolic
```text
East:
P4: [p4] => East(Object3,Object0)
Q-DIR: [P4] => East(Object3,Object0)
The uniquely guaranteed answer is East. It is displayed at C.
Answer: C
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
Objects: Object5, Object4, Object7, Object2, Object8, Object10, Object3, Object11, Object12, Object9, Object0, Object1, Object6
Query: direction Object10 Object12
Consider a map with multiple objects:
Object7 is to the South of Object12.
Object2 is to the South of Object6.
Object12 is to the North of Object11.
Object12 is to the North of Object6.
Object5 is to the North of Object10.
Object12 is to the North of Object0.
Object12 is to the North of Object5.
Object3 is to the South of Object0.
Object7 is to the North of Object1.
Object8 is to the East of Object4.
Object9 is to the South of Object11.
Question: In which direction is Object10 relative to Object12 in every valid map?
A. West
B. East
C. Northeast
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
South:
P5: From p5, Object5 is exactly north of Object10.
P7: From p7, Object12 is exactly north of Object5.
P5-X1: From P5, Object5's x-coordinate is equal to Object10's x-coordinate.
P5-Y1: From P5, Object5's y-coordinate is greater than Object10's y-coordinate.
P7-X1: From P7, Object12's x-coordinate is equal to Object5's x-coordinate.
P7-Y1: From P7, Object12's y-coordinate is greater than Object5's y-coordinate.
PATH-X-1-INV: From P5-X1, Object10's x-coordinate is equal to Object5's x-coordinate.
PATH-X-2-INV: From P7-X1, Object5's x-coordinate is equal to Object12's x-coordinate.
D-X-1: From PATH-X-1-INV, PATH-X-2-INV, Object10's x-coordinate is equal to Object12's x-coordinate.
PATH-Y-1-INV: From P5-Y1, Object10's y-coordinate is less than Object5's y-coordinate.
PATH-Y-2-INV: From P7-Y1, Object5's y-coordinate is less than Object12's y-coordinate.
D-Y-1: From PATH-Y-1-INV, PATH-Y-2-INV, Object10's y-coordinate is less than Object12's y-coordinate.
Q-DIR: From D-X-1, D-Y-1, Object10 is exactly south of Object12.
The uniquely guaranteed answer is South. It is displayed at D.
Answer: D
```
### Symbolic
```text
South:
P5: [p5] => North(Object5,Object10)
P7: [p7] => North(Object12,Object5)
P5-X1: [P5] => x(Object5) = x(Object10)
P5-Y1: [P5] => y(Object5) > y(Object10)
P7-X1: [P7] => x(Object12) = x(Object5)
P7-Y1: [P7] => y(Object12) > y(Object5)
PATH-X-1-INV: [P5-X1] => x(Object10) = x(Object5)
PATH-X-2-INV: [P7-X1] => x(Object5) = x(Object12)
D-X-1: [PATH-X-1-INV, PATH-X-2-INV] => x(Object10) = x(Object12)
PATH-Y-1-INV: [P5-Y1] => y(Object10) < y(Object5)
PATH-Y-2-INV: [P7-Y1] => y(Object5) < y(Object12)
D-Y-1: [PATH-Y-1-INV, PATH-Y-2-INV] => y(Object10) < y(Object12)
Q-DIR: [D-X-1, D-Y-1] => South(Object10,Object12)
The uniquely guaranteed answer is South. It is displayed at D.
Answer: D
```

## contingent-0-base-direction-m0-single
Split: dev; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object4, Object5, Object1, Object2, Object3, Object0
Query: direction Object4 Object3
Consider a map with multiple objects:
Object4 is to the Southeast of Object0.
Object3 is to the Southeast of Object0.
Object1 is to the West of Object3.
Object5 is to the West of Object3.
Object3 is to the East of Object2.
Question: In which direction is Object4 relative to Object3 in every valid map?
A. Northwest
B. East
C. South
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object4 at (4, 1); Object5 at (3, 0); Object1 at (2, 0); Object2 at (1, 0); Object3 at (4, 0); Object0 at (0, 2). In this map, direction=North.
Northeast:
A valid map places Object4 at (5, 1); Object5 at (3, 0); Object1 at (2, 0); Object2 at (1, 0); Object3 at (4, 0); Object0 at (0, 2). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object4:(4,1), Object5:(3,0), Object1:(2,0), Object2:(1,0), Object3:(4,0), Object0:(0,2)} => direction=North.
Northeast:
Valid map {Object4:(5,1), Object5:(3,0), Object1:(2,0), Object2:(1,0), Object3:(4,0), Object0:(0,2)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## contingent-1-base-direction-m0-single
Split: test; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object5, Object2, Object1, Object7, Object0, Object4, Object3, Object8, Object6
Query: direction Object8 Object6
Consider a map with multiple objects:
Object4 is to the South of Object6.
Object1 is to the Southwest of Object8.
Object6 is to the North of Object5.
Object3 is to the Northwest of Object2.
Object1 is to the Southwest of Object6.
Object6 is to the North of Object7.
Object6 is to the North of Object0.
Question: In which direction is Object8 relative to Object6 in every valid map?
A. Northeast
B. Northwest
C. South
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object5 at (2, 6); Object2 at (2, 4); Object1 at (1, 0); Object7 at (2, 3); Object0 at (2, 2); Object4 at (2, 1); Object3 at (0, 5); Object8 at (2, 8); Object6 at (2, 7). In this map, direction=North.
Northeast:
A valid map places Object5 at (2, 6); Object2 at (2, 4); Object1 at (1, 0); Object7 at (2, 3); Object0 at (2, 2); Object4 at (2, 1); Object3 at (0, 5); Object8 at (3, 8); Object6 at (2, 7). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object5:(2,6), Object2:(2,4), Object1:(1,0), Object7:(2,3), Object0:(2,2), Object4:(2,1), Object3:(0,5), Object8:(2,8), Object6:(2,7)} => direction=North.
Northeast:
Valid map {Object5:(2,6), Object2:(2,4), Object1:(1,0), Object7:(2,3), Object0:(2,2), Object4:(2,1), Object3:(0,5), Object8:(3,8), Object6:(2,7)} => direction=Northeast.
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
Objects: Object7, Object8, Object3, Object10, Object1, Object6, Object11, Object0, Object4, Object5, Object9, Object2
Query: direction Object5 Object10
Consider a map with multiple objects:
Object11 is to the East of Object10.
Object10 is to the Northwest of Object6.
Object5 is to the Northwest of Object6.
Object10 is to the West of Object7.
Object1 is to the East of Object10.
Object3 is to the Northwest of Object8.
Object0 is to the East of Object10.
Object9 is to the Southeast of Object2.
Object10 is to the West of Object4.
Question: In which direction is Object5 relative to Object10 in every valid map?
A. Southwest
B. East
C. Northeast
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object7 at (6, 5); Object8 at (6, 3); Object3 at (5, 4); Object10 at (0, 5); Object1 at (4, 5); Object6 at (6, 2); Object11 at (3, 5); Object0 at (2, 5); Object4 at (1, 5); Object5 at (0, 6); Object9 at (6, 0); Object2 at (5, 1). In this map, direction=North.
Northeast:
A valid map places Object7 at (7, 5); Object8 at (7, 3); Object3 at (6, 4); Object10 at (0, 5); Object1 at (5, 5); Object6 at (7, 2); Object11 at (4, 5); Object0 at (3, 5); Object4 at (2, 5); Object5 at (1, 6); Object9 at (7, 0); Object2 at (6, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object7:(6,5), Object8:(6,3), Object3:(5,4), Object10:(0,5), Object1:(4,5), Object6:(6,2), Object11:(3,5), Object0:(2,5), Object4:(1,5), Object5:(0,6), Object9:(6,0), Object2:(5,1)} => direction=North.
Northeast:
Valid map {Object7:(7,5), Object8:(7,3), Object3:(6,4), Object10:(0,5), Object1:(5,5), Object6:(7,2), Object11:(4,5), Object0:(3,5), Object4:(2,5), Object5:(1,6), Object9:(7,0), Object2:(6,1)} => direction=Northeast.
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
Objects: Object3, Object5, Object1, Object4, Object0, Object2
Query: direction Object3 Object5
Consider a map with multiple objects:
Object2 is to the Northeast of Object5.
Object2 is to the Northeast of Object3.
Object4 is to the North of Object5.
Object1 is to the North of Object5.
Object5 is to the South of Object0.
Question: In which direction is Object3 relative to Object5 in every valid map?
A. Southeast
B. North
C. Northeast
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object3 at (0, 4); Object5 at (0, 0); Object1 at (0, 3); Object4 at (0, 2); Object0 at (0, 1); Object2 at (1, 5). In this map, direction=North.
Northeast:
A valid map places Object3 at (1, 4); Object5 at (0, 0); Object1 at (0, 3); Object4 at (0, 2); Object0 at (0, 1); Object2 at (2, 5). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object3:(0,4), Object5:(0,0), Object1:(0,3), Object4:(0,2), Object0:(0,1), Object2:(1,5)} => direction=North.
Northeast:
Valid map {Object3:(1,4), Object5:(0,0), Object1:(0,3), Object4:(0,2), Object0:(0,1), Object2:(2,5)} => direction=Northeast.
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
Objects: Object0, Object1, Object2, Object6, Object7, Object5, Object4, Object3, Object8
Query: direction Object4 Object2
Consider a map with multiple objects:
Object3 is to the North of Object2.
Object6 is to the North of Object2.
Object5 is to the Northeast of Object2.
Object0 is to the East of Object7.
Object4 is to the Southwest of Object5.
Object2 is to the South of Object1.
Object8 is to the North of Object2.
Question: In which direction is Object4 relative to Object2 in every valid map?
A. North
B. South
C. East
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object0 at (1, 6); Object1 at (1, 5); Object2 at (1, 0); Object6 at (1, 4); Object7 at (0, 6); Object5 at (2, 7); Object4 at (1, 3); Object3 at (1, 2); Object8 at (1, 1). In this map, direction=North.
Northeast:
A valid map places Object0 at (1, 5); Object1 at (1, 4); Object2 at (1, 0); Object6 at (1, 3); Object7 at (0, 5); Object5 at (3, 7); Object4 at (2, 6); Object3 at (1, 2); Object8 at (1, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object0:(1,6), Object1:(1,5), Object2:(1,0), Object6:(1,4), Object7:(0,6), Object5:(2,7), Object4:(1,3), Object3:(1,2), Object8:(1,1)} => direction=North.
Northeast:
Valid map {Object0:(1,5), Object1:(1,4), Object2:(1,0), Object6:(1,3), Object7:(0,5), Object5:(3,7), Object4:(2,6), Object3:(1,2), Object8:(1,1)} => direction=Northeast.
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
Objects: Object1, Object2, Object0, Object4, Object3
Query: direction Object4 Object0
Consider a map with multiple objects:
Object2 is to the East of Object4.
Object3 is to the East of Object2.
Object1 is to the East of Object3.
Question: In which direction is Object4 relative to Object0 in every valid map?
A. Southwest
B. East
C. North
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object1 at (3, 1); Object2 at (1, 1); Object0 at (0, 0); Object4 at (0, 1); Object3 at (2, 1). In this map, direction=North.
Northeast:
A valid map places Object1 at (4, 1); Object2 at (2, 1); Object0 at (0, 0); Object4 at (1, 1); Object3 at (3, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object1:(3,1), Object2:(1,1), Object0:(0,0), Object4:(0,1), Object3:(2,1)} => direction=North.
Northeast:
Valid map {Object1:(4,1), Object2:(2,1), Object0:(0,0), Object4:(1,1), Object3:(3,1)} => direction=Northeast.
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
Objects: Object2, Object3, Object1, Object7, Object4, Object5, Object6, Object0
Query: direction Object4 Object2
Consider a map with multiple objects:
Object5 is to the West of Object4.
Object7 is to the East of Object6.
Object5 is to the East of Object7.
Object6 is to the East of Object1.
Object0 is to the Northwest of Object3.
Question: In which direction is Object4 relative to Object2 in every valid map?
A. South
B. West
C. North
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object2 at (4, 2); Object3 at (4, 0); Object1 at (0, 3); Object7 at (2, 3); Object4 at (4, 3); Object5 at (3, 3); Object6 at (1, 3); Object0 at (0, 1). In this map, direction=North.
Northeast:
A valid map places Object2 at (4, 2); Object3 at (4, 0); Object1 at (0, 3); Object7 at (2, 3); Object4 at (5, 3); Object5 at (3, 3); Object6 at (1, 3); Object0 at (0, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object2:(4,2), Object3:(4,0), Object1:(0,3), Object7:(2,3), Object4:(4,3), Object5:(3,3), Object6:(1,3), Object0:(0,1)} => direction=North.
Northeast:
Valid map {Object2:(4,2), Object3:(4,0), Object1:(0,3), Object7:(2,3), Object4:(5,3), Object5:(3,3), Object6:(1,3), Object0:(0,1)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## joint-2-base-direction-m0-single
Split: dev; case: underdetermined; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object2, Object9, Object1, Object6, Object10, Object0, Object7, Object3, Object5, Object8, Object4
Query: direction Object1 Object3
Consider a map with multiple objects:
Object2 is to the South of Object6.
Object7 is to the West of Object9.
Object4 is to the Northwest of Object5.
Object1 is to the West of Object8.
Object7 is to the East of Object10.
Object10 is to the East of Object0.
Object8 is to the West of Object0.
Question: In which direction is Object1 relative to Object3 in every valid map?
A. West
B. Southwest
C. Northeast
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object2 at (5, 4); Object9 at (5, 3); Object1 at (0, 3); Object6 at (5, 5); Object10 at (3, 3); Object0 at (2, 3); Object7 at (4, 3); Object3 at (0, 0); Object5 at (5, 1); Object8 at (1, 3); Object4 at (0, 2). In this map, direction=North.
Northeast:
A valid map places Object2 at (6, 4); Object9 at (6, 3); Object1 at (1, 3); Object6 at (6, 5); Object10 at (4, 3); Object0 at (3, 3); Object7 at (5, 3); Object3 at (0, 0); Object5 at (6, 1); Object8 at (2, 3); Object4 at (1, 2). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object2:(5,4), Object9:(5,3), Object1:(0,3), Object6:(5,5), Object10:(3,3), Object0:(2,3), Object7:(4,3), Object3:(0,0), Object5:(5,1), Object8:(1,3), Object4:(0,2)} => direction=North.
Northeast:
Valid map {Object2:(6,4), Object9:(6,3), Object1:(1,3), Object6:(6,5), Object10:(4,3), Object0:(3,3), Object7:(5,3), Object3:(0,0), Object5:(6,1), Object8:(2,3), Object4:(1,2)} => direction=Northeast.
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
Objects: Object3, Object2, Object4, Object1, Object0
Query: direction Object2 Object4
Consider a map with multiple objects:
Object2 is to the West of Object0.
Object3 is to the West of Object1.
Object3 is to the East of Object0.
Question: In which direction is Object2 relative to Object4 in every valid map?
A. Northwest
B. Southeast
C. West
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object3 at (2, 1); Object2 at (0, 1); Object4 at (0, 0); Object1 at (3, 1); Object0 at (1, 1). In this map, direction=North.
Northeast:
A valid map places Object3 at (3, 1); Object2 at (1, 1); Object4 at (0, 0); Object1 at (4, 1); Object0 at (2, 1). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object3:(2,1), Object2:(0,1), Object4:(0,0), Object1:(3,1), Object0:(1,1)} => direction=North.
Northeast:
Valid map {Object3:(3,1), Object2:(1,1), Object4:(0,0), Object1:(4,1), Object0:(2,1)} => direction=Northeast.
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
Objects: Object4, Object2, Object7, Object5, Object3, Object0, Object6, Object1
Query: direction Object5 Object2
Consider a map with multiple objects:
Object6 is to the North of Object0.
Object7 is to the North of Object6.
Object7 is to the South of Object4.
Object5 is to the North of Object4.
Object1 is to the Southwest of Object3.
Question: In which direction is Object5 relative to Object2 in every valid map?
A. North
B. Northwest
C. Southeast
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
North:
A valid map places Object4 at (1, 6); Object2 at (1, 5); Object7 at (1, 4); Object5 at (1, 7); Object3 at (1, 3); Object0 at (1, 1); Object6 at (1, 2); Object1 at (0, 0). In this map, direction=North.
Northeast:
A valid map places Object4 at (1, 6); Object2 at (0, 1); Object7 at (1, 5); Object5 at (1, 7); Object3 at (1, 4); Object0 at (1, 2); Object6 at (1, 3); Object1 at (0, 0). In this map, direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```
### Symbolic
```text
North:
Valid map {Object4:(1,6), Object2:(1,5), Object7:(1,4), Object5:(1,7), Object3:(1,3), Object0:(1,1), Object6:(1,2), Object1:(0,0)} => direction=North.
Northeast:
Valid map {Object4:(1,6), Object2:(0,1), Object7:(1,5), Object5:(1,7), Object3:(1,4), Object0:(1,2), Object6:(1,3), Object1:(0,0)} => direction=Northeast.
These valid maps give different answers, so there is no invariant answer.
Answer: E
```

## cycle-0-base-direction-m0-single
Split: test; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object2, Object4, Object5, Object1, Object7, Object6, Object3, Object0
Query: direction Object2 Object7
Consider a map with multiple objects:
Object3 is to the North of Object6.
Object7 is to the East of Object0.
Object5 is to the West of Object7.
Object1 is to the South of Object3.
Object1 is to the North of Object6.
Object7 is to the East of Object2.
Object4 is to the West of Object7.
Question: In which direction is Object2 relative to Object7 in every valid map?
A. North
B. West
C. East
D. Southeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
West:
P6: From p6, Object7 is exactly east of Object2.
P6-X1: From P6, Object7's x-coordinate is greater than Object2's x-coordinate.
P6-Y1: From P6, Object7's y-coordinate is equal to Object2's y-coordinate.
Q-DIR: From P6-X1, P6-Y1, Object2 is exactly west of Object7.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
West:
P6: [p6] => East(Object7,Object2)
P6-X1: [P6] => x(Object7) > x(Object2)
P6-Y1: [P6] => y(Object7) = y(Object2)
Q-DIR: [P6-X1, P6-Y1] => West(Object2,Object7)
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```

## cycle-0-inconsistent-direction-m0-single
Split: test; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object1, Object5, Object4, Object0, Object2, Object3, Object7, Object6
Query: direction Object6 Object5
Consider a map with multiple objects:
Object5 is to the East of Object7.
Object4 is to the North of Object3.
Object2 is to the West of Object5.
Object3 is to the North of Object1.
Object1 is to the North of Object4.
Object6 is to the West of Object5.
Object5 is to the East of Object0.
Question: In which direction is Object6 relative to Object5 in every valid map?
A. Northeast
B. North
C. Northwest
D. East
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p2, Object3's y-coordinate is less than Object4's y-coordinate.
From p5, Object4's y-coordinate is less than Object1's y-coordinate.
From p4, Object1's y-coordinate is less than Object3's y-coordinate.
Composing this path gives Object3's y-coordinate less than Object3's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p2, p4, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p2: y(Object3) < y(Object4)
p5: y(Object4) < y(Object1)
p4: y(Object1) < y(Object3)
Transitivity: y(Object3) < y(Object3)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p2, p4, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-1-base-direction-m0-single
Split: dev; case: unique; answer: A

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object1, Object4, Object8, Object7, Object9, Object0, Object3, Object2, Object6, Object11, Object5, Object10
Query: direction Object9 Object8
Consider a map with multiple objects:
Object2 is to the North of Object5.
Object6 is to the North of Object2.
Object8 is to the West of Object3.
Object6 is to the South of Object1.
Object8 is to the West of Object0.
Object7 is to the North of Object4.
Object8 is to the West of Object9.
Object8 is to the West of Object10.
Object11 is to the East of Object8.
Object1 is to the North of Object5.
Question: In which direction is Object9 relative to Object8 in every valid map?
A. East
B. Southwest
C. West
D. Northwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
East:
P7: From p7, Object8 is exactly west of Object9.
P7-X1: From P7, Object8's x-coordinate is less than Object9's x-coordinate.
P7-Y1: From P7, Object8's y-coordinate is equal to Object9's y-coordinate.
Q-DIR: From P7-X1, P7-Y1, Object9 is exactly east of Object8.
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```
### Symbolic
```text
East:
P7: [p7] => West(Object8,Object9)
P7-X1: [P7] => x(Object8) < x(Object9)
P7-Y1: [P7] => y(Object8) = y(Object9)
Q-DIR: [P7-X1, P7-Y1] => East(Object9,Object8)
The uniquely guaranteed answer is East. It is displayed at A.
Answer: A
```

## cycle-1-inconsistent-direction-m0-single
Split: dev; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object2, Object6, Object9, Object1, Object8, Object11, Object3, Object4, Object10, Object5, Object0, Object7
Query: direction Object2 Object11
Consider a map with multiple objects:
Object11 is to the West of Object4.
Object11 is to the West of Object2.
Object5 is to the East of Object11.
Object1 is to the South of Object10.
Object0 is to the South of Object1.
Object0 is to the North of Object8.
Object6 is to the Southeast of Object9.
Object10 is to the South of Object8.
Object7 is to the East of Object11.
Object3 is to the East of Object11.
Question: In which direction is Object2 relative to Object11 in every valid map?
A. West
B. South
C. North
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p4, Object1's y-coordinate is less than Object10's y-coordinate.
From p8, Object10's y-coordinate is less than Object8's y-coordinate.
From p6, Object8's y-coordinate is less than Object0's y-coordinate.
From p5, Object0's y-coordinate is less than Object1's y-coordinate.
Composing this path gives Object1's y-coordinate less than Object1's y-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p4, p5, p6, p8. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p4: y(Object1) < y(Object10)
p8: y(Object10) < y(Object8)
p6: y(Object8) < y(Object0)
p5: y(Object0) < y(Object1)
Transitivity: y(Object1) < y(Object1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p4, p5, p6, p8. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## cycle-2-base-direction-m0-single
Split: train; case: unique; answer: B

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object10, Object15, Object9, Object3, Object2, Object1, Object6, Object5, Object12, Object13, Object0, Object11, Object7, Object4, Object8, Object14
Query: direction Object4 Object6
Consider a map with multiple objects:
Object13 is to the North of Object5.
Object10 is to the West of Object6.
Object5 is to the North of Object11.
Object3 is to the South of Object1.
Object12 is to the West of Object6.
Object6 is to the East of Object8.
Object6 is to the East of Object15.
Object7 is to the West of Object9.
Object14 is to the North of Object2.
Object0 is to the West of Object6.
Object11 is to the North of Object2.
Object6 is to the East of Object4.
Object14 is to the North of Object13.
Question: In which direction is Object4 relative to Object6 in every valid map?
A. Northwest
B. West
C. Southeast
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
West:
P12: From p12, Object6 is exactly east of Object4.
P12-X1: From P12, Object6's x-coordinate is greater than Object4's x-coordinate.
P12-Y1: From P12, Object6's y-coordinate is equal to Object4's y-coordinate.
Q-DIR: From P12-X1, P12-Y1, Object4 is exactly west of Object6.
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
```
### Symbolic
```text
West:
P12: [p12] => East(Object6,Object4)
P12-X1: [P12] => x(Object6) > x(Object4)
P12-Y1: [P12] => y(Object6) = y(Object4)
Q-DIR: [P12-X1, P12-Y1] => West(Object4,Object6)
The uniquely guaranteed answer is West. It is displayed at B.
Answer: B
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
Objects: Object4, Object1, Object5, Object3, Object2, Object0
Query: direction Object2 Object1
Consider a map with multiple objects:
Object4 is to the South of Object1.
Object2 is to the East of Object0.
Object1 is to the South of Object2.
Object1 is to the South of Object5.
Object5 is to the East of Object0.
Object3 is to the South of Object1.
Question: In which direction is Object2 relative to Object1 in every valid map?
A. East
B. South
C. Northeast
D. North
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p4, Object5's x-coordinate is equal to Object1's x-coordinate.
From p3, Object1's x-coordinate is equal to Object2's x-coordinate.
Composing this path gives Object5's x-coordinate equal to Object2's x-coordinate.
From p5, Object5's y-coordinate is equal to Object0's y-coordinate.
From p2, Object0's y-coordinate is equal to Object2's y-coordinate.
Composing this path gives Object5's y-coordinate equal to Object2's y-coordinate.
Object5 and Object2 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p4, p5. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p4: x(Object5) = x(Object1)
p3: x(Object1) = x(Object2)
Transitivity: x(Object5) = x(Object2)
p5: y(Object5) = y(Object0)
p2: y(Object0) = y(Object2)
Transitivity: y(Object5) = y(Object2)
Object5 and Object2 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p4, p5. No minimality claim is made.
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
Objects: Object7, Object0, Object3, Object2, Object8, Object4, Object6, Object1, Object5
Query: direction Object6 Object8
Consider a map with multiple objects:
Object8 is to the North of Object7.
Object2 is to the West of Object4.
Object8 is to the South of Object4.
Object8 is to the South of Object6.
Object8 is to the North of Object3.
Object5 is to the South of Object8.
Object2 is to the West of Object6.
Object0 is to the South of Object1.
Question: In which direction is Object6 relative to Object8 in every valid map?
A. North
B. South
C. Southwest
D. West
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p3, Object4's x-coordinate is equal to Object8's x-coordinate.
From p4, Object8's x-coordinate is equal to Object6's x-coordinate.
Composing this path gives Object4's x-coordinate equal to Object6's x-coordinate.
From p2, Object4's y-coordinate is equal to Object2's y-coordinate.
From p7, Object2's y-coordinate is equal to Object6's y-coordinate.
Composing this path gives Object4's y-coordinate equal to Object6's y-coordinate.
Object4 and Object6 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p4, p7. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p3: x(Object4) = x(Object8)
p4: x(Object8) = x(Object6)
Transitivity: x(Object4) = x(Object6)
p2: y(Object4) = y(Object2)
p7: y(Object2) = y(Object6)
Transitivity: y(Object4) = y(Object6)
Object4 and Object6 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p2, p3, p4, p7. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## coincidence-2-base-direction-m0-single
Split: dev; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object0, Object1, Object5, Object3, Object11, Object4, Object9, Object8, Object10, Object2, Object7
Query: direction Object11 Object5
Consider a map with multiple objects:
Object8 is to the South of Object5.
Object6 is to the North of Object5.
Object5 is to the South of Object2.
Object10 is to the North of Object5.
Object7 is to the North of Object4.
Object5 is to the North of Object11.
Object1 is to the North of Object9.
Object0 is to the East of Object8.
Object11 is to the West of Object0.
Object5 is to the South of Object3.
Question: In which direction is Object11 relative to Object5 in every valid map?
A. West
B. North
C. Northwest
D. Southwest
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p6, Object11's x-coordinate is equal to Object5's x-coordinate.
From p1, Object5's x-coordinate is equal to Object8's x-coordinate.
Composing this path gives Object11's x-coordinate equal to Object8's x-coordinate.
From p9, Object11's y-coordinate is equal to Object0's y-coordinate.
From p8, Object0's y-coordinate is equal to Object8's y-coordinate.
Composing this path gives Object11's y-coordinate equal to Object8's y-coordinate.
Object11 and Object8 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p6, p8, p9. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p6: x(Object11) = x(Object5)
p1: x(Object5) = x(Object8)
Transitivity: x(Object11) = x(Object8)
p9: y(Object11) = y(Object0)
p8: y(Object0) = y(Object8)
Transitivity: y(Object11) = y(Object8)
Object11 and Object8 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p6, p8, p9. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```

## coincidence-3-base-direction-m0-single
Split: train; case: inconsistent; answer: E

### Prompt
```text
Interface: spatialentail-composition-v1
Contract: single
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid map satisfies every premise and these assumptions. Could means in at least one valid map. Under every contract, select E if no valid map exists.
Select the one A-D value that is the invariant answer in every valid map. Select F if that unique answer is unlisted; otherwise select E. For Selection, select E when the premises do not establish exactly one qualifying answer, including when no object or multiple objects qualify.
Eligible objects are independent of the menu. Selection possibilities need not hold simultaneously; Count possibilities must hold jointly in one valid map. Zero is an ordinary Count value.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object3, Object1, Object4, Object5, Object0, Object2
Query: direction Object2 Object1
Consider a map with multiple objects:
Object1 is to the West of Object3.
Object5 is to the West of Object1.
Object4 is to the West of Object1.
Object0 is to the North of Object3.
Object2 is to the East of Object1.
Object0 is to the North of Object2.
Question: In which direction is Object2 relative to Object1 in every valid map?
A. Southwest
B. North
C. East
D. Northeast
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p4, Object3's x-coordinate is equal to Object0's x-coordinate.
From p6, Object0's x-coordinate is equal to Object2's x-coordinate.
Composing this path gives Object3's x-coordinate equal to Object2's x-coordinate.
From p1, Object3's y-coordinate is equal to Object1's y-coordinate.
From p5, Object1's y-coordinate is equal to Object2's y-coordinate.
Composing this path gives Object3's y-coordinate equal to Object2's y-coordinate.
Object3 and Object2 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p4, p5, p6. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p4: x(Object3) = x(Object0)
p6: x(Object0) = x(Object2)
Transitivity: x(Object3) = x(Object2)
p1: y(Object3) = y(Object1)
p5: y(Object1) = y(Object2)
Transitivity: y(Object3) = y(Object2)
Object3 and Object2 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p1, p4, p5, p6. No minimality claim is made.
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
Objects: Object6, Object1, Object5, Object4, Object7, Object0, Object2, Object3, Object8
Query: direction Object2 Object5
Consider a map with multiple objects:
Object5 is to the South of Object8.
Object3 is to the North of Object5.
Object6 is to the North of Object5.
Object4 is to the East of Object2.
Object5 is to the North of Object2.
Object1 is to the North of Object7.
Object0 is to the West of Object4.
Object0 is to the South of Object5.
Question: In which direction is Object2 relative to Object5 in every valid map?
A. Northeast
B. Southeast
C. West
D. South
E. Cannot be determined
F. None of the above
```
### Natural
```text
From p8, Object0's x-coordinate is equal to Object5's x-coordinate.
From p5, Object5's x-coordinate is equal to Object2's x-coordinate.
Composing this path gives Object0's x-coordinate equal to Object2's x-coordinate.
From p7, Object0's y-coordinate is equal to Object4's y-coordinate.
From p4, Object4's y-coordinate is equal to Object2's y-coordinate.
Composing this path gives Object0's y-coordinate equal to Object2's y-coordinate.
Object0 and Object2 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p4, p5, p7, p8. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
### Symbolic
```text
p8: x(Object0) = x(Object5)
p5: x(Object5) = x(Object2)
Transitivity: x(Object0) = x(Object2)
p7: y(Object0) = y(Object4)
p4: y(Object4) = y(Object2)
Transitivity: y(Object0) = y(Object2)
Object0 and Object2 would share both coordinates, violating the distinct-point axiom.
Conflicting premises: p4, p5, p7, p8. No minimality claim is made.
No valid map exists; select E regardless of the query.
Answer: E
```
