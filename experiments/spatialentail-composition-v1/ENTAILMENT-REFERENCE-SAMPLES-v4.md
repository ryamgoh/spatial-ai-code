# Paired Reference Review Samples

Deterministic first occurrence per review category; not a random quality estimate.

## production-b2758-s5-count-m0-unique-entailed

Categories: query-count, contract-unique-entailed, depth-9, escape-None.
Split: extrapolation-test; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object23, Object15, Object21, Object14, Object25, Object12, Object19, Object29, Object4, Object27, Object5, Object7, Object16, Object24, Object8, Object22, Object17, Object9, Object10, Object20, Object11, Object1, Object13, Object26, Object0, Object18, Object28, Object3, Object2
Query: count Southeast Object5
Eligible: Object16, Object6, Object24, Object9, Object25, Object4, Object22, Object10
Consider a map with multiple objects:
Object5 is to the Southeast of Object21.
Object5 is to the Northwest of Object22.
Object10 is to the Northwest of Object5.
Object16 is to the North of Object17.
Object25 is to the Northwest of Object26.
Object27 is to the Southeast of Object20.
Object1 is to the Southeast of Object18.
Object26 is to the Southeast of Object9.
Object14 is to the Southeast of Object8.
Object8 is to the Southeast of Object16.
Object12 is to the Northwest of Object20.
Object28 is to the Northwest of Object24.
Object11 is to the Northwest of Object12.
Object21 is to the Southeast of Object4.
Object5 is to the Northwest of Object6.
Object7 is to the Southeast of Object0.
Object23 is to the Southeast of Object3.
Object25 is to the Southeast of Object4.
Object27 is to the Northwest of Object28.
Object2 is to the Northwest of Object11.
Object29 is to the Northwest of Object13.
Object5 is to the Northwest of Object18.
Object15 is to the East of Object27.
Object13 is to the Northwest of Object19.
Object2 is to the Southeast of Object1.
Object29 is to the Southeast of Object26.
Object14 is to the Northwest of Object21.
Object8 is to the Southeast of Object19.
Question: What exact count of eligible objects southeast of Object5 is guaranteed?
A. 8
B. 4
C. 2
D. 3
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object16, Object8, Object14, Object21 and Object5 (premise sources: 10, 9, 27 and 1). So Object5 has a greater x-coordinate than Object16.
Object16 never qualifies for the queried relation.
Object6 has a greater x-coordinate than Object5 (premise sources: 15).
Object5 has a greater y-coordinate than Object6 (premise sources: 15).
Object6 always qualifies for the queried relation.
The x-coordinates increase strictly along Object5, Object18, Object1, Object2, Object11, Object12, Object20, Object27, Object28 and Object24 (premise sources: 22, 7, 25, 20, 13, 11, 6, 19 and 12). So Object24 has a greater x-coordinate than Object5.
The y-coordinates increase strictly along Object24, Object28, Object27, Object20, Object12, Object11, Object2, Object1, Object18 and Object5 (premise sources: 12, 19, 6, 11, 13, 20, 25, 7 and 22). So Object5 has a greater y-coordinate than Object24.
Object24 always qualifies for the queried relation.
The x-coordinates increase strictly along Object9, Object26, Object29, Object13, Object19, Object8, Object14, Object21 and Object5 (premise sources: 8, 26, 21, 24, 28, 9, 27 and 1). So Object5 has a greater x-coordinate than Object9.
Object9 never qualifies for the queried relation.
The x-coordinates increase strictly along Object25, Object26, Object29, Object13, Object19, Object8, Object14, Object21 and Object5 (premise sources: 5, 26, 21, 24, 28, 9, 27 and 1). So Object5 has a greater x-coordinate than Object25.
Object25 never qualifies for the queried relation.
The x-coordinates increase strictly along Object4, Object21 and Object5 (premise sources: 14 and 1). So Object5 has a greater x-coordinate than Object4.
Object4 never qualifies for the queried relation.
Object22 has a greater x-coordinate than Object5 (premise sources: 2).
Object5 has a greater y-coordinate than Object22 (premise sources: 2).
Object22 always qualifies for the queried relation.
Object5 has a greater x-coordinate than Object10 (premise sources: 3).
Object10 never qualifies for the queried relation.
The qualifying candidates are fixed: Object6, Object24 and Object22. Exactly 3 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object16, Object9, Object25, Object4 and Object10 are northwest of Object5. Object6, Object24 and Object22 are southeast of Object5. Only Object22, Object24 and Object6 qualify in this arrangement. This gives a count of 3.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 8 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

### Symbolic

[p10, p9, p27, p1] x(Object16) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object16) < x(Object5).
NOT SE(Object16,Object5) in every valid arrangement.
[p15] x(Object5) < x(Object6).
[p15] y(Object6) < y(Object5).
SE(Object6,Object5) in every valid arrangement.
[p22, p7, p25, p20, p13, p11, p6, p19, p12] x(Object5) < x(Object18) < x(Object1) < x(Object2) < x(Object11) < x(Object12) < x(Object20) < x(Object27) < x(Object28) < x(Object24) => x(Object5) < x(Object24).
[p12, p19, p6, p11, p13, p20, p25, p7, p22] y(Object24) < y(Object28) < y(Object27) < y(Object20) < y(Object12) < y(Object11) < y(Object2) < y(Object1) < y(Object18) < y(Object5) => y(Object24) < y(Object5).
SE(Object24,Object5) in every valid arrangement.
[p8, p26, p21, p24, p28, p9, p27, p1] x(Object9) < x(Object26) < x(Object29) < x(Object13) < x(Object19) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object9) < x(Object5).
NOT SE(Object9,Object5) in every valid arrangement.
[p5, p26, p21, p24, p28, p9, p27, p1] x(Object25) < x(Object26) < x(Object29) < x(Object13) < x(Object19) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object25) < x(Object5).
NOT SE(Object25,Object5) in every valid arrangement.
[p14, p1] x(Object4) < x(Object21) < x(Object5) => x(Object4) < x(Object5).
NOT SE(Object4,Object5) in every valid arrangement.
[p2] x(Object5) < x(Object22).
[p2] y(Object22) < y(Object5).
SE(Object22,Object5) in every valid arrangement.
[p3] x(Object10) < x(Object5).
NOT SE(Object10,Object5) in every valid arrangement.
Fixed qualifying set={Object6, Object24, Object22}; count=3 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Object16,Object5) & NW(Object9,Object5) & NW(Object25,Object5) & NW(Object4,Object5) & NW(Object10,Object5). SE(Object6,Object5) & SE(Object24,Object5) & SE(Object22,Object5). Qualifying objects = {Object22, Object24, Object6}. count=3.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The claim that count=8 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

## production-b2758-s5-count-m0-listed-entailed

Categories: contract-listed-entailed.
Split: extrapolation-test; contract: listed-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: listed-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all displayed A-D values that are entailed. Exclude every displayed value that is not entailed, including claims that could hold but are not guaranteed. Select F if no displayed value is entailed.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object23, Object15, Object21, Object14, Object25, Object12, Object19, Object29, Object4, Object27, Object5, Object7, Object16, Object24, Object8, Object22, Object17, Object9, Object10, Object20, Object11, Object1, Object13, Object26, Object0, Object18, Object28, Object3, Object2
Query: count Southeast Object5
Eligible: Object16, Object6, Object24, Object9, Object25, Object4, Object22, Object10
Consider a map with multiple objects:
Object5 is to the Southeast of Object21.
Object5 is to the Northwest of Object22.
Object10 is to the Northwest of Object5.
Object16 is to the North of Object17.
Object25 is to the Northwest of Object26.
Object27 is to the Southeast of Object20.
Object1 is to the Southeast of Object18.
Object26 is to the Southeast of Object9.
Object14 is to the Southeast of Object8.
Object8 is to the Southeast of Object16.
Object12 is to the Northwest of Object20.
Object28 is to the Northwest of Object24.
Object11 is to the Northwest of Object12.
Object21 is to the Southeast of Object4.
Object5 is to the Northwest of Object6.
Object7 is to the Southeast of Object0.
Object23 is to the Southeast of Object3.
Object25 is to the Southeast of Object4.
Object27 is to the Northwest of Object28.
Object2 is to the Northwest of Object11.
Object29 is to the Northwest of Object13.
Object5 is to the Northwest of Object18.
Object15 is to the East of Object27.
Object13 is to the Northwest of Object19.
Object2 is to the Southeast of Object1.
Object29 is to the Southeast of Object26.
Object14 is to the Northwest of Object21.
Object8 is to the Southeast of Object19.
Question: Which listed exact counts of eligible objects southeast of Object5 are entailed? Select all that apply.
A. 8
B. 4
C. 2
D. 3
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object16, Object8, Object14, Object21 and Object5 (premise sources: 10, 9, 27 and 1). So Object5 has a greater x-coordinate than Object16.
Object16 never qualifies for the queried relation.
Object6 has a greater x-coordinate than Object5 (premise sources: 15).
Object5 has a greater y-coordinate than Object6 (premise sources: 15).
Object6 always qualifies for the queried relation.
The x-coordinates increase strictly along Object5, Object18, Object1, Object2, Object11, Object12, Object20, Object27, Object28 and Object24 (premise sources: 22, 7, 25, 20, 13, 11, 6, 19 and 12). So Object24 has a greater x-coordinate than Object5.
The y-coordinates increase strictly along Object24, Object28, Object27, Object20, Object12, Object11, Object2, Object1, Object18 and Object5 (premise sources: 12, 19, 6, 11, 13, 20, 25, 7 and 22). So Object5 has a greater y-coordinate than Object24.
Object24 always qualifies for the queried relation.
The x-coordinates increase strictly along Object9, Object26, Object29, Object13, Object19, Object8, Object14, Object21 and Object5 (premise sources: 8, 26, 21, 24, 28, 9, 27 and 1). So Object5 has a greater x-coordinate than Object9.
Object9 never qualifies for the queried relation.
The x-coordinates increase strictly along Object25, Object26, Object29, Object13, Object19, Object8, Object14, Object21 and Object5 (premise sources: 5, 26, 21, 24, 28, 9, 27 and 1). So Object5 has a greater x-coordinate than Object25.
Object25 never qualifies for the queried relation.
The x-coordinates increase strictly along Object4, Object21 and Object5 (premise sources: 14 and 1). So Object5 has a greater x-coordinate than Object4.
Object4 never qualifies for the queried relation.
Object22 has a greater x-coordinate than Object5 (premise sources: 2).
Object5 has a greater y-coordinate than Object22 (premise sources: 2).
Object22 always qualifies for the queried relation.
Object5 has a greater x-coordinate than Object10 (premise sources: 3).
Object10 never qualifies for the queried relation.
The qualifying candidates are fixed: Object6, Object24 and Object22. Exactly 3 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object16, Object9, Object25, Object4 and Object10 are northwest of Object5. Object6, Object24 and Object22 are southeast of Object5. Only Object22, Object24 and Object6 qualify in this arrangement. This gives a count of 3.

The claim that exactly 8 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The displayed entailed-answer set is {3}.

Answer: D

### Symbolic

[p10, p9, p27, p1] x(Object16) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object16) < x(Object5).
NOT SE(Object16,Object5) in every valid arrangement.
[p15] x(Object5) < x(Object6).
[p15] y(Object6) < y(Object5).
SE(Object6,Object5) in every valid arrangement.
[p22, p7, p25, p20, p13, p11, p6, p19, p12] x(Object5) < x(Object18) < x(Object1) < x(Object2) < x(Object11) < x(Object12) < x(Object20) < x(Object27) < x(Object28) < x(Object24) => x(Object5) < x(Object24).
[p12, p19, p6, p11, p13, p20, p25, p7, p22] y(Object24) < y(Object28) < y(Object27) < y(Object20) < y(Object12) < y(Object11) < y(Object2) < y(Object1) < y(Object18) < y(Object5) => y(Object24) < y(Object5).
SE(Object24,Object5) in every valid arrangement.
[p8, p26, p21, p24, p28, p9, p27, p1] x(Object9) < x(Object26) < x(Object29) < x(Object13) < x(Object19) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object9) < x(Object5).
NOT SE(Object9,Object5) in every valid arrangement.
[p5, p26, p21, p24, p28, p9, p27, p1] x(Object25) < x(Object26) < x(Object29) < x(Object13) < x(Object19) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object25) < x(Object5).
NOT SE(Object25,Object5) in every valid arrangement.
[p14, p1] x(Object4) < x(Object21) < x(Object5) => x(Object4) < x(Object5).
NOT SE(Object4,Object5) in every valid arrangement.
[p2] x(Object5) < x(Object22).
[p2] y(Object22) < y(Object5).
SE(Object22,Object5) in every valid arrangement.
[p3] x(Object10) < x(Object5).
NOT SE(Object10,Object5) in every valid arrangement.
Fixed qualifying set={Object6, Object24, Object22}; count=3 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Object16,Object5) & NW(Object9,Object5) & NW(Object25,Object5) & NW(Object4,Object5) & NW(Object10,Object5). SE(Object6,Object5) & SE(Object24,Object5) & SE(Object22,Object5). Qualifying objects = {Object22, Object24, Object6}. count=3.

The claim that count=8 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The displayed entailed-answer set is {3}.

Answer: D

## production-b2758-s5-count-m0-exhaustive-entailed

Categories: contract-exhaustive-entailed.
Split: extrapolation-test; contract: exhaustive-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: exhaustive-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all A-D values only if they cover the complete nonempty set of individually entailed domain answers. Select F if any entailed answer is missing or no domain answer is entailed. Non-entailed answers outside the menu do not make coverage incomplete.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object6, Object23, Object15, Object21, Object14, Object25, Object12, Object19, Object29, Object4, Object27, Object5, Object7, Object16, Object24, Object8, Object22, Object17, Object9, Object10, Object20, Object11, Object1, Object13, Object26, Object0, Object18, Object28, Object3, Object2
Query: count Southeast Object5
Eligible: Object16, Object6, Object24, Object9, Object25, Object4, Object22, Object10
Consider a map with multiple objects:
Object5 is to the Southeast of Object21.
Object5 is to the Northwest of Object22.
Object10 is to the Northwest of Object5.
Object16 is to the North of Object17.
Object25 is to the Northwest of Object26.
Object27 is to the Southeast of Object20.
Object1 is to the Southeast of Object18.
Object26 is to the Southeast of Object9.
Object14 is to the Southeast of Object8.
Object8 is to the Southeast of Object16.
Object12 is to the Northwest of Object20.
Object28 is to the Northwest of Object24.
Object11 is to the Northwest of Object12.
Object21 is to the Southeast of Object4.
Object5 is to the Northwest of Object6.
Object7 is to the Southeast of Object0.
Object23 is to the Southeast of Object3.
Object25 is to the Southeast of Object4.
Object27 is to the Northwest of Object28.
Object2 is to the Northwest of Object11.
Object29 is to the Northwest of Object13.
Object5 is to the Northwest of Object18.
Object15 is to the East of Object27.
Object13 is to the Northwest of Object19.
Object2 is to the Southeast of Object1.
Object29 is to the Southeast of Object26.
Object14 is to the Northwest of Object21.
Object8 is to the Southeast of Object19.
Question: Which exact counts of eligible objects southeast of Object5 are entailed? Select the complete entailed-count set.
A. 8
B. 4
C. 2
D. 3
E. Cannot be determined
F. No complete entailed answer can be formed from A-D.

### Natural

The x-coordinates increase strictly along Object16, Object8, Object14, Object21 and Object5 (premise sources: 10, 9, 27 and 1). So Object5 has a greater x-coordinate than Object16.
Object16 never qualifies for the queried relation.
Object6 has a greater x-coordinate than Object5 (premise sources: 15).
Object5 has a greater y-coordinate than Object6 (premise sources: 15).
Object6 always qualifies for the queried relation.
The x-coordinates increase strictly along Object5, Object18, Object1, Object2, Object11, Object12, Object20, Object27, Object28 and Object24 (premise sources: 22, 7, 25, 20, 13, 11, 6, 19 and 12). So Object24 has a greater x-coordinate than Object5.
The y-coordinates increase strictly along Object24, Object28, Object27, Object20, Object12, Object11, Object2, Object1, Object18 and Object5 (premise sources: 12, 19, 6, 11, 13, 20, 25, 7 and 22). So Object5 has a greater y-coordinate than Object24.
Object24 always qualifies for the queried relation.
The x-coordinates increase strictly along Object9, Object26, Object29, Object13, Object19, Object8, Object14, Object21 and Object5 (premise sources: 8, 26, 21, 24, 28, 9, 27 and 1). So Object5 has a greater x-coordinate than Object9.
Object9 never qualifies for the queried relation.
The x-coordinates increase strictly along Object25, Object26, Object29, Object13, Object19, Object8, Object14, Object21 and Object5 (premise sources: 5, 26, 21, 24, 28, 9, 27 and 1). So Object5 has a greater x-coordinate than Object25.
Object25 never qualifies for the queried relation.
The x-coordinates increase strictly along Object4, Object21 and Object5 (premise sources: 14 and 1). So Object5 has a greater x-coordinate than Object4.
Object4 never qualifies for the queried relation.
Object22 has a greater x-coordinate than Object5 (premise sources: 2).
Object5 has a greater y-coordinate than Object22 (premise sources: 2).
Object22 always qualifies for the queried relation.
Object5 has a greater x-coordinate than Object10 (premise sources: 3).
Object10 never qualifies for the queried relation.
The qualifying candidates are fixed: Object6, Object24 and Object22. Exactly 3 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object16, Object9, Object25, Object4 and Object10 are northwest of Object5. Object6, Object24 and Object22 are southeast of Object5. Only Object22, Object24 and Object6 qualify in this arrangement. This gives a count of 3.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 8 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

### Symbolic

[p10, p9, p27, p1] x(Object16) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object16) < x(Object5).
NOT SE(Object16,Object5) in every valid arrangement.
[p15] x(Object5) < x(Object6).
[p15] y(Object6) < y(Object5).
SE(Object6,Object5) in every valid arrangement.
[p22, p7, p25, p20, p13, p11, p6, p19, p12] x(Object5) < x(Object18) < x(Object1) < x(Object2) < x(Object11) < x(Object12) < x(Object20) < x(Object27) < x(Object28) < x(Object24) => x(Object5) < x(Object24).
[p12, p19, p6, p11, p13, p20, p25, p7, p22] y(Object24) < y(Object28) < y(Object27) < y(Object20) < y(Object12) < y(Object11) < y(Object2) < y(Object1) < y(Object18) < y(Object5) => y(Object24) < y(Object5).
SE(Object24,Object5) in every valid arrangement.
[p8, p26, p21, p24, p28, p9, p27, p1] x(Object9) < x(Object26) < x(Object29) < x(Object13) < x(Object19) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object9) < x(Object5).
NOT SE(Object9,Object5) in every valid arrangement.
[p5, p26, p21, p24, p28, p9, p27, p1] x(Object25) < x(Object26) < x(Object29) < x(Object13) < x(Object19) < x(Object8) < x(Object14) < x(Object21) < x(Object5) => x(Object25) < x(Object5).
NOT SE(Object25,Object5) in every valid arrangement.
[p14, p1] x(Object4) < x(Object21) < x(Object5) => x(Object4) < x(Object5).
NOT SE(Object4,Object5) in every valid arrangement.
[p2] x(Object5) < x(Object22).
[p2] y(Object22) < y(Object5).
SE(Object22,Object5) in every valid arrangement.
[p3] x(Object10) < x(Object5).
NOT SE(Object10,Object5) in every valid arrangement.
Fixed qualifying set={Object6, Object24, Object22}; count=3 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Object16,Object5) & NW(Object9,Object5) & NW(Object25,Object5) & NW(Object4,Object5) & NW(Object10,Object5). SE(Object6,Object5) & SE(Object24,Object5) & SE(Object22,Object5). Qualifying objects = {Object22, Object24, Object6}. count=3.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The claim that count=8 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {3}.

Answer: D

## production-b416-s4-direction-m0-unique-entailed

Categories: query-direction, depth-5.
Split: train; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object23, Object1, Object8, Object19, Object25, Object18, Object21, Object6, Object10, Object4, Object0, Object26, Object27, Object13, Object12, Object20, Object17, Object9, Object16, Object11, Object22, Object5, Object3, Object7, Object2, Object24, Object14, Object15
Query: direction Object16 Object25
Consider a map with multiple objects:
Object9 is to the Northeast of Object3.
Object9 is to the Southwest of Object20.
Object24 is to the Southeast of Object23.
Object10 is to the Northwest of Object19.
Object16 is to the Southwest of Object15.
Object13 is to the Northeast of Object25.
Object21 is to the Northeast of Object11.
Object25 is to the Southwest of Object11.
Object14 is to the Northeast of Object11.
Object27 is to the Southwest of Object2.
Object17 is to the Southwest of Object7.
Object7 is to the Southwest of Object1.
Object5 is to the Southeast of Object24.
Object25 is to the Southeast of Object15.
Object5 is to the Northeast of Object16.
Object8 is to the Southwest of Object4.
Object26 is to the Northwest of Object23.
Object7 is to the Southwest of Object3.
Object27 is to the Northeast of Object1.
Object13 is to the Southwest of Object8.
Object0 is to the Northeast of Object21.
Object12 is to the Northwest of Object18.
Object22 is to the Northeast of Object0.
Object26 is to the Southeast of Object25.
Object25 is to the Southwest of Object17.
Object6 is to the North of Object27.
Question: Which single direction of Object16 relative to Object25 is entailed?
A. Northwest
B. Southeast
C. East
D. Southwest
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object16, Object15 and Object25 (premise sources: 5 and 14). So Object25 has a greater x-coordinate than Object16.
The y-coordinates increase strictly along Object16, Object5, Object24, Object23, Object26 and Object25 (premise sources: 15, 13, 3, 17 and 24). So Object25 has a greater y-coordinate than Object16.
Therefore, Object16 is southwest of Object25 in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. From left to right, the order is Object16, Object15 and Object25. From bottom to top, the order is Object16, Object25 and Object15. So Object16 is southwest of Object25.

The claim that Object16 is north of Object25 is false in Case 1, so it is not entailed.

The claim that Object16 is northeast of Object25 is false in Case 1, so it is not entailed.

The claim that Object16 is east of Object25 is false in Case 1, so it is not entailed.

The claim that Object16 is southeast of Object25 is false in Case 1, so it is not entailed.

The claim that Object16 is south of Object25 is false in Case 1, so it is not entailed.

The claim that Object16 is west of Object25 is false in Case 1, so it is not entailed.

The claim that Object16 is northwest of Object25 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: D

### Symbolic

[p5, p14] x(Object16) < x(Object15) < x(Object25) => x(Object16) < x(Object25).
[p15, p13, p3, p17, p24] y(Object16) < y(Object5) < y(Object24) < y(Object23) < y(Object26) < y(Object25) => y(Object16) < y(Object25).
SW(Object16,Object25) in every valid arrangement.

Case 1 (counterexample satisfying all premises): x(Object16) < x(Object15) < x(Object25). y(Object16) < y(Object25) < y(Object15). => SW(Object16,Object25).

The claim that N(Object16,Object25) is false in Case 1, so it is not entailed.

The claim that NE(Object16,Object25) is false in Case 1, so it is not entailed.

The claim that E(Object16,Object25) is false in Case 1, so it is not entailed.

The claim that SE(Object16,Object25) is false in Case 1, so it is not entailed.

The claim that S(Object16,Object25) is false in Case 1, so it is not entailed.

The claim that W(Object16,Object25) is false in Case 1, so it is not entailed.

The claim that NW(Object16,Object25) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: D

## production-b416-s4-which-m0-unique-entailed

Categories: query-which.
Split: train; contract: unique-entailed; answer: A.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object23, Object1, Object8, Object19, Object25, Object18, Object21, Object6, Object10, Object4, Object0, Object26, Object27, Object13, Object12, Object20, Object17, Object9, Object16, Object11, Object22, Object5, Object3, Object7, Object2, Object24, Object14, Object15
Query: which Southwest Object25
Eligible: Object22, Object16, Object2, Object4, Object14, Object20
Consider a map with multiple objects:
Object9 is to the Northeast of Object3.
Object9 is to the Southwest of Object20.
Object24 is to the Southeast of Object23.
Object10 is to the Northwest of Object19.
Object16 is to the Southwest of Object15.
Object13 is to the Northeast of Object25.
Object21 is to the Northeast of Object11.
Object25 is to the Southwest of Object11.
Object14 is to the Northeast of Object11.
Object27 is to the Southwest of Object2.
Object17 is to the Southwest of Object7.
Object7 is to the Southwest of Object1.
Object5 is to the Southeast of Object24.
Object25 is to the Southeast of Object15.
Object5 is to the Northeast of Object16.
Object8 is to the Southwest of Object4.
Object26 is to the Northwest of Object23.
Object7 is to the Southwest of Object3.
Object27 is to the Northeast of Object1.
Object13 is to the Southwest of Object8.
Object0 is to the Northeast of Object21.
Object12 is to the Northwest of Object18.
Object22 is to the Northeast of Object0.
Object26 is to the Southeast of Object25.
Object25 is to the Southwest of Object17.
Object6 is to the North of Object27.
Question: Which single eligible candidate is individually guaranteed to be southwest of Object25?
A. Object16
B. Object14
C. Object22
D. Object4
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object16, Object15 and Object25 (premise sources: 5 and 14). So Object25 has a greater x-coordinate than Object16.
The y-coordinates increase strictly along Object16, Object5, Object24, Object23, Object26 and Object25 (premise sources: 15, 13, 3, 17 and 24). So Object25 has a greater y-coordinate than Object16.
Therefore, Object16 is southwest of Object25 in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object22, Object2, Object4, Object14 and Object20 are northeast of Object25. Object16 is southwest of Object25. Only Object16 qualifies in this arrangement.

The claim that Object22 is southwest of Object25 is false in Case 1, so it is not entailed.

The claim that Object2 is southwest of Object25 is false in Case 1, so it is not entailed.

The claim that Object4 is southwest of Object25 is false in Case 1, so it is not entailed.

The claim that Object14 is southwest of Object25 is false in Case 1, so it is not entailed.

The claim that Object20 is southwest of Object25 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Object16}.

Answer: A

### Symbolic

[p5, p14] x(Object16) < x(Object15) < x(Object25) => x(Object16) < x(Object25).
[p15, p13, p3, p17, p24] y(Object16) < y(Object5) < y(Object24) < y(Object23) < y(Object26) < y(Object25) => y(Object16) < y(Object25).
SW(Object16,Object25) in every valid arrangement.

Case 1 (counterexample satisfying all premises): NE(Object22,Object25) & NE(Object2,Object25) & NE(Object4,Object25) & NE(Object14,Object25) & NE(Object20,Object25). SW(Object16,Object25). Qualifying objects = {Object16}.

The claim that SW(Object22,Object25) is false in Case 1, so it is not entailed.

The claim that SW(Object2,Object25) is false in Case 1, so it is not entailed.

The claim that SW(Object4,Object25) is false in Case 1, so it is not entailed.

The claim that SW(Object14,Object25) is false in Case 1, so it is not entailed.

The claim that SW(Object20,Object25) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Object16}.

Answer: A

## production-b1953-s6-direction-m0-unique-entailed

Categories: depth-3.
Split: in-range-test; contract: unique-entailed; answer: C.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object17, Object9, Object19, Object8, Object20, Object18, Object6, Object13, Object21, Object15, Object0, Object3, Object1, Object5, Object14, Object2, Object7, Object11, Object12, Object4, Object10, Object16
Query: direction Object5 Object7
Consider a map with multiple objects:
Object19 is to the South of Object21.
Object14 is to the North of Object19.
Object1 is to the Northeast of Object17.
Object7 is to the South of Object10.
Object7 is to the South of Object3.
Object5 is to the Southeast of Object17.
Object4 is to the North of Object7.
Object7 is to the South of Object19.
Object16 is to the South of Object8.
Object3 is to the South of Object18.
Object18 is to the South of Object10.
Object7 is to the Northeast of Object1.
Object21 is to the South of Object11.
Object0 is to the South of Object7.
Object14 is to the South of Object20.
Object6 is to the South of Object2.
Object17 is to the Southeast of Object13.
Object0 is to the South of Object5.
Object9 is to the North of Object19.
Object12 is to the North of Object15.
Object15 is to the North of Object19.
Question: Which single direction of Object5 relative to Object7 is entailed?
A. Northwest
B. East
C. South
D. North
E. Cannot be determined
F. None of the above

### Natural

Object5, Object0 and Object7 share the same x-coordinate (premise sources: 18 and 14).
The y-coordinates increase strictly along Object5, Object17, Object1 and Object7 (premise sources: 6, 3 and 12). So Object7 has a greater y-coordinate than Object5.
Therefore, Object5 is south of Object7 in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object0, Object5 and Object7 share the same horizontal position. From bottom to top, the order is Object0, Object5 and Object7. So Object5 is south of Object7.

The claim that Object5 is north of Object7 is false in Case 1, so it is not entailed.

The claim that Object5 is northeast of Object7 is false in Case 1, so it is not entailed.

The claim that Object5 is east of Object7 is false in Case 1, so it is not entailed.

The claim that Object5 is southeast of Object7 is false in Case 1, so it is not entailed.

The claim that Object5 is southwest of Object7 is false in Case 1, so it is not entailed.

The claim that Object5 is west of Object7 is false in Case 1, so it is not entailed.

The claim that Object5 is northwest of Object7 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {South}.

Answer: C

### Symbolic

[p18, p14] x(Object5) = x(Object0) = x(Object7) => x(Object5) = x(Object7).
[p6, p3, p12] y(Object5) < y(Object17) < y(Object1) < y(Object7) => y(Object5) < y(Object7).
S(Object5,Object7) in every valid arrangement.

Case 1 (counterexample satisfying all premises): x(Object0) = x(Object5) = x(Object7). y(Object0) < y(Object5) < y(Object7). => S(Object5,Object7).

The claim that N(Object5,Object7) is false in Case 1, so it is not entailed.

The claim that NE(Object5,Object7) is false in Case 1, so it is not entailed.

The claim that E(Object5,Object7) is false in Case 1, so it is not entailed.

The claim that SE(Object5,Object7) is false in Case 1, so it is not entailed.

The claim that SW(Object5,Object7) is false in Case 1, so it is not entailed.

The claim that W(Object5,Object7) is false in Case 1, so it is not entailed.

The claim that NW(Object5,Object7) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {South}.

Answer: C

## production-b1492-s0-count-m0-unique-entailed

Categories: depth-6.
Split: dev; contract: unique-entailed; answer: A.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object10, Object8, Object17, Object13, Object2, Object6, Object11, Object5, Object15, Object9, Object3, Object7, Object0, Object18, Object14, Object19, Object16, Object1, Object4, Object12
Query: count South Object15
Eligible: Object3, Object5, Object0, Object16, Object8
Consider a map with multiple objects:
Object17 is to the North of Object4.
Object14 is to the South of Object7.
Object18 is to the North of Object17.
Object8 is to the North of Object13.
Object2 is to the South of Object3.
Object16 is to the North of Object3.
Object14 is to the North of Object18.
Object13 is to the South of Object9.
Object0 is to the South of Object15.
Object2 is to the North of Object15.
Object15 is to the North of Object7.
Object10 is to the East of Object1.
Object11 is to the North of Object9.
Object19 is to the Northeast of Object11.
Object4 is to the North of Object5.
Object2 is to the South of Object12.
Object13 is to the North of Object12.
Object11 is to the North of Object6.
Object16 is to the North of Object11.
Question: What exact count of eligible objects south of Object15 is guaranteed?
A. 2
B. 1
C. 4
D. 0
E. Cannot be determined
F. None of the above

### Natural

The y-coordinates increase strictly along Object15, Object2 and Object3 (premise sources: 10 and 5). So Object3 has a greater y-coordinate than Object15.
Object3 never qualifies for the queried relation.
Object5, Object4, Object17, Object18, Object14, Object7 and Object15 share the same x-coordinate (premise sources: 15, 1, 3, 7, 2 and 11).
The y-coordinates increase strictly along Object5, Object4, Object17, Object18, Object14, Object7 and Object15 (premise sources: 15, 1, 3, 7, 2 and 11). So Object15 has a greater y-coordinate than Object5.
Object5 always qualifies for the queried relation.
Object0 and Object15 share the same x-coordinate (premise sources: 9).
Object15 has a greater y-coordinate than Object0 (premise sources: 9).
Object0 always qualifies for the queried relation.
The y-coordinates increase strictly along Object15, Object2, Object3 and Object16 (premise sources: 10, 5 and 6). So Object16 has a greater y-coordinate than Object15.
Object16 never qualifies for the queried relation.
The y-coordinates increase strictly along Object15, Object2, Object12, Object13 and Object8 (premise sources: 10, 16, 17 and 4). So Object8 has a greater y-coordinate than Object15.
Object8 never qualifies for the queried relation.
The qualifying candidates are fixed: Object5 and Object0. Exactly 2 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object3, Object16 and Object8 are north of Object15. Object5 and Object0 are south of Object15. Only Object0 and Object5 qualify in this arrangement. This gives a count of 2.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: A

### Symbolic

[p10, p5] y(Object15) < y(Object2) < y(Object3) => y(Object15) < y(Object3).
NOT S(Object3,Object15) in every valid arrangement.
[p15, p1, p3, p7, p2, p11] x(Object5) = x(Object4) = x(Object17) = x(Object18) = x(Object14) = x(Object7) = x(Object15) => x(Object5) = x(Object15).
[p15, p1, p3, p7, p2, p11] y(Object5) < y(Object4) < y(Object17) < y(Object18) < y(Object14) < y(Object7) < y(Object15) => y(Object5) < y(Object15).
S(Object5,Object15) in every valid arrangement.
[p9] x(Object0) = x(Object15).
[p9] y(Object0) < y(Object15).
S(Object0,Object15) in every valid arrangement.
[p10, p5, p6] y(Object15) < y(Object2) < y(Object3) < y(Object16) => y(Object15) < y(Object16).
NOT S(Object16,Object15) in every valid arrangement.
[p10, p16, p17, p4] y(Object15) < y(Object2) < y(Object12) < y(Object13) < y(Object8) => y(Object15) < y(Object8).
NOT S(Object8,Object15) in every valid arrangement.
Fixed qualifying set={Object5, Object0}; count=2 in every valid arrangement.

Case 1 (counterexample satisfying all premises): N(Object3,Object15) & N(Object16,Object15) & N(Object8,Object15). S(Object5,Object15) & S(Object0,Object15). Qualifying objects = {Object0, Object5}. count=2.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: A

## production-b724-s1-count-m0-unique-entailed

Categories: depth-7.
Split: train; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object1, Object20, Object3, Object8, Object22, Object25, Object18, Object13, Object10, Object0, Object17, Object24, Object2, Object6, Object9, Object15, Object16, Object23, Object7, Object21, Object4, Object14, Object19, Object11, Object5, Object12
Query: count Southwest Object20
Eligible: Object2, Object0, Object21, Object25, Object22, Object9, Object16
Consider a map with multiple objects:
Object10 is to the Southeast of Object14.
Object11 is to the Northwest of Object17.
Object14 is to the Southeast of Object18.
Object6 is to the Southwest of Object7.
Object4 is to the Northeast of Object22.
Object18 is to the Southeast of Object20.
Object6 is to the Northeast of Object5.
Object20 is to the Southeast of Object4.
Object20 is to the Southwest of Object25.
Object13 is to the Northeast of Object23.
Object24 is to the Southeast of Object17.
Object0 is to the Northeast of Object13.
Object20 is to the Southwest of Object23.
Object25 is to the Northeast of Object21.
Object8 is to the Northeast of Object13.
Object8 is to the Southwest of Object2.
Object19 is to the Northeast of Object15.
Object11 is to the Southeast of Object10.
Object15 is to the Southwest of Object21.
Object16 is to the Southwest of Object20.
Object9 is to the Northeast of Object7.
Object22 is to the Southwest of Object24.
Object1 is to the East of Object12.
Object3 is to the Northeast of Object6.
Object23 is to the Southwest of Object15.
Object19 is to the Southwest of Object5.
Question: What exact count of eligible objects southwest of Object20 is guaranteed?
A. 3
B. 5
C. 6
D. 2
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object20, Object23, Object13, Object8 and Object2 (premise sources: 13, 10, 15 and 16). So Object2 has a greater x-coordinate than Object20.
Object2 never qualifies for the queried relation.
The x-coordinates increase strictly along Object20, Object23, Object13 and Object0 (premise sources: 13, 10 and 12). So Object0 has a greater x-coordinate than Object20.
Object0 never qualifies for the queried relation.
The x-coordinates increase strictly along Object20, Object23, Object15 and Object21 (premise sources: 13, 25 and 19). So Object21 has a greater x-coordinate than Object20.
Object21 never qualifies for the queried relation.
Object25 has a greater x-coordinate than Object20 (premise sources: 9).
Object25 never qualifies for the queried relation.
The x-coordinates increase strictly along Object22, Object4 and Object20 (premise sources: 5 and 8). So Object20 has a greater x-coordinate than Object22.
The y-coordinates increase strictly along Object22, Object24, Object17, Object11, Object10, Object14, Object18 and Object20 (premise sources: 22, 11, 2, 18, 1, 3 and 6). So Object20 has a greater y-coordinate than Object22.
Object22 always qualifies for the queried relation.
The x-coordinates increase strictly along Object20, Object23, Object15, Object19, Object5, Object6, Object7 and Object9 (premise sources: 13, 25, 17, 26, 7, 4 and 21). So Object9 has a greater x-coordinate than Object20.
Object9 never qualifies for the queried relation.
Object20 has a greater x-coordinate than Object16 (premise sources: 20).
Object20 has a greater y-coordinate than Object16 (premise sources: 20).
Object16 always qualifies for the queried relation.
The qualifying candidates are fixed: Object22 and Object16. Exactly 2 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object2, Object0, Object21, Object25 and Object9 are northeast of Object20. Object22 and Object16 are southwest of Object20. Only Object16 and Object22 qualify in this arrangement. This gives a count of 2.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: D

### Symbolic

[p13, p10, p15, p16] x(Object20) < x(Object23) < x(Object13) < x(Object8) < x(Object2) => x(Object20) < x(Object2).
NOT SW(Object2,Object20) in every valid arrangement.
[p13, p10, p12] x(Object20) < x(Object23) < x(Object13) < x(Object0) => x(Object20) < x(Object0).
NOT SW(Object0,Object20) in every valid arrangement.
[p13, p25, p19] x(Object20) < x(Object23) < x(Object15) < x(Object21) => x(Object20) < x(Object21).
NOT SW(Object21,Object20) in every valid arrangement.
[p9] x(Object20) < x(Object25).
NOT SW(Object25,Object20) in every valid arrangement.
[p5, p8] x(Object22) < x(Object4) < x(Object20) => x(Object22) < x(Object20).
[p22, p11, p2, p18, p1, p3, p6] y(Object22) < y(Object24) < y(Object17) < y(Object11) < y(Object10) < y(Object14) < y(Object18) < y(Object20) => y(Object22) < y(Object20).
SW(Object22,Object20) in every valid arrangement.
[p13, p25, p17, p26, p7, p4, p21] x(Object20) < x(Object23) < x(Object15) < x(Object19) < x(Object5) < x(Object6) < x(Object7) < x(Object9) => x(Object20) < x(Object9).
NOT SW(Object9,Object20) in every valid arrangement.
[p20] x(Object16) < x(Object20).
[p20] y(Object16) < y(Object20).
SW(Object16,Object20) in every valid arrangement.
Fixed qualifying set={Object22, Object16}; count=2 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NE(Object2,Object20) & NE(Object0,Object20) & NE(Object21,Object20) & NE(Object25,Object20) & NE(Object9,Object20). SW(Object22,Object20) & SW(Object16,Object20). Qualifying objects = {Object16, Object22}. count=2.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {2}.

Answer: D

## production-b994-s0-count-m0-unique-entailed

Categories: depth-2.
Split: dev; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object13, Object6, Object9, Object2, Object10, Object11, Object5, Object7, Object4, Object8, Object3, Object0, Object15, Object12, Object14, Object1
Query: count Northeast Object12
Eligible: Object11, Object8, Object15, Object3, Object0, Object6
Consider a map with multiple objects:
Object0 is to the Northeast of Object15.
Object10 is to the South of Object7.
Object3 is to the Northeast of Object4.
Object9 is to the Southwest of Object10.
Object2 is to the West of Object13.
Object11 is to the Southwest of Object8.
Object12 is to the Southwest of Object4.
Object10 is to the Southwest of Object12.
Object15 is to the Southwest of Object5.
Object10 is to the Northeast of Object8.
Object1 is to the Northeast of Object5.
Object6 is to the Northeast of Object0.
Object11 is to the Southwest of Object6.
Object14 is to the Northeast of Object11.
Object14 is to the Southwest of Object12.
Object12 is to the Northeast of Object5.
Object12 is to the Northeast of Object0.
Object6 is to the Southwest of Object10.
Question: What exact count of eligible objects northeast of Object12 is guaranteed?
A. 3
B. 0
C. 2
D. 1
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object11, Object14 and Object12 (premise sources: 14 and 15). So Object12 has a greater x-coordinate than Object11.
Object11 never qualifies for the queried relation.
The x-coordinates increase strictly along Object8, Object10 and Object12 (premise sources: 10 and 8). So Object12 has a greater x-coordinate than Object8.
Object8 never qualifies for the queried relation.
The x-coordinates increase strictly along Object15, Object0 and Object12 (premise sources: 1 and 17). So Object12 has a greater x-coordinate than Object15.
Object15 never qualifies for the queried relation.
The x-coordinates increase strictly along Object12, Object4 and Object3 (premise sources: 7 and 3). So Object3 has a greater x-coordinate than Object12.
The y-coordinates increase strictly along Object12, Object4 and Object3 (premise sources: 7 and 3). So Object3 has a greater y-coordinate than Object12.
Object3 always qualifies for the queried relation.
Object12 has a greater x-coordinate than Object0 (premise sources: 17).
Object0 never qualifies for the queried relation.
The x-coordinates increase strictly along Object6, Object10 and Object12 (premise sources: 18 and 8). So Object12 has a greater x-coordinate than Object6.
Object6 never qualifies for the queried relation.
The qualifying candidates are fixed: Object3. Exactly 1 qualifies in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object11, Object8, Object15, Object0 and Object6 are southwest of Object12. Object3 is northeast of Object12. Only Object3 qualifies in this arrangement. This gives a count of 1.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: D

### Symbolic

[p14, p15] x(Object11) < x(Object14) < x(Object12) => x(Object11) < x(Object12).
NOT NE(Object11,Object12) in every valid arrangement.
[p10, p8] x(Object8) < x(Object10) < x(Object12) => x(Object8) < x(Object12).
NOT NE(Object8,Object12) in every valid arrangement.
[p1, p17] x(Object15) < x(Object0) < x(Object12) => x(Object15) < x(Object12).
NOT NE(Object15,Object12) in every valid arrangement.
[p7, p3] x(Object12) < x(Object4) < x(Object3) => x(Object12) < x(Object3).
[p7, p3] y(Object12) < y(Object4) < y(Object3) => y(Object12) < y(Object3).
NE(Object3,Object12) in every valid arrangement.
[p17] x(Object0) < x(Object12).
NOT NE(Object0,Object12) in every valid arrangement.
[p18, p8] x(Object6) < x(Object10) < x(Object12) => x(Object6) < x(Object12).
NOT NE(Object6,Object12) in every valid arrangement.
Fixed qualifying set={Object3}; count=1 in every valid arrangement.

Case 1 (counterexample satisfying all premises): SW(Object11,Object12) & SW(Object8,Object12) & SW(Object15,Object12) & SW(Object0,Object12) & SW(Object6,Object12). NE(Object3,Object12). Qualifying objects = {Object3}. count=1.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: D

## production-b11-s7-count-m0-unique-entailed

Categories: depth-1.
Split: train; contract: unique-entailed; answer: C.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object13, Object12, Object1, Object11, Object5, Object7, Object2, Object4, Object10, Object8, Object6, Object3, Object9
Query: count South Object0
Eligible: Object9, Object7, Object5, Object3, Object10, Object8, Object2
Consider a map with multiple objects:
Object7 is to the North of Object3.
Object0 is to the South of Object10.
Object3 is to the North of Object0.
Object8 is to the South of Object10.
Object5 is to the South of Object10.
Object9 is to the West of Object6.
Object8 is to the North of Object0.
Object4 is to the Northeast of Object5.
Object1 is to the Northwest of Object12.
Object0 is to the South of Object9.
Object5 is to the North of Object0.
Object2 is to the North of Object0.
Object3 is to the South of Object2.
Object11 is to the East of Object13.
Object0 is to the South of Object7.
Question: What exact count of eligible objects south of Object0 is guaranteed?
A. 7
B. 5
C. 0
D. 1
E. Cannot be determined
F. None of the above

### Natural

Object9 has a greater y-coordinate than Object0 (premise sources: 10).
Object9 never qualifies for the queried relation.
Object7 has a greater y-coordinate than Object0 (premise sources: 15).
Object7 never qualifies for the queried relation.
Object5 has a greater y-coordinate than Object0 (premise sources: 11).
Object5 never qualifies for the queried relation.
Object3 has a greater y-coordinate than Object0 (premise sources: 3).
Object3 never qualifies for the queried relation.
Object10 has a greater y-coordinate than Object0 (premise sources: 2).
Object10 never qualifies for the queried relation.
Object8 has a greater y-coordinate than Object0 (premise sources: 7).
Object8 never qualifies for the queried relation.
Object2 has a greater y-coordinate than Object0 (premise sources: 12).
Object2 never qualifies for the queried relation.
The qualifying candidates are fixed: none. Exactly 0 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object9, Object7, Object5, Object3, Object10, Object8 and Object2 are north of Object0. No eligible object qualifies in this arrangement. This gives a count of 0.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 7 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: C

### Symbolic

[p10] y(Object0) < y(Object9).
NOT S(Object9,Object0) in every valid arrangement.
[p15] y(Object0) < y(Object7).
NOT S(Object7,Object0) in every valid arrangement.
[p11] y(Object0) < y(Object5).
NOT S(Object5,Object0) in every valid arrangement.
[p3] y(Object0) < y(Object3).
NOT S(Object3,Object0) in every valid arrangement.
[p2] y(Object0) < y(Object10).
NOT S(Object10,Object0) in every valid arrangement.
[p7] y(Object0) < y(Object8).
NOT S(Object8,Object0) in every valid arrangement.
[p12] y(Object0) < y(Object2).
NOT S(Object2,Object0) in every valid arrangement.
Fixed qualifying set={}; count=0 in every valid arrangement.

Case 1 (counterexample satisfying all premises): N(Object9,Object0) & N(Object7,Object0) & N(Object5,Object0) & N(Object3,Object0) & N(Object10,Object0) & N(Object8,Object0) & N(Object2,Object0). Qualifying objects = {}. count=0.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The claim that count=7 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: C

## production-b534-s15-direction-m1-unique-entailed

Categories: escape-unique-unlisted, omission.
Split: train; contract: unique-entailed; answer: F.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object26, Object23, Object20, Object0, Object6, Object13, Object10, Object14, Object27, Object25, Object18, Object12, Object8, Object2, Object11, Object1, Object15, Object17, Object4, Object5, Object9, Object22, Object19, Object24, Object3, Object7, Object16, Object21, Object28
Query: direction Object20 Object18
Consider a map with multiple objects:
Object27 is to the South of Object0.
Object22 is to the Northeast of Object16.
Object17 is to the Northeast of Object12.
Object13 is to the Northwest of Object1.
Object22 is to the Southwest of Object11.
Object27 is to the North of Object23.
Object4 is to the South of Object7.
Object25 is to the South of Object19.
Object10 is to the South of Object2.
Object18 is to the South of Object21.
Object5 is to the North of Object1.
Object25 is to the South of Object3.
Object11 is to the Southwest of Object18.
Object17 is to the Southwest of Object16.
Object9 is to the North of Object26.
Object20 is to the North of Object6.
Object18 is to the South of Object15.
Object4 is to the North of Object0.
Object19 is to the South of Object1.
Object14 is to the North of Object25.
Object0 is to the South of Object26.
Object21 is to the South of Object23.
Object9 is to the North of Object5.
Object20 is to the Southeast of Object12.
Object28 is to the Northeast of Object24.
Object25 is to the North of Object21.
Object6 is to the South of Object18.
Object8 is to the Northwest of Object3.
Question: Which single direction of Object20 relative to Object18 is entailed?
A. West
B. Northeast
C. Southwest
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Object20, Object6 and Object18 share the same x-coordinate (premise sources: 16 and 27).
The y-coordinates increase strictly along Object20, Object12, Object17, Object16, Object22, Object11 and Object18 (premise sources: 24, 3, 14, 2, 5 and 13). So Object18 has a greater y-coordinate than Object20.
Therefore, Object20 is south of Object18 in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

### Symbolic

[p16, p27] x(Object20) = x(Object6) = x(Object18) => x(Object20) = x(Object18).
[p24, p3, p14, p2, p5, p13] y(Object20) < y(Object12) < y(Object17) < y(Object16) < y(Object22) < y(Object11) < y(Object18) => y(Object20) < y(Object18).
S(Object20,Object18) in every valid arrangement.

The sole entailed answer is unlisted.

Answer: F

## production-b534-s15-direction-m1-listed-entailed

Categories: escape-no-listed-entailments.
Split: train; contract: listed-entailed; answer: F.

### Question

Interface: spatialentail-entailment-v1
Contract: listed-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all displayed A-D values that are entailed. Exclude every displayed value that is not entailed, including claims that could hold but are not guaranteed. Select F if no displayed value is entailed.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object26, Object23, Object20, Object0, Object6, Object13, Object10, Object14, Object27, Object25, Object18, Object12, Object8, Object2, Object11, Object1, Object15, Object17, Object4, Object5, Object9, Object22, Object19, Object24, Object3, Object7, Object16, Object21, Object28
Query: direction Object20 Object18
Consider a map with multiple objects:
Object27 is to the South of Object0.
Object22 is to the Northeast of Object16.
Object17 is to the Northeast of Object12.
Object13 is to the Northwest of Object1.
Object22 is to the Southwest of Object11.
Object27 is to the North of Object23.
Object4 is to the South of Object7.
Object25 is to the South of Object19.
Object10 is to the South of Object2.
Object18 is to the South of Object21.
Object5 is to the North of Object1.
Object25 is to the South of Object3.
Object11 is to the Southwest of Object18.
Object17 is to the Southwest of Object16.
Object9 is to the North of Object26.
Object20 is to the North of Object6.
Object18 is to the South of Object15.
Object4 is to the North of Object0.
Object19 is to the South of Object1.
Object14 is to the North of Object25.
Object0 is to the South of Object26.
Object21 is to the South of Object23.
Object9 is to the North of Object5.
Object20 is to the Southeast of Object12.
Object28 is to the Northeast of Object24.
Object25 is to the North of Object21.
Object6 is to the South of Object18.
Object8 is to the Northwest of Object3.
Question: Which listed directions of Object20 relative to Object18 are entailed? Select all that apply.
A. West
B. Northeast
C. Southwest
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Case 1: A counterexample arrangement satisfying all premises is the following. Object18, Object20 and Object6 share the same horizontal position. From bottom to top, the order is Object6, Object20 and Object18. So Object20 is south of Object18.

The claim that Object20 is west of Object18 is false in Case 1, so it is not entailed.

The claim that Object20 is northeast of Object18 is false in Case 1, so it is not entailed.

The claim that Object20 is southwest of Object18 is false in Case 1, so it is not entailed.

The claim that Object20 is northwest of Object18 is false in Case 1, so it is not entailed.

No displayed answer is entailed.

Answer: F

### Symbolic

Case 1 (counterexample satisfying all premises): x(Object18) = x(Object20) = x(Object6). y(Object6) < y(Object20) < y(Object18). => S(Object20,Object18).

The claim that W(Object20,Object18) is false in Case 1, so it is not entailed.

The claim that NE(Object20,Object18) is false in Case 1, so it is not entailed.

The claim that SW(Object20,Object18) is false in Case 1, so it is not entailed.

The claim that NW(Object20,Object18) is false in Case 1, so it is not entailed.

No displayed answer is entailed.

Answer: F

## production-b534-s15-direction-m1-exhaustive-entailed

Categories: escape-incomplete-entailed-coverage.
Split: train; contract: exhaustive-entailed; answer: F.

### Question

Interface: spatialentail-entailment-v1
Contract: exhaustive-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select all A-D values only if they cover the complete nonempty set of individually entailed domain answers. Select F if any entailed answer is missing or no domain answer is entailed. Non-entailed answers outside the menu do not make coverage incomplete.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object26, Object23, Object20, Object0, Object6, Object13, Object10, Object14, Object27, Object25, Object18, Object12, Object8, Object2, Object11, Object1, Object15, Object17, Object4, Object5, Object9, Object22, Object19, Object24, Object3, Object7, Object16, Object21, Object28
Query: direction Object20 Object18
Consider a map with multiple objects:
Object27 is to the South of Object0.
Object22 is to the Northeast of Object16.
Object17 is to the Northeast of Object12.
Object13 is to the Northwest of Object1.
Object22 is to the Southwest of Object11.
Object27 is to the North of Object23.
Object4 is to the South of Object7.
Object25 is to the South of Object19.
Object10 is to the South of Object2.
Object18 is to the South of Object21.
Object5 is to the North of Object1.
Object25 is to the South of Object3.
Object11 is to the Southwest of Object18.
Object17 is to the Southwest of Object16.
Object9 is to the North of Object26.
Object20 is to the North of Object6.
Object18 is to the South of Object15.
Object4 is to the North of Object0.
Object19 is to the South of Object1.
Object14 is to the North of Object25.
Object0 is to the South of Object26.
Object21 is to the South of Object23.
Object9 is to the North of Object5.
Object20 is to the Southeast of Object12.
Object28 is to the Northeast of Object24.
Object25 is to the North of Object21.
Object6 is to the South of Object18.
Object8 is to the Northwest of Object3.
Question: Select the complete set of individually entailed directions of Object20 relative to Object18.
A. West
B. Northeast
C. Southwest
D. Northwest
E. Cannot be determined
F. No complete entailed answer can be formed from A-D.

### Natural

Object20, Object6 and Object18 share the same x-coordinate (premise sources: 16 and 27).
The y-coordinates increase strictly along Object20, Object12, Object17, Object16, Object22, Object11 and Object18 (premise sources: 24, 3, 14, 2, 5 and 13). So Object18 has a greater y-coordinate than Object20.
Therefore, Object20 is south of Object18 in every valid arrangement.

An entailed answer is missing from A-D, so the complete entailed set cannot be represented.

Answer: F

### Symbolic

[p16, p27] x(Object20) = x(Object6) = x(Object18) => x(Object20) = x(Object18).
[p24, p3, p14, p2, p5, p13] y(Object20) < y(Object12) < y(Object17) < y(Object16) < y(Object22) < y(Object11) < y(Object18) => y(Object20) < y(Object18).
S(Object20,Object18) in every valid arrangement.

An entailed answer is missing from A-D, so the complete entailed set cannot be represented.

Answer: F

## production-b2554-s6-direction-m0-unique-entailed

Categories: depth-8.
Split: in-range-test; contract: unique-entailed; answer: C.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object20, Object25, Object2, Object18, Object7, Object26, Object15, Object6, Object12, Object16, Object9, Object19, Object5, Object1, Object27, Object24, Object11, Object23, Object14, Object10, Object0, Object17, Object8, Object13, Object4, Object22, Object21, Object3
Query: direction Object19 Object9
Consider a map with multiple objects:
Object27 is to the Southwest of Object14.
Object20 is to the North of Object15.
Object5 is to the Northeast of Object16.
Object5 is to the Northeast of Object13.
Object1 is to the Southwest of Object7.
Object12 is to the Northeast of Object9.
Object5 is to the Northeast of Object2.
Object14 is to the Southwest of Object2.
Object19 is to the South of Object6.
Object22 is to the Southwest of Object26.
Object21 is to the Southwest of Object13.
Object25 is to the Southwest of Object1.
Object23 is to the Southwest of Object17.
Object18 is to the West of Object24.
Object16 is to the Northeast of Object9.
Object4 is to the South of Object8.
Object8 is to the West of Object15.
Object11 is to the North of Object3.
Object3 is to the East of Object20.
Object11 is to the West of Object9.
Object1 is to the West of Object10.
Object4 is to the East of Object6.
Object1 is to the Southwest of Object21.
Object1 is to the Southwest of Object22.
Object0 is to the Southwest of Object2.
Object25 is to the Northeast of Object9.
Object12 is to the Southwest of Object7.
Object27 is to the Northeast of Object26.
Question: Which single direction of Object19 relative to Object9 is entailed?
A. Northwest
B. North
C. Southwest
D. Northeast
E. Cannot be determined
F. None of the above

### Natural

Following the x-coordinates, Object19 shares its x-coordinate with Object6; Object6 has a smaller x-coordinate than Object4; Object4 shares its x-coordinate with Object8; Object8 has a smaller x-coordinate than Object15; Object15 shares its x-coordinate with Object20; Object20 has a smaller x-coordinate than Object3; Object3 shares its x-coordinate with Object11; Object11 has a smaller x-coordinate than Object9 (premise sources: 9, 22, 16, 17, 2, 19, 18 and 20). Therefore, Object9 has a greater x-coordinate than Object19.
Following the y-coordinates, Object19 has a smaller y-coordinate than Object6; Object6 shares its y-coordinate with Object4; Object4 has a smaller y-coordinate than Object8; Object8 shares its y-coordinate with Object15; Object15 has a smaller y-coordinate than Object20; Object20 shares its y-coordinate with Object3; Object3 has a smaller y-coordinate than Object11; Object11 shares its y-coordinate with Object9 (premise sources: 9, 22, 16, 17, 2, 19, 18 and 20). Therefore, Object9 has a greater y-coordinate than Object19.
Therefore, Object19 is southwest of Object9 in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. From left to right, the order is Object19 and Object9. From bottom to top, the order is Object19 and Object9. So Object19 is southwest of Object9.

The claim that Object19 is north of Object9 is false in Case 1, so it is not entailed.

The claim that Object19 is northeast of Object9 is false in Case 1, so it is not entailed.

The claim that Object19 is east of Object9 is false in Case 1, so it is not entailed.

The claim that Object19 is southeast of Object9 is false in Case 1, so it is not entailed.

The claim that Object19 is south of Object9 is false in Case 1, so it is not entailed.

The claim that Object19 is west of Object9 is false in Case 1, so it is not entailed.

The claim that Object19 is northwest of Object9 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: C

### Symbolic

[p9, p22, p16, p17, p2, p19, p18, p20] x(Object19) = x(Object6) < x(Object4) = x(Object8) < x(Object15) = x(Object20) < x(Object3) = x(Object11) < x(Object9) => x(Object19) < x(Object9).
[p9, p22, p16, p17, p2, p19, p18, p20] y(Object19) < y(Object6) = y(Object4) < y(Object8) = y(Object15) < y(Object20) = y(Object3) < y(Object11) = y(Object9) => y(Object19) < y(Object9).
SW(Object19,Object9) in every valid arrangement.

Case 1 (counterexample satisfying all premises): x(Object19) < x(Object9). y(Object19) < y(Object9). => SW(Object19,Object9).

The claim that N(Object19,Object9) is false in Case 1, so it is not entailed.

The claim that NE(Object19,Object9) is false in Case 1, so it is not entailed.

The claim that E(Object19,Object9) is false in Case 1, so it is not entailed.

The claim that SE(Object19,Object9) is false in Case 1, so it is not entailed.

The claim that S(Object19,Object9) is false in Case 1, so it is not entailed.

The claim that W(Object19,Object9) is false in Case 1, so it is not entailed.

The claim that NW(Object19,Object9) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Southwest}.

Answer: C

## production-b328-s4-count-m0-unique-entailed

Categories: depth-4.
Split: train; contract: unique-entailed; answer: B.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object10, Object18, Object12, Object0, Object9, Object2, Object1, Object6, Object15, Object17, Object5, Object7, Object3, Object14, Object11, Object8, Object13, Object4, Object16
Query: count Southeast Object15
Eligible: Object4, Object17, Object14, Object18, Object16
Consider a map with multiple objects:
Object14 is to the Northwest of Object9.
Object12 is to the East of Object18.
Object18 is to the Northwest of Object7.
Object10 is to the West of Object3.
Object2 is to the Southeast of Object8.
Object18 is to the Southeast of Object14.
Object0 is to the North of Object5.
Object7 is to the Northwest of Object8.
Object6 is to the Southwest of Object1.
Object13 is to the East of Object9.
Object15 is to the Southeast of Object17.
Object4 is to the Southeast of Object18.
Object16 is to the East of Object1.
Object8 is to the Southeast of Object9.
Object2 is to the Northwest of Object15.
Object18 is to the Southeast of Object17.
Object4 is to the Northwest of Object11.
Object3 is to the North of Object1.
Object11 is to the Northwest of Object15.
Object15 is to the North of Object10.
Question: What exact count of eligible objects southeast of Object15 is guaranteed?
A. 3
B. 1
C. 2
D. 0
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object4, Object11 and Object15 (premise sources: 17 and 19). So Object15 has a greater x-coordinate than Object4.
Object4 never qualifies for the queried relation.
Object15 has a greater x-coordinate than Object17 (premise sources: 11).
Object17 never qualifies for the queried relation.
The x-coordinates increase strictly along Object14, Object18, Object4, Object11 and Object15 (premise sources: 6, 12, 17 and 19). So Object15 has a greater x-coordinate than Object14.
Object14 never qualifies for the queried relation.
The x-coordinates increase strictly along Object18, Object4, Object11 and Object15 (premise sources: 12, 17 and 19). So Object15 has a greater x-coordinate than Object18.
Object18 never qualifies for the queried relation.
Following the x-coordinates, Object15 shares its x-coordinate with Object10; Object10 has a smaller x-coordinate than Object3; Object3 shares its x-coordinate with Object1; Object1 has a smaller x-coordinate than Object16 (premise sources: 20, 4, 18 and 13). Therefore, Object16 has a greater x-coordinate than Object15.
Following the y-coordinates, Object16 shares its y-coordinate with Object1; Object1 has a smaller y-coordinate than Object3; Object3 shares its y-coordinate with Object10; Object10 has a smaller y-coordinate than Object15 (premise sources: 13, 18, 4 and 20). Therefore, Object15 has a greater y-coordinate than Object16.
Object16 always qualifies for the queried relation.
The qualifying candidates are fixed: Object16. Exactly 1 qualifies in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object4, Object17, Object14 and Object18 are northwest of Object15. Object16 is southeast of Object15. Only Object16 qualifies in this arrangement. This gives a count of 1.

The claim that exactly 0 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: B

### Symbolic

[p17, p19] x(Object4) < x(Object11) < x(Object15) => x(Object4) < x(Object15).
NOT SE(Object4,Object15) in every valid arrangement.
[p11] x(Object17) < x(Object15).
NOT SE(Object17,Object15) in every valid arrangement.
[p6, p12, p17, p19] x(Object14) < x(Object18) < x(Object4) < x(Object11) < x(Object15) => x(Object14) < x(Object15).
NOT SE(Object14,Object15) in every valid arrangement.
[p12, p17, p19] x(Object18) < x(Object4) < x(Object11) < x(Object15) => x(Object18) < x(Object15).
NOT SE(Object18,Object15) in every valid arrangement.
[p20, p4, p18, p13] x(Object15) = x(Object10) < x(Object3) = x(Object1) < x(Object16) => x(Object15) < x(Object16).
[p13, p18, p4, p20] y(Object16) = y(Object1) < y(Object3) = y(Object10) < y(Object15) => y(Object16) < y(Object15).
SE(Object16,Object15) in every valid arrangement.
Fixed qualifying set={Object16}; count=1 in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Object4,Object15) & NW(Object17,Object15) & NW(Object14,Object15) & NW(Object18,Object15). SE(Object16,Object15). Qualifying objects = {Object16}. count=1.

The claim that count=0 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {1}.

Answer: B

## production-b2905-s118-direction-m0-unique-entailed

Categories: depth-None, escape-inconsistent.
Split: train; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5
Query: direction Object1 Object4
Consider a map with multiple objects:
Object4 is to the South of Object2.
Object4 is to the North of Object5.
Object5 is to the South of Object1.
Object0 is to the West of Object5.
Object3 is to the Northwest of Object4.
Object1 is to the Northwest of Object2.
Question: Which single direction of Object1 relative to Object4 is entailed?
A. Southeast
B. North
C. East
D. Northeast
E. Cannot be determined
F. None of the above

### Natural

From p6, Object1's x-coordinate is less than Object2's x-coordinate.
From p1, Object2's x-coordinate is equal to Object4's x-coordinate.
From p2, Object4's x-coordinate is equal to Object5's x-coordinate.
From p3, Object5's x-coordinate is equal to Object1's x-coordinate.
Composing this path gives Object1's x-coordinate less than Object1's x-coordinate.
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p2, p3, p6. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

### Symbolic

p6: x(Object1) < x(Object2)
p1: x(Object2) = x(Object4)
p2: x(Object4) = x(Object5)
p3: x(Object5) = x(Object1)
Transitivity: x(Object1) < x(Object1)
A strict order cannot be less than itself: contradiction.
Conflicting premises: p1, p2, p3, p6. No minimality claim is made.

The premises are inconsistent; every contract requires E.

Answer: E

## production-b2925-s2-which-m0-unique-entailed

Categories: escape-multiple-entailed.
Split: in-range-test; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object2, Object3, Object5, Object10, Object0, Object1, Object9, Object6, Object4, Object8, Object7
Query: which South Object1
Eligible: Object7, Object2, Object6, Object10, Object4, Object5, Object0
Consider a map with multiple objects:
Object8 is to the South of Object6.
Object1 is to the South of Object8.
Object8 is to the South of Object5.
Object4 is to the South of Object1.
Object1 is to the South of Object0.
Object1 is to the South of Object7.
Object10 is to the North of Object0.
Object1 is to the North of Object9.
Object10 is to the North of Object8.
Object9 is to the North of Object2.
Object3 is to the South of Object4.
Object10 is to the North of Object6.
Question: Which single eligible candidate is individually guaranteed to be south of Object1?
A. Object4
B. Object2
C. Object7
D. Object5
E. Cannot be determined
F. None of the above

### Natural

Object2, Object9 and Object1 share the same x-coordinate (premise sources: 10 and 8).
The y-coordinates increase strictly along Object2, Object9 and Object1 (premise sources: 10 and 8). So Object1 has a greater y-coordinate than Object2.
Therefore, Object2 is south of Object1 in every valid arrangement.

Object4 and Object1 share the same x-coordinate (premise sources: 4).
Object1 has a greater y-coordinate than Object4 (premise sources: 4).
Therefore, Object4 is south of Object1 in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

### Symbolic

[p10, p8] x(Object2) = x(Object9) = x(Object1) => x(Object2) = x(Object1).
[p10, p8] y(Object2) < y(Object9) < y(Object1) => y(Object2) < y(Object1).
S(Object2,Object1) in every valid arrangement.

[p4] x(Object4) = x(Object1).
[p4] y(Object4) < y(Object1).
S(Object4,Object1) in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

## production-b2925-s2-which-m0-unique-entailed-position1

Categories: position-variant.
Split: in-range-test; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object2, Object3, Object5, Object10, Object0, Object1, Object9, Object6, Object4, Object8, Object7
Query: which South Object1
Eligible: Object7, Object2, Object6, Object10, Object4, Object5, Object0
Consider a map with multiple objects:
Object8 is to the South of Object6.
Object1 is to the South of Object8.
Object8 is to the South of Object5.
Object4 is to the South of Object1.
Object1 is to the South of Object0.
Object1 is to the South of Object7.
Object10 is to the North of Object0.
Object1 is to the North of Object9.
Object10 is to the North of Object8.
Object9 is to the North of Object2.
Object3 is to the South of Object4.
Object10 is to the North of Object6.
Question: Which single eligible candidate is individually guaranteed to be south of Object1?
A. Object4
B. Object7
C. Object2
D. Object5
E. Cannot be determined
F. None of the above

### Natural

Object2, Object9 and Object1 share the same x-coordinate (premise sources: 10 and 8).
The y-coordinates increase strictly along Object2, Object9 and Object1 (premise sources: 10 and 8). So Object1 has a greater y-coordinate than Object2.
Therefore, Object2 is south of Object1 in every valid arrangement.

Object4 and Object1 share the same x-coordinate (premise sources: 4).
Object1 has a greater y-coordinate than Object4 (premise sources: 4).
Therefore, Object4 is south of Object1 in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

### Symbolic

[p10, p8] x(Object2) = x(Object9) = x(Object1) => x(Object2) = x(Object1).
[p10, p8] y(Object2) < y(Object9) < y(Object1) => y(Object2) < y(Object1).
S(Object2,Object1) in every valid arrangement.

[p4] x(Object4) = x(Object1).
[p4] y(Object4) < y(Object1).
S(Object4,Object1) in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

## production-b2922-s132-direction-m0-unique-entailed

Categories: escape-no-entailed-answer.
Split: in-range-test; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6, Object7
Query: direction Object3 Object2
Consider a map with multiple objects:
Object6 is to the East of Object0.
Object2 is to the South of Object6.
Object7 is to the Southeast of Object6.
Object4 is to the Northeast of Object0.
Object3 is to the Southeast of Object6.
Object5 is to the Southwest of Object4.
Object2 is to the Northeast of Object5.
Question: Which single direction of Object3 relative to Object2 is entailed?
A. Southeast
B. Northeast
C. North
D. Northwest
E. Cannot be determined
F. None of the above

### Natural

Case 1: A counterexample arrangement satisfying all premises is the following. Object2 and Object6 lie to the left of Object3. Object2 and Object6 share the same horizontal position. From bottom to top, the order is Object3, Object2 and Object6. So Object3 is southeast of Object2.

Case 2: A counterexample arrangement satisfying all premises is the following. Object2 and Object6 lie to the left of Object3. Object2 and Object6 share the same horizontal position. From bottom to top, the order is Object2, Object3 and Object6. So Object3 is northeast of Object2.

The claim that Object3 is north of Object2 is false in Case 1, so it is not entailed.

The claim that Object3 is northeast of Object2 is false in Case 1, so it is not entailed.

The claim that Object3 is east of Object2 is false in Case 1, so it is not entailed.

The claim that Object3 is southeast of Object2 is false in Case 2, so it is not entailed.

The claim that Object3 is south of Object2 is false in Case 1, so it is not entailed.

The claim that Object3 is southwest of Object2 is false in Case 1, so it is not entailed.

The claim that Object3 is west of Object2 is false in Case 1, so it is not entailed.

The claim that Object3 is northwest of Object2 is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

### Symbolic

Case 1 (counterexample satisfying all premises): x(Object2) = x(Object6) < x(Object3). y(Object3) < y(Object2) < y(Object6). => SE(Object3,Object2).

Case 2 (counterexample satisfying all premises): x(Object2) = x(Object6) < x(Object3). y(Object2) < y(Object3) < y(Object6). => NE(Object3,Object2).

The claim that N(Object3,Object2) is false in Case 1, so it is not entailed.

The claim that NE(Object3,Object2) is false in Case 1, so it is not entailed.

The claim that E(Object3,Object2) is false in Case 1, so it is not entailed.

The claim that SE(Object3,Object2) is false in Case 2, so it is not entailed.

The claim that S(Object3,Object2) is false in Case 1, so it is not entailed.

The claim that SW(Object3,Object2) is false in Case 1, so it is not entailed.

The claim that W(Object3,Object2) is false in Case 1, so it is not entailed.

The claim that NW(Object3,Object2) is false in Case 1, so it is not entailed.

No domain answer is entailed.

Answer: E

## production-b2907-s32-which-m0-unique-entailed

Categories: three-entailed.
Split: train; contract: unique-entailed; answer: E.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object12, Object9, Object10, Object5, Object7, Object2, Object1, Object3, Object13, Object6, Object4, Object11, Object8, Object0
Query: which Northeast Object5
Eligible: Object3, Object4, Object0, Object11, Object1, Object13, Object7
Consider a map with multiple objects:
Object7 is to the Southwest of Object5.
Object13 is to the Southwest of Object8.
Object5 is to the Northeast of Object8.
Object5 is to the Southwest of Object4.
Object0 is to the Northeast of Object5.
Object2 is to the Northeast of Object1.
Object5 is to the Northeast of Object11.
Object7 is to the Southwest of Object11.
Object10 is to the West of Object12.
Object5 is to the Northeast of Object2.
Object1 is to the Northwest of Object9.
Object6 is to the Southwest of Object3.
Object6 is to the Northeast of Object5.
Question: Which single eligible candidate is individually guaranteed to be northeast of Object5?
A. Object4
B. Object1
C. Object3
D. Object0
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object5, Object6 and Object3 (premise sources: 13 and 12). So Object3 has a greater x-coordinate than Object5.
The y-coordinates increase strictly along Object5, Object6 and Object3 (premise sources: 13 and 12). So Object3 has a greater y-coordinate than Object5.
Therefore, Object3 is northeast of Object5 in every valid arrangement.

Object4 has a greater x-coordinate than Object5 (premise sources: 4).
Object4 has a greater y-coordinate than Object5 (premise sources: 4).
Therefore, Object4 is northeast of Object5 in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

### Symbolic

[p13, p12] x(Object5) < x(Object6) < x(Object3) => x(Object5) < x(Object3).
[p13, p12] y(Object5) < y(Object6) < y(Object3) => y(Object5) < y(Object3).
NE(Object3,Object5) in every valid arrangement.

[p4] x(Object5) < x(Object4).
[p4] y(Object5) < y(Object4).
NE(Object4,Object5) in every valid arrangement.

At least two domain answers are entailed; Unique requires exactly one.

Answer: E

## production-b2816-s0-count-m0-unique-entailed

Categories: depth-10.
Split: extrapolation-test; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object14, Object21, Object28, Object9, Object18, Object6, Object0, Object25, Object17, Object8, Object7, Object13, Object16, Object27, Object12, Object15, Object2, Object3, Object11, Object19, Object10, Object24, Object23, Object22, Object5, Object20, Object1, Object4, Object26
Query: count East Object25
Eligible: Object0, Object3, Object12, Object22, Object17, Object10
Consider a map with multiple objects:
Object28 is to the East of Object3.
Object9 is to the West of Object18.
Object23 is to the East of Object22.
Object11 is to the West of Object2.
Object13 is to the West of Object7.
Object15 is to the East of Object17.
Object18 is to the West of Object1.
Object6 is to the West of Object8.
Object20 is to the South of Object16.
Object11 is to the East of Object10.
Object23 is to the East of Object11.
Object26 is to the West of Object14.
Object27 is to the West of Object25.
Object13 is to the East of Object5.
Object17 is to the West of Object25.
Object16 is to the East of Object1.
Object13 is to the East of Object4.
Object16 is to the West of Object6.
Object7 is to the West of Object25.
Object19 is to the West of Object11.
Object8 is to the West of Object27.
Object21 is to the Northeast of Object24.
Object26 is to the East of Object0.
Object5 is to the East of Object23.
Object12 is to the West of Object19.
Object12 is to the East of Object3.
Object9 is to the East of Object14.
Object28 is to the West of Object4.
Question: What exact count of eligible objects east of Object25 is guaranteed?
A. 4
B. 3
C. 5
D. 0
E. Cannot be determined
F. None of the above

### Natural

The x-coordinates increase strictly along Object0, Object26, Object14, Object9, Object18, Object1, Object16, Object6, Object8, Object27 and Object25 (premise sources: 23, 12, 27, 2, 7, 16, 18, 8, 21 and 13). So Object25 has a greater x-coordinate than Object0.
Object0 never qualifies for the queried relation.
The x-coordinates increase strictly along Object3, Object28, Object4, Object13, Object7 and Object25 (premise sources: 1, 28, 17, 5 and 19). So Object25 has a greater x-coordinate than Object3.
Object3 never qualifies for the queried relation.
The x-coordinates increase strictly along Object12, Object19, Object11, Object23, Object5, Object13, Object7 and Object25 (premise sources: 25, 20, 11, 24, 14, 5 and 19). So Object25 has a greater x-coordinate than Object12.
Object12 never qualifies for the queried relation.
The x-coordinates increase strictly along Object22, Object23, Object5, Object13, Object7 and Object25 (premise sources: 3, 24, 14, 5 and 19). So Object25 has a greater x-coordinate than Object22.
Object22 never qualifies for the queried relation.
Object25 has a greater x-coordinate than Object17 (premise sources: 15).
Object17 never qualifies for the queried relation.
The x-coordinates increase strictly along Object10, Object11, Object23, Object5, Object13, Object7 and Object25 (premise sources: 10, 11, 24, 14, 5 and 19). So Object25 has a greater x-coordinate than Object10.
Object10 never qualifies for the queried relation.
The qualifying candidates are fixed: none. Exactly 0 qualify in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object0, Object3, Object12, Object22, Object17 and Object10 are west of Object25. No eligible object qualifies in this arrangement. This gives a count of 0.

The claim that exactly 1 eligible candidate qualifies is false in Case 1, so it is not entailed.

The claim that exactly 2 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 3 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 4 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 5 eligible candidates qualify is false in Case 1, so it is not entailed.

The claim that exactly 6 eligible candidates qualify is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: D

### Symbolic

[p23, p12, p27, p2, p7, p16, p18, p8, p21, p13] x(Object0) < x(Object26) < x(Object14) < x(Object9) < x(Object18) < x(Object1) < x(Object16) < x(Object6) < x(Object8) < x(Object27) < x(Object25) => x(Object0) < x(Object25).
NOT E(Object0,Object25) in every valid arrangement.
[p1, p28, p17, p5, p19] x(Object3) < x(Object28) < x(Object4) < x(Object13) < x(Object7) < x(Object25) => x(Object3) < x(Object25).
NOT E(Object3,Object25) in every valid arrangement.
[p25, p20, p11, p24, p14, p5, p19] x(Object12) < x(Object19) < x(Object11) < x(Object23) < x(Object5) < x(Object13) < x(Object7) < x(Object25) => x(Object12) < x(Object25).
NOT E(Object12,Object25) in every valid arrangement.
[p3, p24, p14, p5, p19] x(Object22) < x(Object23) < x(Object5) < x(Object13) < x(Object7) < x(Object25) => x(Object22) < x(Object25).
NOT E(Object22,Object25) in every valid arrangement.
[p15] x(Object17) < x(Object25).
NOT E(Object17,Object25) in every valid arrangement.
[p10, p11, p24, p14, p5, p19] x(Object10) < x(Object11) < x(Object23) < x(Object5) < x(Object13) < x(Object7) < x(Object25) => x(Object10) < x(Object25).
NOT E(Object10,Object25) in every valid arrangement.
Fixed qualifying set={}; count=0 in every valid arrangement.

Case 1 (counterexample satisfying all premises): W(Object0,Object25) & W(Object3,Object25) & W(Object12,Object25) & W(Object22,Object25) & W(Object17,Object25) & W(Object10,Object25). Qualifying objects = {}. count=0.

The claim that count=1 is false in Case 1, so it is not entailed.

The claim that count=2 is false in Case 1, so it is not entailed.

The claim that count=3 is false in Case 1, so it is not entailed.

The claim that count=4 is false in Case 1, so it is not entailed.

The claim that count=5 is false in Case 1, so it is not entailed.

The claim that count=6 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {0}.

Answer: D

## production-b2904-s345-which-m0-unique-entailed

Categories: sole-entailed-with-contingent.
Split: train; contract: unique-entailed; answer: D.

### Question

Interface: spatialentail-entailment-v1
Contract: unique-entailed
Named objects are distinct points on fixed global x/y axes. No two objects coincide; sharing one coordinate is allowed. Directions are exact coordinate comparisons: North means equal x and greater y; Northeast means greater x and greater y. There are no distances, unit steps, grid spacing, or map bounds. Unstated relations are unknown, not false. A valid arrangement satisfies every premise and these assumptions. A claim is entailed only when it holds in every valid arrangement. A claim that is merely compatible with the premises is not necessarily entailed. Under every contract, select E if no valid arrangement exists.
Select the one A-D value if exactly one domain answer is entailed. Select F if that sole entailed answer is unlisted. Select E if zero or multiple domain answers are entailed. For Selection, count individually entailed candidates; an additional candidate that is not guaranteed does not invalidate the sole entailed candidate.
Eligible objects are independent of the menu. Count concerns all eligible objects jointly. Zero is an ordinary exact Count answer.
End with Answer: <letters>. Use ascending uppercase letters without spaces or punctuation. Unique requires one letter; other contracts allow a nonempty subset of A-D. E and F must each appear alone.
Objects: Object0, Object1, Object2, Object3, Object4, Object5, Object6
Query: which North Object0
Eligible: Object1, Object2, Object3, Object4, Object5, Object6
Consider a map with multiple objects:
Object4 is to the Northeast of Object6.
Object3 is to the West of Object4.
Object1 is to the Northwest of Object5.
Object3 is to the North of Object0.
Object3 is to the Northeast of Object6.
Object1 is to the West of Object2.
Question: Which single eligible candidate is individually guaranteed to be north of Object0?
A. Object4
B. Object1
C. Object2
D. Object3
E. Cannot be determined
F. None of the above

### Natural

Object0 and Object3 share the same x-coordinate (premise sources: 4).
Object3 has a greater y-coordinate than Object0 (premise sources: 4).
Therefore, Object3 is north of Object0 in every valid arrangement.

Case 1: A counterexample arrangement satisfying all premises is the following. Object1, Object2, Object5 and Object6 are northwest of Object0. Object3 is north of Object0. Object4 is northeast of Object0. Only Object3 qualifies in this arrangement.

The claim that Object1 is north of Object0 is false in Case 1, so it is not entailed.

The claim that Object2 is north of Object0 is false in Case 1, so it is not entailed.

The claim that Object4 is north of Object0 is false in Case 1, so it is not entailed.

The claim that Object5 is north of Object0 is false in Case 1, so it is not entailed.

The claim that Object6 is north of Object0 is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Object3}.

Answer: D

### Symbolic

[p4] x(Object0) = x(Object3).
[p4] y(Object0) < y(Object3).
N(Object3,Object0) in every valid arrangement.

Case 1 (counterexample satisfying all premises): NW(Object1,Object0) & NW(Object2,Object0) & NW(Object5,Object0) & NW(Object6,Object0). N(Object3,Object0). NE(Object4,Object0). Qualifying objects = {Object3}.

The claim that N(Object1,Object0) is false in Case 1, so it is not entailed.

The claim that N(Object2,Object0) is false in Case 1, so it is not entailed.

The claim that N(Object4,Object0) is false in Case 1, so it is not entailed.

The claim that N(Object5,Object0) is false in Case 1, so it is not entailed.

The claim that N(Object6,Object0) is false in Case 1, so it is not entailed.

The complete entailed-answer set is {Object3}.

Answer: D
