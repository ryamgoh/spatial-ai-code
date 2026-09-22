# Progress Report: Experiments 10–14

## Purpose

This phase of the project focused on making the spatial-reasoning task more
precise, identifying the model's remaining weaknesses, and building a reliable
starting point for GRPO.

The work progressed through five stages:

1. correct the answer semantics;
2. identify hidden failure modes;
3. rebalance supervised training;
4. test harder structural generalization; and
5. prepare a controlled GRPO feasibility test.

## Experiment 10: Adding a valid “none” answer

We added a fifth option, **None of these is proven**, so the model would not be
forced to choose an unsupported answer from A–D. The 1,500 SpatialMap examples
were regraded under this rule: 369 received the new answer and 1,131 retained an
A–D answer.

We also prepared nested 1.5K, 5K, and 20K synthetic training sets for 0.8B, 2B,
and 4B models. This established a cleaner five-option task, but no completed
Experiment 10 result table is recorded in the repository.

## Experiment 11: Finding the difficult cases

We introduced a solver-generated dataset with explicit question subtypes. The
completed 4B model trained on 1.5K examples achieved:

- **93.9%** strict accuracy on the 4K synthetic test;
- **96.1%** strict accuracy on SpatialMap v6.

The overall score hid a concentrated weakness. Three direction cases accounted
for 188 of 245 errors:

| Case | Accuracy | Main difficulty |
|---|---:|---|
| Contradictory cycle | 36.8% | Detecting that conflicting relations invalidate the answer |
| Incomplete direction pair | 55.6% | Avoiding an unsupported choice when only one compatible direction is listed |
| Two-answer direction | 86.5% | Returning both directions when only one axis is known |

Most other question types were already above 96%. This showed that the next
step should target specific policies rather than add more easy examples.

## Experiment 12: Rebalancing the supervised curriculum

We revised the training setup by:

- stating the three difficult direction rules explicitly;
- balancing direction, object-selection, and counting questions;
- balancing subtypes within each question family;
- separating training, validation, and test data; and
- selecting checkpoints using a hard validation subset.

Later notes report about **97%** matched-test accuracy for the 1.5K model and
**96%** for the 6K model. These values should be treated as approximate because
the native Experiment 12 result report is not committed.

When transferred to the harder V13 diagnostic, the 1.5K and 6K models scored
52.9% and 43.7% under two-stage evaluation. Strong matched-test performance
therefore did not imply strong structural generalization.

## Experiment 13: Harder reasoning and better traces

We built V13 to test capabilities missing from the earlier data:

- cardinal, diagonal, and mixed relations;
- controlled proof depth;
- independent X- and Y-axis proofs;
- disconnected and query-branch distractors;
- global inconsistency detection; and
- matched closed-loop and open-chain cases.

### Curriculum development

The first 400-row probe improved ordinary reasoning but reduced closed-loop
accuracy from 59.8% to 22.6%. A second probe increased the number and variety of
closed-loop and matched open-chain examples while keeping the training budget
fixed. It reached 68.7% overall, 74.5% on closed loops, and 61.4% on open-chain
controls.

The curriculum then scaled successfully:

| Model | V13 accuracy | Depth 5 | Closed loop | Open chain |
|---|---:|---:|---:|---:|
| Untuned 4B | 42.0% | 72.9% | 59.8% | 7.4% |
| Probe v2 | 68.7% | 68.1% | 74.5% | 61.4% |
| V13 1.5K | 85.4% | 94.4% | 91.9% | 82.6% |
| V13 6K | **94.9%** | **100.0%** | **99.8%** | **95.0%** |

### Structural breakpoint

The harder V13.1 evaluation extended proofs and loops to lengths 6, 8, and 10
and added stronger interference. The 6K model fell to 82.4%. Adding 2K targeted
examples raised this only to 85.6%, while length-10 performance remained near
63%. More examples alone did not solve the problem.

### Trace representation

We compared two reasoning traces using the same 8K prompts, answers, data order,
optimizer, and context length:

- **Full state:** rewrite the complete X/Y graph after every premise.
- **Delta state:** record only the affected graph component, then print the
  complete graph once before answering.

| Trace | Original V13 | V13.1 | Length/depth 10 |
|---|---:|---:|---:|
| Full state | 95.5% | 85.6% | 63.2% |
| Delta state | **96.2%** | **97.8%** | **97.5%** |

This result indicates that repeated full-state serialization created an
artificial long-context reasoning bottleneck. Delta-state traces are now the
default representation.

## Experiment 14: Preparing GRPO

Experiment 14 asks whether outcome-only GRPO can improve hard spatial problems
without requiring a gold reasoning trace and without damaging the capabilities
learned through SFT.

The current setup contains:

- 680 prompt-only training examples;
- a disjoint 300-example hard holdout;
- four sampled answers per prompt;
- an exact answer-set reward from the symbolic solver;
- a small answer-format reward; and
- retention checks on V13 and V13.1.

Before training, a calibration stage checks whether the four answers to the same
prompt contain both successes and failures. GRPO cannot learn from a group whose
answers all receive the same reward.

The GRPO pipeline and evaluation protocol are implemented. No completed GRPO
result is recorded yet, so the current claim is limited to feasibility and
experimental readiness.

## Current position

The main result so far is not that additional data always improves reasoning.
The experiments show that task semantics, curriculum balance, evaluation
difficulty, and trace representation can each determine the apparent capability
of the model.

The strongest current model is the 4B delta-state SFT model, with 96.2% on V13
and 97.8% on V13.1. This provides the starting policy for the GRPO feasibility
test.

## Question types and example reasoning

The task represents space with two ordered graphs:

- X-axis: west `<` east;
- Y-axis: south `<` north.

For example, `B < A` on X means A is east of B. A diagonal answer requires a
result from both axes.

### Direction questions

#### `dir-1`: both axes are known

```text
X-proof: B < A -> A is east of B
Y-proof: B < A -> A is north of B
Composition: east + north = northeast
Answer: the option containing Northeast
```

#### `dir-2`: only one axis is known

This case comes directly from the original SpatialEval item
`spatialmap.tqa.2003.0`. Its relevant statements are:

```text
Sally's Salon is Southeast of Recycle Center.
Andy's Autos is Northwest of Sally's Salon.
Andy's Autos is Southeast of Recycle Center.
Nightingale Novelties is Southwest of Andy's Autos.
Nightingale Novelties is Southwest of Sally's Salon.

Question: Where is Recycle Center relative to Nightingale Novelties?
Options: Southeast, Southwest, Northeast, Northwest
```

The dataset's original single answer was **Northeast**, corresponding to one
underlying grid arrangement. Once the grid is removed, however, the prose proves
only that Recycle Center is **north** of Nightingale Novelties. It never fixes
which one is farther east. Both layouts below satisfy the relevant statements:

```text
Layout 1                              Layout 2

North ↑                               North ↑

....A....                             ....A....
.........                             .........
......B..                             ......B..
.......C.                             .......C.
..D......                             .....D...

East →                                East →

A is northeast of D.                  A is northwest of D.
```

Legend: `A` = Recycle Center, `B` = Andy's Autos, `C` = Sally's Salon, and
`D` = Nightingale Novelties. Dots are empty grid cells. In both layouts:

- Sally is southeast of Recycle Center;
- Andy is southeast of Recycle Center and northwest of Sally; and
- Nightingale is southwest of both Andy and Sally.

The corresponding symbolic deduction is:

```text
Y-proof: Nightingale < Andy < Recycle Center -> Recycle Center is North
X-proof: no path orders Recycle Center and Nightingale -> East/West unknown
Possible directions: Northeast or Northwest
Answer: C, D
```

If the query is reversed—Nightingale relative to Recycle Center—the same
missing X-axis fact gives **Southeast or Southwest**:

```text
North ↑

....A....
.........
.........
..D...D..
..SW..SE.

East →
```

The key distinction is between the original hidden map and the textual task. A
map assigns exact coordinates, so one diagonal happens to be true. The prose
contains only qualitative inequalities. If those inequalities determine one
axis but not the other, selecting the hidden-map direction would add information
that was never stated. Under text-only entailment, both compatible options must
therefore be accepted.

#### `dir-undetermined`: neither axis is known

```text
X-proof: no path
Y-proof: no path
No complete direction is proven
Answer: Cannot be determined
```

#### `dir-incomplete`: one possible direction is missing

Suppose east is known, so the valid pair is northeast or southeast, but only
northeast is listed. Selecting it would invent a north relation.

```text
Known axis: east
Unknown axis: north or south
Required pair: {northeast, southeast}
Only northeast is listed
Answer: Cannot be determined
```

#### `dir-omit`: the proven direction is absent

```text
X-proof -> east
Y-proof -> north
Derived direction: northeast
Northeast is not listed
Answer: None of the Options
```

### Which-object questions

`which-1` through `which-4` use the same rule. The number states how many listed
objects satisfy the requested relation.

Example: “Which objects are north of R?”

```text
R < A on Y -> A is north -> In
No path from R to B -> B is not proven north -> Out
R < C on Y -> C is north -> In
R < D on Y -> D is north -> In
Answer: A, C, D
```

This is a `which-3` example. Keeping one, two, or all four proven objects gives
`which-1`, `which-2`, or `which-4`. For diagonal queries, each selected object
must satisfy both required axes.

For `which-0`, none of the listed objects is proven:

```text
Check every listed object against all required axes
Proven listed set: empty
Answer: None of the Options
```

### Count questions

For `count-1`, the derived count is present in the menu. Zero is a valid count.

```text
Proven objects northwest of R: {A, C, D}
Count: 3
Option “3” is listed
Answer: the letter containing 3
```

For `count-omit`, the derived count is absent:

```text
Proven set: {A, C, D}
Count: 3
No option contains 3
Answer: None of the Options
```

### Globally inconsistent worlds

A cycle on either axis invalidates the complete V13 world, even when the cycle
is disconnected from the queried objects.

```text
X relations: A < B, B < C, C < A
Cycle detected: A < B < C < A
The premise set is inconsistent
Answer: Cannot be determined
```

This rule applies to direction, which-object, and count questions.

### Matched open-chain control

An open chain has a similar shape but no contradiction. It must be solved
normally.

```text
A < B, B < C, C < D
No cycle detected
Continue with the normal direction, which-object, or count rule
```

These controls test whether the model detects actual inconsistency rather than
rejecting every long chain.

### Full-state and delta-state traces

Consider the same SpatialEval-style question under both trace formats:

```text
A is west of B.
A is south of E.
B is southwest of C.
E is south of F.
C is west of D.
F is southwest of G.
D is west of J.
G is south of J.
H is north of B.
K is southwest of L.

Question: In which direction is J relative to A?
Options:
A. Northeast
B. Southeast
C. Southwest
D. Northwest
E. Cannot be determined
```

This example maintains two independent proof chains:

```text
X proof: A < B < C < D < J
Y proof: A < E < F < G < J
```

The diagonal statements also create secondary state: `B < C` on Y and `F < G`
on X. `B < H` is a branch attached to the query graph, while `K < L` is a
disconnected distractor. The answer is **A. Northeast**.

#### Full-state reasoning

The full-state trace rewrites everything known after every premise:

```text
<think>
Step 1: A is west of B.
X-Extraction: A < B
Y-Extraction: none
X-State: A < B
Y-State: Empty

Step 2: A is south of E.
X-Extraction: none
Y-Extraction: A < E
X-State: A < B
Y-State: A < E

Step 3: B is southwest of C.
X-Extraction: B < C
Y-Extraction: B < C
X-State: A < B < C
Y-State: A < E; B < C

Step 4: E is south of F.
Y-Extraction: E < F
X-State: A < B < C
Y-State: A < E < F; B < C

Step 5: C is west of D.
X-Extraction: C < D
X-State: A < B < C < D
Y-State: A < E < F; B < C

Step 6: F is southwest of G.
X-Extraction: F < G
Y-Extraction: F < G
X-State: A < B < C < D; F < G
Y-State: A < E < F < G; B < C

Step 7: D is west of J.
X-Extraction: D < J
X-State: A < B < C < D < J; F < G
Y-State: A < E < F < G; B < C

Step 8: G is south of J.
Y-Extraction: G < J
X-State: A < B < C < D < J; F < G
Y-State: A < E < F < G < J; B < C

Step 9: H is north of B.
Y-Extraction: B < H
X-State: A < B < C < D < J; F < G
Y-State: A < E < F < G < J; B < C; B < H

Step 10: K is southwest of L.
X-Extraction: K < L
Y-Extraction: K < L
X-State: A < B < C < D < J; F < G; K < L
Y-State: A < E < F < G < J; B < C; B < H; K < L

Final Deduction:
X-proof: A < B < C < D < J -> J is east of A.
Y-proof: A < E < F < G < J -> J is north of A.
Composition: east + north = Northeast.
</think>
Answer: A
```

#### Delta-state reasoning

The delta-state trace prints only the component affected by each premise, then
prints the complete state once before answering:

```text
<think>
Step 1: A is west of B.
X-Extraction: A < B
X-Affected Component: A < B
Conflict Status: X=no; Y=no

Step 2: A is south of E.
Y-Extraction: A < E
Y-Affected Component: A < E
Conflict Status: X=no; Y=no

Step 3: B is southwest of C.
X-Extraction: B < C
Y-Extraction: B < C
X-Affected Component: A < B < C
Y-Affected Component: B < C
Conflict Status: X=no; Y=no

Step 4: E is south of F.
Y-Extraction: E < F
Y-Affected Component: A < E < F
Conflict Status: X=no; Y=no

Step 5: C is west of D.
X-Extraction: C < D
X-Affected Component: A < B < C < D
Conflict Status: X=no; Y=no

Step 6: F is southwest of G.
X-Extraction: F < G
Y-Extraction: F < G
X-Affected Component: F < G
Y-Affected Component: A < E < F < G
Conflict Status: X=no; Y=no

Step 7: D is west of J.
X-Extraction: D < J
X-Affected Component: A < B < C < D < J
Conflict Status: X=no; Y=no

Step 8: G is south of J.
Y-Extraction: G < J
Y-Affected Component: A < E < F < G < J
Conflict Status: X=no; Y=no

Step 9: H is north of B.
Y-Extraction: B < H
Y-Affected Component: B < {C, H}
Conflict Status: X=no; Y=no

Step 10: K is southwest of L.
X-Extraction: K < L
Y-Extraction: K < L
X-Affected Component: K < L
Y-Affected Component: K < L
Conflict Status: X=no; Y=no

Complete State:
Final X-State: A < B < C < D < J; F < G; K < L
Final Y-State: A < E < F < G < J; B < C; B < H; K < L

Final Deduction:
X-proof: A < B < C < D < J -> J is east of A.
Y-proof: A < E < F < G < J -> J is north of A.
Composition: east + north = Northeast.
</think>
Answer: A
```

Both traces reach the same proof and answer. Full-state repeatedly copies every
known relation. Delta-state shows local updates and reconstructs the complete
state once at the end. The difference becomes substantial on long problems.
