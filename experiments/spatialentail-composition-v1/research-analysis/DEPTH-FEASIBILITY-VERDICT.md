# Depth Feasibility Verdict

## Decision

**Keep the pivot. Do not generate or seal the production training/benchmark
sets yet.** The first bounded pilot establishes that the existing solver and
evidence infrastructure can support the requested depth descriptors through
10, including fixed-membership Selection and Count. It also exposes a concrete
renderer token blocker and unfinished sampling/diversity requirements.

This is the requested pre-production report. No model training, GPU inference,
production export, new seal, commit or push was performed. Prior packages and
the v2 seal still verify unchanged.

## What Ran

The [protocol](DEPTH-FEASIBILITY-PROTOCOL.md) was written before execution.
Seed 20261101, 5-second timeout per solver call, no retries:

- 60 depth-controlled premise sets, six at each requested depth 1-10.
- 12 random-world, 12 random-atom and ten targeted semantic controls.
- 94 premise sets, 282 query attempts; three question types per set.
- Full paired contracts/styles; omission menus added on the fixed one-in-four
  base-ID hash policy when a valid omission variant exists.
- Existing proof/certificate verification, full solver labels, independent
  coordinate checks, prompt round trips, paired factual alignment, rename/order
  checks and individual/joint designated-distractor removals.
- Exact pinned Qwen3.5-4B tokenizer measurement for both styles, with the
  existing 4,096-token Stage-1 and 8,192-context limits.

Result: **247 accepted bases, 35 token-rejected bases, 861 views per style**.
All 94 stories retain at least one query. There are 45 source families and
94 distinct canonical story groups. No split-conflict, unsupported-evidence
or verification-failed attempts occurred in this finite schedule.

The [measurement tables](DEPTH-FEASIBILITY-MEASUREMENTS.md) include every
depth/role count, observed answer-size combination, missing literal gold,
token maximum and rejection. Raw complete records are in
[`depth-feasibility-v1`](../depth-feasibility-v1/).

## Depth Result

All 180 controlled queries passed evidence and requested-depth checks before
token admission. The depth is the declared shortest atomic axis-support
descriptor, not a proved global minimum of reasoning effort.

| Requested Depth | Direction Admitted | Selection Admitted | Count Admitted |
| --- | ---: | ---: | ---: |
| Each of 1-4 | 6/6 | 6/6 | 6/6 |
| 5 | 1/6 | 6/6 | 6/6 |
| Each of 6-10 | 0/6 | 6/6 | 6/6 |

Selection and Count include positive and negative memberships. Each admitted
controlled Selection/Count base has a maximum-depth support ablation changing
the complete semantic answer; Count checks changes to possible counts, not
just member identities. This dependence test is not proof that no shorter
alternative algorithm can solve the query.

### The Direction Blocker

The current Exhaustive renderer repeats exclusions for all seven alternative
directions, often repeating most of the long axis derivation. At depth 10,
the largest references reached **9,062 Natural / 7,920 Symbolic tokens**.
This violates the Stage-1 cap; some full sequences also exceed context.

Concrete case: `depth-5-sample-0-direction-m0-single` needs 787 Natural target
tokens, whereas its full-menu Exhaustive counterpart needs 4,342. They have
the same sole possible direction; the difference is repeated evidence, not
a more difficult semantic answer.

The correct next change is a separately tested compact proof selection:
derive the entailed exact direction once, establish that the other exact
directions are excluded by those coordinate comparisons, then apply the
contract/menu rule. Keep the full checked exclusion evidence in audit records.
Apply the same factual selection to Natural and Symbolic. This must also cover
Listed when a unique answer is absent, without silently changing the contract.
Do not increase token limits, truncate, or drop only the failing contract/style
to make the current run appear successful.

All Selection and Count references in this run fit: maxima were 2,247/1,977
tokens for Selection and 2,589/2,281 for Count (Natural/Symbolic).
This is evidence for these scaffolds, not a guarantee for richer future graphs.

## Gold Result

E/F answers are valid labels, not inherently bad data. Their mixture must be
controlled by semantic cases and menu policy, not independently balanced
across contracts.

| Contract | E/F Share in Accepted Pilot |
| --- | ---: |
| Unique | 51.9% |
| Listed | 21.6% |
| Exhaustive | 32.4% |
| Combined | 35.3% (304/861) |

The training-role-selected contract-majority baseline gives 26.0% accuracy
on the provisional in-range test, versus 51.3% on the old sealed test.
**This is not a controlled improvement result**: the datasets differ, this
pilot is small, and token rejection removed many unique Direction examples.
The point is that the mixture can change substantially without falsifying gold.
Unique Selection still has only nine substantive training-role views here;
that is not an adequate final quota.

The observed combinations confirm the user's concern:

- Unique returns one substantive value, never two to four.
- All controlled Direction and fixed Count queries have one possible value.
- Fixed Selection can have multiple possible members.
- The random/targeted variable Count controls supply returned sizes 2-4.
- Listed can return four displayed possibilities when the full set has more
  than four; Exhaustive then returns F.
- E/F have null substantive cardinality. A count value of 4 is a singleton
  answer, not four correct options.

Training has all six Unique and all 17 Listed/Exhaustive strings, but dev and
in-range test do not. Train and in-range test cover all 33 semantic cells;
dev covers only 24. In particular, the random source-role assignment did not
place any inconsistent control in dev. Production needs deterministic
pre-admission category scheduling and feasible-position coverage, not another
unconstrained random draw.

## Diversity and Benchmark Limits

The controlled cohort has 60 canonical identities and 50 graph degree profiles,
but those numbers overstate its conceptual variety: it uses a single long
backbone, direct other memberships, optional leaf branches, and disconnected
distractors. It does not yet test independently composed x/y networks or
multiple essential deep memberships. Those are implementation gaps, not reasons
to claim the pivot is invalid or launch another research direction.

Six in-range source families deliberately group depth variants; six disjoint
families cover depths 9/10. There are only two controlled source families per
in-range role, so this is not a credible final generalization benchmark.
Production needs many independently generated families, not more renamed or
lengthened copies of these twelve.

The underlying query's depth is not necessarily the difficulty of its final
letter. A Unique multiple-match escape can be established by two easy matches;
Exhaustive F can be established by one unlisted witness. Depth analysis must
separate these from substantive full-menu cases. Even a Listed substantive
answer can omit a deep excluded candidate, so report menu-relevant support
coverage rather than assuming every view exercises the maximum underlying depth.

Independent reference-backend enumeration was not run for these larger
problems. Verification uses Z3 plus proof replay, direct witness checks and
metamorphic checks. The BFS path finder additionally passed independent
small-graph path enumeration tests.

## Inspection and Tests

1,176 distinct selected regression tests passed across the final test runs:
36 new depth/report tests, 282 existing composition/difficulty tests, and
858 tokenizer/solver/proof/certificate tests. Lint and whitespace checks passed.

Six full Natural/Symbolic pairs were inspected: the three query types at
`depth-1-sample-0`, Count at `depth-6-sample-0`, Direction at
`depth-10-sample-0`, and zero Count at `depth-10-sample-1`.
Their full prompts and paired references are in the
[example supplement](DEPTH-FEASIBILITY-EXAMPLES.md).
No factual discrepancy was found in these six pairs. This is a spot-check,
not the pivot's complete category/depth release audit.

## Proposed Next Work

1. Implement and test compact unique-Direction evidence for all three contracts
   in a new versioned rendering path, preserving the old sealed sources.
2. Add independently composed axes and multiple deep-member constructions;
   check shortcut resistance, semantic dependence, and label-structure
   correlations. Do not solve coverage by merely renaming this backbone.
3. Schedule semantic strata into provisional roles before admission, including
   contradictions in dev, and deterministically cover feasible gold positions.
4. Run one newly documented bounded pilot and report again. Only then settle
   numeric proportions, total sizes, and final attempt caps.

For planning, a depth-cohort starting target could be 40 distinct bases per
query/depth in training, ten in dev and ten in in-range test, and 20 per
query/depth at 9/10. That is **1,560 depth-controlled bases**, before additional
semantic coverage. Require multiple independent families per cell; menu/style
copies do not count. These are proposal numbers, not a justified final sample
size, approved generation budget, or instruction to run production.

Do not freeze semantic proportions or a universal E/F threshold from this
token-biased pilot. Final sizes remain pending the corrected feasibility
results and user review. No new pivot is needed; production approval remains
blocked by the concrete items above.
