# Question-Led Explanation Results

## Verdict

Implemented and verified on the existing development examples. No questions
were generated, no selection quotas changed, and nothing was sealed or trained.
The prose now starts with the question's purpose, presents guaranteed matches
first within fixed-membership evidence, and groups shared branches and exclusion
reasons. Both styles retain the same selected deductions.

This is a more readable deterministic justification, not a claim of human-like
discovery or a transcript of how a model internally solves the scene.

## Verification

- Final composition and pivot regressions: 841 passed, 1 skipped. Ruff passed
  for changed Python files, and the final diff has no whitespace errors.
- Replayed all 300 saved bases and 1,719 paired views from version 4.
- Zero failures; all references pass the existing tokenizer admission limits.
- Preserved prompts, menus, answers, semantic labels, family metadata, depth
  annotations, qualitative cases and audit witnesses.
- Compared exact evidence-unit inventories, allowing presentation reordering
  but no removal or addition of substantive deductions.
- Independently expanded grouped branch sentences into the original cited
  paths; checked grouped exclusion signs, candidates and query directions.
- The saved outlines contain 880 grouped exclusions across views, and the
  printed outputs contain 97 grouped branch paragraphs per style.
- Fresh outline construction and text auditing also passed 180 pairs covering
  all 60 fixed expansion constructions under three contracts.
- After replay, fixed repeated organization losing grouping. Revalidated every
  saved pair against the final sources, including idempotence; all outputs are
  exactly unchanged. The original replay manifest remains unchanged.

See [final-source revalidation](PROSE-ORGANIZATION-REVALIDATION-v5.json),
[replay manifest](../shared-proof-replay-v5/manifest.json) and
[full token summary](../shared-proof-replay-v5/summary.json).

## Token Counts

Actual target tokens under the pinned Qwen3.5-4B tokenizer:

| Measure | Natural v4 | Natural v5 | Symbolic v4 | Symbolic v5 |
|---|---:|---:|---:|---:|
| All-view mean | 258.2 | 251.7 | 220.2 | 229.1 |
| All-view maximum | 930 | 925 | 790 | 794 |
| Shared-exclusion depth 3, sample 0, Unique full menu | 515 | 277 | 394 | 257 |
| Shared-exclusion depth 10, sample 0, Unique full menu | 654 | 486 | 594 | 510 |

Grouping reduces repeated work in branch-heavy examples. Task introductions
add text, especially in short or already concise examples; not every response
becomes shorter. Token reduction is not the acceptance criterion for this
presentation refinement.

## Inspection

Read the [nine actual saved pairs](../shared-proof-replay-v5/examples.md):
Direction, Selection and Count, plus all three expansion patterns at depths 3
and 10. The primary coding assistant inspected these; this is not an independent
human fluency evaluation.

In the depth-3 shared-exclusion example, the explanation establishes Object2
first, derives Object3's position once, groups five candidates sharing Object3's
horizontal position, and then rules out the other eligible candidates together.
Their individual premise correspondence remains explicit in both styles.

Remaining limitations are visible rather than hidden: the prose is templated,
long object lists and citations remain, and isolated candidates may still get
individual proof paragraphs. Count's joint-subset explanation and scenario
wording were not redesigned. The auditor covers specified evidence-bearing
grammar, not arbitrary free-form prose or model internal representations.

## Next Gate

Review the actual wording before production generation. The previously agreed
dataset sizes and remaining sampling decisions are unaffected by this change.
