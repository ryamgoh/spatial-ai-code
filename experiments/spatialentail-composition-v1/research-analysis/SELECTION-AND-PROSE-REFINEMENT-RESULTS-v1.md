# Selection and Shared-Proof Refinement Results

Development-only replay and subset selection. No new stories, split assignment, production quota adoption, seal or training. All previous result artifacts remain unchanged.

## Selection

Same deduplicated pool and 70 bundles per task: 210 bundles and 630 views per style. Every bundle retains all three contracts. The new policies require the exact 123 fine cells present in the old-pool selection, not merely any 123 cells.

| Policy | Fine cells | Unique Selection E/F | Overall E/F | Single-base cells |
| --- | --- | --- | --- | --- |
| Old pool | 123 | 50/70 (71.4%) | 175/630 (27.8%) | 60 |
| Expanded, E/F first | 99 | 17/70 (24.3%) | 117/630 (18.6%) | 54 |
| coverage_only | 123 | 18/70 (25.7%) | 126/630 (20.0%) | 90 |
| coverage_and_pattern_caps | 123 | 29/70 (41.4%) | 137/630 (21.7%) | 78 |

Both new policies reported optimality at every MILP stage with zero relative gap. The spatial solver is still Z3; HiGHS only selects existing bundles. Optimality is solver-reported numerical optimality for this fixed pool and policy, not a formal rational certificate.

Coverage-only selects 33/70 questions from the three new patterns. The capped policy sets seven per pattern, selecting 21/70 (30%). Seven is the predeclared diagnostic setting, not an approved production quota. No cap was relaxed.

| Selection source | Coverage only | Coverage + caps |
| --- | --- | --- |
| contingent | 0 | 1 |
| controlled-depth | 23 | 24 |
| match-chain | 7 | 7 |
| multiple | 1 | 0 |
| random-atoms | 3 | 7 |
| random-world | 10 | 16 |
| rival-chain | 12 | 7 |
| shared-exclusion | 14 | 7 |
| singleton | 0 | 1 |

| Substantive Unique Selection depth | Coverage only | Coverage + caps |
| --- | --- | --- |
| 1 | 5 | 5 |
| 2 | 3 | 3 |
| 3 | 6 | 5 |
| 4 | 4 | 3 |
| 5 | 7 | 3 |
| 6 | 6 | 5 |
| 7 | 5 | 5 |
| 8 | 6 | 3 |
| 9 | 6 | 5 |
| 10 | 4 | 4 |

No protected category is missing. All depths 1-10 remain represented, but these are development examples, not held-out benchmark families. Caps apply only to the three additions; the capped subset still contains 24 older controlled-depth Selection bases and 23 random-world/random-atom Selection bases.

Coverage counts do not imply adequate replication: 78/123 cells in the capped subset have only one distinct base, versus 60/123 in the old subset. Distinct bases are not necessarily independent construction families. No independent-family minimum has been established.

## Shared-Proof Replay

Accepted replay: 300/300 bases; 1719 paired views; 80 changed pairs. The replay includes the one duplicate expansion base deliberately, to test all saved outputs.

Repeated unconditional axis subpaths are now stated once as a numbered comparison. Later deductions cite that endpoint comparison and their remaining premise links. Equality paths may be reused in reverse; strict-order paths may not. Assumption/contradiction contexts are not factored.

Original full paths, premise indices, depth metadata, questions, menus, golds, semantic cases and source IDs remain unchanged. Removing display-only factoring recovers the original selected evidence exactly. Numerical witness maps remain audit-only.

| Style | Old mean | New mean | Old maximum | New maximum |
| --- | --- | --- | --- | --- |
| natural | 265.9 | 258.2 | 1127 | 930 |
| symbolic | 224.0 | 220.2 | 812 | 790 |

These are target-token counts for all 1,719 views per style under the pinned Qwen3.5-4B tokenizer. The response/context limits are unchanged.

All 80 changed pairs belong to shared-exclusion constructions; the other 1,639
pairs are byte-identical. Natural is shorter in 79 views and longer in one
(by six tokens). Symbolic is shorter in 68 views, unchanged in token count in
1,645 views, and longer in six (by at most 18 tokens). Comparison labels can
cost more tokens than they save on short chains, even when they remove
repeated links. No universal token reduction is claimed.

For the depth-10 shared-exclusion group, maximum target lengths fell from
1,127 to 662 Natural and 812 to 602 Symbolic. These group maxima are distinct
from the individual Unique example lengths in the table below.

## Verification

- Full composition/pivot suite: **796 passed, 1 skipped**. Changed Python files
  passed Ruff; tracked changes passed whitespace checks.
- Independently verified **67 hashes** for the selection run and **72 hashes**
  for the replay, covering their recorded inputs, current source snapshot,
  output artifacts, and the replay's pinned tokenizer.
- Independently reconstructed both selected subsets and rechecked exact cell
  preservation, task quotas, base/stratum restrictions, pattern caps and saved
  metrics. No solver constraints were relaxed.
- Re-audited all saved replay outputs: **5,628 expanded printed paths** and
  **1,537 qualitative scenarios per style**, with **zero printed numerical
  witness maps**. The auditor checks actual cited premise links, prior comparison
  definitions, expanded endpoints, full witness completions and cross-style
  agreement. It does not prove arbitrary conclusion prose or human fluency.
- Rebuilt solver evidence and checked unchanged semantics during replay.
  Display-only factoring was applied to the saved selected outline; removing
  it recovers that outline exactly. Questions, gold, menus, measured depths,
  source IDs and substantive-depth admission flags were preserved.
- Mutation tests reject missing/forward/incorrect comparison references,
  definitions and printed links even when both styles are corrupted together.
  Additional direct checks rejected wrong axes/endpoints/signs and strict-order
  reversal. Temporary-assumption contexts remain unfactored.

## Interpretation

The coverage regression is fixed by an explicit hard constraint. The pattern cap reduces concentration at a measurable cost in E/F; it does not create new diversity. The capped result is the more conservative development example, not an automatically adopted training distribution.

Shared-path rendering removes repeated trunk derivations while retaining a checked inclusion/exclusion argument for every required candidate. Remaining per-candidate phrasing is deliberately regular. This is generated readable evidence, not a claim of human prose evaluation or a globally shortest proof.

Production sizes, supported cells, family isolation and replication minima still need explicit approval. The unobserved Direction size-four and varying-identity invariant Count cases, and the indirect-coincidence certificate limitation, are unchanged. This pass does not authorize further sampling or sealing.

## Artifacts

- `../selection-refinement-v1/`: constraints, selected IDs, metrics and hashes.
- `../shared-proof-replay-v4/`: all replayed outputs, token summaries and hashes.
- `SHARED-PROOF-EXAMPLES-v4.md`: actual saved paired explanations.
- `SELECTION-AND-PROSE-REFINEMENT-PROTOCOL-v1.md`: fixed pre-run policy.

## Saved Example Lengths

| Pattern | Depth | Style | Old tokens | New tokens |
| --- | --- | --- | --- | --- |
| shared-exclusion | 3 | natural | 554 | 515 |
| shared-exclusion | 3 | symbolic | 394 | 394 |
| shared-exclusion | 10 | natural | 990 | 654 |
| shared-exclusion | 10 | symbolic | 804 | 594 |
| match-chain | 10 | natural | 407 | 407 |
| match-chain | 10 | symbolic | 400 | 400 |
| rival-chain | 10 | natural | 356 | 356 |
| rival-chain | 10 | symbolic | 320 | 320 |
