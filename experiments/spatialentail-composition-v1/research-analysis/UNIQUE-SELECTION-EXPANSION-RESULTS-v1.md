# Bounded Unique Selection Expansion Results

Development feasibility only. No production data, benchmark assignment, seal or training. Old pool, references and selection artifacts remain unchanged.

## Fixed Attempt Results

60 new Selection-only attempts: three patterns, depths 1-10, two variants per pattern/depth. Seed 20261031, no retries. Full certificates, qualitative paired renderings, printed-evidence audit, prompt round trips, actual tokenizer admission, and semantic/depth dependence checks use the existing pipeline.

| Status | Bases |
| --- | --- |
| accepted | 60 |

Distinct added query signatures after old-pool/within-expansion deduplication: **59**. Duplicate exclusions: 1. Accepted views per style before deduplication: 360. Each base has a covered menu and an omitted-answer menu, each with all three contracts. Those omission F answers are deliberate candidate alternatives, not forced training exposure.

Available substantive Unique Selection bases: **22 -> 81**.

## Patterns

- Match chain: a long inclusion argument for the only match; direct exclusions.
- Rival chain: a direct match; one long exclusion argument for a rival.
- Shared exclusion trunk: a direct match; several rivals depend on shared exclusion support.

These remain a small template family, not general random graphs. Names, premise order/orientation, quarter-turn rotation, candidate counts and branch exclusions vary. No distractors are added to inflate graph diversity. At depth 1 some patterns collapse. Match/rival chains can also share unlabelled topology while putting the difficult reasoning in different membership decisions.

| Depth | match-chain | rival-chain | shared-exclusion |
| --- | --- | --- | --- |
| 1 | 2 | 2 | 2 |
| 2 | 2 | 2 | 2 |
| 3 | 2 | 2 | 2 |
| 4 | 2 | 2 | 2 |
| 5 | 2 | 2 | 2 |
| 6 | 2 | 2 | 2 |
| 7 | 2 | 2 | 2 |
| 8 | 1 | 2 | 2 |
| 9 | 2 | 2 | 2 |
| 10 | 2 | 2 | 2 |

Counts above are retained added bases after signature deduplication, not independent construction families. There are only **6 conservative source-family IDs** across the expansion. Related depth variants are development examples and cannot become independent final held-out families.

## Token Lengths

| Style | Mean target tokens | Maximum |
| --- | --- | --- |
| natural | 323.5 | 1127 |
| symbolic | 266.1 | 812 |

Pinned Qwen3.5-4B tokenizer; lengths include the final answer, exclude prompt and audit metadata. The existing response/context limits are unchanged.

## Same Selection Policy

Combined-pool MILP result: **optimal**. Same 70 bundles per task, one bundle per base, protected observed semantic/escape/cardinality/depth categories, and within-stratum topology cap. Objective order is unchanged. No E/F percentage was imposed after inspecting the outcomes.

| Metric | Old pool selection | Expanded pool selection |
| --- | --- | --- |
| Views per style | 630 | 630 |
| Fine cells | 123 | 99 |
| Task/topology combinations | 202 | 205 |
| Source families | 53 | 59 |
| Single-base cells | 60 | 54 |

| Task/contract | Previous E/F | New E/F | New substantive |
| --- | --- | --- | --- |
| direction/single | 25/70 | 25/70 (35.7%) | 45 |
| direction/all-possible | 22/70 | 22/70 (31.4%) | 48 |
| direction/visible-possible | 4/70 | 4/70 (5.7%) | 66 |
| which/single | 50/70 | 17/70 (24.3%) | 53 |
| which/all-possible | 20/70 | 3/70 (4.3%) | 67 |
| which/visible-possible | 11/70 | 3/70 (4.3%) | 67 |
| count/single | 33/70 | 33/70 (47.1%) | 37 |
| count/all-possible | 8/70 | 8/70 (11.4%) | 62 |
| count/visible-possible | 2/70 | 2/70 (2.9%) | 68 |

| Selection Unique depth | Previous | New |
| --- | --- | --- |
| 1 | 4 | 6 |
| 2 | 1 | 3 |
| 3 | 2 | 6 |
| 4 | 2 | 4 |
| 5 | 1 | 7 |
| 6 | 2 | 6 |
| 7 | 2 | 5 |
| 8 | 2 | 6 |
| 9 | 2 | 6 |
| 10 | 2 | 4 |

Missing protected categories: 0 -> 0. Before metrics are recomputed against the combined candidate inventory; any newly available feature is therefore visible as a gap in the old subset.

| Retained Selection source | Bases |
| --- | --- |
| contingent | 1 |
| controlled-depth | 23 |
| match-chain | 7 |
| random-atoms | 5 |
| random-world | 7 |
| rival-chain | 12 |
| shared-exclusion | 14 |
| singleton | 1 |

Total E/F: 175/630 -> 117/630. This is a selection diagnostic, not a proposed production mix.

## Interpretation

The targeted supply expansion worked: Unique Selection E/F fell from 71.4% to
24.3%, while all protected categories and substantive depths 1-10 remained
represented. Overall E/F fell from 27.8% to 18.6%.

This is not an across-the-board coverage improvement. Fine-cell coverage fell
from 123 to 99 because the unchanged lexicographic selector minimizes E/F
before maximizing fine cells. Protecting individual semantic, cardinality and
depth categories does not protect every intersection of those categories.
The larger candidate pool did not remove the old selection; the optimization
policy preferred a different tradeoff.

The three new patterns occupy 33/70 selected Selection bases (47.1%):
7 match-chain, 12 rival-chain and 14 shared-exclusion. Another 23 are existing
controlled-depth constructions; only 12 are random-world or random-atom bases.
Thus improved answer balance does not establish broad construction diversity.
Production selection needs explicit coverage and concentration requirements,
not simply the lowest attainable E/F rate.

## Verification and Explanation Inspection

- Full composition/pivot regression suite: **772 passed, 1 skipped**.
- All **67 checked hashes** matched: recorded sources and inputs, expansion
  outputs, pinned tokenizer files, and old-pool outputs.
- Rebuilt the combined inventory from saved records and independently rechecked
  the 210 selected bundle IDs, exact task quotas, one-menu-per-base rule,
  within-stratum topology caps and protected coverage. Recomputed metrics match
  the saved summary after JSON normalization.
- All four HiGHS objective stages reported zero relative gap. Objective values
  were 117 E/F, 99 fine cells, 205 task/topology combinations and 52477 seeded
  rank sum. These are solver-reported numerical optima, not formal rational
  optimality certificates. Spatial semantics still use Z3.
- Inspected actual Natural/Symbolic pairs for every pattern at depths 3 and 10.
  Match-chain and rival-chain examples place the long argument in the intended
  inclusion or exclusion decision. Unlisted eligible rivals are correctly
  checked for Unique.

The shared-exclusion examples expose remaining repetition: the renderer prints
the common trunk again for each rival, and mixed equality/inequality chains are
wordy in Natural. These outputs pass the evidence checks, contain no numerical
witness maps, and fit the token limits, but they are not fully polished prose.
Factoring shared support would require a separately verified renderer change;
it was not silently introduced during this fixed-pipeline comparison.

## Limits and Next Decision

The main question is whether targeted unique-match supply improves coverage and balance without editing answers or discarding required categories. An improvement does not establish final scale, split isolation or independent-family minima. The expanded pool intentionally overrepresents these three constructions.

Unobserved Exhaustive Direction size-four and invariant Count with changing member identities remain uninvestigated by this Selection-only pass. The indirect-coincidence certificate limitation remains unchanged.

Stop for review before any further sampling. Agree production supported cells, construction/family limits and quotas only after reviewing these results. No automatic generation or sealing follows this report.

## Artifacts

- `../unique-selection-expansion-v1/manifest.json`: fixed schedule and hashes.
- `../unique-selection-expansion-v1/attempts.jsonl`: all new inputs, evidence and paired outputs.
- `../unique-selection-expansion-v1/summary.json`: duplicates, selection IDs, objective bounds and comparison.
- `UNIQUE-SELECTION-EXPANSION-EXAMPLES-v1.md`: six substantive paired examples.
