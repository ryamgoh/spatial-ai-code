# Second Bounded Supply Proposal

Status: approved for one fixed second batch. The user chose **Approve second
batch**: at most 1,600 slots with the source/query counts below, followed by
supply and selection checks. Answer targets and source/depth requirements stay
unchanged. The first 2,048-slot authorization remains exhausted.

## Why The Simpler Task Still Has A Shortage

Unique-only removed two answer contracts and their explanations. The new
one-menu design also removes duplicate menu exposure from training. Neither
change guarantees a specified joint distribution over answer reasons, source
types, proof depths and families.

The solver did not fail: the first supplementary batch had zero verification
failures. Its 815 accepted questions were not all in the scarce categories.
In particular, 640 Which slots yielded 27 sole-entailment and 14 multiple-
entailment questions, only 41 usable questions for the deficient combined group.
The earlier budget was an uncalibrated resource cap, not a measured-yield
success guarantee. This proposal uses that completed batch's observed yields.

Keep correctness/evidence checks separate from design quotas. Exact source
percentages are a study-design choice, not a theorem needed for entailment.
The user has chosen to preserve them; this proposal does not silently change
them or add further balancing requirements.

## Remaining Needs

- Direction: at least 3 additional random-world no-entailment questions to
  reach its 300-slot random-world E requirement.
- Which: at least 49 additional sole/multiple-entailment questions in random
  sources to clear the currently binding source-capped combined-group bound.
- Count: no supplementary slots; its checked aggregate supply conditions pass.

These are necessary lower bounds, not sufficient budgets for all joint gates.
An accepted question in a different category remains valid but does not
necessarily help the deficient group.

## Proposed Fixed Cap

| Query | Source | Attempts | Relevant accepted outcomes in previous batch | Previous yield |
|---|---|---:|---|---:|
| Direction | Random-world | 64 | No entailment: 129 / 768 | 16.80% |
| Which | Random-world | 1,024 | Sole or multiple entailment: 29 / 384 | 7.55% |
| Which | Random-atoms | 512 | Sole or multiple entailment: 12 / 256 | 4.69% |
| Total | | 1,600 | | |

Straight-line extrapolation gives about 11 relevant Direction questions and
101 relevant Which questions (about 77 random-world and 24 random-atoms).
The previous Which breakdown projects about 68 sole and 33 multiple outcomes.
These are planning estimates, not guaranteed counts, confidence bounds or
independent-trial probability estimates. New overlaps and evidence exclusions
can lower the yield.

The rates include previously rejected slots, including missing-object and
held-out overlap exclusions; do not multiply by an additional admission rate.
The final retained source proportions remain 70/15/10/5. The attempt proportions
above are not new dataset proportions.

## Execution Boundaries

After approval, write one immutable second-batch manifest with exact new seed
ranges, source/query slots, source-code and input fingerprints. Verify seed
disjointness against both the original and first supplementary inventories.
Use a new batch identity; never overwrite or resume the exhausted first batch.

Keep the existing random samplers and all-scene admission rules. Do not add
premises to mention missing objects, delete declared objects, relabel constructed
cases as random, or retry a seed until its answer has the desired category.
Unmentioned-object and held-out-overlap cases are rejected and consume slots.
Do not change the seeded omission-menu eligibility to manufacture F supply.

Generate only the declared Direction/Which query slots. Use all verification,
token and independent replay gates. Quarantine conflicts with existing split
family assignments, canonical scenes and the external benchmark; no split moves.
Record every attempted slot, outcome and rejection. Accepted off-target semantic
cases may enter the candidate inventory, but receive no false shortage credit.

After all 1,600 slots, stop generation and recheck combined supply bounds. If
they fail, report the failure; no third batch or automatic refill is authorized.
If they pass, replay the selected-explanation evidence needed from preserved
candidates and run joint one-menu selection. Historical compact token/proof
metadata alone cannot certify the new explanation-depth predicate or short
prompt export.

Require exact 6,000 distinct questions, 1,200/600/200 per query, approved E
subtypes, and all retained source, controlled depth, Count-value, family,
negative-evidence and selected-explanation depth gates. Passing aggregate supply
does not replace validating an actual selected set.

Evaluation selections remain unchanged. No sealing or training is authorized.

## Approval Recorded

The user approved only this second fixed batch of at most 1,600 candidate query
attempts, followed by the stated supply and selection checks. No new criteria,
quota relaxations, sampler changes or unbounded generation are authorized.

Evidence: [first batch result](UNIQUE-TRAINING-REPAIR-RESULT-v1.md),
[post-batch supply](UNIQUE-TRAINING-REPAIR-SUPPLY-v1.json), and
[source-proportion diagnostic](UNIQUE-SOURCE-PROPORTION-DIAGNOSTIC-v1.md).
