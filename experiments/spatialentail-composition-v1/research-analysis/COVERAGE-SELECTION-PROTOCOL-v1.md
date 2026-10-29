# Coverage-Protected Selection Check

Existing `balance-feasibility-v1` pool only. No sampling, re-rendering, production
export, split assignment or sealing. Preserve all original artifacts.

## Fixed Policy

- Select exactly 70 complete menu bundles per task, matching the prior diagnostic.
- Select at most one bundle per base; retain all three contracts and both styles.
- Require at least one example for every observed task/contract/semantic-case,
  task/contract/returned-cardinality, task/contract/escape-reason, and
  task/contract/Count-zero-presence category.
- Require at least one substantive depth-supported view for every observed
  task/contract/depth, including development depths 9/10. This is an availability
  check, not a train/test assignment.
- Allow at most one selected base per topology within a stratum defined by task,
  semantic case, base depth, complete possible-set size and Count-zero presence.
  Different semantics or depths can therefore share a topology. This is still a
  diagnostic cap, not a production independence guarantee.
- Subject to these hard constraints, lexicographically minimize total E/F views,
  maximize covered fine-grained matrix cells, minimize repeated topology use
  across strata, then prefer lower seeded bundle ranks.
- Use Z3 Optimize with a 120-second limit. Unknown, infeasibility or an unproven
  objective bound must be reported, never silently relaxed.

The hard minima are one for a bounded availability check, not sufficient
production quotas. Preserve observed categories, not every literal permutation
or every full matrix intersection. Report losses in the latter explicitly.
Compare selected counts, depth retention, semantic coverage, E/F, and diversity
with the original 70-per-task diagnostic. Also report the pool's maximum
available substantive Unique bases; selection cannot create missing examples.

Verify source artifact hashes before and after selection. Recompute selection
constraints independently in Python after solving. Save IDs and diagnostics,
not training examples. Stop for review with specific remaining shortages.
