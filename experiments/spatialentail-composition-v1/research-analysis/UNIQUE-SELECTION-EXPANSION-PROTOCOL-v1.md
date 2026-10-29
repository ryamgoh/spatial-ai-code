# Bounded Unique Selection Expansion

Development only. No production data, held-out benchmark, seal or training.
Preserve the existing 240-base pool and selection results.

## Fixed Schedule

Seed 20261031; 60 new Selection base attempts. Three patterns, depths 1-10,
two variants per pattern/depth. No replacement, seed search or retry-to-quota.

1. **Match chain:** the sole qualifying candidate has a depth-d northward
   chain; other candidates have direct east/west/south exclusions.
2. **Rival chain:** the sole match is direct; one excluded rival has a
   depth-d southward chain. Other rivals have direct exclusions.
3. **Shared exclusion trunk:** the sole match is direct; a southward trunk
   of length d-1 supports branches to all rivals. Branch relations vary
   between south/east/west when the trunk is nonempty.

Each candidate set has 5-7 objects, including exactly one guaranteed match.
Randomize candidate count, irrelevant-rival exclusion directions, quarter-turn
rotation, statement orientation, names and premise order. No disconnected or
attached distractors are added to inflate topology variety. This is a small
pattern library, not unrestricted random graph generation.

Depth 1 necessarily collapses some patterns; canonical collisions are reported
and excluded from independent supply gains. Pattern/sample family IDs group
related depths conservatively. None of these development families or renamed
copies may be used as final held-out benchmark families.

## Admission

Use the existing full-semantic/certificate/qualitative-renderer pipeline,
5000 ms solver-call timeout, pinned Qwen tokenizer and existing token limits.
Require consistent semantics with possible=certain={target}. Require measured
fixed-membership depth equal requested depth and the existing premise-ablation
answer-dependence check. Verify matched texts, prompt round trips, renaming and
printed evidence. Keep failures and their stage.

Use the existing bounded menu inventory, not a new menu distribution. For each
unique answer this includes covered and omitted menus, all three contracts and
both styles. Omission F variants are diagnostic and do not fill substantive
depth slots.

## Comparison

Report independent query signatures, collision counts against the old pool
and within the expansion, pattern and family counts, accepted depths, token
lengths, and literal unique-match supply before/after. Query signatures normalize
name/order/inverse statements but not arbitrary rotations.

Re-run the unchanged coverage-protected MILP selector on the combined pool at
70 bundles per task. Use the same constraints, objective order, and 120 seconds
per objective as the previous selector. Exclude duplicate query signatures
from added supply, with deterministic first-by-base-ID retention.
The optimizer may change other task selections to preserve complete coverage;
do not label a better E/F rate as overall production readiness.

Save additions and comparison separately. No old examples are rewritten.
Inspect representative explanations from each pattern and stop for feedback.
Unobserved Direction/Count cells and the inherited certificate boundary remain
separate unresolved items; this pass does not claim to fix them.
