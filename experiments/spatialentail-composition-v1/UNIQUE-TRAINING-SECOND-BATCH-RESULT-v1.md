# Second Bounded Batch Result

Status: second batch completed; necessary supply bounds pass; final selection
and evidence replay are not yet complete. No dataset seal or training authorization.

## Batch Outcome

| Outcome | Fixed query slots |
|---|---:|
| Accepted and independently replayed | 585 |
| Pre-admission rejection: declared objects absent from premises | 975 |
| Held-out canonical scene overlap exclusion | 40 |
| Total | 1,600 |

No verification failures, token rejections or automatic replacement slots.
Both supplementary batch authorizations are now exhausted. No third batch
is authorized.

New relevant supply:

- Direction random-world: 11 no-entailment questions, plus 10 sole-entailment.
- Which random-world: 68 sole, 27 multiple, 279 no-entailment.
- Which random-atoms: 20 sole, 11 multiple, 56 no-entailment, 103 inconsistent.
- No Count questions were generated.

The deficient Which group gained 126 sole/multiple-entailment questions.
All accepted categories remain candidate supply; acceptance alone does not
mean a question will be selected.

## Supply Gate

The combined pool now has 15,450 training candidates. All checked necessary
answer-subset bounds pass under the unchanged source and depth/semantic stratum
quotas. This removes the previous aggregate Direction and Which obstructions.

It does not prove a feasible 6,000-question selection: family floors,
controlled Count-value quotas, coverage and selected-explanation depth evidence
must still hold jointly. No target or numerical minimum has been relaxed.

The next process reconstructs existing original candidates and replays
supplementary evidence under the short Unique protocol, verifies preserved
canonical questions/semantics/menus/measurements, then attempts one-menu
selection with all retained constraints. This is reference materialization
and verification of existing candidates, not new seed generation.

Older compact metadata is not silently treated as proof that an F explanation
meets the new selected-explanation depth predicate. Natural/Symbolic references
and token admission must be checked on reconstructed output.

## Provenance

- [Second manifest](UNIQUE-TRAINING-REPAIR-MANIFEST-v2.json).
- [Admission and combined supply receipt](UNIQUE-TRAINING-REPAIR-ADMISSION-v2.json).
- Manifest SHA-256:
  `a1450d50b539c2ea5d9b1e5c5b49ee27ca80b0164a65bc1eba608d6c5cb8d9ba`.
- Full and compact evidence paths and hashes are in the admission receipt.
- The runner verified fixed inputs before and after execution. A separate
  post-run check confirmed the manifest, full evidence and all recorded prior
  input hashes, including evaluation inputs.
- First-batch artifacts were not edited or rerun.

The second-batch guards passed seven tests together with the first runner's
guards. Reconstruction, selected-menu and selected-explanation audit tests
passed nine tests. Existing entailment and shared integration regression
results are recorded in the first repair report.

## Remaining Gate

Evidence replay and exact selection are in progress. The intended selection
receipt is `UNIQUE-TRAINING-SELECTION-REPLAY-v1.json`; a running marker or absent
receipt is not success. Require a completed validated selection before claiming
the 1,200/600/200 allocation is established.

The active goal remains incomplete. Evaluation selections, answer semantics,
source proportions, family isolation and the L1-L10 study design remain unchanged.
