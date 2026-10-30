# Question-Led Explanation Organization

## Scope

Development-only presentation refinement authorized after inspection of
SHARED-PROOF-EXAMPLES-v4.md. No production sampling, dataset sealing, training,
or change to answer semantics or depth definitions.

## Changes

- Begin with the question's task, without asserting the answer in advance.
- Within uninterrupted unconditional membership evidence, present entailed
  candidates before impossible candidates. Keep contingent cases and temporary
  assumptions in their original scopes.
- Group exclusions only when the selected paths establish the same conflicting
  axis relation to the reference. Retain every candidate's evidence.
- Print consecutive branches with identical shared support together, with
  explicit candidate-to-premise correspondence. Independently expand their
  citations into the original paths.
- Use horizontal position, height, left/right and above/below in Natural path
  prose. Symbolic retains explicit axis comparisons and identical grouping.
- Preserve version-3 and version-4 rendering for historical records.

These are answer justifications, not simulated discovery transcripts, evidence
that a model formed a complete internal map, or proof of human-like fluency.
All scene premises remain in the question, including irrelevant information.
Neither style is required to narrate every premise.

## Fixed Verification

Replay all 300 saved bases from shared-proof-replay-v4 (1,719 pairs), without
resampling. Check source hashes, rebuild spatial evidence, and preserve prompts,
menus, gold, semantic labels, family metadata and depth support.

Compare exact evidence-unit multisets before and after organization; only
ordering and presentation grouping may change. Preserve qualitative scenarios
and audit witnesses. The independent printed-text auditor checks grouped branch
citations, expanded paths, grouped exclusion signs and candidate membership,
as well as existing scenario, assumption and cross-style checks.

Use the pinned Qwen3.5-4B tokenizer and existing response/context admission
limits. Report all failures without replacements. Inspect all three expansion
patterns at depths 3 and 10 plus Direction and Count examples.

Run composition and pivot regressions, including deliberate corruption of
grouped candidates, relations, citations and missing evidence. Stop for user
review of actual paired outputs before production generation.

## Command

```sh
python -m spatial.v2.composition_refinement \
  --task question-led \
  --output experiments/spatialentail-composition-v1/shared-proof-replay-v5 \
  --tokenizer /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```
