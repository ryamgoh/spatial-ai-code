# SpatialMap Unique External Set

One external evaluation set under `unique-entailed`, not three contracts.
Status: generated and independently audited; unsealed and not used for training.

## Contents

| Query | Primary base questions | Omission companions | Total views per style |
|---|---:|---:|---:|
| Direction | 500 | 79 | 579 |
| Which | 500 | 66 | 566 |
| Count | 500 | 45 | 545 |
| Total | 1,500 | 190 | 1,690 |

The 1,500 questions come from 500 original scenes. Natural and Symbolic use
the same question IDs, menus, and labels. Omission companions are correlated
views of existing questions, not additional independent scenes.

## Presentation And Semantics

Prompt version: `plain-unique-system-rules-v1`.
Study scope: `unique-only-entailment-v1`.
The user prompt has the original scene sentences in one paragraph, a short
question, and A-F options. Original names, sentence order, and introductory
`is in the map` declarations are preserved; whitespace is normalized.

Examples:

- "In which direction is Planetarium Prints relative to Police Supply Store?"
- "Which object is southwest of Ice Queen Ice Cream?"
- "How many objects are southeast of Oz Oddities?"

Entailment, all-nonreference Which/Count scope, and complete E/F rules are in
the shared system message. No "guaranteed" or "Select E/F" instructions are
repeated in the question. Premises, menus, and labels are not altered by moving
those rules. Existing Natural/Symbolic reference prose is preserved.

Historical Corr SINGLE is not the current contract. Labels were freshly
computed under our exact-axis, distinct-point, open-world semantics.
No original oracle letter is used as a substitute for verification.

## Verified Distribution

| Query | Primary A-D | Primary E | Primary F | All-menu E | All-menu F |
|---|---:|---:|---:|---:|---:|
| Direction | 332 | 168 | 0 | 168 | 79 |
| Which | 244 | 256 | 0 | 277 | 45 |
| Count | 195 | 305 | 0 | 305 | 45 |

The 21 additional Which E views are omission companions of multiple-entailment
questions: hiding an entailed object does not make the full-scene answer unique.

## Checks

- 900 entailment, prompt, training-message, and evaluation regression tests passed.
- All 1,500 source paragraphs round-trip with their original names and sentences.
- Full production label, source, reference, and token checks passed.
- All 1,690 views have matched Natural/Symbolic references.
- Maximum reference target tokens: Natural 755; Symbolic 762.
- Maximum Stage-2 context reservation: 4,772 of 8,192 tokens.
- Independent audit: 1,500 bases, 500 stories, 1,690 views, 1,023 entailed
  claims, 8,477 checked counterexamples, and 3,380 style-specific model inputs.
- Comparison with the preserved `spatialmap-entailment-all-five-v3` Unique
  subset: identical 1,690 IDs, zero menu changes, zero label changes.
  This comparison is to the prior entailment adaptation, not native Corr SINGLE.

See `../EXTERNAL-UNIQUE-INDEPENDENT-AUDIT-v1.json` for the independent receipt
and `manifest.json` for source, software, tokenizer, and output hashes.

## Files

- `natural-inputs.jsonl`, `symbolic-inputs.jsonl`: model messages without golds.
- `references.jsonl`: protected labels, references and source metadata.
- `evidence.jsonl`: solver evidence retained for audit, not model input.
- `manifest.json`: versioned provenance and counts.

This set is external evaluation only. It cannot be used for SFT, dev checkpoint
selection, or to infer model transfer accuracy before experiments are run.
No dataset seal, training, or GPU evaluation was performed.
