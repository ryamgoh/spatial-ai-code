# SpatialEntail consensus-rejection human audit

Status: planned execution protocol, 2026-10-10. No review can begin until real,
formally validated items and a frozen quality-review packet exist.

## Purpose

The human is an appeal reviewer for items rejected independently by both model
judges, not a third primary judge over the complete sample. This minimizes
human involvement while checking whether unanimous automated rejection is a
false positive. It does not estimate overall human acceptability, false
acceptance rates, or population-level LLM–human agreement.

Formal semantics, certificate validity, replay, round trip, split integrity,
menu validity, and context admission remain checker-authoritative. The quality
review concerns prompt clarity and the checked Natural trace.

## Start gate and frozen packet

Freeze a sample of at least the larger of 2% of the formally admitted maximum
candidate pool or `5C` examples, where `C` is the number of frozen reporting
cells and every cell contributes at least five items. Every sampled item must already pass the
automatic checks.

The immutable packet records:

- packet, rubric, and judge-prompt versions;
- pool and benchmark-registry fingerprints;
- sampling seed, cell coverage, and randomized order;
- one opaque `review_item_id` and payload hash per item; and
- a sealed crosswalk to base ID, cell, query, semantic status, provenance, and
  audit metadata.

Each visible payload contains only the opaque ID, prompt, and checked Natural
trace. It excludes the source route, real sample ID, cell/capability tags,
Symbolic trace, coordinates, solver state, expected hypothesis, and answer
metadata not already present in the task.

## Sequential model screening

Run the two terminal-agent judges sequentially over the same immutable packet:

1. `DeepSeek-V4-Pro`;
2. `GPT-5.5`.

Each run uses the `trae` provider, starts from a fresh isolated context, and
receives the identical frozen instructions and one item at a time. The second
judge never sees the first judge's output. Pin provider/model identifiers,
system-prompt version, rubric, output schema, execution order, and software
revision. Record any generation control unavailable through the terminal-agent
interface as unavailable rather than inventing a value.

Each judge returns `approve` or `reject`, criterion-level scores, and fixed
reason codes for a rejection. Store the two streams separately and join them by
opaque ID only after both are locked.

## Routing rule

| Judge pair | Automated disposition | Human action |
|---|---|---|
| approve / approve | Accept for the reviewed quality gate. | None. |
| approve / reject, in either order | Hold out and revise or replace; record model disagreement. | None. |
| reject / reject | Pending human appeal review. | Present one blinded Approve/Reject menu. |

The split-verdict rule is deliberately conservative. It avoids accepting a
disputed item and respects the requirement that the human is contacted only
after unanimous rejection. A revised or replacement item enters a new packet
version and is screened again by both model judges.

## Human appeal menu

Present only unanimous-rejection items, exactly one at a time, in a separate
terminal thread. The interaction shows:

1. opaque review ID and progress among unanimous rejections;
2. the frozen anonymized prompt;
3. the frozen checked Natural trace; and
4. an interactive **Approve** / **Reject** menu.

The human necessarily knows that the item reached the appeal queue, but does
not see either model identity, verdict explanation, reason codes, confidence,
or other items. This is payload-blinded adjudication, not an independent
source-blinded third rating.

`Approve` means the unanimous model rejection is judged a false positive and
the item is restored with disposition `human_override_accept`. `Reject` confirms
that the item should be removed or revised. A rejection requires one or more
fixed reasons:

| Code | Human-facing reason |
|---|---|
| `PROMPT_UNCLEAR` | Prompt requires unstated information or is hard to understand. |
| `CONTRACT_UNCLEAR` | Requested answer cardinality or menu policy is unclear. |
| `TRACE_UNREADABLE` | Explanation is difficult to follow or exposes implementation internals. |
| `NOT_SELF_CONTAINED` | Trace omits evidence needed for its conclusion. |
| `REDUNDANT` | Repetition is excessive and not semantically necessary. |
| `ANSWER_LEAKAGE` | The answer appears before the evidence intended to justify it. |
| `OTHER_UNACCEPTABLE` | Another human-facing defect makes the item unsuitable. |

An optional note is capped at 240 characters and never replaces the fixed
reason. The reviewer may stop and resume between items. No decision is
overwritten: corrections append a superseding event.

## Records and reporting

Each model or human event records packet ID/fingerprint, rubric and instruction
versions, opaque item ID, payload hash, disposition or verdict, reason codes,
and timestamps. Human resume is allowed only when packet fingerprint,
instructions, and reviewer identity match.

Report:

- double-approve, split, and double-reject counts and rates;
- agreement and Cohen's kappa between the two model judges;
- criterion-score association between the two model judges where defined;
- human approval/rejection counts among double-rejects;
- the human override rate among double-rejects;
- final accepted, held-out, revised, and rejected counts by reporting cell; and
- every re-audit packet version.

Do not report Fleiss' kappa, overall human acceptability, LLM–human kappa, or a
human false-acceptance rate. The human sample is selected by the model verdicts,
so those quantities are not identifiable. If later work needs them, it requires
a separately sampled human-calibration study.

Release requires 100% automatic formal validity. Within the quality sample, an
item passes only through double approval or human override of double rejection;
a split verdict is held out. The final acceptance threshold and per-cell
revision trigger must be frozen with the benchmark-cell registry before the
review begins.
