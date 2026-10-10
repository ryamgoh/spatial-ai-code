# SpatialEntail Natural-quality judge prompt v1

Review each item independently. The input contains only an opaque item ID, a
spatial-reasoning prompt, a formally checked Natural trace, and payload hashes.
Do not inspect repository metadata, a crosswalk, another judge's output, or
expected experimental results.

This is a presentation-quality review. Do not redo or override formal theorem
checking. Judge whether the supplied problem and explanation are suitable for
the declared research and supervised-training use.

For every item, return all seven Boolean criteria:

- `prompt_clear`: the task is understandable without unstated information;
- `contract_clear`: the requested answer cardinality and menu policy are clear;
- `trace_readable`: the reasoning can be followed without decoding internal
  implementation details;
- `self_contained`: the trace states the evidence needed for its conclusion;
- `nonredundant`: repetition is limited to what the reasoning requires;
- `no_answer_leakage`: the answer is not asserted before its intended evidence;
- `acceptable`: the complete item is suitable for the declared use.

Return `approve` only when all seven criteria are true. Otherwise return
`reject`, set `acceptable` to false, and include one or more of these reason
codes:

- `PROMPT_UNCLEAR`
- `CONTRACT_UNCLEAR`
- `TRACE_UNREADABLE`
- `NOT_SELF_CONTAINED`
- `REDUNDANT`
- `ANSWER_LEAKAGE`
- `OTHER_UNACCEPTABLE`

Write one JSON object with this exact shape:

```json
{
  "schema": "spatialentail-judge-run-v1",
  "provider": "trae",
  "model": "MODEL_NAME",
  "run_id": "RUN_ID",
  "packet_fingerprint": "PACKET_FINGERPRINT",
  "judge_prompt_version": "quality-judge-v1",
  "sequence_position": 1,
  "isolated_context": true,
  "decisions": [
    {
      "review_item_id": "R00001",
      "payload_hash": "ITEM_PAYLOAD_HASH",
      "verdict": "approve",
      "reason_codes": [],
      "criteria": {
        "prompt_clear": true,
        "contract_clear": true,
        "trace_readable": true,
        "self_contained": true,
        "nonredundant": true,
        "no_answer_leakage": true,
        "acceptable": true
      }
    }
  ]
}
```

Preserve every item ID and payload hash exactly and emit exactly one decision
per packet item. Use `DeepSeek-V4-Pro` with sequence position 1 for the first
run and `GPT-5.5` with sequence position 2 for the second. Do not include prose
outside the JSON artifact.
