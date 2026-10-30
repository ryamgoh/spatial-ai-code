# Direction Coverage: Depth-Visible Menu Correction

Preserve the completed version-1 run and all its failure records. Its spatial,
evidence, text and token checks passed, but some rival-chain cases failed the
additional requirement for substantive printed-depth support under every
contract. When the Listed menu omits the deep rival, the reference correctly
does not use that rival's proof. Such a view must not count toward that depth.

## Correction

Use exactly the same 240 saved scenes, questions, depths and family IDs.
No resampling, new constructions, weaker checks or quota changes.

For records failing the version-1 check, rerun full admission with this
deterministic menu rule:

- Start with the original menu inventory and original seed.
- If a menu includes the sole match, but no rival attaining the measured maximum
  depth, replace its first nonmatching option with the lexically first such rival.
- Leave match-omitting menus unchanged. Do not change the winning option's
  position, possible-value coverage, menu size or number of menus.
- Preserve all three contracts and both explanation styles.

Recheck consistency, full semantics, certificates, transformations, measured
and printed depths, premise ablation, prose audits and actual tokenizer limits.
Reuse previously admitted records byte-equivalently at the JSON-object level.
Verify source output hashes and exact serialized-problem equality for all
240 records. Save version 2 separately, recording rebuilt and inherited counts.
No further retry loop is authorized by this protocol.

Test the actual previously failing Southwest depth-3 rival-chain case with
both menu policies. Assert that the old view correctly fails the printed-depth
requirement, the corrected view passes, and the scene, semantics, menu coverage
and gold letters remain unchanged.

```sh
python -m spatial.v2.composition_direction_coverage \
  --source experiments/spatialentail-composition-v1/direction-coverage-v1 \
  --output experiments/spatialentail-composition-v1/direction-coverage-v2 \
  --tokenizer /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```
