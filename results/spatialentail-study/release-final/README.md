# Frozen SpatialEntail benchmark source

The four committed evaluation modules use explicit public names:

- `bench-core.jsonl` — standard task coverage;
- `bench-composition-matched.jsonl` — new instances from composition families
  represented in training;
- `bench-composition-heldout.jsonl` — composition families reserved from
  training; and
- `bench-depth.jsonl` — Direction, Selection, and Count at depths 3–8.

`manifest.json` preserves the sealed internal role keys `bench-comp` and
`bench-composition` for provenance compatibility. The evaluation preparer maps
those immutable keys to the unambiguous public filenames and reporting labels.
