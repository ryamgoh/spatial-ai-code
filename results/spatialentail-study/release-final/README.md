# Frozen SpatialEntail benchmark source

The committed benchmark uses three formal groups:

- **CORE** (1,300): `bench-core.jsonl`;
- **COMP** (1,200): `bench-composition.jsonl`; and
- **SHIFT** (1,800): `bench-shift.jsonl`, with held-out composition (900) and
  depth ladder (900) distinguished by row metadata for diagnostic reporting.

`manifest.json` preserves the sealed internal role keys `bench-comp`,
`bench-composition`, and `bench-depth` for provenance compatibility. The
evaluation preparer maps those immutable keys to the public CORE/COMP/SHIFT
hierarchy.
