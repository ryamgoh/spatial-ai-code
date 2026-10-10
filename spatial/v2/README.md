# Spatial V2

This is the active SpatialEntail implementation.

The supported certificate acceptance contract is recorded in
[`docs/spatialentail/certificates.md`](../../docs/spatialentail/certificates.md).

- `solver.py` defines the structured Boolean spatial language and semantic
  oracle.
- `text.py` parses and renders the controlled `X is DIR of Y` grammar composed
  with `NOT`, `AND`, `OR`, `IF ... THEN`, and `IFF`.
- `proofs.py` and the certificate modules define independently replayed
  evidence. Its deterministic constructor handles Boolean closure, finite
  nested case splits, axis reasoning, and explicit or spatial contradictions.
- `symbolic_trace_codec.py` defines the versioned JSON grammar for proof,
  answer-set, and menu traces and parses model output back into replay-checked
  certificates.
  The grammar contract is documented in
  [`docs/spatialentail/trace-formats.md`](../../docs/spatialentail/trace-formats.md).
- The answer-certificate renderers expose separate compact training and
  coordinate-bearing audit views from checked evidence. Natural narrates the
  checked dependencies/scopes that Symbolic encodes; rendered information and
  token costs still need auditing before claiming a notation-only ablation. Only Symbolic model output has replay scoring.
- `supervision_controls.py` derives checked-trace, answer-only, and deliberately
  corrupted Symbolic controls from one row. Corruption changes semantic
  evidence while retaining valid syntax and the gold decision, and must fail
  replay. Answer-only evidence is absent, not invalid. Context admission uses
  the selected tokenizer/template and rejects whole paired groups on overflow.
  Workload manifests report lengths and matching multipliers, not measured compute.
- `difficulty.py` measures proof depth and support for workload controls.
- `generation.py` emits training rows only after certificate checking and text
  round-trip validation. Each row records its actual `world-first`, `proof-template-first`, or
  `premise-first` proposal route. The premise-first route samples visible
  relations without coordinates and obtains witnesses only after solving.
  `certificate-first` remains a stronger, unclaimed provenance.
- `generate_all.py`, `generate_matrix.py`, and `audit_spatialeval.py` are module
  entry points.
- `tests/` contains the active semantic and generation contracts.

Run commands from the repository root, for example:

```bash
python -m spatial.v2.generate_all --help
pytest spatial/v2 -q
```

Do not import from `spatial.v1` here. V1 is retained only for reproducing older
experiments.

The runnable bounded Direction pilot is documented in
[`experiments/15-v2-ablation/README.md`](../../experiments/15-v2-ablation/README.md).
It separates train/development/test groups, supports explicit holdout cells, and
uses the admitted evaluation prompt verbatim. Which and Count API support does
not imply that all query families fit or are included in that pilot. No model
training result is established by generation smoke checks.
