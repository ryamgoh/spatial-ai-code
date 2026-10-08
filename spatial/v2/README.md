# Spatial V2

This is the active SpatialEntail implementation.

- `solver.py` defines the structured Boolean spatial language and semantic
  oracle.
- `text.py` parses and renders the controlled `X is DIR of Y` grammar composed
  with `NOT`, `AND`, `OR`, `IF ... THEN`, and `IFF`.
- `proofs.py` and the certificate modules define independently replayed
  evidence. Its deterministic constructor handles Boolean closure, one-level
  case splits, axis reasoning, and explicit or spatial contradictions.
- `difficulty.py` measures proof depth and support for workload controls.
- `audit_evidence.py` and `audit_rendering.py` produce diagnostic reports; they
  are not training explanation modules.
- `generation.py` emits training rows only after certificate checking and text
  round-trip validation.
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
