# Temporary corrected SpatialEntail data transfer

`spatialentail-prepared-data-v2.tar.gz` contains the six schema-uniform
Train/Dev JSONL files for the prepared SFT jobs. It temporarily travels through
Git because the generated data directory is ignored.

From the repository root on the server:

```sh
shasum -a 256 artifacts/transfer/spatialentail-prepared-data-v2.tar.gz
tar -xzf artifacts/transfer/spatialentail-prepared-data-v2.tar.gz
```

Expected SHA-256:

```text
b6bde6bf29b1c3db467804c10bcd91e1ce4a5323397916da3adbf0da66352cd7
```

This extraction replaces the earlier metadata-rich files that caused the
PyArrow schema-cast failure. Remove the archive from Git after the server copy
is confirmed.
