# Temporary SpatialEntail data transfer

`spatialentail-prepared-data.tar.gz` is a temporary transport copy of the six
generated Train/Dev JSONL files. It is committed only because the generated
data directory is intentionally ignored and the Slurm server is reached through
Git.

From the repository root on the server, verify and extract it:

```sh
shasum -a 256 artifacts/transfer/spatialentail-prepared-data.tar.gz
tar -xzf artifacts/transfer/spatialentail-prepared-data.tar.gz
```

Expected SHA-256:

```text
140d1c809a328e7ec6843e4d6055725042f8d4f67d8a66e411760b9075520d52
```

After confirming that
`experiments/spatialentail-sft/prepared-final/data/` contains all six JSONL
files and their hashes pass the job preflight, remove this archive from Git in
a follow-up commit.
