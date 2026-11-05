#!/usr/bin/env bash
# Download the Sachs et al. (2005) flow-cytometry data (853 x 11) from the
# bnlearn repository, check it against the file used in the paper (shape,
# column names, sum of all values) and write data/sachs_data.csv.
# Needs curl and a python3 with pandas (e.g. the activated causal_ci env).
set -euo pipefail
cd "$(dirname "$0")"
URL="https://www.bnlearn.com/research/sachs05/sachs.data.txt"
tmp=$(mktemp)
curl -sSfL "$URL" -o "$tmp"
python3 - "$tmp" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], sep=r"\s+")
cols = ["Raf","Mek","Plcg","PIP2","PIP3","Erk","Akt","PKA","PKC","P38","Jnk"]
assert d.shape == (853, 11), d.shape
assert list(d.columns) == cols, list(d.columns)
total = round(float(d.values.sum()), 6)
EXPECTED_SUM = 801429.24
assert abs(total - EXPECTED_SUM) < 1e-6, (total, EXPECTED_SUM)
d.to_csv("sachs_data.csv", index=False)
print(f"sachs_data.csv written: 853 x 11, value checksum {total} matches the paper's file.")
PY
rm -f "$tmp"
