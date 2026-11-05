#!/usr/bin/env bash
# Build the release zip from release/MANIFEST.txt; nothing outside the manifest
# is shipped. Files are staged in a temporary directory under
# pair-ci_replication/, so the zip unpacks to that folder.
# Usage: bash release/make_release.sh OUT.zip
set -euo pipefail
out="${1:?usage: make_release.sh OUT.zip}"
out=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$out")
cd "$(dirname "$0")/.."
stage=$(mktemp -d)/pair-ci_replication
mkdir -p "$stage"
python3 - "$stage" <<'PY'
import fnmatch, os, shutil, sys
stage = sys.argv[1]
inc, exc = [], []
for line in open("release/MANIFEST.txt"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    (exc if line.startswith("!") else inc).append(line.lstrip("!"))
def excluded(path):
    return any(fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, pat.rstrip("/") + "/*")
               or fnmatch.fnmatch("/" + path, "*/" + pat.replace("**/", "")) for pat in exc) \
        or any(part in ("__pycache__",) for part in path.split("/")) or path.endswith((".pyc", ".DS_Store"))
files = []
for item in inc:
    if item.endswith("/"):
        for root, _, names in os.walk(item):
            files += [os.path.join(root, n) for n in names]
    else:
        files.append(item)
n = 0
for f in sorted(set(files)):
    if excluded(f) or not os.path.isfile(f):
        continue
    dst = os.path.join(stage, f); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(f, dst); n += 1
print(f"staged {n} files")
PY
( cd "$(dirname "$stage")" && rm -f "$out" && zip -qr "$out" pair-ci_replication )
echo "wrote $out"
