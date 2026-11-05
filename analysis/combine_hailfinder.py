#!/usr/bin/env python
"""Merge per-cell HAILFINDER CSVs into results/hailfinder_results.csv.

Each (rate, mechanism) job writes `results/hailfinder_{rate}_{mechanism}.csv`;
this script concatenates them into the file that build_tables.py reads for
the HAILFINDER results (Section 5, Appendix I). Missing cells are listed
but are not an error, so partial merges work while jobs are running.

Usage:
    python analysis/combine_hailfinder.py
    python analysis/combine_hailfinder.py --results-dir results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RATES = [0.1, 0.2, 0.4]
MECHANISMS = ["complete", "mar", "mnar", "mixed"]


def cell_path(results_dir: Path, rate: float, mech: str, suffix: str) -> Path:
    return results_dir / f"hailfinder_{rate}_{mech}{suffix}.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str,
                        default="results/hailfinder_results.csv")
    parser.add_argument("--suffix", type=str, default="",
                        help="Filename suffix for per-cell inputs, e.g. "
                             "'_harm' to read hailfinder_0.1_mar_harm.csv.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.output)

    frames = []
    missing = []
    found = []
    for rate in RATES:
        for mech in MECHANISMS:
            if mech == "complete" and rate > 0.1:
                continue
            p = cell_path(results_dir, rate, mech, args.suffix)
            if p.exists():
                frames.append(pd.read_csv(p))
                found.append((rate, mech, len(frames[-1])))
            else:
                missing.append((rate, mech))

    if not frames:
        print("No per-cell CSVs found. Nothing to combine.", file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)

    print(f"Merged {len(frames)} cell(s) into {out}")
    print(f"Total rows: {len(df)}")
    for rate, mech, nrows in found:
        print(f"  {rate} {mech}: {nrows} rows")
    if missing:
        print("\nMissing cells (jobs may still be running):")
        for rate, mech in missing:
            print(f"  {rate} {mech}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
