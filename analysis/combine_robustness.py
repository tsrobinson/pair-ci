"""Combine and summarise the robustness grid (missingness rate x n).

Concatenates results/robustness_{mechanism}_{rate}_{n}.csv into
results/robustness_results.csv, prints median SHD, F1 and runtime by
method, and prints LaTeX tables of median SHD (robustness to missingness
rate and sample size, Appendix F). Run from the package root:
python analysis/combine_robustness.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import pandas as pd
from pathlib import Path


MISS_RATES = [0.1, 0.3, 0.5]
N_OBS_GRID = [500, 1000, 2000]
MECHANISMS = ["mar", "mnar"]


def load_and_combine():
    dfs = []
    missing = []
    for mech in MECHANISMS:
        for mr in MISS_RATES:
            for nobs in N_OBS_GRID:
                fname = f"results/robustness_{mech}_{mr}_{nobs}.csv"
                if Path(fname).exists():
                    df = pd.read_csv(fname)
                    print(f"  Loaded {fname}: {len(df)} rows")
                    dfs.append(df)
                else:
                    missing.append(fname)
                    print(f"  WARNING: {fname} not found")
    if missing:
        print(f"\n  Missing {len(missing)} of {len(MECHANISMS)*len(MISS_RATES)*len(N_OBS_GRID)} files")
    if not dfs:
        print("No results found.")
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv("results/robustness_results.csv", index=False)
    print(f"\nCombined: {len(combined)} rows -> results/robustness_results.csv")
    return combined


def summarise(df):
    df_ok = df[~df["failed"]].copy()
    if len(df_ok) == 0:
        print("No successful runs.")
        return

    method_labels = {
        "ci_oracle_cached": "Ours",
        "testwise": "Testwise",
        "impute_fisherz": "Impute-FZ",
        "complete_case": "Complete case",
        "impute_ci_oracle": "Impute-CI",
    }
    df_ok["method_label"] = df_ok["method"].map(method_labels).fillna(df_ok["method"])

    print(f"\n{'='*60}")
    print("ROBUSTNESS: Median SHD by (n_obs, miss_rate)")
    print(f"{'='*60}")
    for mech in MECHANISMS:
        print(f"\n--- {mech.upper()} ---")
        sub = df_ok[df_ok["mechanism"] == mech]
        for nobs in N_OBS_GRID:
            print(f"\n  n = {nobs}")
            s = sub[sub["n_obs"] == nobs]
            pivot = (s.groupby(["method_label", "miss_rate"])["shd"]
                     .agg(["median", lambda x: x.quantile(0.25),
                           lambda x: x.quantile(0.75)])
                     .round(1))
            pivot.columns = ["median", "Q25", "Q75"]
            for mr in MISS_RATES:
                print(f"    miss_rate = {mr}")
                try:
                    sub_mr = s[s["miss_rate"] == mr]
                    tbl = (sub_mr.groupby("method_label")["shd"]
                           .agg(["median", "mean"])
                           .round(1)
                           .sort_values("median"))
                    print(f"    {tbl.to_string()}")
                except Exception:
                    print("    (no data)")

    print(f"\n{'='*60}")
    print("ROBUSTNESS: F1 by (n_obs, miss_rate)")
    print(f"{'='*60}")
    for mech in MECHANISMS:
        print(f"\n--- {mech.upper()} ---")
        sub = df_ok[df_ok["mechanism"] == mech]
        pivot = (sub.groupby(["method_label", "n_obs", "miss_rate"])["f1"]
                 .median().unstack("miss_rate")
                 .round(3))
        print(pivot.to_string())

    print(f"\n{'='*60}")
    print("ROBUSTNESS: Runtime (seconds, median)")
    print(f"{'='*60}")
    print(df_ok.groupby(["method_label", "n_obs"])["runtime"]
          .median().unstack("n_obs").round(1).to_string())


def generate_latex_table(df):
    """Print one LaTeX table per mechanism of median SHD by miss_rate and
    n_obs. The quartiles are computed but not printed."""
    df_ok = df[~df["failed"]].copy()
    method_labels = {
        "ci_oracle_cached": "Ours",
        "testwise": "Testwise",
        "impute_fisherz": "Impute-FZ",
        "complete_case": "Complete case",
    }
    methods_order = ["ci_oracle_cached", "testwise", "impute_fisherz", "complete_case"]

    for mech in MECHANISMS:
        sub = df_ok[df_ok["mechanism"] == mech]
        print(f"\n% LaTeX table: Robustness {mech.upper()}")
        print(r"\begin{tabular}{l" + "c" * (len(MISS_RATES) * len(N_OBS_GRID)) + "}")
        print(r"\toprule")
        # Header row 1: n_obs groups
        header1 = "& " + " & ".join(
            [f"\\multicolumn{{{len(MISS_RATES)}}}{{c}}{{$n={n}$}}"
             for n in N_OBS_GRID])
        print(header1 + r" \\")
        # Header row 2: miss_rate within each n
        cmidrules = " ".join(
            [f"\\cmidrule(lr){{{2+i*len(MISS_RATES)}-{1+(i+1)*len(MISS_RATES)}}}"
             for i in range(len(N_OBS_GRID))])
        print(cmidrules)
        header2 = "Method & " + " & ".join(
            [f"{mr:.0%}" for _ in N_OBS_GRID for mr in MISS_RATES])
        print(header2 + r" \\")
        print(r"\midrule")

        for method in methods_order:
            label = method_labels.get(method, method)
            vals = []
            for nobs in N_OBS_GRID:
                for mr in MISS_RATES:
                    s = sub[(sub["method"] == method) &
                            (sub["n_obs"] == nobs) &
                            (sub["miss_rate"] == mr)]["shd"]
                    if len(s) > 0:
                        med = s.median()
                        q25 = s.quantile(0.25)
                        q75 = s.quantile(0.75)
                        vals.append(f"{med:.0f}")
                    else:
                        vals.append("---")
            print(f"{label} & " + " & ".join(vals) + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    print("=== Combining Robustness Results ===\n")
    df = load_and_combine()
    if df is not None:
        summarise(df)
        generate_latex_table(df)
