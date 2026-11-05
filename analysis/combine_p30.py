"""Combine and summarise the p = 30 graph-recovery results.

Concatenates results/p30_{dgp}_{mechanism}.csv into
results/p30_full_results.csv, prints median SHD and F1 by method and
mechanism, and prints p = 10/20/30 scaling tables of SHD and runtime
(scaling experiments, Section 5). Needs the combined p = 10 and p = 20
files for the scaling tables. Run from the package root:
python analysis/combine_p30.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import pandas as pd
from pathlib import Path


def load_and_combine():
    dfs = []
    missing = []
    for dgp in ["linear", "nonlinear"]:
        for mech in ["mar", "mnar", "mixed", "complete"]:
            fname = f"results/p30_{dgp}_{mech}.csv"
            if Path(fname).exists():
                df = pd.read_csv(fname)
                print(f"  Loaded {fname}: {len(df)} rows")
                dfs.append(df)
            else:
                missing.append(fname)
                print(f"  WARNING: {fname} not found")
    if missing:
        print(f"\n  Missing {len(missing)} of 8 files: {missing}")
    if not dfs:
        print("No results found.")
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv("results/p30_full_results.csv", index=False)
    print(f"\nCombined: {len(combined)} rows -> results/p30_full_results.csv")
    return combined


def summarise(df):
    df_ok = df[~df["failed"]].copy()
    if len(df_ok) == 0:
        print("No successful runs.")
        return
    for dgp in ["linear", "nonlinear"]:
        print(f"\n{'='*60}")
        print(f"p=30 Graph Recovery -- {dgp.upper()} edges (median SHD)")
        print(f"{'='*60}")
        sub = df_ok[df_ok["dgp"] == dgp]
        if len(sub) == 0:
            print("  No data")
            continue
        pivot = (sub.groupby(["method", "mechanism"])["shd"]
                 .median().unstack("mechanism")
                 .reindex(columns=["complete", "mar", "mnar", "mixed"]))
        pivot["overall"] = pivot.median(axis=1)
        pivot = pivot.sort_values("overall")
        print(pivot.round(1).to_string())

        print(f"\nMedian F1:")
        print(sub.groupby(["method", "mechanism"])["f1"]
              .median().unstack("mechanism")
              .reindex(columns=["complete", "mar", "mnar", "mixed"])
              .round(3).to_string())

    print(f"\n{'='*60}")
    print("Runtime per PC run (seconds, median)")
    print(f"{'='*60}")
    print(df_ok.groupby(["dgp", "method"])["runtime"]
          .median().unstack("method").round(1).to_string())


def scaling_comparison():
    files = {
        10: ("results/p10_linear_results.csv", "results/p10_nonlinear_results.csv"),
        20: ("results/p20_full_results.csv", None),
        30: ("results/p30_full_results.csv", None),
    }
    all_dfs = []
    for p_val, (f1, f2) in files.items():
        if p_val == 10:
            if Path(f1).exists():
                df = pd.read_csv(f1)
                df["p"] = 10
                df["dgp"] = "linear"
                all_dfs.append(df)
            if f2 and Path(f2).exists():
                df = pd.read_csv(f2)
                df["p"] = 10
                df["dgp"] = "nonlinear"
                all_dfs.append(df)
        else:
            if Path(f1).exists():
                df = pd.read_csv(f1)
                if "p" not in df.columns:
                    df["p"] = p_val
                all_dfs.append(df)

    if not all_dfs:
        print("No data for scaling comparison.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_ok = combined[~combined["failed"]].copy()

    method_map = {
        "ci_oracle_cached": "ours", "ci_oracle_fast": "ours",
        "testwise": "testwise", "impute_fisherz": "impute_fisherz",
        "complete_case": "complete_case",
    }
    combined_ok["method_clean"] = (combined_ok["method"]
                                   .map(method_map)
                                   .fillna(combined_ok["method"]))

    print(f"\n{'='*60}")
    print("SCALING COMPARISON: p=10 vs p=20 vs p=30 (median SHD)")
    print(f"{'='*60}")

    for dgp in ["linear", "nonlinear"]:
        print(f"\n--- {dgp.upper()} ---")
        sub = combined_ok[combined_ok["dgp"] == dgp]
        pivot = (sub.groupby(["method_clean", "p"])["shd"]
                 .median().unstack("p"))
        print(pivot.round(1).to_string())

    labels = {"ours": "Ours", "testwise": "Testwise",
              "impute_fisherz": "Impute-FZ",
              "complete_case": "Complete case"}
    methods_order = ["ours", "testwise", "impute_fisherz", "complete_case"]

    # SHD scaling table
    for dgp in ["linear", "nonlinear"]:
        sub = combined_ok[combined_ok["dgp"] == dgp]
        pivot = (sub.groupby(["method_clean", "p"])["shd"]
                 .median().unstack("p"))
        print(f"\n% LaTeX: scaling table ({dgp})")
        print(r"\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
        print(r"\toprule")
        header = "Method & " + " & ".join([f"$p={c}$" for c in pivot.columns]) + r" \\"
        print(header)
        print(r"\midrule")
        for method in methods_order:
            if method in pivot.index:
                row = pivot.loc[method]
                vals = " & ".join([f"{v:.0f}" if not pd.isna(v) else "---"
                                   for v in row])
                print(f"{labels.get(method, method)} & {vals} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

    # F1 scaling table
    print(f"\n{'='*60}")
    print("SCALING COMPARISON: F1 (median)")
    print(f"{'='*60}")
    for dgp in ["linear", "nonlinear"]:
        sub = combined_ok[combined_ok["dgp"] == dgp]
        pivot = (sub.groupby(["method_clean", "p"])["f1"]
                 .median().unstack("p"))
        print(f"\n--- {dgp.upper()} F1 ---")
        print(pivot.round(3).to_string())

    # Runtime scaling table
    print(f"\n{'='*60}")
    print("SCALING COMPARISON: Runtime (seconds, median)")
    print(f"{'='*60}")
    for dgp in ["linear", "nonlinear"]:
        sub = combined_ok[combined_ok["dgp"] == dgp]
        pivot = (sub.groupby(["method_clean", "p"])["runtime"]
                 .median().unstack("p"))
        print(f"\n--- {dgp.upper()} Runtime ---")
        print(pivot.round(1).to_string())

        print(f"\n% LaTeX: runtime table ({dgp})")
        print(r"\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
        print(r"\toprule")
        header = "Method & " + " & ".join([f"$p={c}$" for c in pivot.columns]) + r" \\"
        print(header)
        print(r"\midrule")
        for method in methods_order:
            if method in pivot.index:
                row = pivot.loc[method]
                vals = " & ".join([f"{v:.1f}" if not pd.isna(v) else "---"
                                   for v in row])
                print(f"{labels.get(method, method)} & {vals} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    print("=== Combining p=30 Results ===\n")
    df = load_and_combine()
    if df is not None:
        summarise(df)
        scaling_comparison()
