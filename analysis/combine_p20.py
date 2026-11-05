"""Combine and summarise the p = 20 graph-recovery results.

Concatenates results/p20_{dgp}_{mechanism}.csv into
results/p20_full_results.csv, prints median SHD, F1 and runtime by method
and mechanism, compares with p = 10, and prints LaTeX tables (scaling
experiments, Section 5). Run from the package root:
python analysis/combine_p20.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def load_and_combine():
    dfs = []
    missing = []
    for dgp in ["linear", "nonlinear"]:
        for mech in ["mar", "mnar", "mixed", "complete"]:
            fname = f"results/p20_{dgp}_{mech}.csv"
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
    combined.to_csv("results/p20_full_results.csv", index=False)
    print(f"\nCombined: {len(combined)} rows -> results/p20_full_results.csv")
    return combined


def summarise(df):
    df_ok = df[~df["failed"]].copy()
    if len(df_ok) == 0:
        print("No successful runs.")
        return
    for dgp in ["linear", "nonlinear"]:
        print(f"\n{'='*60}")
        print(f"p=20 Graph Recovery -- {dgp.upper()} edges (median SHD)")
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


def compare_with_p10(df_p20, p10_linear_path, p10_nonlinear_path):
    if not Path(p10_linear_path).exists() or not Path(p10_nonlinear_path).exists():
        print("\nSkipping p=10 comparison (files not found)")
        return
    df_lin10 = pd.read_csv(p10_linear_path)
    df_nonlin10 = pd.read_csv(p10_nonlinear_path)
    df_lin10["p"] = 10
    df_lin10["dgp"] = "linear"
    df_nonlin10["p"] = 10
    df_nonlin10["dgp"] = "nonlinear"
    df_p10 = pd.concat([df_lin10, df_nonlin10], ignore_index=True)
    df_p20_copy = df_p20.copy()
    df_p20_copy["p"] = 20
    combined = pd.concat([df_p10, df_p20_copy], ignore_index=True)
    combined_ok = combined[~combined["failed"]]

    print(f"\n{'='*60}")
    print("p=10 vs p=20 Comparison (median SHD)")
    print(f"{'='*60}")
    for dgp in ["linear", "nonlinear"]:
        print(f"\n--- {dgp.upper()} ---")
        sub = combined_ok[combined_ok["dgp"] == dgp].copy()
        method_map = {
            "ci_oracle_cached": "ours", "ci_oracle_fast": "ours",
            "testwise": "testwise", "impute_fisherz": "impute_fisherz",
            "complete_case": "complete_case",
        }
        sub["method_clean"] = sub["method"].map(method_map).fillna(sub["method"])
        pivot = (sub.groupby(["method_clean", "p"])["shd"]
                 .median().unstack("p"))
        print(pivot.round(1).to_string())


def generate_latex_table(df):
    df_ok = df[~df["failed"]].copy()
    method_labels = {
        "ci_oracle_fast": "Ours (FastCI)", "testwise": "Testwise deletion",
        "impute_fisherz": "Impute + Fisher-Z", "complete_case": "Complete case",
    }
    methods_order = ["ci_oracle_fast", "testwise", "impute_fisherz", "complete_case"]
    mechs = ["complete", "mar", "mnar", "mixed"]

    for dgp in ["linear", "nonlinear"]:
        sub = df_ok[df_ok["dgp"] == dgp]
        if len(sub) == 0:
            continue

        # SHD table
        pivot = (sub.groupby(["method", "mechanism"])["shd"]
                 .median().unstack("mechanism").reindex(columns=mechs))
        print(f"\n% LaTeX table: p=20 {dgp} (SHD)")
        print(r"\begin{tabular}{lcccc}")
        print(r"\toprule")
        print(r"Method & Complete & MAR & MNAR & Mixed \\")
        print(r"\midrule")
        for method in methods_order:
            if method in pivot.index:
                row = pivot.loc[method]
                label = method_labels.get(method, method)
                vals = " & ".join([f"{v:.0f}" if not pd.isna(v) else "---"
                                   for v in row])
                print(f"{label} & {vals} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

        # SHD with IQR
        print(f"\n% LaTeX table: p=20 {dgp} (SHD with IQR)")
        print(r"\begin{tabular}{lcccc}")
        print(r"\toprule")
        print(r"Method & Complete & MAR & MNAR & Mixed \\")
        print(r"\midrule")
        for method in methods_order:
            label = method_labels.get(method, method)
            cells = []
            for mech in mechs:
                v = sub[(sub["method"] == method) & (sub["mechanism"] == mech)]["shd"]
                if len(v) > 0:
                    cells.append(f"{v.median():.0f} ({v.quantile(0.25):.0f}--{v.quantile(0.75):.0f})")
                else:
                    cells.append("---")
            print(f"{label} & " + " & ".join(cells) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

        # F1 table
        print(f"\n% LaTeX table: p=20 {dgp} (F1)")
        for method in methods_order:
            label = method_labels.get(method, method)
            cells = []
            for mech in mechs:
                v = sub[(sub["method"] == method) & (sub["mechanism"] == mech)]["f1"]
                if len(v) > 0:
                    cells.append(f"{v.median():.3f}")
                else:
                    cells.append("---")
            print(f"  {label:25s} " + "  ".join(cells))

    # Runtime comparison
    print(f"\n% Runtime comparison (seconds, median)")
    for dgp in ["linear", "nonlinear"]:
        sub = df_ok[df_ok["dgp"] == dgp]
        print(f"  {dgp}:")
        for method in methods_order:
            v = sub[sub["method"] == method]["runtime"]
            if len(v) > 0:
                label = method_labels.get(method, method)
                print(f"    {label:25s}  {v.median():.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p10_linear", type=str,
                        default="results/p10_linear_results.csv")
    parser.add_argument("--p10_nonlinear", type=str,
                        default="results/p10_nonlinear_results.csv")
    args = parser.parse_args()

    print("=== Combining p=20 Results ===\n")
    df = load_and_combine()
    if df is not None:
        summarise(df)
        compare_with_p10(df, args.p10_linear, args.p10_nonlinear)
        generate_latex_table(df)
