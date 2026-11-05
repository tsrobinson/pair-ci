"""Combine the p = 10 graph-recovery results and make their figures.

For each DGP (linear, nonlinear), concatenates
results/p10_{dgp}_{mechanism}.csv into results/p10_{dgp}_results.csv,
prints median SHD by method, and writes figures and LaTeX tables to
figures/ with plot_kill_switch.py (graph recovery, Section 5). Run from
the package root: python analysis/combine_p10.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from pathlib import Path

from dgps import analyze_kill_switch
from plot_kill_switch import plot_shd, plot_pairwise, generate_latex_table, generate_extended_tables


def combine_dgp(dgp_prefix, label):
    mechs = ["mar", "mnar", "mixed", "complete"]
    dfs = []
    for mech in mechs:
        f = Path(f"results/{dgp_prefix}_{mech}.csv")
        if f.exists():
            df = pd.read_csv(f)
            print(f"Loaded {f}: {len(df)} rows")
            dfs.append(df)
        else:
            print(f"WARNING: {f} not found -- job may still be running")

    if not dfs:
        print(f"No results found for {label}.")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    out_path = f"results/{dgp_prefix}_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nCombined: {len(combined)} rows -> {out_path}")
    return combined


if __name__ == "__main__":
    print("=" * 60)
    print("p=10 LINEAR")
    print("=" * 60)
    df_linear = combine_dgp("p10_linear", "linear")
    if df_linear is not None:
        analyze_kill_switch(df_linear)
        out = Path("figures")
        out.mkdir(parents=True, exist_ok=True)
        plot_shd(df_linear, out, prefix="linear_")
        plot_pairwise(df_linear, out, prefix="linear_")
        generate_latex_table(df_linear, out, prefix="linear_")
        generate_extended_tables(df_linear, out, prefix="linear_")
        print(f"\nLinear figures saved to {out}/ with 'linear_' prefix")

    print("\n" + "=" * 60)
    print("p=10 NONLINEAR")
    print("=" * 60)
    df_nonlinear = combine_dgp("p10_nonlinear", "nonlinear")
    if df_nonlinear is not None:
        analyze_kill_switch(df_nonlinear)
        out = Path("figures")
        out.mkdir(parents=True, exist_ok=True)
        plot_shd(df_nonlinear, out, prefix="nonlinear_")
        plot_pairwise(df_nonlinear, out, prefix="nonlinear_")
        generate_latex_table(df_nonlinear, out, prefix="nonlinear_")
        generate_extended_tables(df_nonlinear, out, prefix="nonlinear_")
        print(f"\nNonlinear figures saved to {out}/ with 'nonlinear_' prefix")
