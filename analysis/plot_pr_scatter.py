"""
Precision-recall scatter of skeleton recovery (Appendix G).

A 2 x 3 grid (DGP x p) of per-replicate skeleton precision against recall,
with per-method medians, F1 contours, and the corner with precision > 0.8
and recall < 0.3 shaded: methods that reach a competitive SHD by returning
few edges (e.g. MissDAG on nonlinear DGPs, complete case at high
missingness) sit there. Reads results/p{P}_{dgp}_{mech}.csv and, for the
FZ-Rubin and MissDAG baselines, results/newbase_p{P}_{dgp}_{mech}.csv;
missing files are skipped.

Usage:
    python analysis/plot_pr_scatter.py
    python analysis/plot_pr_scatter.py --results-dir results/
    python analysis/plot_pr_scatter.py --output figures/pr_scatter.pdf
"""

import os
import sys
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Method display config (matches generate_figures.py colours) ───────

METHOD_INFO = {
    "ci_oracle_fast":   {"name": "PAIR-CI",       "color": "#2196F3", "marker": "o"},
    "ci_oracle_cached": {"name": "PAIR-CI",       "color": "#2196F3", "marker": "o"},
    "testwise":         {"name": "Test-wise",     "color": "#4CAF50", "marker": "s"},
    "impute_fisherz":   {"name": "FZ-vote",       "color": "#FF9800", "marker": "^"},
    "complete_case":    {"name": "Complete case", "color": "#9E9E9E", "marker": "v"},
    "micd_pooled":      {"name": "FZ-Rubin",      "color": "#673AB7", "marker": "D"},
    "mvpc":             {"name": "MVPC",          "color": "#795548", "marker": "P"},
    "missdag":          {"name": "MissDAG",       "color": "#E91E63", "marker": "X"},
}

# Legend order; ci_oracle_cached is plotted together with ci_oracle_fast.
METHOD_ORDER = [
    "ci_oracle_fast",
    "testwise",
    "impute_fisherz",
    "complete_case",
    "micd_pooled",
    "mvpc",
    "missdag",
]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ── Data loading ──────────────────────────────────────────────────────

# Result files of the main graph experiments (p10_linear_mar.csv, ...) and
# of experiments/new_baselines.py (newbase_p10_linear_mar.csv, ...).
EXISTING_RE = re.compile(r"^p(\d+)_(linear|nonlinear)_(mar|mnar|mixed|complete)\.csv$")
NEWBASE_RE = re.compile(r"^newbase_p(\d+)_(linear|nonlinear)_(mar|mnar|mixed|complete)\.csv$")
MVPC_RE = re.compile(r"^mvpc_p(\d+)_(linear|nonlinear)_(mar|mnar|mixed|complete)\.csv$")


def load_results(results_dir):
    """Concatenate all matching CSVs, adding dgp and p from the filename
    where the file lacks them."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return pd.DataFrame()

    rows = []
    for f in sorted(results_dir.iterdir()):
        if not f.is_file():
            continue
        m = EXISTING_RE.match(f.name) or NEWBASE_RE.match(f.name) or MVPC_RE.match(f.name)
        if not m:
            continue
        p, dgp, mech = int(m.group(1)), m.group(2), m.group(3)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  skip {f.name}: read failed ({e})")
            continue
        if "dgp" not in df.columns:
            df["dgp"] = dgp
        if "p" not in df.columns:
            df["p"] = p
        df["_source"] = ("newbase" if NEWBASE_RE.match(f.name)
                         else "mvpc" if MVPC_RE.match(f.name) else "existing")
        rows.append(df)
        print(f"  loaded {f.name}: {len(df)} rows ({df['_source'].iloc[0]})")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ── Plotting ──────────────────────────────────────────────────────────


def _draw_giveup_zone(ax):
    """Shade the high-precision low-recall corner."""
    rect = mpatches.Rectangle(
        (0.0, 0.8), 0.3, 0.2,
        linewidth=0, color="red", alpha=0.10, zorder=0,
    )
    ax.add_patch(rect)
    ax.text(0.02, 0.98, '"declare nothing"\nzone',
            fontsize=7, color="#7a0000", alpha=0.9,
            ha="left", va="top", style="italic", zorder=1)


def _draw_f1_contours(ax, levels=(0.2, 0.4, 0.6, 0.8)):
    """Draw F1 contour lines (F1 = 2PR/(P+R)) as light reference grid."""
    rs = np.linspace(0.01, 1.0, 200)
    for f1 in levels:
        # P = f1 * R / (2R - f1), valid where 2R > f1
        valid = 2 * rs > f1
        ps = np.where(valid, f1 * rs / (2 * rs - f1), np.nan)
        ps = np.where((ps >= 0) & (ps <= 1), ps, np.nan)
        ax.plot(rs, ps, color="gray", alpha=0.30, linewidth=0.6,
                linestyle=":", zorder=0)
        # Annotate near the diagonal
        idx = np.argmin(np.abs(rs - f1))
        if 0 <= ps[idx] <= 1:
            ax.text(rs[idx] + 0.01, ps[idx] - 0.02, f"F1={f1}",
                    fontsize=6, color="gray", alpha=0.7, zorder=1)


def plot_pr_scatter(df, output_path):
    """2x3 grid of P-R scatter plots: dgp x p."""
    setup_style()

    # Rows whose `failed` flag is missing are dropped as well.
    df = df[~df["failed"].fillna(True)].copy()
    df = df[df["precision"].notna() & df["recall"].notna()]
    if len(df) == 0:
        print("No non-failed rows with valid precision/recall — nothing to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5),
                              sharex=True, sharey=True)

    dgps = ["linear", "nonlinear"]
    ps = [10, 20, 30]

    # Track which methods we've actually plotted, for the shared legend.
    plotted_methods = set()

    for di, dgp in enumerate(dgps):
        for pi, p in enumerate(ps):
            ax = axes[di, pi]
            sub = df[(df["dgp"] == dgp) & (df["p"] == p)]

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{dgp.title()} DGP, $p={p}$")
            if di == 1:
                ax.set_xlabel("Recall")
            if pi == 0:
                ax.set_ylabel("Precision")
            ax.grid(True, alpha=0.20, linestyle=":")

            _draw_giveup_zone(ax)
            _draw_f1_contours(ax)

            if len(sub) == 0:
                ax.text(0.5, 0.5, "(no data yet)", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=10)
                continue

            # Per-replicate points; both ci_oracle labels are PAIR-CI.
            for method, info in METHOD_INFO.items():
                if method == "ci_oracle_cached":
                    continue
                if method == "ci_oracle_fast":
                    ms = sub[sub["method"].isin(
                        ["ci_oracle_fast", "ci_oracle_cached"])]
                else:
                    ms = sub[sub["method"] == method]
                if len(ms) == 0:
                    continue
                ax.scatter(ms["recall"], ms["precision"],
                           s=10, alpha=0.18, color=info["color"],
                           edgecolors="none", zorder=2)

            # Median markers per method (large, on top)
            for method in METHOD_ORDER:
                info = METHOD_INFO[method]
                if method == "ci_oracle_fast":
                    ms = sub[sub["method"].isin(
                        ["ci_oracle_fast", "ci_oracle_cached"])]
                else:
                    ms = sub[sub["method"] == method]
                if len(ms) == 0:
                    continue
                med_p = ms["precision"].median()
                med_r = ms["recall"].median()
                ax.scatter(med_r, med_p,
                           s=140, color=info["color"], marker=info["marker"],
                           edgecolors="black", linewidths=0.9, zorder=10)
                plotted_methods.add(method)

    # Shared legend (one entry per method, sorted by METHOD_ORDER)
    handles = []
    labels = []
    for method in METHOD_ORDER:
        if method not in plotted_methods:
            continue
        info = METHOD_INFO[method]
        handle = plt.Line2D(
            [0], [0],
            marker=info["marker"], color="w",
            markerfacecolor=info["color"], markeredgecolor="black",
            markersize=9, markeredgewidth=0.9,
            label=info["name"],
        )
        handles.append(handle)
        labels.append(info["name"])

    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=min(len(handles), 6),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved: {output_path}")
    print(f"Saved: {png_path}")


# ── Main ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="figures/pr_scatter.pdf")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}/ ...")
    df = load_results(args.results_dir)
    print(f"\nTotal rows: {len(df)}")
    if len(df) == 0:
        sys.exit(0)

    print(f"Methods present:    {sorted(df['method'].unique())}")
    print(f"DGPs present:       {sorted(df['dgp'].unique())}")
    print(f"p values present:   {sorted(df['p'].unique())}")
    print(f"Sources:            {sorted(df['_source'].unique())}")

    cells = (df.groupby(["dgp", "p"])
             .size()
             .reset_index(name="rows")
             .sort_values(["dgp", "p"]))
    print("\nRows per (dgp, p) cell:")
    print(cells.to_string(index=False))

    plot_pr_scatter(df, args.output)
