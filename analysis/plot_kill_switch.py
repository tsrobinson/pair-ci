"""Figures and tables for the p = 10 graph-recovery experiment.

SHD box plots by missingness mechanism, pairwise SHD of PAIR-CI against
FZ-vote, and LaTeX tables of median SHD and F1. "Kill switch" is the
experiment's internal name. Called by combine_p10.py.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

METHOD_NAMES = {
    "ci_oracle_cached": "Ours (Integrated)", "complete_case": "Complete Case",
    "impute_fisherz": "Impute->Fisher-Z", "testwise": "Testwise Deletion",
    "impute_ci_oracle": "Impute->CI Oracle",
}
METHOD_COLORS = {
    "ci_oracle_cached": "#2196F3", "complete_case": "#9E9E9E",
    "impute_fisherz": "#FF9800", "testwise": "#4CAF50",
    "impute_ci_oracle": "#9C27B0",
}
METHOD_ORDER = ["ci_oracle_cached", "impute_fisherz", "impute_ci_oracle",
                "testwise", "complete_case"]
MECH_NAMES = {"complete": "Complete\n(Control)", "mar": "MAR",
              "mnar": "MNAR", "mixed": "Mixed"}
MECH_ORDER = ["complete", "mar", "mnar", "mixed"]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def plot_shd(df, out, prefix=""):
    """SHD box plots by method, one panel per mechanism."""
    setup_style()
    dv = df[~df["failed"]].copy()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ai, mech in enumerate(MECH_ORDER):
        ax = axes[ai]
        dm = dv[dv["mechanism"] == mech]
        data_list, colors = [], []
        for method in METHOD_ORDER:
            v = dm[dm["method"] == method]["shd"].dropna()
            if len(v) > 0:
                data_list.append(v.values)
                colors.append(METHOD_COLORS[method])
        if data_list:
            bp = ax.boxplot(data_list, positions=range(len(data_list)), widths=0.6,
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker=".", markersize=3, alpha=0.3))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
        ax.set_title(MECH_NAMES[mech], fontweight="bold")
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(
            [METHOD_NAMES[m].replace("->", "\n->") for m in METHOD_ORDER],
            rotation=45, ha="right", fontsize=8)
        if ai == 0:
            ax.set_ylabel("SHD (lower=better)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Graph Recovery by Missingness Mechanism",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = Path(out)
    fig.savefig(out / f"{prefix}shd_by_mechanism.pdf")
    fig.savefig(out / f"{prefix}shd_by_mechanism.png")
    print(f"  Saved: {prefix}shd_by_mechanism.pdf/png")
    plt.close()


def plot_pairwise(df, out, prefix=""):
    """PAIR-CI against FZ-vote SHD on matched (graph, dataset) pairs."""
    setup_style()
    dv = df[~df["failed"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ai, mech in enumerate(["mar", "mnar", "mixed"]):
        ax = axes[ai]
        ours = (dv[(dv["mechanism"] == mech) & (dv["method"] == "ci_oracle_cached")]
                [["graph", "dataset", "shd"]].rename(columns={"shd": "shd_ours"}))
        base = (dv[(dv["mechanism"] == mech) & (dv["method"] == "impute_fisherz")]
                [["graph", "dataset", "shd"]].rename(columns={"shd": "shd_base"}))
        mg = ours.merge(base, on=["graph", "dataset"])
        if len(mg) == 0:
            continue
        ax.scatter(mg["shd_base"], mg["shd_ours"], alpha=0.4, s=20,
                   c="#2196F3", edgecolors="none")
        lim = max(mg["shd_base"].max(), mg["shd_ours"].max()) + 1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
        nb = (mg["shd_ours"] < mg["shd_base"]).sum()
        nw = (mg["shd_ours"] > mg["shd_base"]).sum()
        nt = (mg["shd_ours"] == mg["shd_base"]).sum()
        ax.text(0.05, 0.95, f"Better:{nb}\nTied:{nt}\nWorse:{nw}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("Impute->Fisher-Z (SHD)")
        ax.set_title(mech.upper(), fontweight="bold")
        if ai == 0:
            ax.set_ylabel("Ours (SHD)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle("Pairwise SHD: Ours vs Impute-then-Fisher-Z",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(out)
    fig.savefig(out / f"{prefix}pairwise_comparison.pdf")
    fig.savefig(out / f"{prefix}pairwise_comparison.png")
    print(f"  Saved: {prefix}pairwise_comparison.pdf/png")
    plt.close()


def generate_latex_table(df, out, prefix=""):
    """Write median SHD and F1 by method and mechanism as LaTeX."""
    dv = df[~df["failed"]].copy()
    rows = []
    for method in METHOD_ORDER:
        row = {"Method": METHOD_NAMES[method]}
        for mech in MECH_ORDER:
            v = dv[(dv["mechanism"] == mech) & (dv["method"] == method)]
            row[f"{mech}_shd"] = f"{v['shd'].median():.1f}"
            row[f"{mech}_f1"] = f"{v['f1'].median():.3f}"
        rows.append(row)
    latex = "\\begin{tabular}{l" + "cc" * len(MECH_ORDER) + "}\n\\toprule\n"
    latex += " & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{MECH_NAMES[m].replace(chr(10), ' ')}}}"
        for m in MECH_ORDER) + " \\\\\n\\midrule\n"
    for r in rows:
        latex += r["Method"]
        for mech in MECH_ORDER:
            latex += f" & {r[f'{mech}_shd']} & {r[f'{mech}_f1']}"
        latex += " \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    out = Path(out)
    with open(out / f"{prefix}results_table.tex", "w") as f:
        f.write(latex)
    print(f"  Saved: {prefix}results_table.tex")


def generate_extended_tables(df, out, prefix=""):
    """Print SHD (IQR), F1, precision, recall and runtime by method; write
    the SHD (IQR) table as LaTeX."""
    dv = df[~df["failed"]].copy()
    out = Path(out)

    # --- SHD with IQR ---
    print("\n  SHD with IQR:")
    for method in METHOD_ORDER:
        label = METHOD_NAMES[method]
        vals = []
        for mech in MECH_ORDER:
            v = dv[(dv["mechanism"] == mech) & (dv["method"] == method)]["shd"]
            if len(v) > 0:
                med = v.median()
                q25 = v.quantile(0.25)
                q75 = v.quantile(0.75)
                vals.append(f"{med:.0f} ({q25:.0f}-{q75:.0f})")
            else:
                vals.append("---")
        print(f"    {label:25s} " + "  ".join(f"{v:18s}" for v in vals))

    # --- F1 table ---
    print("\n  F1 (median):")
    for method in METHOD_ORDER:
        label = METHOD_NAMES[method]
        vals = []
        for mech in MECH_ORDER:
            v = dv[(dv["mechanism"] == mech) & (dv["method"] == method)]["f1"]
            if len(v) > 0:
                vals.append(f"{v.median():.3f}")
            else:
                vals.append("---")
        print(f"    {label:25s} " + "  ".join(f"{v:8s}" for v in vals))

    # --- Precision / Recall ---
    print("\n  Precision / Recall (median):")
    for method in METHOD_ORDER:
        label = METHOD_NAMES[method]
        for mech in MECH_ORDER:
            v = dv[(dv["mechanism"] == mech) & (dv["method"] == method)]
            if len(v) > 0:
                prec = v["precision"].median()
                rec = v["recall"].median()
                print(f"    {label:25s} {mech:10s}  prec={prec:.3f}  rec={rec:.3f}")

    # --- Runtime ---
    print("\n  Runtime (seconds, median):")
    for method in METHOD_ORDER:
        label = METHOD_NAMES[method]
        v = dv[dv["method"] == method]["runtime"]
        if len(v) > 0:
            print(f"    {label:25s}  {v.median():.1f}s")

    # --- LaTeX: SHD with IQR ---
    latex_lines = [
        "% SHD with IQR",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Complete & MAR & MNAR & Mixed \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        label = METHOD_NAMES[method]
        cells = []
        for mech in MECH_ORDER:
            v = dv[(dv["mechanism"] == mech) & (dv["method"] == method)]["shd"]
            if len(v) > 0:
                med = v.median()
                q25 = v.quantile(0.25)
                q75 = v.quantile(0.75)
                cells.append(f"{med:.0f} ({q25:.0f}--{q75:.0f})")
            else:
                cells.append("---")
        latex_lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    latex_lines += ["\\bottomrule", "\\end{tabular}"]
    with open(out / f"{prefix}results_table_iqr.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  Saved: {prefix}results_table_iqr.tex")
