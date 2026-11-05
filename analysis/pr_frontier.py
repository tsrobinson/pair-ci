"""
Precision-recall operating-point frontiers (Appendix T).

For each benchmark cell the alpha-tunable baselines (test-wise deletion,
complete-case, FZ-vote, FZ-Rubin) are swept over alpha and each swept point
is the median (recall, precision) over replicates at that alpha; the points
are joined in alpha order.  PAIR-CI is shown at its deployed alpha = 0.05
(and along its own alpha sweep where one was run, p = 10); MVPC has no alpha
and sits at its native operating point.  Iso-F1 contours are drawn in the
background so that "same F1 at a sparser operating point" can be read off
directly.  Median edge counts at the sparse end of each curve are annotated
because the sparsity-matched comparison (tab_sparsity_matched) is about
edge counts, not alpha.

Panels: (a) Erdős–Rényi p = 10, nonlinear, MNAR 30%; (b) ALARM nonlinear,
MNAR 20%; (c) HAILFINDER, MNAR 20%.  Also writes
results/pr_frontier_points.csv with every plotted point.

Colours are colour-blind safe; PAIR-CI and MVPC are drawn in ink, and
every series also differs in marker shape.

    python analysis/pr_frontier.py --output figures/pr_frontier.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tables import load_csv  # noqa: E402
from tables_revision import load_network, load_er, cell_stats  # noqa: E402

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = {  # colour, marker, line style, label
    # FZ-vote and FZ-Rubin (both Fisher-Z on imputed data) share a hue and
    # differ in marker and line style.
    "testwise":       ("#2a78d6", "o", "-",  "Test-wise"),
    "impute_fisherz": ("#eb6834", "s", "-",  "FZ-vote"),
    "micd_pooled":    ("#eb6834", "^", "--", "FZ-Rubin"),
    "complete_case":  ("#1baf7a", "v", "-",  "Complete case"),
    "mvpc":           (INK,       "D", "",   "MVPC (native)"),
    "ours":           (INK,       "*", "",   "PAIR-CI"),
}
SWEEP_ORDER = ["testwise", "impute_fisherz", "micd_pooled", "complete_case"]


def sweep_points(sw: pd.DataFrame, method: str) -> pd.DataFrame:
    """Median recall, precision, F1 and edge count at each alpha."""
    s = sw[sw["method"] == method].groupby("alpha").agg(
        recall=("recall", "median"), precision=("precision", "median"),
        f1=("f1", "median"), edges=("n_edges_out", "median")).reset_index()
    return s.sort_values("alpha")


def panel_data(results_dir: Path) -> list[dict]:
    """Sweeps, PAIR-CI points and the MVPC point for the three panels."""
    panels = []
    # (a) p=10 nonlinear MNAR: baselines from the p=10 sweep, PAIR-CI from its own sweep
    sw = load_csv(results_dir, "alpha_sweep_baselines_p10.csv")
    sw = sw[(~sw["failed"].astype(bool)) & (sw["mechanism"] == "mnar") & (sw["dgp"] == "nonlinear")]
    ours_sw = load_csv(results_dir, "alpha_sensitivity_results.csv")
    ours_sw = ours_sw[(~ours_sw["failed"].astype(bool)) & (ours_sw["mechanism"] == "mnar")]
    ours_pts = ours_sw.groupby("alpha").agg(recall=("recall", "median"), precision=("precision", "median"),
                                            f1=("f1", "median")).reset_index().sort_values("alpha")
    er = load_er(results_dir, 10, "nonlinear")
    mv = cell_stats(er[(er["method_key"] == "mvpc") & (er["mechanism"] == "mnar")])
    o_main = cell_stats(er[(er["method_key"] == "ours") & (er["mechanism"] == "mnar")])
    ours_pts["edges"] = np.where(np.isclose(ours_pts["alpha"], 0.05), o_main["edges"], np.nan)
    panels.append(dict(title=r"(a) Erdős–Rényi $p{=}10$, nonlinear, MNAR",
                       sweeps={m: sweep_points(sw, m) for m in SWEEP_ORDER},
                       ours=ours_pts, ours_alpha=0.05,
                       mvpc=(mv["rec"], mv["prec"], mv["edges"])))
    for key, fname, title in (("alarm_nonlinear", "alpha_sweep_baselines_alarm_nonlinear.csv", "(b) ALARM nonlinear, MNAR 20%"),
                              ("hailfinder", "alpha_sweep_baselines_hailfinder.csv", "(c) HAILFINDER, MNAR 20%")):
        sw = load_csv(results_dir, fname)
        sw = sw[(~sw["failed"].astype(bool)) & (sw["mechanism"] == "mnar")]
        d, _ = load_network(results_dir, key)
        d = d[(d["rate"] == 0.2) & (d["mechanism"] == "mnar")]
        o = cell_stats(d[d["method_key"] == "ours"])
        mv = cell_stats(d[d["method_key"] == "mvpc"])
        ours_pts = pd.DataFrame(dict(alpha=[0.05], recall=[o["rec"]], precision=[o["prec"]], f1=[o["f1"]], edges=[o["edges"]]))
        panels.append(dict(title=title, sweeps={m: sweep_points(sw, m) for m in SWEEP_ORDER},
                           ours=ours_pts, ours_alpha=0.05, mvpc=(mv["rec"], mv["prec"], mv["edges"])))
    return panels


def draw(panels: list[dict], out_path: Path):
    """Draw the panels and return every plotted point as a DataFrame."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    rr = np.linspace(0.01, 1, 200)
    rows = []
    for ax, P in zip(axes, panels):
        for f in (0.3, 0.5, 0.7, 0.9):
            with np.errstate(divide="ignore", invalid="ignore"):
                pp = f * rr / (2 * rr - f)
            ok = (pp > 0) & (pp <= 1.02)
            ax.plot(rr[ok], pp[ok], color=GRID, lw=0.8, zorder=0)
            xs = rr[ok]
            if len(xs):
                ax.text(xs[-1] - 0.01, pp[ok][-1] + 0.012, f"F1 = {f:.1f}", color=MUTED, fontsize=6.5, ha="right")
        notes = ["Sparse end of each sweep:"]
        for m in SWEEP_ORDER:
            col, mk, ls, lab = SERIES[m]
            s = P["sweeps"][m]
            if s.empty:
                continue
            ax.plot(s["recall"], s["precision"], color=col, lw=1.6, ls=ls, zorder=2, alpha=0.9)
            ax.scatter(s["recall"], s["precision"], color=col, marker=mk, s=26, zorder=3,
                       edgecolor="white", linewidth=0.6, label=lab)
            sparse = s.iloc[0]
            notes.append(f"{lab}: {sparse['edges']:g} edges at $\\alpha$={sparse['alpha']:g}")
            for _, r in s.iterrows():
                rows.append(dict(panel=P["title"], method=m, alpha=r["alpha"], recall=r["recall"],
                                 precision=r["precision"], f1=r["f1"], edges=r["edges"]))
        rec, prec, edges = P["mvpc"]
        col, mk, _, lab = SERIES["mvpc"]
        ax.scatter([rec], [prec], facecolor="white", edgecolor=col, marker=mk, s=42, zorder=4, linewidth=1.2, label=lab)
        ax.annotate(f"MVPC, {edges:g} edges", (rec, prec), xytext=(5, -9), textcoords="offset points",
                    fontsize=6, color=col)
        rows.append(dict(panel=P["title"], method="mvpc", alpha=np.nan, recall=rec, precision=prec, f1=np.nan, edges=edges))
        o = P["ours"]
        col, mk, _, lab = SERIES["ours"]
        if len(o) > 1:
            ax.plot(o["recall"], o["precision"], color=col, lw=1.4, ls="--", zorder=4)
            ax.scatter(o["recall"], o["precision"], color=col, marker="o", s=18, zorder=5)
        pt = o[np.isclose(o["alpha"], P["ours_alpha"])].iloc[0]
        ax.scatter([pt["recall"]], [pt["precision"]], color=col, marker=mk, s=150, zorder=6,
                   edgecolor="white", linewidth=0.8, label=lab + r" ($\alpha{=}0.05$)")
        etxt = f", {pt['edges']:g} edges" if "edges" in pt and not pd.isna(pt.get("edges", np.nan)) else ""
        ax.annotate(f"PAIR-CI{etxt}", (pt["recall"], pt["precision"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color=col, fontweight="bold")
        for _, r in o.iterrows():
            rows.append(dict(panel=P["title"], method="ours", alpha=r["alpha"], recall=r["recall"],
                             precision=r["precision"], f1=r["f1"], edges=r.get("edges", np.nan)))
        ax.text(0.98, 0.02, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color=MUTED, linespacing=1.3)
        ax.set_title(P["title"], fontsize=9, loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Skeleton recall")
        ax.grid(alpha=0.25, lw=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Skeleton precision")
    axes[0].legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.92, ncol=1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output", default="figures/pr_frontier.pdf")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    panels = panel_data(results_dir)
    pts = draw(panels, out)
    pts.to_csv(results_dir / "pr_frontier_points.csv", index=False)
    print(f"Saved {out} (+ .png) and {results_dir / 'pr_frontier_points.csv'}")
    pd.set_option("display.width", 200)
    print(pts.round(3).to_string())


if __name__ == "__main__":
    main()
