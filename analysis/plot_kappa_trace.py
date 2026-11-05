"""Figure 4 (Appendix N): PAIR-CI false positive rate against the
estimated kappa product, with a fitted power law.

Reads results/kappa_trace.csv (48 adversarial cells, from
compute_kappa_trace.py); writes figures/kappa_fpr_trace.{pdf,png} and
results/kappa_fit.csv, the fit summaries quoted in the paper. The x
variable is the product of Definition 9's three factors with kappa_imp
squared, as in the first-order expansion:

    kappa-hat = kappa_imp^2 * |kappa_Y| * |kappa_Z|

On the inflated cells (FPR > 0.05), log(FPR - 0.05) is regressed on
log(kappa-hat) by OLS and FPR = 0.05 + exp(a) kappa-hat^b is drawn; the
slope b is compared with the first-order expansion. The Spearman
correlation uses all cells. Products below 1e-4 are drawn at 1e-4 (x axis
only). `--x kappa_yz` plots kappa_Y * kappa_Z instead.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, ".."))

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
PALETTE = {"linear": "#2a78d6", "rf": "#e34948"}  # colour-blind-safe pair
MARKERS = {"full": "o", "missing": "s"}
IMPUTER_LABELS = {"linear": "Linear MICE", "rf": "Random-forest MICE"}


def kappa_product(df: pd.DataFrame) -> pd.Series:
    """kappa_imp^2 |kappa_Y| |kappa_Z| per cell."""
    return df["kappa_imp"] ** 2 * df["kappa_Y"].abs() * df["kappa_Z"].abs()


def fit_stats(df: pd.DataFrame, x: pd.Series) -> dict:
    """Spearman correlation, medians and the log-log fit on FPR > 0.05."""
    infl = df["fpr"] > 0.05
    lx, ly = np.log(x[infl]), np.log(df.loc[infl, "fpr"] - 0.05)
    res = stats.linregress(lx, ly)
    return dict(
        n_cells=len(df), n_inflated=int(infl.sum()),
        spearman=stats.spearmanr(x, df["fpr"])[0],
        median_product_calibrated=x[~infl].median(),
        median_product_inflated=x[infl].median(),
        median_fpr_calibrated=df.loc[~infl, "fpr"].median(),
        median_fpr_inflated=df.loc[infl, "fpr"].median(),
        slope=res.slope, slope_se=res.stderr, intercept=res.intercept, r2=res.rvalue ** 2,
        x_min_inflated=x[infl].min(), x_max_inflated=x[infl].max(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/kappa_trace.csv")
    parser.add_argument("--output", default="figures/kappa_fpr_trace.pdf")
    parser.add_argument("--stats-output", default="results/kappa_fit.csv")
    parser.add_argument("--x", choices=["product", "kappa_yz"], default="product")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(ROOT, args.input))
    x = kappa_product(df) if args.x == "product" else df["kappa_yz"]
    S = fit_stats(df, x)
    pd.DataFrame([S]).to_csv(os.path.join(ROOT, args.stats_output), index=False)

    out_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    xf = np.maximum(x, 1e-4)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axhspan(0.0, 0.05, color="#cde2fb", alpha=0.35, zorder=0, lw=0)
    ax.axhline(0.05, color=MUTED, linestyle=":", linewidth=0.9, label=r"Nominal $\alpha = 0.05$")
    for imputer in ("linear", "rf"):
        for cond in ("full", "missing"):
            sub = df[(df["imputer"] == imputer) & (df["cond_strategy"] == cond)]
            if sub.empty:
                continue
            y = sub["fpr"].values
            yerr = np.array([y - sub["fpr_lo"].values, sub["fpr_hi"].values - y])
            ax.errorbar(xf[sub.index], y, yerr=yerr, fmt=MARKERS[cond], color=PALETTE[imputer],
                        markersize=6, capsize=2, linewidth=0.8,
                        markerfacecolor="white" if cond == "missing" else PALETTE[imputer],
                        markeredgewidth=1.0, label=f"{IMPUTER_LABELS[imputer]}, {cond} conditioning")
    grid = np.logspace(np.log10(S["x_min_inflated"]), np.log10(S["x_max_inflated"]), 200)
    fit = 0.05 + np.exp(S["intercept"]) * grid ** S["slope"]
    keep = fit <= 1.0  # draw the fit only where it is a valid rate
    ax.plot(grid[keep], fit[keep], color=INK, lw=1.6, zorder=5,
            label=rf"Fit on inflated cells: slope {S['slope']:.2f} $\pm$ {S['slope_se']:.2f}")
    ax.text(0.03, 0.60, rf"Spearman $\rho$ = {S['spearman']:.2f} over {S['n_cells']} cells",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    xlabel = (r"Estimated $\hat\kappa$ product: $\hat\kappa_{\mathrm{imp}}^2\,|\hat\kappa_Y|\,|\hat\kappa_Z|$"
              if args.x == "product" else r"Residual $\kappa_Y \cdot \kappa_Z$ (after paired cancellation)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PAIR-CI false positive rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25, linewidth=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out_path} (+ .png); stats -> {args.stats_output}")
    for k, v in S.items():
        print(f"  {k:<28} {v:.4g}" if isinstance(v, float) else f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
