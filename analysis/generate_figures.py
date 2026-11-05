"""
Power curves of the standalone CI tests (Appendix D).

Reads results/ci_calibration_results.csv and writes figures/power_{dgp}.pdf:
rejection rate against signal strength, one panel per missingness
mechanism, for PAIR-CI and the imputed Fisher-Z, GCM and KCI baselines.

Usage:
  python analysis/generate_figures.py
  python analysis/generate_figures.py --results-dir results/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Style ────────────────────────────────────────────────────────────

COLORS = {
    "Ours": "#2196F3",
    "Fisher-Z (Imputed)": "#FF9800",
    "Fisher-Z (Rubin's Rules)": "#FF5722",
    "GCM (Imputed)": "#9C27B0",
    "KCI (Imputed)": "#795548",
    "Fisher-Z (Complete Case)": "#FFB74D",
    "GCM (Complete Case)": "#CE93D8",
    "KCI (Complete Case)": "#BCAAA4",
}

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def load(results_dir, name, required=True):
    path = Path(results_dir) / name
    if not path.exists():
        if required:
            print(f"  *** MISSING: {name}")
        return None
    return pd.read_csv(path)


# ── Power curves by DGP ─────────────────────────────────────────────

def plot_power_curves(df, out_dir):
    """Power vs signal strength, one figure per DGP, panels by mechanism."""
    setup_style()
    df = df[df["failed"] == False]

    method_variants = [
        ("Ours", "Ours", "Integrated"),
        ("Fisher-Z (Imputed)", "Fisher-Z", "Imputed"),
        ("GCM (Imputed)", "GCM", "Imputed"),
        ("KCI (Imputed)", "KCI", "Imputed"),
    ]

    dgp_files = {
        "linear_gaussian": "power_linear_gaussian.pdf",
        "post_nonlinear": "power_post_nonlinear.pdf",
        "latent_confounder": "power_latent_confounder.pdf",
    }

    mechanisms = ["complete", "mar", "mnar"]
    mech_labels = {"complete": "Complete", "mar": "MAR", "mnar": "MNAR"}
    signals = sorted(df["signal"].unique())

    for dgp, fname in dgp_files.items():
        dgp_data = df[df["dgp"] == dgp]
        if len(dgp_data) == 0:
            print(f"  Skipping {fname} — no data for {dgp}")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        for mi, mech in enumerate(mechanisms):
            ax = axes[mi]
            sub = dgp_data[dgp_data["mechanism"] == mech]
            for label, method, variant in method_variants:
                msub = sub[(sub["method"] == method) &
                           (sub["variant"] == variant)]
                if len(msub) == 0:
                    continue
                power = msub.groupby("signal")["rejected"].mean()
                power = power.reindex(signals, fill_value=np.nan)
                ax.plot(power.index, power.values, "o-", markersize=3,
                        color=COLORS.get(label, "#999"), label=label,
                        linewidth=1.5)

            ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8,
                       alpha=0.5)
            ax.set_xlabel("Signal strength")
            if mi == 0:
                ax.set_ylabel("Rejection rate")
            ax.set_title(mech_labels[mech])
            ax.set_ylim(-0.02, 1.05)
            ax.set_xticks(signals)
            if mi == 2:
                ax.legend(loc="center right", framealpha=0.9)

        plt.tight_layout()
        fig.savefig(out_dir / fname)
        plt.close(fig)
        print(f"  Saved {out_dir / fname}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  FIGURE GENERATION")
    print(f"  Results: {results_dir.resolve()}")
    print(f"  Output:  {fig_dir.resolve()}")
    print("=" * 60)

    calib = load(results_dir, "ci_calibration_results.csv")
    if calib is not None:
        print("\nPower curves:")
        plot_power_curves(calib, fig_dir)
    else:
        print("\nSkipping power curves — no data")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
