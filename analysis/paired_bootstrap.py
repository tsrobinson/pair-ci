"""
Paired, cluster-bootstrap confidence intervals for the headline
graph-recovery contrasts (Appendix T).

Estimand.  For a benchmark cell (DGP x p, or a network at 20% missingness)
and a metric (SHD or skeleton F1), the mean over the experiment's datasets
of the paired difference  PAIR-CI minus baseline,  pooled over the
missingness mechanisms MAR / MNAR / mixed.  The datasets are the replicate
draws actually used in the experiment (bit-identical across methods, except
MVPC on HAILFINDER, which is matched in distribution only -- flagged in the
output).  Negative SHD differences and positive F1 differences favour
PAIR-CI.

Uncertainty.  Nonparametric percentile bootstrap over clusters, B = 10,000,
fixed seed.  On the Erdős–Rényi grids the cluster is the random graph
(10 graphs x 20 datasets x 3 mechanisms); on the fixed network topologies
it is the replicate index (one dataset per rep and mechanism).  The
interval is conditional on the DGP grid and treats datasets as
exchangeable within cluster; it does not account for the finite set of
graphs beyond the cluster resampling.

Outputs results/paired_bootstrap_contrasts.csv and
tables/tab_bootstrap_contrasts.tex:

    python analysis/paired_bootstrap.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables_revision import load_er, load_network, NETWORKS  # noqa: E402

BASELINES = [("testwise", "Test-wise"), ("complete_case", "Complete case"),
             ("impute_fisherz", "FZ-vote"), ("micd_pooled", "FZ-Rubin"), ("mvpc", "MVPC")]
MECHS = ("mar", "mnar", "mixed")


def cluster_bootstrap(diffs: pd.Series, clusters: pd.Series, B: int, rng: np.random.RandomState):
    """Return (mean, 2.5%, 97.5%) of the mean difference, resampling
    whole clusters with replacement."""
    ids = np.asarray(sorted(clusters.unique()))
    by = {c: diffs[clusters == c].values for c in ids}
    boots = np.empty(B)
    for b in range(B):
        samp = rng.choice(ids, len(ids), replace=True)
        boots[b] = np.concatenate([by[c] for c in samp]).mean()
    return diffs.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def contrasts(results_dir: Path, B: int, seed: int) -> pd.DataFrame:
    # One RandomState serves every contrast in loop order, so adding or
    # reordering cells changes the intervals that follow.
    rng = np.random.RandomState(seed)
    rows = []
    for dgp in ("linear", "nonlinear"):
        for p in (10, 20, 30):
            df = load_er(results_dir, p, dgp)
            if df is None:
                continue
            df = df[df["mechanism"].isin(MECHS)]
            for metric in ("shd", "f1"):
                piv = df.pivot_table(index=["graph", "dataset", "mechanism"], columns="method_key", values=metric)
                for key, label in BASELINES:
                    if key not in piv.columns:
                        continue
                    d = (piv["ours"] - piv[key]).dropna()
                    cl = d.index.get_level_values("graph").to_series(index=d.index)
                    mean, lo, hi = cluster_bootstrap(d, cl, B, rng)
                    rows.append(dict(cell=f"ER {dgp} p={p}", metric=metric, baseline=key, baseline_label=label,
                                     mean_diff=mean, ci_lo=lo, ci_hi=hi, n_pairs=len(d), n_clusters=cl.nunique(),
                                     cluster="graph", paired="bit-identical datasets"))
    for net, spec in NETWORKS.items():
        df, _ = load_network(results_dir, net)
        df = df[(df["rate"] == 0.2) & (df["mechanism"].isin(MECHS))]
        for metric in ("shd", "f1"):
            piv = df.pivot_table(index=["rep", "mechanism"], columns="method_key", values=metric)
            for key, label in BASELINES:
                if key not in piv.columns:
                    continue
                d = (piv["ours"] - piv[key]).dropna()
                cl = d.index.get_level_values("rep").to_series(index=d.index)
                mean, lo, hi = cluster_bootstrap(d, cl, B, rng)
                paired = "matched in distribution" if (net == "hailfinder" and key == "mvpc") else "bit-identical datasets"
                rows.append(dict(cell=spec["label"] + " 20\\%", metric=metric, baseline=key, baseline_label=label,
                                 mean_diff=mean, ci_lo=lo, ci_hi=hi, n_pairs=len(d), n_clusters=cl.nunique(),
                                 cluster="replicate", paired=paired))
    return pd.DataFrame(rows)


def to_tex(C: pd.DataFrame, metrics=(("shd", "SHD"),)) -> str:
    """LaTeX table of the contrasts; SHD only by default, as in the paper
    (the F1 rows are always in the CSV). Intervals covering 0 are greyed;
    an asterisk marks contrasts matched in distribution only."""
    cells = list(dict.fromkeys(C["cell"]))
    lines = [r"\begin{tabular}{ll" + "c" * len(BASELINES) + "}", r"    \toprule",
             "    Benchmark & Metric & " + " & ".join(l for _, l in BASELINES) + r" \\", r"    \midrule"]
    for ci, cell in enumerate(cells):
        for metric, mlab in metrics:
            sub = C[(C["cell"] == cell) & (C["metric"] == metric)].set_index("baseline")
            vals = []
            for key, _ in BASELINES:
                if key not in sub.index:
                    vals.append("--")
                    continue
                r = sub.loc[key]
                s = f"{r.mean_diff:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"
                if r.ci_lo <= 0 <= r.ci_hi:
                    s = f"\\textcolor{{gray}}{{{s}}}"
                if r.paired != "bit-identical datasets":
                    s += r"$^\ast$"
                vals.append(s)
            lines.append(f"    {cell if metric == 'shd' else '':<26} & {mlab} & " + " & ".join(vals) + r" \\")
        if ci < len(cells) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="tables")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    results_dir, output_dir = Path(args.results_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    C = contrasts(results_dir, args.B, args.seed)
    C.to_csv(results_dir / "paired_bootstrap_contrasts.csv", index=False)
    (output_dir / "tab_bootstrap_contrasts.tex").write_text(to_tex(C) + "\n")
    pd.set_option("display.width", 200)
    print(C[["cell", "metric", "baseline", "mean_diff", "ci_lo", "ci_hi", "n_pairs", "n_clusters"]].round(3).to_string())
    print(f"\nB={args.B}, seed={args.seed}; wrote {results_dir / 'paired_bootstrap_contrasts.csv'} and {output_dir / 'tab_bootstrap_contrasts.tex'}")


if __name__ == "__main__":
    main()
