"""Calibration and power of PAIR-CI as the number of imputations M varies.

Checks whether more imputations raise power at weak signals by increasing
the Barnard-Rubin degrees of freedom. PAIR-CI (general harness, K = 5,
cached imputation) runs with M in {3, 5, 10, 20} on the same datasets: the
linear-Gaussian and post-nonlinear DGPs, |X| = 5, n in {500, 1000}, signal
in {0, 0.3, 0.6, 1.0}, under complete data, MAR and MNAR (30% per column),
50 replicates per cell. The estimands are the rejection rates at
alpha = 0.05.

    python experiments/m_sensitivity.py --n_reps 50 --Ms 3 5 10 20 --K 5 \
        --dgps linear_gaussian post_nonlinear --sample_sizes 500 1000 \
        --dim_x 5 --signals 0 0.3 0.6 1.0 --mechanisms complete mar mnar \
        --output results/m_sensitivity_results.csv

Feeds App. M and App. T.2. Takes about 6-12 hours on 16 cores.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

from itertools import product

from dgps import DGP_FUNCTIONS, inject_missingness_calibration
from ci_oracle import GeneralCITest


def run_single_test(data_miss, z_idx, y_idx, cond_set, M, K, rng):
    ci = GeneralCITest(
        data_miss,
        n_imputations=M,
        n_folds=K,
        imputation_strategy="cached",
        random_state=rng.randint(0, 2**31),
    )
    stat, pval = ci.test(x_idx=z_idx, y_idx=y_idx, cond_set=cond_set)
    return stat, pval


def run_experiment(args):
    rng = np.random.RandomState(args.seed)

    Ms = args.Ms
    K = args.K
    dgps = args.dgps
    sample_sizes = args.sample_sizes
    dim_x = args.dim_x
    signals = args.signals
    mechanisms = args.mechanisms
    miss_rate = args.miss_rate
    alpha = args.alpha
    n_reps = args.n_reps

    # The M values are looped inside each replicate, so within a
    # (dgp, n, signal, mechanism) cell every M sees the same (data, mask)
    # and the comparison across M is paired.
    outer_configs = list(product(dgps, sample_sizes, signals, mechanisms))
    print(f"M values: {Ms}")
    print(f"K (fixed): {K}")
    print(f"DGPs: {dgps}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"dim_x: {dim_x}")
    print(f"Signals: {signals}")
    print(f"Mechanisms: {mechanisms}")
    print(f"Outer configs: {len(outer_configs)}")
    print(f"Reps per config: {n_reps}")
    print(f"Total tests: {len(outer_configs) * n_reps * len(Ms)}")
    print()

    all_results = []
    t_start = time.time()

    for i, (dgp, n, signal, mechanism) in enumerate(outer_configs):
        elapsed = (time.time() - t_start) / 60
        print(f"[{elapsed:6.1f} min] Config {i+1}/{len(outer_configs)}: "
              f"{dgp}, n={n}, sig={signal}, {mechanism}")

        dgp_fn = DGP_FUNCTIONS[dgp]
        for rep in range(n_reps):
            X, Y, Z = dgp_fn(n, dim_x, signal, rng)
            X_miss = inject_missingness_calibration(X, mechanism, miss_rate, rng)
            data = np.column_stack([Z, Y, X_miss])
            z_idx = 0
            y_idx = 1
            cond_set = tuple(range(2, 2 + dim_x))
            # One test seed per M, drawn from the main stream after the
            # data: identical data, distinct seeds.
            m_seeds = [rng.randint(0, 2**31) for _ in Ms]

            for M, m_seed in zip(Ms, m_seeds):
                m_rng = np.random.RandomState(m_seed)
                try:
                    stat, pval = run_single_test(data, z_idx, y_idx, cond_set,
                                                 M, K, m_rng)
                    all_results.append(dict(
                        dgp=dgp, n=n, dim_x=dim_x, signal=signal,
                        mechanism=mechanism, M=M, K=K,
                        statistic=stat, p_value=pval,
                        rejected=pval < alpha, failed=False, rep=rep,
                    ))
                except Exception:
                    all_results.append(dict(
                        dgp=dgp, n=n, dim_x=dim_x, signal=signal,
                        mechanism=mechanism, M=M, K=K,
                        statistic=np.nan, p_value=np.nan,
                        rejected=False, failed=True, rep=rep,
                    ))

    return pd.DataFrame(all_results)


def analyze(df, alpha=0.05):
    df = df[~df["failed"]].copy()

    print("\n" + "=" * 70)
    print("M-SENSITIVITY RESULTS")
    print("=" * 70)

    # FPR at signal=0
    null = df[df["signal"] == 0]
    print("\n--- FPR under null (signal=0, target: 5%) ---\n")
    fpr = null.groupby(["M", "mechanism"])["rejected"].mean().unstack("mechanism")
    col_order = [c for c in ["complete", "mar", "mnar"] if c in fpr.columns]
    print(fpr[col_order].round(3).to_string())

    # Power at each signal level
    for sig in sorted(df[df["signal"] > 0]["signal"].unique()):
        alt = df[df["signal"] == sig]
        print(f"\n--- Power at signal={sig} ---\n")
        pwr = alt.groupby(["M", "mechanism"])["rejected"].mean().unstack("mechanism")
        col_order = [c for c in ["complete", "mar", "mnar"] if c in pwr.columns]
        print(pwr[col_order].round(3).to_string())

    # Breakdown by DGP at signal=0.3 (the weak-signal case)
    weak = df[df["signal"] == 0.3]
    if len(weak) > 0:
        print("\n--- Power at signal=0.3 by DGP ---\n")
        for dgp in sorted(weak["dgp"].unique()):
            sub = weak[weak["dgp"] == dgp]
            print(f"\n{dgp}:")
            pwr = sub.groupby(["M", "mechanism"])["rejected"].mean().unstack("mechanism")
            col_order = [c for c in ["complete", "mar", "mnar"] if c in pwr.columns]
            print(pwr[col_order].round(3).to_string())

    # Breakdown by sample size at signal=0.3
    if len(weak) > 0:
        print("\n--- Power at signal=0.3 by sample size ---\n")
        for n in sorted(weak["n"].unique()):
            sub = weak[weak["n"] == n]
            print(f"\nn={n}:")
            pwr = sub.groupby(["M", "mechanism"])["rejected"].mean().unstack("mechanism")
            col_order = [c for c in ["complete", "mar", "mnar"] if c in pwr.columns]
            print(pwr[col_order].round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M-sensitivity: power recovery via more imputations")
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--Ms", nargs="+", type=int, default=[3, 5, 10, 20])
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--dgps", nargs="+",
                        default=["linear_gaussian", "post_nonlinear"])
    parser.add_argument("--sample_sizes", nargs="+", type=int,
                        default=[500, 1000])
    parser.add_argument("--dim_x", type=int, default=5)
    parser.add_argument("--signals", nargs="+", type=float,
                        default=[0, 0.3, 0.6, 1.0])
    parser.add_argument("--mechanisms", nargs="+",
                        default=["complete", "mar", "mnar"])
    parser.add_argument("--miss_rate", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/m_sensitivity_results.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("M-SENSITIVITY EXPERIMENT")
    print("=" * 70)
    print(f"Output: {args.output}\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    t0 = time.time()
    df = run_experiment(args)
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    analyze(df, args.alpha)
