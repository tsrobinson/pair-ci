"""Calibration of the early-stopping rule in PAIR-CI's fast variant.

Runs the fast variant (ExtraTrees, M = 5, K = 5, 100 trees) on the same
null datasets with and without early stopping, which skips the remaining
imputations once |t| > 4 after two. DGPs are linear Gaussian and
post-nonlinear with signal 0, n = 1000, |X| = 5, under complete data, MAR
and MNAR (30% per column), 100 replicates each. The estimands are the false
positive rate of each rule at alpha = 0.05 and their decision agreement.

    python experiments/early_stopping_calibration.py --n_reps 100 \
        --output results/early_stopping_calibration.csv

Feeds App. P.
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import time
import argparse

from ci_oracle_fast import FastCITest
from dgps import dgp_linear_gaussian, dgp_post_nonlinear, inject_missingness_calibration


def run_comparison(n_reps=100, n_obs=1000, dim_x=5, alpha=0.05, seed=42):
    """FPR of the fast variant with and without early stopping, under H0."""
    rng = np.random.RandomState(seed)
    results = []

    for dgp_name, dgp_fn in [("linear_gaussian", dgp_linear_gaussian),
                               ("post_nonlinear", dgp_post_nonlinear)]:
        for mechanism in ["complete", "mar", "mnar"]:
            for rep in range(n_reps):
                X, Y, Z = dgp_fn(n_obs, dim_x, 0.0, rng)  # signal=0 (null)
                data = np.column_stack([Z, Y, X])

                if mechanism != "complete":
                    X_miss = inject_missingness_calibration(
                        data[:, 2:], mechanism, 0.3, rng)
                    data[:, 2:] = X_miss

                # Both runs share this seed, so they differ only in the
                # stopping rule.
                rs = rng.randint(0, 2**31)
                cond = tuple(range(2, 2 + dim_x))

                # Default: stop after 2 imputations if |t| > 4.
                ci_es = FastCITest(data, n_imputations=5, n_folds=5, n_trees=100,
                                   early_stop=True, early_stop_after=2,
                                   early_stop_threshold=4.0, random_state=rs)
                t_es, p_es = ci_es.test(0, 1, cond)

                ci_no = FastCITest(data, n_imputations=5, n_folds=5, n_trees=100,
                                   early_stop=False, random_state=rs)
                t_no, p_no = ci_no.test(0, 1, cond)

                results.append({
                    "dgp": dgp_name, "mechanism": mechanism, "rep": rep,
                    "t_early_stop": t_es, "p_early_stop": p_es,
                    "reject_early_stop": p_es < alpha,
                    "t_no_early_stop": t_no, "p_no_early_stop": p_no,
                    "reject_no_early_stop": p_no < alpha,
                })

            n_done = len([r for r in results if r["dgp"] == dgp_name
                         and r["mechanism"] == mechanism])
            print(f"  {dgp_name} {mechanism}: {n_done} reps done")

    return pd.DataFrame(results)


def analyze(df):
    print("\n" + "=" * 60)
    print("EARLY STOPPING CALIBRATION CHECK (signal=0, target=0.05)")
    print("=" * 60)

    print("\nFPR with early stopping:")
    print(df.groupby(["dgp", "mechanism"])["reject_early_stop"]
          .mean().unstack("mechanism").round(4).to_string())

    print("\nFPR without early stopping:")
    print(df.groupby(["dgp", "mechanism"])["reject_no_early_stop"]
          .mean().unstack("mechanism").round(4).to_string())

    print("\nDifference (early_stop - no_early_stop):")
    es = df.groupby(["dgp", "mechanism"])["reject_early_stop"].mean()
    no = df.groupby(["dgp", "mechanism"])["reject_no_early_stop"].mean()
    print((es - no).unstack("mechanism").round(4).to_string())

    print("\nDecision agreement:")
    df["agree"] = df["reject_early_stop"] == df["reject_no_early_stop"]
    print(df.groupby(["dgp", "mechanism"])["agree"]
          .mean().unstack("mechanism").round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/early_stopping_calibration.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("Early stopping calibration check\n")
    t0 = time.time()
    df = run_comparison(n_reps=args.n_reps, seed=args.seed)
    print(f"\nTime: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    analyze(df)
