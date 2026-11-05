"""Cached versus per-query imputation on two five-variable linear DAGs.

A small check of imputation caching: a chain and a diamond DAG (n = 1000),
columns 1 and 2 masked at a 30% rate under MAR or MNAR, 20 datasets per
(graph, mechanism). PAIR-CI (general variant, M = 5, K = 5) runs each known
CI and CD query with cached imputation (strategy A) and with per-query
re-imputation (strategy B) from the same seed; the estimands are the
agreement of the two decisions at alpha = 0.05 and |p_A - p_B|.

    python experiments/caching_benchmark.py \
        --output results/caching_benchmark_results.csv

No analysis script reads the output; the agreement rates in Section 3.2 and
App. C come from caching_validation.py.
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import time
import argparse

from ci_oracle import GeneralCITest


def generate_linear_dag(n, rng, graph_type="chain"):
    """Five-variable linear Gaussian DAG with known CI and CD queries.

    "chain" is X0 -> X1 -> X2 -> X3 -> X4; "fork" is the diamond
    X0 -> {X1, X2} -> X3 -> X4. Returns (data, true_ci, true_cd), the last
    two as lists of (x, y, cond_set).
    """
    if graph_type == "chain":
        X0 = rng.randn(n)
        X1 = 0.8 * X0 + rng.randn(n) * 0.5
        X2 = 0.7 * X1 + rng.randn(n) * 0.5
        X3 = 0.6 * X2 + rng.randn(n) * 0.5
        X4 = 0.9 * X3 + rng.randn(n) * 0.5
        data = np.column_stack([X0, X1, X2, X3, X4])
        true_ci = [(0, 2, (1,)), (0, 3, (1,)), (0, 3, (2,)), (0, 4, (1,)),
                   (1, 3, (2,)), (1, 4, (2,)), (2, 4, (3,))]
        true_cd = [(0, 1, ()), (1, 2, ()), (2, 3, ()), (3, 4, ()), (0, 2, ())]
    elif graph_type == "fork":
        X0 = rng.randn(n)
        X1 = 0.8 * X0 + rng.randn(n) * 0.5
        X2 = 0.7 * X0 + rng.randn(n) * 0.5
        X3 = 0.6 * X1 + 0.5 * X2 + rng.randn(n) * 0.5
        X4 = 0.9 * X3 + rng.randn(n) * 0.5
        data = np.column_stack([X0, X1, X2, X3, X4])
        true_ci = [(1, 2, (0,)), (0, 3, (1, 2)), (0, 4, (1, 2)),
                   (1, 4, (3,)), (2, 4, (3,))]
        true_cd = [(0, 1, ()), (0, 2, ()), (1, 3, ()), (2, 3, ()),
                   (3, 4, ()), (1, 2, (3,))]
    return data, true_ci, true_cd


def inject_missingness_local(data, mechanism, rate, rng):
    """Mask columns 1 and 2: MAR driven by column 0, or MNAR self-masking."""
    data = data.copy()
    n = data.shape[0]
    for col in [1, 2]:
        if mechanism == "mar":
            logit = data[:, 0] - np.quantile(data[:, 0], 1 - rate)
            prob = 1 / (1 + np.exp(-3 * logit))
        elif mechanism == "mnar":
            logit = data[:, col] - np.quantile(data[:, col], 1 - rate)
            prob = 1 / (1 + np.exp(-3 * logit))
        else:
            raise ValueError(f"Unknown: {mechanism}")
        data[rng.rand(n) < prob, col] = np.nan
    return data


def run_benchmark(n_datasets=20, n_obs=1000, miss_rate=0.3,
                  n_imputations=5, n_folds=5, alpha=0.05):
    results = []
    rng = np.random.RandomState(123)
    configs = [("chain", "mar"), ("chain", "mnar"),
               ("fork", "mar"), ("fork", "mnar")]

    for gt, mm in configs:
        print(f"\n{'='*60}\nGraph: {gt}, Missingness: {mm}\n{'='*60}")
        for d in range(n_datasets):
            data, tci, tcd = generate_linear_dag(n_obs, rng, gt)
            data_miss = inject_missingness_local(data, mm, miss_rate, rng)
            # Both strategies get the same seed, so they differ only in
            # where imputation happens.
            rs = rng.randint(0, 2**31)
            ci_a = GeneralCITest(data_miss, n_imputations=n_imputations,
                                 n_folds=n_folds, imputation_strategy="cached",
                                 random_state=rs)
            ci_b = GeneralCITest(data_miss, n_imputations=n_imputations,
                                 n_folds=n_folds, imputation_strategy="per_query",
                                 random_state=rs)
            all_tests = ([(x, y, s, True) for x, y, s in tci]
                         + [(x, y, s, False) for x, y, s in tcd])
            for x, y, s, is_ci in all_tests:
                t0 = time.time()
                sa, pa = ci_a.test(x, y, s)
                ta = time.time() - t0
                t0 = time.time()
                sb, pb = ci_b.test(x, y, s)
                tb = time.time() - t0
                da = "reject" if pa < alpha else "accept"
                db = "reject" if pb < alpha else "accept"
                results.append({
                    "graph": gt, "mechanism": mm, "dataset": d,
                    "x": x, "y": y, "cond_set": str(s), "true_ci": is_ci,
                    "stat_a": sa, "pval_a": pa, "decision_a": da,
                    "stat_b": sb, "pval_b": pb, "decision_b": db,
                    "agree": da == db, "pval_diff": abs(pa - pb),
                    "time_a": ta, "time_b": tb,
                })
        df_t = pd.DataFrame([r for r in results
                             if r["graph"] == gt and r["mechanism"] == mm])
        print(f"  Agreement: {df_t['agree'].mean():.1%}")

    return pd.DataFrame(results)


def summarize_results(df):
    print(f"\n{'='*70}\nCACHING BENCHMARK SUMMARY\n{'='*70}")
    print(f"\nOverall agreement: {df['agree'].mean():.1%}")
    print(f"Mean |p-diff|: {df['pval_diff'].mean():.4f}")
    print(f"Max |p-diff|: {df['pval_diff'].max():.4f}")
    s = df.groupby(["graph", "mechanism"]).agg(
        agreement=("agree", "mean"), mean_pdiff=("pval_diff", "mean"),
        mean_time_a=("time_a", "mean"),
        mean_time_b=("time_b", "mean")).round(4)
    print(f"\n{s.to_string()}")
    agree = df["agree"].mean()
    print(f"\n{'='*70}\nRECOMMENDATION\n{'='*70}")
    if agree > 0.95:
        print("Strategy A (cached) is acceptable. Bias is negligible.")
    elif agree > 0.90:
        print("Strategy A shows minor discrepancies. Use B for main results.")
    else:
        print("Strategy A shows meaningful bias. Use Strategy B everywhere.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="results/caching_benchmark_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Running Imputation Caching Benchmark...\n")
    df = run_benchmark()
    summarize_results(df)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
