"""Cached versus per-query imputation at p = 5, 10 and 20.

For each p, 20 datasets, each from a fresh Erdos-Renyi DAG (edge
probability 0.4, 0.25 and 0.2) with nonlinear edges and n = 1000;
max(2, p // 3) non-root variables are masked at a 30% per-column rate under
MAR. PAIR-CI (general variant, M = 5, K = 5) tests every pair (i, j)
marginally and given one other variable, once with cached imputation and
once with per-query re-imputation from the same seed. The estimands are the
per-dataset agreement of the two decisions at alpha = 0.05 and the median
|p_cached - p_perquery|.

    python experiments/caching_validation.py \
        --output results/caching_validation_large_p.csv

Feeds the agreement rates in Section 3.2 and App. C.
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

from ci_oracle import GeneralCITest
from dgps import (
    generate_erdos_renyi_dag, inject_missingness,
    sample_from_dag_nonlinear,
)


def run_caching_comparison(p, n_datasets=20, n_obs=1000, edge_prob=None,
                           miss_rate=0.3, alpha=0.05, seed=42):
    if edge_prob is None:
        edge_prob = {5: 0.4, 10: 0.25, 20: 0.2}.get(p, 0.2)

    n_miss_cols = max(2, p // 3)
    rng = np.random.RandomState(seed)
    all_results = []

    for d in range(n_datasets):
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        data = sample_from_dag_nonlinear(edges, p, n_obs, rng)

        has_parent = [j for j in range(p)
                      if any(True for (i, c) in edges if c == j)]
        if len(has_parent) >= n_miss_cols:
            miss_cols = sorted(rng.choice(has_parent, n_miss_cols, replace=False))
        else:
            miss_cols = sorted(rng.choice(range(p), n_miss_cols, replace=False))
        data_miss = inject_missingness(data, "mar", miss_cols, miss_rate, rng)

        # Same seed for both strategies, so they differ only in where
        # imputation happens.
        rs = rng.randint(0, 2**31)

        ci_cached = GeneralCITest(
            data_miss, n_imputations=5, n_folds=5,
            imputation_strategy="cached", random_state=rs)
        ci_perquery = GeneralCITest(
            data_miss, n_imputations=5, n_folds=5,
            imputation_strategy="per_query", random_state=rs)

        n_tests = 0
        n_agree = 0
        p_diffs = []

        for i in range(p):
            for j in range(i + 1, p):
                # The conditioning column must differ from both endpoints,
                # or the full model would see a copy of y among its features.
                valid_cond_cols = [c for c in range(p) if c != i and c != j]
                cond_singleton = (valid_cond_cols[(i + j) % len(valid_cond_cols)],)
                for cond in [(), cond_singleton]:
                    try:
                        _, p_cached = ci_cached.test(i, j, cond)
                        _, p_perquery = ci_perquery.test(i, j, cond)

                        n_tests += 1
                        if (p_cached < alpha) == (p_perquery < alpha):
                            n_agree += 1
                        p_diffs.append(abs(p_cached - p_perquery))
                    except Exception:
                        continue

        agreement = n_agree / n_tests if n_tests > 0 else np.nan
        median_diff = np.median(p_diffs) if p_diffs else np.nan

        all_results.append({
            "p": p, "dataset": d, "n_tests": n_tests,
            "n_agree": n_agree, "agreement": agreement,
            "median_p_diff": median_diff,
            "mean_p_diff": np.mean(p_diffs) if p_diffs else np.nan,
            "max_p_diff": np.max(p_diffs) if p_diffs else np.nan,
        })

        print(f"  p={p}, dataset {d+1}/{n_datasets}: "
              f"{n_tests} tests, {agreement:.1%} agreement, "
              f"median |dp| = {median_diff:.4f}")

    return pd.DataFrame(all_results)


def analyze(df):
    print("\n" + "=" * 60)
    print("CACHING VALIDATION SUMMARY")
    print("=" * 60)
    for p_val in sorted(df["p"].unique()):
        sub = df[df["p"] == p_val]
        print(f"\np = {p_val}:")
        print(f"  Agreement: {sub['agreement'].mean():.1%} "
              f"(range: {sub['agreement'].min():.1%}-{sub['agreement'].max():.1%})")
        print(f"  Median |dp|: {sub['median_p_diff'].mean():.4f}")
        print(f"  Mean |dp|: {sub['mean_p_diff'].mean():.4f}")
        print(f"  Max |dp|: {sub['max_p_diff'].max():.4f}")
        print(f"  Total tests: {sub['n_tests'].sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/caching_validation_large_p.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Caching Validation at Larger p\n")
    all_dfs = []
    t0 = time.time()
    for p in [5, 10, 20]:
        print(f"\n--- p = {p} ---")
        df = run_caching_comparison(p=p, n_datasets=args.n_datasets, seed=args.seed)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print(f"Saved to {args.output}")
    analyze(combined)
