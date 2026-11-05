"""Sensitivity of PAIR-CI graph recovery to M, K, trees and leaf size.

The p = 10 Erdos-Renyi setting (edge probability 0.25, nonlinear edges,
n = 1000, three incomplete non-root variables at 30% MAR; 10 graphs x 20
datasets as run). PAIR-CI's fast variant (ExtraTrees, early stopping) is
run on each dataset under 11 configurations that vary one hyperparameter
at a time around M = 5, K = 5, 100 trees, min_samples_leaf = 5: M in
{3, 5, 10}, K in {3, 5, 10}, trees in {20, 50, 100, 200}, leaf in
{1, 5, 10, 20}. Test-wise deletion and FZ-vote run on the same data as
references. Estimands are median SHD, skeleton F1 and runtime per
configuration.

    python experiments/sensitivity.py --n_graphs 10 --n_datasets 20 \
        --output results/sensitivity_results.csv

Feeds App. E.1 and App. T.2.
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

from ci_oracle_fast import FastCITest
from pc_integration import (
    run_pc_complete_case, run_pc_impute_then_test,
    run_pc_testwise_deletion, structural_hamming_distance,
    edge_metrics, dag_to_adjacency, _pc_skeleton, _orient_edges,
    PCResult,
)
from dgps import (
    generate_erdos_renyi_dag, inject_missingness,
    sample_from_dag_nonlinear,
)


def run_pc_with_config(data, alpha, M, K, n_trees, min_leaf, random_state):
    t0 = time.time()
    p = data.shape[1]
    ci = FastCITest(
        data, n_imputations=M, n_folds=K, n_trees=n_trees,
        min_samples_leaf=min_leaf, early_stop=True,
        random_state=random_state,
    )
    adj, sep, nt = _pc_skeleton(p, ci, alpha, None, False)
    adj = _orient_edges(adj, sep, p, False)
    return PCResult(adj, sep, nt, time.time() - t0, "ci_oracle_fast")


# One-at-a-time grid around (M=5, K=5, n_trees=100, min_samples_leaf=5),
# deduplicated to 11 configurations. The pivot is the fast variant used at
# p = 20 and 30; the main p = 10 runs use the general variant (K = 10).
CONFIGS = list(set([
    (3, 5, 100, 5), (5, 5, 100, 5), (10, 5, 100, 5),                    # M
    (5, 3, 100, 5), (5, 5, 100, 5), (5, 10, 100, 5),                    # K
    (5, 5, 20, 5), (5, 5, 50, 5), (5, 5, 100, 5), (5, 5, 200, 5),       # trees
    (5, 5, 100, 1), (5, 5, 100, 5), (5, 5, 100, 10), (5, 5, 100, 20),   # leaf
]))
CONFIGS.sort()


def run_sensitivity(n_graphs=5, n_datasets=10, n_obs=1000, p=10,
                    edge_prob=0.25, miss_rate=0.3, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    all_results = []

    for g in range(n_graphs):
        print(f"\nGraph {g+1}/{n_graphs}")
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)

        has_parent = [j for j in range(p) if any(True for (i, c) in edges if c == j)]
        if len(has_parent) >= 3:
            miss_cols = sorted(rng.choice(has_parent, 3, replace=False))
        else:
            miss_cols = sorted(rng.choice(range(p), 3, replace=False))

        for d in range(n_datasets):
            data = sample_from_dag_nonlinear(edges, p, n_obs, rng)
            data_miss = inject_missingness(data, "mar", miss_cols, miss_rate, rng)
            actual_miss = np.mean(np.isnan(data_miss))
            # One seed per dataset, shared by every configuration and both
            # baselines.
            base_seed = rng.randint(0, 2**31)

            for ci, (M, K, nt, ml) in enumerate(CONFIGS):
                config_name = f"M={M}_K={K}_trees={nt}_leaf={ml}"
                try:
                    result = run_pc_with_config(
                        data_miss, alpha, M, K, nt, ml, base_seed)
                    shd = structural_hamming_distance(adj_true, result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append(dict(
                        graph=g, dataset=d, config=config_name,
                        M=M, K=K, n_trees=nt, min_leaf=ml,
                        shd=shd, precision=em["precision"],
                        recall=em["recall"], f1=em["f1"],
                        runtime=result.runtime_seconds,
                        actual_miss_rate=actual_miss, failed=False))
                except Exception:
                    all_results.append(dict(
                        graph=g, dataset=d, config=config_name,
                        M=M, K=K, n_trees=nt, min_leaf=ml,
                        shd=np.nan, precision=np.nan,
                        recall=np.nan, f1=np.nan, runtime=np.nan,
                        actual_miss_rate=actual_miss, failed=True))

            for name, fn, kw in [
                ("testwise", run_pc_testwise_deletion, dict(alpha=alpha)),
                ("impute_fisherz", run_pc_impute_then_test,
                 dict(alpha=alpha, ci_method="fisherz", n_imputations=5,
                      random_state=base_seed)),
            ]:
                try:
                    result = fn(data_miss, **kw)
                    shd = structural_hamming_distance(adj_true, result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append(dict(
                        graph=g, dataset=d, config=name,
                        M=0, K=0, n_trees=0, min_leaf=0,
                        shd=shd, precision=em["precision"],
                        recall=em["recall"], f1=em["f1"],
                        runtime=result.runtime_seconds,
                        actual_miss_rate=actual_miss, failed=False))
                except Exception:
                    all_results.append(dict(
                        graph=g, dataset=d, config=name,
                        M=0, K=0, n_trees=0, min_leaf=0,
                        shd=np.nan, precision=np.nan,
                        recall=np.nan, f1=np.nan, runtime=np.nan,
                        actual_miss_rate=actual_miss, failed=True))

        print(f"  {n_datasets} datasets x {len(CONFIGS)} configs done")

    return pd.DataFrame(all_results)


def analyze(df):
    df_ok = df[~df["failed"]]
    our = df_ok[df_ok["M"] > 0]
    baselines = df_ok[df_ok["M"] == 0]

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    print("\nBaselines (for reference):")
    print(baselines.groupby("config")["shd"].agg(["median", "mean"]).round(1))

    print("\n--- Vary M (K=5, trees=100, leaf=5) ---")
    sub = our[(our["K"] == 5) & (our["n_trees"] == 100) & (our["min_leaf"] == 5)]
    print(sub.groupby("M")["shd"].agg(["median", "mean", "std"]).round(1))

    print("\n--- Vary K (M=5, trees=100, leaf=5) ---")
    sub = our[(our["M"] == 5) & (our["n_trees"] == 100) & (our["min_leaf"] == 5)]
    print(sub.groupby("K")["shd"].agg(["median", "mean", "std"]).round(1))

    print("\n--- Vary n_trees (M=5, K=5, leaf=5) ---")
    sub = our[(our["M"] == 5) & (our["K"] == 5) & (our["min_leaf"] == 5)]
    print(sub.groupby("n_trees")["shd"].agg(["median", "mean", "std"]).round(1))

    print("\n--- Vary min_leaf (M=5, K=5, trees=100) ---")
    sub = our[(our["M"] == 5) & (our["K"] == 5) & (our["n_trees"] == 100)]
    print(sub.groupby("min_leaf")["shd"].agg(["median", "mean", "std"]).round(1))

    print("\n--- Runtime by config (seconds, median) ---")
    print(our.groupby("config")["runtime"].median().sort_values().round(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, default=5)
    parser.add_argument("--n_datasets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/sensitivity_results.csv")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_graphs = 2
        args.n_datasets = 3

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Sensitivity: {args.n_graphs} graphs x {args.n_datasets} datasets "
          f"x {len(CONFIGS)} configs")
    print(f"Configs: {CONFIGS}\n")

    t0 = time.time()
    df = run_sensitivity(args.n_graphs, args.n_datasets, seed=args.seed)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    analyze(df)
