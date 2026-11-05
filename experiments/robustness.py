"""Graph recovery at p = 10 across missingness rates and sample sizes.

The p = 10 Erdos-Renyi setting of graph_recovery_p10.py with nonlinear
edges (10 graphs, edge probability 0.25, three incomplete non-root
variables, 20 datasets per graph), swept over per-column rates of 10%, 30%
and 50%, n in {500, 1000, 2000} and MAR or MNAR. PAIR-CI (general variant,
M = 5, K = 10, cached imputation) is compared with complete-case PC,
test-wise deletion and FZ-vote; estimands are SHD and skeleton precision,
recall and F1. One job per (mechanism, rate, n):

    python experiments/robustness.py --mechanism mar --miss_rate 0.3 \
        --n_obs 1000 --n_graphs 10 --n_datasets 20 \
        --output results/robustness_mar_0.3_1000.csv

analysis/combine_robustness.py merges the jobs into
results/robustness_results.csv, which feeds App. F (Table robustness_mar).
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import time
import argparse

from pc_integration import (
    run_pc_with_ci_oracle, run_pc_complete_case, run_pc_impute_then_test,
    run_pc_testwise_deletion, structural_hamming_distance, edge_metrics,
    dag_to_adjacency,
)
from dgps import (
    generate_erdos_renyi_dag, inject_missingness,
    sample_from_dag_nonlinear,
)


MISS_RATES = [0.1, 0.3, 0.5]
N_OBS_GRID = [500, 1000, 2000]
MECHANISMS = ["mar", "mnar"]


def run_all_methods(data_miss, alpha, n_imputations, n_folds, random_state):
    results = {}
    rng = np.random.RandomState(random_state)
    for name, fn, kw in [
        ("ci_oracle_cached", run_pc_with_ci_oracle,
         dict(alpha=alpha, n_imputations=n_imputations, n_folds=n_folds,
              imputation_strategy="cached", random_state=rng.randint(0, 2**31))),
        ("complete_case", run_pc_complete_case, dict(alpha=alpha)),
        ("impute_fisherz", run_pc_impute_then_test,
         dict(alpha=alpha, n_imputations=n_imputations,
              random_state=rng.randint(0, 2**31))),
        ("testwise", run_pc_testwise_deletion, dict(alpha=alpha)),
    ]:
        try:
            results[name] = fn(data_miss, **kw)
        except Exception as e:
            print(f"    {name} failed: {e}")
            results[name] = None
    return results


def run_robustness(mechanism, miss_rate, n_obs, n_graphs=10, n_datasets=20,
                   p=10, edge_prob=0.25, n_miss_cols=3, alpha=0.05,
                   n_imputations=5, n_folds=5, seed=42):
    """Graph recovery at p = 10 for one mechanism, rate and sample size."""
    rng = np.random.RandomState(seed)
    all_results = []

    for g in range(n_graphs):
        print(f"Graph {g+1}/{n_graphs}, mechanism={mechanism}, "
              f"miss_rate={miss_rate}, n_obs={n_obs}")
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)

        has_parent = [j for j in range(p) if any(True for (i, c) in edges if c == j)]
        if len(has_parent) >= n_miss_cols:
            miss_cols = sorted(rng.choice(has_parent, n_miss_cols, replace=False))
        else:
            miss_cols = sorted(rng.choice(range(p), n_miss_cols, replace=False))

        # Draw the datasets of every (mechanism, rate, n) and run only the
        # requested one, so that all 18 jobs share graphs and the RNG
        # stream.
        for mech in MECHANISMS:
            for mr in MISS_RATES:
                for no in N_OBS_GRID:
                    for d in range(n_datasets):
                        data = sample_from_dag_nonlinear(edges, p, no, rng)
                        data_miss = inject_missingness(
                            data, mech, miss_cols, mr, rng)
                        actual_miss = np.mean(np.isnan(data_miss))
                        method_seed = rng.randint(0, 2**31)

                        if mech != mechanism or mr != miss_rate or no != n_obs:
                            continue

                        results = run_all_methods(
                            data_miss, alpha, n_imputations, n_folds,
                            method_seed)

                        for mname, result in results.items():
                            if result is None:
                                all_results.append({
                                    "graph": g, "dataset": d,
                                    "mechanism": mech, "method": mname,
                                    "n_obs": no, "miss_rate": mr,
                                    "shd": np.nan, "precision": np.nan,
                                    "recall": np.nan, "f1": np.nan,
                                    "runtime": np.nan, "n_ci_tests": np.nan,
                                    "n_true_edges": len(edges),
                                    "actual_miss_rate": actual_miss,
                                    "failed": True,
                                })
                                continue
                            shd = structural_hamming_distance(
                                adj_true, result.adjacency_matrix)
                            em = edge_metrics(adj_true, result.adjacency_matrix)
                            all_results.append({
                                "graph": g, "dataset": d,
                                "mechanism": mech, "method": mname,
                                "n_obs": no, "miss_rate": mr,
                                "shd": shd,
                                "precision": em["precision"],
                                "recall": em["recall"],
                                "f1": em["f1"],
                                "runtime": result.runtime_seconds,
                                "n_ci_tests": result.n_ci_tests,
                                "n_true_edges": len(edges),
                                "actual_miss_rate": actual_miss,
                                "failed": False,
                            })

    return pd.DataFrame(all_results)


def analyze(csv_path):
    """Print median SHD grouped by method."""
    df = pd.read_csv(csv_path)
    print("\n=== Median SHD by method ===")
    summary = (df[~df["failed"]]
               .groupby("method")["shd"]
               .median()
               .sort_values())
    print(summary.to_string())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robustness: p=10 graph recovery across miss_rates and sample sizes")
    parser.add_argument("--miss_rate", type=float, required=True,
                        choices=MISS_RATES,
                        help="Missingness rate (0.1, 0.3, or 0.5)")
    parser.add_argument("--n_obs", type=int, required=True,
                        choices=N_OBS_GRID,
                        help="Sample size (500, 1000, or 2000)")
    parser.add_argument("--mechanism", type=str, required=True,
                        choices=MECHANISMS,
                        help="Missingness mechanism (mar or mnar)")
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Robustness: mechanism={args.mechanism}, miss_rate={args.miss_rate}, "
          f"n_obs={args.n_obs}, {args.n_graphs} graphs x {args.n_datasets} datasets")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.time()
    df = run_robustness(
        mechanism=args.mechanism, miss_rate=args.miss_rate,
        n_obs=args.n_obs, n_graphs=args.n_graphs,
        n_datasets=args.n_datasets, p=10, edge_prob=0.25,
        n_miss_cols=3, alpha=0.05, n_imputations=5,
        n_folds=10, seed=args.seed,
    )
    elapsed = (time.time() - t0) / 60
    print(f"\nTime: {elapsed:.1f} min")

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    analyze(args.output)
