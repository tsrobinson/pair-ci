"""Graph recovery on Erdos-Renyi DAGs at p = 10 (general PAIR-CI variant).

Ten random DAGs (p = 10, edge probability 0.25) with linear Gaussian or
nonlinear edges; three non-root variables are incomplete at a 30% per-column
rate under one of MAR, MNAR or mixed, or the data are complete; n = 1000
and 20 datasets per graph. PAIR-CI (general variant: M = 5, K = 10, cached
imputation; random forests on nonlinear edges, a ridge learner on linear
ones) is compared with complete-case PC, test-wise deletion and FZ-vote.
Estimands are SHD and skeleton precision, recall and F1, summarised as
medians. One job per mechanism and DGP:

    python experiments/graph_recovery_p10.py --mechanism mnar \
        --dgp nonlinear --n_graphs 10 --n_datasets 20 \
        --output results/p10_nonlinear_mnar.csv

analysis/combine_p10.py merges the jobs into results/p10_{dgp}_results.csv,
which feeds Table 2 and App. F.
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
    sample_from_dag, sample_from_dag_nonlinear,
)
from ci_oracle_linear import LinearLearnerCITest


def run_all_methods(data_miss, alpha, n_imputations, n_folds, random_state,
                    dgp="nonlinear"):
    results = {}
    rng = np.random.RandomState(random_state)
    # Ridge learner on linear edges, random forests on nonlinear ones.
    ci_cls = LinearLearnerCITest if dgp == "linear" else None
    methods = [
        ("ci_oracle_cached", run_pc_with_ci_oracle,
         dict(alpha=alpha, n_imputations=n_imputations, n_folds=n_folds,
              imputation_strategy="cached", random_state=rng.randint(0, 2**31),
              ci_test_cls=ci_cls)),
        ("complete_case", run_pc_complete_case, dict(alpha=alpha)),
        ("impute_fisherz", run_pc_impute_then_test,
         dict(alpha=alpha, n_imputations=n_imputations,
              random_state=rng.randint(0, 2**31))),
        ("testwise", run_pc_testwise_deletion, dict(alpha=alpha)),
    ]
    for name, fn, kw in methods:
        try:
            results[name] = fn(data_miss, **kw)
        except Exception as e:
            print(f"    {name} failed: {e}")
            results[name] = None
    return results


def run_single_mechanism(mechanism, dgp, n_graphs=10, n_datasets=20,
                         n_obs=1000, p=10, edge_prob=0.25, miss_rate=0.3,
                         n_miss_cols=3, alpha=0.05, n_imputations=5,
                         n_folds=10, seed=42):
    rng = np.random.RandomState(seed)
    all_results = []

    sample_fn = sample_from_dag if dgp == "linear" else sample_from_dag_nonlinear

    for g in range(n_graphs):
        print(f"Graph {g+1}/{n_graphs}, mechanism={mechanism}, dgp={dgp}")
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)

        has_parent = [j for j in range(p) if any(True for (i, c) in edges if c == j)]
        if len(has_parent) >= n_miss_cols:
            miss_cols = sorted(rng.choice(has_parent, n_miss_cols, replace=False))
        else:
            miss_cols = sorted(rng.choice(range(p), n_miss_cols, replace=False))

        # Draw the datasets of all four mechanisms and run only `mechanism`,
        # so that the per-mechanism jobs share graphs and the RNG stream
        # (new_baselines.py and mvpc_baseline.py replay it).
        for mech in ["mar", "mnar", "mixed", "complete"]:
            for d in range(n_datasets):
                data = sample_fn(edges, p, n_obs, rng)
                data_miss = inject_missingness(data, mech, miss_cols, miss_rate, rng)
                actual_miss = np.mean(np.isnan(data_miss))
                method_seed = rng.randint(0, 2**31)

                if mech != mechanism:
                    continue

                mr = run_all_methods(data_miss, alpha, n_imputations, n_folds,
                                     method_seed, dgp=dgp)

                for mname, result in mr.items():
                    if result is None:
                        all_results.append({
                            "graph": g, "dataset": d, "mechanism": mech,
                            "dgp": dgp, "method": mname, "p": p,
                            "n_obs": n_obs, "shd": np.nan,
                            "precision": np.nan, "recall": np.nan,
                            "f1": np.nan, "tp": np.nan, "fp": np.nan,
                            "fn": np.nan, "runtime": np.nan,
                            "n_ci_tests": np.nan,
                            "n_true_edges": len(edges),
                            "actual_miss_rate": actual_miss, "failed": True,
                        })
                        continue
                    shd = structural_hamming_distance(adj_true, result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append({
                        "graph": g, "dataset": d, "mechanism": mech,
                        "dgp": dgp, "method": mname, "p": p,
                        "n_obs": n_obs, "shd": shd,
                        "precision": em["precision"], "recall": em["recall"],
                        "f1": em["f1"], "tp": em["tp"], "fp": em["fp"],
                        "fn": em["fn"], "runtime": result.runtime_seconds,
                        "n_ci_tests": result.n_ci_tests,
                        "n_true_edges": len(edges),
                        "actual_miss_rate": actual_miss, "failed": False,
                    })

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="p=10 graph recovery: single mechanism + DGP runner")
    parser.add_argument("--mechanism", type=str, required=True,
                        choices=["mar", "mnar", "mixed", "complete"])
    parser.add_argument("--dgp", type=str, required=True,
                        choices=["linear", "nonlinear"])
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--n_obs", type=int, default=1000)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--edge_prob", type=float, default=0.25)
    parser.add_argument("--miss_rate", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_imputations", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"p=10 kill-switch: dgp={args.dgp}, mechanism={args.mechanism}, "
          f"{args.n_graphs} graphs x {args.n_datasets} datasets")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.time()
    df = run_single_mechanism(
        mechanism=args.mechanism, dgp=args.dgp,
        n_graphs=args.n_graphs, n_datasets=args.n_datasets,
        n_obs=args.n_obs, p=args.p, edge_prob=args.edge_prob,
        miss_rate=args.miss_rate, n_miss_cols=3,
        alpha=args.alpha, n_imputations=args.n_imputations,
        n_folds=args.n_folds, seed=args.seed,
    )
    print(f"\nTime: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
