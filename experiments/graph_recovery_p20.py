"""Graph recovery on Erdos-Renyi DAGs at p = 20 (fast PAIR-CI variant).

Ten random DAGs (p = 20, edge probability 0.2) with linear Gaussian or
nonlinear edges; 6 non-root variables are incomplete at a 30% per-column
rate under one of MAR, MNAR or mixed, or the data are complete; n = 1000
and 20 datasets per graph. PAIR-CI (fast variant: M = 5, K = 5, 100 trees,
early stopping; ExtraTrees on nonlinear edges, a ridge learner on linear
ones) is compared with complete-case PC, test-wise deletion and FZ-vote.
Estimands are SHD and skeleton precision, recall and F1, summarised as
medians. One job per mechanism and DGP:

    python experiments/graph_recovery_p20.py --mechanism mnar \
        --dgp nonlinear --n_graphs 10 --n_datasets 20 \
        --output results/p20_nonlinear_mnar.csv

analysis/combine_p20.py merges the jobs into results/p20_full_results.csv,
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
import warnings
warnings.filterwarnings("ignore")

from pc_integration import (
    run_pc_with_ci_oracle, run_pc_complete_case, run_pc_impute_then_test,
    run_pc_testwise_deletion, structural_hamming_distance,
    edge_metrics, dag_to_adjacency, _pc_skeleton, _orient_edges,
    PCResult,
)
from dgps import (
    generate_erdos_renyi_dag, sample_from_dag, inject_missingness,
    sample_from_dag_nonlinear,
)
from ci_oracle_fast import FastCITest
from ci_oracle_linear import LinearLearnerCITest, LinearLearnerFastCITest


def run_pc_with_fast_ci(data, alpha=0.05, random_state=None, verbose=False,
                        ci_test_cls=None):
    t_start = time.time()
    p = data.shape[1]
    cls = ci_test_cls if ci_test_cls is not None else FastCITest
    ci_test = cls(
        data, n_imputations=5, n_folds=5, n_trees=100,
        early_stop=True, early_stop_after=2, early_stop_threshold=4.0,
        random_state=random_state,
    )
    adj, sep_sets, n_tests = _pc_skeleton(p, ci_test, alpha, None, verbose)
    adj_oriented = _orient_edges(adj, sep_sets, p, verbose)
    return PCResult(
        adjacency_matrix=adj_oriented, sep_sets=sep_sets,
        n_ci_tests=n_tests, runtime_seconds=time.time() - t_start,
        method="ci_oracle_fast",
    )


def run_all_methods_p20(data_miss, alpha, random_state, dgp="nonlinear"):
    results = {}
    rng = np.random.RandomState(random_state)
    # On linear edges, the fast harness (K = 5, early stopping) with
    # Ridge(alpha=1e-6) as learner: essentially the OLS residualisation
    # inside Fisher-Z. The general variant (K = 10) took about six days per
    # mechanism here, beyond the cluster's two-day limit.
    ci_cls = LinearLearnerFastCITest if dgp == "linear" else None
    methods = [
        ("ci_oracle_fast", run_pc_with_fast_ci,
         dict(alpha=alpha, random_state=rng.randint(0, 2**31),
              ci_test_cls=ci_cls)),
        ("complete_case", run_pc_complete_case, dict(alpha=alpha)),
        ("impute_fisherz", run_pc_impute_then_test,
         dict(alpha=alpha, n_imputations=5,
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
                         n_obs=1000, p=20, edge_prob=0.2, miss_rate=0.3,
                         n_miss_cols=6, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    all_results = []

    sample_fn = sample_from_dag if dgp == "linear" else sample_from_dag_nonlinear

    for g in range(n_graphs):
        t_graph = time.time()
        print(f"Graph {g+1}/{n_graphs}, mechanism={mechanism}, dgp={dgp}")

        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)
        n_true_edges = len(edges)
        print(f"  True edges: {n_true_edges}")

        has_parent = [j for j in range(p)
                      if any(True for (i, c) in edges if c == j)]
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
                data_miss = inject_missingness(data, mech, miss_cols,
                                               miss_rate, rng)
                actual_miss = np.mean(np.isnan(data_miss))
                method_seed = rng.randint(0, 2**31)

                if mech != mechanism:
                    continue

                print(f"  Dataset {d+1}/{n_datasets} "
                      f"(miss={actual_miss:.1%})...", end=" ", flush=True)
                t_ds = time.time()

                mr = run_all_methods_p20(data_miss, alpha, method_seed,
                                          dgp=dgp)

                for mname, result in mr.items():
                    if result is None:
                        all_results.append({
                            "graph": g, "dataset": d, "mechanism": mech,
                            "dgp": dgp, "method": mname, "p": p,
                            "n_obs": n_obs, "shd": np.nan,
                            "precision": np.nan, "recall": np.nan,
                            "f1": np.nan, "tp": np.nan, "fp": np.nan,
                            "fn": np.nan, "runtime": np.nan,
                            "n_true_edges": n_true_edges,
                            "n_ci_tests": np.nan,
                            "actual_miss_rate": actual_miss,
                            "failed": True,
                        })
                        continue
                    shd = structural_hamming_distance(
                        adj_true, result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append({
                        "graph": g, "dataset": d, "mechanism": mech,
                        "dgp": dgp, "method": mname, "p": p,
                        "n_obs": n_obs, "shd": shd,
                        "precision": em["precision"],
                        "recall": em["recall"],
                        "f1": em["f1"],
                        "tp": em["tp"], "fp": em["fp"], "fn": em["fn"],
                        "runtime": result.runtime_seconds,
                        "n_true_edges": n_true_edges,
                        "n_ci_tests": result.n_ci_tests,
                        "actual_miss_rate": actual_miss,
                        "failed": False,
                    })

                print(f"{time.time()-t_ds:.0f}s")

        print(f"  Graph {g+1} done in {(time.time()-t_graph)/60:.1f} min")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="p=20 graph recovery: single mechanism + DGP runner")
    parser.add_argument("--mechanism", type=str, required=True,
                        choices=["mar", "mnar", "mixed", "complete"])
    parser.add_argument("--dgp", type=str, required=True,
                        choices=["linear", "nonlinear"])
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--n_obs", type=int, default=1000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--edge_prob", type=float, default=0.2)
    parser.add_argument("--miss_rate", type=float, default=0.3)
    parser.add_argument("--n_miss_cols", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== p=20 Kill-Switch ===")
    print(f"DGP: {args.dgp}, Mechanism: {args.mechanism}")
    print(f"Graphs: {args.n_graphs}, Datasets: {args.n_datasets}")
    print(f"p={args.p}, n={args.n_obs}, edge_prob={args.edge_prob}")
    print(f"Miss cols: {args.n_miss_cols}, Miss rate: {args.miss_rate}\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.time()
    df = run_single_mechanism(
        mechanism=args.mechanism, dgp=args.dgp,
        n_graphs=args.n_graphs, n_datasets=args.n_datasets,
        n_obs=args.n_obs, p=args.p, edge_prob=args.edge_prob,
        miss_rate=args.miss_rate, n_miss_cols=args.n_miss_cols,
        alpha=args.alpha, seed=args.seed,
    )

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"Results: {len(df)} rows")

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    if len(df) > 0 and not df["shd"].isna().all():
        summary = (df[~df["failed"]]
                   .groupby("method")["shd"]
                   .agg(["median", "mean", "std"])
                   .round(1))
        print(f"\nSHD summary ({args.dgp}, {args.mechanism}):")
        print(summary)
