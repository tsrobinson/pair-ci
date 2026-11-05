"""Bayle versus Nadeau-Bengio within-imputation variance in PAIR-CI.

Runs PAIR-CI (general harness, M = 5, K = 5, cached imputation) with each
variance estimator on the same data and seed.
--mode calibration: the calibration DGPs, n in {500, 1000, 2000}, |X| = 5,
signal in {0, 0.3, 0.6, 1.0}, complete data, MAR and MNAR (30% per
column), 50 replicates; estimands are rejection rates at alpha = 0.05.
--mode graph: p = 10 Erdos-Renyi DAGs with nonlinear edges, three
incomplete non-root variables at 30% MAR or MNAR (10 graphs x 20 datasets
as run); estimands are median SHD, precision, recall and F1.

    python experiments/variance_comparison.py --mode both --n_reps 50 \
        --n_graphs 10 --n_datasets 20 \
        --output_calib results/variance_comparison_calibration.csv \
        --output_graph results/variance_comparison_graph.csv

Feeds App. K.
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
from pc_integration import (
    run_pc_with_ci_oracle, run_pc_complete_case, run_pc_impute_then_test,
    run_pc_testwise_deletion, structural_hamming_distance, edge_metrics,
    dag_to_adjacency,
)
from dgps import (
    dgp_linear_gaussian, dgp_post_nonlinear, dgp_latent_confounder,
    generate_erdos_renyi_dag, sample_from_dag_nonlinear,
    inject_missingness, inject_missingness_calibration,
    DGP_FUNCTIONS,
)


def run_calibration_comparison(n_reps=50, sample_sizes=(500, 1000, 2000),
                               signals=(0,), dim_x=5, alpha=0.05, seed=42):
    """Rejection rates of PAIR-CI under the NB and Bayle estimators."""
    rng = np.random.RandomState(seed)
    results = []

    for dgp_name, dgp_fn in DGP_FUNCTIONS.items():
        for n_obs in sample_sizes:
            for signal in signals:
                for mechanism in ["complete", "mar", "mnar"]:
                    for rep in range(n_reps):
                        X, Y, Z = dgp_fn(n_obs, dim_x, signal, rng)
                        data = np.column_stack([Z, Y, X])

                        if mechanism != "complete":
                            X_miss = inject_missingness_calibration(
                                data[:, 2:], mechanism, 0.3, rng)
                            data[:, 2:] = X_miss

                        # Both estimators share this seed: same imputations,
                        # folds and placebo, so only the variance differs.
                        rs = rng.randint(0, 2**31)
                        cond = tuple(range(2, 2 + dim_x))

                        for var_est in ["nadeau_bengio", "bayle"]:
                            ci = GeneralCITest(
                                data, n_imputations=5, n_folds=5,
                                variance_estimator=var_est,
                                random_state=rs)
                            t_stat, p_val = ci.test(0, 1, cond)
                            results.append({
                                "dgp": dgp_name, "n_obs": n_obs,
                                "signal": signal, "mechanism": mechanism,
                                "rep": rep, "variance_estimator": var_est,
                                "t_stat": t_stat, "p_value": p_val,
                                "reject": p_val < alpha,
                            })

                    done = len([r for r in results if r["dgp"] == dgp_name
                               and r["n_obs"] == n_obs and r["signal"] == signal
                               and r["mechanism"] == mechanism])
                    print(f"  {dgp_name} n={n_obs} signal={signal} "
                          f"{mechanism}: {done} results")

    return pd.DataFrame(results)


def run_graph_comparison(n_graphs=5, n_datasets=10, n_obs=1000, p=10,
                         edge_prob=0.25, miss_rate=0.3, alpha=0.05, seed=42):
    """Graph recovery with the NB and Bayle estimators at p = 10."""
    rng = np.random.RandomState(seed)
    results = []

    for g in range(n_graphs):
        print(f"Graph {g+1}/{n_graphs}")
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)

        has_parent = [j for j in range(p) if any(True for (i, c) in edges if c == j)]
        if len(has_parent) >= 3:
            miss_cols = sorted(rng.choice(has_parent, 3, replace=False))
        else:
            miss_cols = sorted(rng.choice(range(p), 3, replace=False))

        for mech in ["mar", "mnar"]:
            for d in range(n_datasets):
                data = sample_from_dag_nonlinear(edges, p, n_obs, rng)
                data_miss = inject_missingness(data, mech, miss_cols,
                                               miss_rate, rng)
                rs = rng.randint(0, 2**31)

                for var_est in ["nadeau_bengio", "bayle"]:
                    try:
                        ci = GeneralCITest(
                            data_miss, n_imputations=5, n_folds=5,
                            imputation_strategy="cached",
                            variance_estimator=var_est,
                            random_state=rs)
                        from pc_integration import _pc_skeleton, _orient_edges, PCResult
                        t0 = time.time()
                        adj, sep, nt = _pc_skeleton(p, ci, alpha, None, False)
                        adj = _orient_edges(adj, sep, p, False)
                        runtime = time.time() - t0
                        shd = structural_hamming_distance(adj_true, adj)
                        em = edge_metrics(adj_true, adj)
                        results.append({
                            "graph": g, "dataset": d, "mechanism": mech,
                            "variance_estimator": var_est, "shd": shd,
                            "precision": em["precision"], "recall": em["recall"],
                            "f1": em["f1"], "runtime": runtime,
                            "n_ci_tests": nt, "failed": False,
                        })
                    except Exception as e:
                        results.append({
                            "graph": g, "dataset": d, "mechanism": mech,
                            "variance_estimator": var_est, "shd": np.nan,
                            "precision": np.nan, "recall": np.nan,
                            "f1": np.nan, "runtime": np.nan,
                            "n_ci_tests": np.nan, "failed": True,
                        })

    return pd.DataFrame(results)


def analyze_calibration(df):
    print("\n" + "=" * 60)
    print("VARIANCE ESTIMATOR COMPARISON: Calibration")
    print("=" * 60)
    null = df[df["signal"] == 0]
    if len(null) > 0:
        print("\nFPR (signal=0, target=0.05):")
        pivot = (null.groupby(["variance_estimator", "mechanism"])["reject"]
                 .mean().unstack("mechanism").round(4))
        print(pivot.to_string())

        print("\nFPR by sample size:")
        pivot2 = (null.groupby(["variance_estimator", "n_obs"])["reject"]
                  .mean().unstack("n_obs").round(4))
        print(pivot2.to_string())

    alt = df[df["signal"] > 0]
    if len(alt) > 0:
        print("\nPower (signal>0):")
        pivot3 = (alt.groupby(["variance_estimator", "signal", "n_obs"])["reject"]
                  .mean().unstack("n_obs").round(3))
        print(pivot3.to_string())


def analyze_graph(df):
    print("\n" + "=" * 60)
    print("VARIANCE ESTIMATOR COMPARISON: Graph Recovery")
    print("=" * 60)
    ok = df[~df["failed"]]
    print("\nMedian SHD:")
    print(ok.groupby(["variance_estimator", "mechanism"])["shd"]
          .median().unstack("mechanism").round(1).to_string())
    print("\nMedian precision:")
    print(ok.groupby(["variance_estimator", "mechanism"])["precision"]
          .median().unstack("mechanism").round(3).to_string())
    print("\nMedian recall:")
    print(ok.groupby(["variance_estimator", "mechanism"])["recall"]
          .median().unstack("mechanism").round(3).to_string())
    print("\nMedian F1:")
    print(ok.groupby(["variance_estimator", "mechanism"])["f1"]
          .median().unstack("mechanism").round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Nadeau-Bengio vs Bayle variance estimators")
    parser.add_argument("--mode", choices=["calibration", "graph", "both"],
                        default="both")
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--n_graphs", type=int, default=5)
    parser.add_argument("--n_datasets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_calib", type=str,
                        default="results/variance_comparison_calibration.csv")
    parser.add_argument("--output_graph", type=str,
                        default="results/variance_comparison_graph.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.mode in ("calibration", "both"):
        print("=== Calibration comparison ===\n")
        t0 = time.time()
        df_cal = run_calibration_comparison(
            n_reps=args.n_reps,
            sample_sizes=(500, 1000, 2000),
            signals=(0, 0.3, 0.6, 1.0),
            seed=args.seed)
        print(f"\nTime: {(time.time()-t0)/60:.1f} min")
        df_cal.to_csv(args.output_calib, index=False)
        analyze_calibration(df_cal)

    if args.mode in ("graph", "both"):
        print("\n=== Graph recovery comparison ===\n")
        t0 = time.time()
        df_graph = run_graph_comparison(
            n_graphs=args.n_graphs, n_datasets=args.n_datasets,
            seed=args.seed)
        print(f"\nTime: {(time.time()-t0)/60:.1f} min")
        df_graph.to_csv(args.output_graph, index=False)
        analyze_graph(df_graph)
