"""Graph recovery at p = 10 with every non-root variable incomplete.

Supplementary diagnostic; not reported in the paper. As
graph_recovery_p10.py (same four methods, DGP samplers and metrics), except
that every variable with a parent is incomplete, so at a 30% per-column
rate about 20-25% of all entries are missing, on 7-8 of the 10 columns.
MAR or MNAR, linear or nonlinear edges, n = 1000, 5 graphs x 10 datasets by
default. The estimands are median SHD and mean skeleton F1 by method.

    python experiments/heavy_missingness_p10.py --dgp nonlinear \
        --mechanism mnar --output results/heavy_miss_p10_nonlinear_mnar.csv

Results are saved after each graph.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import argparse
import time
import numpy as np
import pandas as pd

from pc_integration import (
    structural_hamming_distance, edge_metrics, dag_to_adjacency,
)
from dgps import (
    generate_erdos_renyi_dag, inject_missingness,
    sample_from_dag, sample_from_dag_nonlinear,
)
from graph_recovery_p10 import run_all_methods


def run(mechanism, dgp, n_graphs, n_datasets, n_obs, p, edge_prob,
        miss_rate, alpha, n_imputations, n_folds, seed, output):
    rng = np.random.RandomState(seed)
    sample_fn = sample_from_dag if dgp == "linear" else sample_from_dag_nonlinear
    all_results = []

    for g in range(n_graphs):
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)
        non_root = sorted({c for (_, c) in edges})
        print(f"Graph {g+1}/{n_graphs}: {len(edges)} edges, "
              f"{len(non_root)}/{p} non-root (all incomplete), "
              f"mechanism={mechanism}, dgp={dgp}", flush=True)

        # Draw the datasets of both mechanisms and run only `mechanism`, so
        # that the MAR and MNAR jobs share graphs (as in graph_recovery_p10).
        for mech in ["mar", "mnar"]:
            for d in range(n_datasets):
                data = sample_fn(edges, p, n_obs, rng)
                data_miss = inject_missingness(
                    data, mech, non_root, miss_rate, rng)
                actual_miss = np.mean(np.isnan(data_miss))
                method_seed = rng.randint(0, 2**31)

                if mech != mechanism:
                    continue

                mr = run_all_methods(data_miss, alpha, n_imputations,
                                     n_folds, method_seed, dgp=dgp)
                for mname, result in mr.items():
                    base = {
                        "graph": g, "dataset": d, "mechanism": mech,
                        "dgp": dgp, "method": mname, "p": p, "n_obs": n_obs,
                        "n_miss_cols": len(non_root),
                        "n_true_edges": len(edges),
                        "actual_miss_rate": actual_miss,
                    }
                    if result is None:
                        all_results.append({
                            **base, "shd": np.nan, "precision": np.nan,
                            "recall": np.nan, "f1": np.nan, "tp": np.nan,
                            "fp": np.nan, "fn": np.nan, "runtime": np.nan,
                            "n_ci_tests": np.nan, "failed": True,
                        })
                        continue
                    shd = structural_hamming_distance(
                        adj_true, result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append({
                        **base, "shd": shd, "precision": em["precision"],
                        "recall": em["recall"], "f1": em["f1"],
                        "tp": em["tp"], "fp": em["fp"], "fn": em["fn"],
                        "runtime": result.runtime_seconds,
                        "n_ci_tests": result.n_ci_tests, "failed": False,
                    })
        pd.DataFrame(all_results).to_csv(output, index=False)  # incremental

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="p=10 recovery, all non-root variables incomplete")
    ap.add_argument("--mechanism", required=True, choices=["mar", "mnar"])
    ap.add_argument("--dgp", required=True, choices=["linear", "nonlinear"])
    ap.add_argument("--n_graphs", type=int, default=5)
    ap.add_argument("--n_datasets", type=int, default=10)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--edge_prob", type=float, default=0.25)
    ap.add_argument("--miss_rate", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_imputations", type=int, default=5)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_graphs, args.n_datasets, args.n_obs = 1, 2, 400

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    df = run(args.mechanism, args.dgp, args.n_graphs, args.n_datasets,
             args.n_obs, args.p, args.edge_prob, args.miss_rate,
             args.alpha, args.n_imputations, args.n_folds, args.seed,
             args.output)
    print(f"\nTime: {(time.time() - t0) / 60:.1f} min; saved {args.output}")
    ok = df[~df["failed"]]
    if len(ok):
        print(f"\nOverall missing-cell fraction: "
              f"{ok['actual_miss_rate'].mean():.3f}")
        print("\nMedian SHD / mean F1 by method:")
        print(ok.groupby("method")[["shd", "f1"]]
              .agg({"shd": "median", "f1": "mean"}).round(3).to_string())
