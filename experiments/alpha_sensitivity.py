"""PAIR-CI graph recovery across significance levels at p = 10.

Ten Erdos-Renyi DAGs (p = 10, edge probability 0.25, nonlinear edges),
n = 1000, three incomplete non-root variables at a 30% per-column rate under
MAR and MNAR, 20 datasets per graph and mechanism. PAIR-CI (fast variant:
ExtraTrees, M = 5, K = 5, early stopping) is run at alpha in {0.01, 0.05,
0.10, 0.15, 0.20} on each dataset; the estimands are median SHD and skeleton
precision, recall and F1 per alpha.

    python experiments/alpha_sensitivity.py --n_graphs 10 --n_datasets 20 \
        --output results/alpha_sensitivity_results.csv

Feeds App. T.1 (PAIR-CI's frontier in Figure pr_frontier) and App. T.2.
alpha_sweep_baselines.py replays this RNG stream for the baselines.
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
    structural_hamming_distance, edge_metrics, dag_to_adjacency,
    _pc_skeleton, _orient_edges, PCResult,
)
from dgps import (
    generate_erdos_renyi_dag, inject_missingness,
    sample_from_dag_nonlinear,
)


def run_pc_at_alpha(data, alpha, random_state):
    t0 = time.time()
    p = data.shape[1]
    ci = FastCITest(data, n_imputations=5, n_folds=5, n_trees=100,
                    early_stop=True, early_stop_after=2,
                    early_stop_threshold=4.0, random_state=random_state)
    adj, sep, nt = _pc_skeleton(p, ci, alpha, None, False)
    adj = _orient_edges(adj, sep, p, False)
    return PCResult(adj, sep, nt, time.time() - t0, "ci_oracle_fast")


def run_alpha_sensitivity(n_graphs=5, n_datasets=10, n_obs=1000, p=10,
                          edge_prob=0.25, miss_rate=0.3, seed=42):
    rng = np.random.RandomState(seed)
    all_results = []

    alphas = [0.01, 0.05, 0.10, 0.15, 0.20]
    mechanisms = ["mar", "mnar"]

    for g in range(n_graphs):
        print(f"Graph {g+1}/{n_graphs}")
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)

        has_parent = [j for j in range(p)
                      if any(True for (i, c) in edges if c == j)]
        if len(has_parent) >= 3:
            miss_cols = sorted(rng.choice(has_parent, 3, replace=False))
        else:
            miss_cols = sorted(rng.choice(range(p), 3, replace=False))

        for mech in mechanisms:
            for d in range(n_datasets):
                data = sample_from_dag_nonlinear(edges, p, n_obs, rng)
                data_miss = inject_missingness(data, mech, miss_cols,
                                               miss_rate, rng)
                # One seed per dataset, shared by every alpha, so the sweep
                # varies only the level.
                base_seed = rng.randint(0, 2**31)

                for alpha in alphas:
                    try:
                        result = run_pc_at_alpha(data_miss, alpha, base_seed)
                        shd = structural_hamming_distance(
                            adj_true, result.adjacency_matrix)
                        em = edge_metrics(adj_true, result.adjacency_matrix)
                        all_results.append(dict(
                            graph=g, dataset=d, mechanism=mech,
                            alpha=alpha, shd=shd,
                            precision=em["precision"],
                            recall=em["recall"], f1=em["f1"],
                            n_ci_tests=result.n_ci_tests,
                            runtime=result.runtime_seconds,
                            failed=False))
                    except Exception:
                        all_results.append(dict(
                            graph=g, dataset=d, mechanism=mech,
                            alpha=alpha, shd=np.nan,
                            precision=np.nan, recall=np.nan, f1=np.nan,
                            n_ci_tests=np.nan, runtime=np.nan,
                            failed=True))

    return pd.DataFrame(all_results)


def analyze(df):
    df_ok = df[~df["failed"]]
    print("\n" + "=" * 60)
    print("ALPHA SENSITIVITY")
    print("=" * 60)

    print("\nMedian SHD by alpha:")
    print((df_ok.groupby(["alpha", "mechanism"])["shd"]
           .median().unstack("mechanism")).round(1).to_string())

    print("\nMedian precision by alpha:")
    print((df_ok.groupby(["alpha", "mechanism"])["precision"]
           .median().unstack("mechanism")).round(3).to_string())

    print("\nMedian recall by alpha:")
    print((df_ok.groupby(["alpha", "mechanism"])["recall"]
           .median().unstack("mechanism")).round(3).to_string())

    print("\nMedian F1 by alpha:")
    print((df_ok.groupby(["alpha", "mechanism"])["f1"]
           .median().unstack("mechanism")).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, default=5)
    parser.add_argument("--n_datasets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/alpha_sensitivity_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Alpha sensitivity: {args.n_graphs} graphs x "
          f"{args.n_datasets} datasets x 5 alphas x 2 mechanisms\n")

    t0 = time.time()
    df = run_alpha_sensitivity(
        n_graphs=args.n_graphs, n_datasets=args.n_datasets, seed=args.seed)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    analyze(df)
