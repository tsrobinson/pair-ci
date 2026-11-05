"""Graph recovery on the Sachs DAG (p = 11, 17 edges) with synthetic
nonlinear edges.

Uses the 17-edge DAG of sachs.py as ground truth and simulates data with
alarm.py's nonlinear edge functions (n = 1000), so that
nonlinearity can be separated from scale and topology when compared with
the real-data run in sachs.py. Four randomly chosen non-root variables are
made incomplete at per-column rates of 10%, 20% and 40% under MAR, MNAR and
mixed mechanisms, plus a complete-data control. PAIR-CI (fast variant) is
compared with complete-case PC, test-wise deletion and FZ-vote; estimands
are median SHD and skeleton F1 over 20 replicates. As released, each
(rate, mechanism) cell ran as its own job:

    python experiments/sachs_nonlinear.py --n_obs 1000 --n_reps 20 \
        --dgp nonlinear --rate 0.2 --mechanism mnar \
        --output results/sachs_nonlinear_0.2_mnar.csv

analysis/combine_sachs_nonlinear.py merges the cells into
results/sachs_nonlinear_results.csv, which feeds App. J (Table
sachs_nonlinear).
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ci_oracle_fast import FastCITest
from pc_integration import (
    run_pc_complete_case, run_pc_impute_then_test,
    run_pc_testwise_deletion, structural_hamming_distance,
    edge_metrics, _pc_skeleton, _orient_edges, PCResult,
)
from dgps import inject_missingness

# These imports rely on experiments/ being sys.path[0], as it is when the
# script is run as `python experiments/sachs_nonlinear.py`.
from alarm import sample_network_data
from sachs import SACHS_NODES, SACHS_EDGES, get_sachs_adjacency


def run_pc_fast(data, alpha=0.05, random_state=None, ci_test_cls=None):
    t0 = time.time()
    p = data.shape[1]
    cls = ci_test_cls if ci_test_cls is not None else FastCITest
    ci = cls(data, n_imputations=5, n_folds=5, n_trees=100,
            early_stop=True, early_stop_after=2,
            early_stop_threshold=4.0, random_state=random_state)
    adj, sep, nt = _pc_skeleton(p, ci, alpha, None, False)
    adj = _orient_edges(adj, sep, p, False)
    return PCResult(adj, sep, nt, time.time() - t0, "ci_oracle_fast")


def run_all_methods(data_miss, alpha, seed, dgp="nonlinear"):
    from ci_oracle_linear import LinearLearnerFastCITest
    results = {}
    rng = np.random.RandomState(seed)
    ci_cls = LinearLearnerFastCITest if dgp == "linear" else None
    methods = [
        ("ci_oracle_fast", run_pc_fast,
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


def run_sachs_synthetic(n_obs=1000, n_reps=20, seed=42, alpha=0.05,
                        dgp="nonlinear", rate_filter=None, mech_filter=None):
    """Run every (rate, mechanism) cell, or one cell if rate_filter and
    mech_filter are both set.

    In a single-cell run, the skipped cells still draw each replicate's
    data, incomplete columns and method seed, but not its missingness mask.
    A per-cell job therefore reproduces a single-job run only for the 10%
    complete and 10% MAR cells; later cells get different, identically
    distributed datasets. The released results come from per-cell jobs,
    whereas new_baselines.py --benchmark sachs replays the single-job
    stream, so FZ-Rubin and MissDAG see datasets matched to PAIR-CI's in
    distribution only.
    """
    adj_true = get_sachs_adjacency()
    p = len(SACHS_NODES)
    n_true_edges = len(SACHS_EDGES)
    if rate_filter is None and mech_filter is None:
        print(f"Sachs DAG (synthetic): {p} nodes, {n_true_edges} true edges, dgp={dgp}")
    else:
        print(f"Sachs synthetic cell: rate={rate_filter}, mech={mech_filter}, dgp={dgp}")

    rng = np.random.RandomState(seed)
    all_results = []

    # Four of the non-root variables are incomplete (~36% of the 11).
    idx = {name: i for i, name in enumerate(SACHS_NODES)}
    has_parent = sorted({idx[child] for _, child in SACHS_EDGES})
    n_miss_cols = 4

    for rate in [0.1, 0.2, 0.4]:
        for mech in ["complete", "mar", "mnar", "mixed"]:
            if mech == "complete" and rate > 0.1:
                continue

            run_this_cell = (
                (rate_filter is None or rate == rate_filter)
                and (mech_filter is None or mech == mech_filter)
            )

            if not run_this_cell:
                # Advance the RNG past this cell's replicates, without the
                # masking draws of inject_missingness (see the docstring).
                for _ in range(n_reps):
                    _ = sample_network_data(SACHS_NODES, SACHS_EDGES,
                                            n_obs, rng, dgp=dgp)
                    _ = sorted(rng.choice(has_parent, n_miss_cols,
                                          replace=False))
                    _ = rng.randint(0, 2**31)
                continue

            print(f"Rate={rate}, Mechanism={mech}...", end=" ", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                data = sample_network_data(SACHS_NODES, SACHS_EDGES,
                                           n_obs, rng, dgp=dgp)
                miss_cols = sorted(rng.choice(has_parent, n_miss_cols,
                                              replace=False))
                if mech == "complete":
                    data_miss = data.copy()
                else:
                    data_miss = inject_missingness(
                        data, mech, miss_cols, rate, rng)
                actual_miss = float(np.mean(np.isnan(data_miss)))
                mr = run_all_methods(data_miss, alpha, rng.randint(0, 2**31),
                                     dgp=dgp)
                for mname, result in mr.items():
                    if result is None:
                        all_results.append(dict(
                            rate=rate, mechanism=mech, rep=rep, method=mname,
                            dgp=dgp, p=p, n_obs=n_obs,
                            n_true_edges=n_true_edges,
                            shd=np.nan, precision=np.nan, recall=np.nan,
                            f1=np.nan, runtime=np.nan, n_ci_tests=np.nan,
                            actual_miss_rate=actual_miss, failed=True))
                        continue
                    shd = structural_hamming_distance(adj_true,
                                                      result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append(dict(
                        rate=rate, mechanism=mech, rep=rep, method=mname,
                        dgp=dgp, p=p, n_obs=n_obs,
                        n_true_edges=n_true_edges,
                        shd=shd, precision=em["precision"],
                        recall=em["recall"], f1=em["f1"],
                        runtime=result.runtime_seconds,
                        n_ci_tests=result.n_ci_tests,
                        actual_miss_rate=actual_miss, failed=False))
            print(f"done ({(time.time()-t0)/60:.1f} min)")

    return pd.DataFrame(all_results)


def analyze(df):
    df_ok = df[~df["failed"]]
    print("\n" + "=" * 60)
    print("SACHS SYNTHETIC NONLINEAR RESULTS")
    print("=" * 60)
    for rate in sorted(df_ok["rate"].unique()):
        sub = df_ok[df_ok["rate"] == rate]
        print(f"\n--- {rate:.0%} missingness ---")
        print("Median SHD:")
        print(sub.groupby(["method", "mechanism"])["shd"]
              .median().unstack("mechanism").round(1).to_string())
        print("Median F1:")
        print(sub.groupby(["method", "mechanism"])["f1"]
              .median().unstack("mechanism").round(3).to_string())
        print("Median runtime (s):")
        print(sub.groupby("method")["runtime"]
              .median().round(1).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obs", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--dgp", type=str, default="nonlinear",
                        choices=["linear", "nonlinear"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=None,
                        choices=[0.1, 0.2, 0.4],
                        help="Run only this missingness rate (for parallel "
                             "per-cell jobs). If unset, runs all rates.")
    parser.add_argument("--mechanism", type=str, default=None,
                        choices=["complete", "mar", "mnar", "mixed"],
                        help="Run only this missingness mechanism. If unset, "
                             "runs all mechanisms.")
    parser.add_argument("--output", type=str,
                        default="results/sachs_nonlinear_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.rate is not None and args.mechanism is not None:
        print(f"Sachs synthetic per-cell: rate={args.rate}, "
              f"mech={args.mechanism}, reps={args.n_reps}, dgp={args.dgp}\n")
    else:
        print(f"Sachs synthetic experiment: {args.n_reps} reps, "
              f"n={args.n_obs}, dgp={args.dgp}\n")
    t0 = time.time()
    df = run_sachs_synthetic(n_obs=args.n_obs, n_reps=args.n_reps,
                              seed=args.seed, alpha=args.alpha,
                              dgp=args.dgp,
                              rate_filter=args.rate,
                              mech_filter=args.mechanism)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    if len(df) > 0:
        analyze(df)
