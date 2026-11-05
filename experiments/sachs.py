"""Graph recovery on the Sachs protein-signalling data (p = 11, 17 edges).

Uses the observational flow-cytometry data of Sachs et al. (2005), 853 cells
on the raw scale (data/sachs_data.csv, created by data/fetch_sachs_data.sh),
against the 17-edge ground-truth DAG below. Each replicate is a bootstrap
resample of all 853 rows; four randomly chosen non-root variables are then
made incomplete at per-column rates of 10%, 20% and 40% under MAR, MNAR and
mixed mechanisms, plus a complete-data control. PAIR-CI (fast variant) is
compared with complete-case PC, test-wise deletion and FZ-vote; estimands
are median SHD and skeleton F1 over 20 replicates.

    python experiments/sachs.py --n_reps 20 --data data/sachs_data.csv \
        --output results/sachs_results.csv

Feeds App. J (Table sachs_linear). FZ-Rubin and MissDAG are run on the
same bootstrap datasets by new_baselines.py --benchmark sachs --dgp real.
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
    edge_metrics, _pc_skeleton, _orient_edges, PCResult,
)
from dgps import inject_missingness

# ------------------------------------------------------------------
# Ground truth (Sachs et al. 2005, 17 edges)
# ------------------------------------------------------------------

SACHS_NODES = [
    "Raf", "Mek", "Plcg", "PIP2", "PIP3",
    "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"
]

SACHS_EDGES = [
    ("PKC", "Raf"),  ("PKC", "Mek"),  ("PKC", "PKA"),
    ("PKC", "Jnk"),  ("PKC", "P38"),  ("PKA", "Raf"),
    ("PKA", "Mek"),  ("PKA", "Erk"),  ("PKA", "Akt"),
    ("PKA", "Jnk"),  ("PKA", "P38"),  ("Raf", "Mek"),
    ("Mek", "Erk"),  ("Erk", "Akt"),  ("Plcg", "PIP2"),
    ("Plcg", "PIP3"), ("PIP3", "PIP2"),
]


def get_sachs_adjacency():
    p = len(SACHS_NODES)
    idx = {name: i for i, name in enumerate(SACHS_NODES)}
    adj = np.zeros((p, p), dtype=int)
    for parent, child in SACHS_EDGES:
        adj[idx[parent], idx[child]] = 1
    return adj


def run_pc_fast(data, alpha=0.05, random_state=None, ci_test_cls=None,
                method_label="ci_oracle_fast"):
    t0 = time.time()
    p = data.shape[1]
    cls = ci_test_cls if ci_test_cls is not None else FastCITest
    ci = cls(data, n_imputations=5, n_folds=5, n_trees=100,
            early_stop=True, early_stop_after=2,
            early_stop_threshold=4.0, random_state=random_state)
    adj, sep, nt = _pc_skeleton(p, ci, alpha, None, False)
    adj = _orient_edges(adj, sep, p, False)
    return PCResult(adj, sep, nt, time.time() - t0, method_label)


def run_all_methods(data_miss, alpha, seed, dgp="real",
                    learner="rf"):
    from ci_oracle_linear import LinearLearnerFastCITest
    results = {}
    rng = np.random.RandomState(seed)
    # learner="rf" is the default FastCITest (ExtraTrees); "ridge" swaps in
    # LinearLearnerFastCITest, Ridge(alpha=1e-6). The synthetic benchmarks
    # use ridge on linear edges and ExtraTrees on nonlinear ones. On the
    # real data the default is ExtraTrees; --learner ridge checks whether a
    # linear learner, the assumption Fisher-Z makes, fits these data better.
    # The data are used on their raw (untransformed) scale.
    if learner == "ridge" or dgp == "linear":
        ours_ci_cls = LinearLearnerFastCITest
        ours_label = "ci_oracle_fast_ridge" if learner == "ridge" else "ci_oracle_fast"
    else:
        ours_ci_cls = None
        ours_label = "ci_oracle_fast"
    methods = [
        (ours_label, run_pc_fast,
         dict(alpha=alpha, random_state=rng.randint(0, 2**31),
              ci_test_cls=ours_ci_cls, method_label=ours_label)),
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


def standardise_columns(df):
    name_map = {
        "praf": "Raf", "raf": "Raf",
        "pmek": "Mek", "mek": "Mek",
        "plcg": "Plcg",
        "pip2": "PIP2", "pip3": "PIP3",
        "p44.42": "Erk", "p44/42": "Erk", "erk": "Erk",
        "pakts473": "Akt", "akt": "Akt",
        "pka": "PKA", "pkc": "PKC",
        "p38": "P38", "pjnk": "Jnk", "jnk": "Jnk",
    }
    df = df.rename(columns={c: name_map.get(c, name_map.get(c.lower(), c))
                            for c in df.columns})
    missing = [n for n in SACHS_NODES if n not in df.columns]
    assert not missing, f"Missing columns: {missing}. Got: {list(df.columns)}"
    return df[SACHS_NODES].astype(float)


def run_sachs_experiment(data_path, n_reps=20, seed=42, alpha=0.05, learner="rf"):
    df = standardise_columns(pd.read_csv(data_path))
    adj_true = get_sachs_adjacency()
    p, n = df.shape[1], df.shape[0]
    print(f"Data: {n} obs x {p} vars, {len(SACHS_EDGES)} true edges")

    rng = np.random.RandomState(seed)
    all_results = []

    # Four incomplete columns drawn from the non-root variables, as in
    # sachs_nonlinear.py and new_baselines.run_sachs.
    node_idx = {name: i for i, name in enumerate(SACHS_NODES)}
    has_parent = sorted({node_idx[child] for _, child in SACHS_EDGES})
    n_miss_cols = 4

    for rate in [0.1, 0.2, 0.4]:
        for mech in ["complete", "mar", "mnar", "mixed"]:
            if mech == "complete" and rate > 0.1:
                continue
            print(f"Rate={rate}, Mechanism={mech}...", end=" ", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                idx = rng.choice(n, size=n, replace=True)
                data_boot = df.values[idx]
                miss_cols = sorted(rng.choice(has_parent, n_miss_cols,
                                              replace=False))
                if mech == "complete":
                    data_miss = data_boot.copy()
                else:
                    data_miss = inject_missingness(
                        data_boot, mech, miss_cols, rate, rng)
                actual_miss = np.mean(np.isnan(data_miss))
                mr = run_all_methods(data_miss, alpha, rng.randint(0, 2**31),
                                     learner=learner)
                for mname, result in mr.items():
                    if result is None:
                        all_results.append(dict(
                            rate=rate, mechanism=mech, rep=rep, method=mname,
                            shd=np.nan, precision=np.nan, recall=np.nan,
                            f1=np.nan, runtime=np.nan, n_ci_tests=np.nan,
                            actual_miss_rate=actual_miss, failed=True))
                        continue
                    shd = structural_hamming_distance(adj_true, result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append(dict(
                        rate=rate, mechanism=mech, rep=rep, method=mname,
                        shd=shd, precision=em["precision"], recall=em["recall"],
                        f1=em["f1"], runtime=result.runtime_seconds,
                        n_ci_tests=result.n_ci_tests,
                        actual_miss_rate=actual_miss, failed=False))
            print(f"done ({(time.time()-t0)/60:.1f} min)")

    return pd.DataFrame(all_results)


def analyze(df):
    df_ok = df[~df["failed"]]
    print("\n" + "=" * 60)
    print("SACHS RESULTS")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="data/sachs_data.csv")
    parser.add_argument("--output", type=str, default="results/sachs_results.csv")
    parser.add_argument("--learner", type=str, default="rf",
                        choices=["rf", "ridge"],
                        help="Base learner for ci_oracle_fast. "
                             "'rf' (default) uses ExtraTrees; 'ridge' uses "
                             "Ridge(alpha=1e-6) via LinearLearnerFastCITest, "
                             "renamed to ci_oracle_fast_ridge in the output "
                             "CSV for distinct merging.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Sachs experiment: {args.n_reps} reps\n")
    t0 = time.time()
    df = run_sachs_experiment(args.data, args.n_reps, args.seed, args.alpha,
                              learner=args.learner)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    analyze(df)
