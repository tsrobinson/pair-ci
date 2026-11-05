"""Graph recovery on the ALARM network (p = 37, 46 edges).

Data are simulated from the ALARM DAG (Beinlich et al. 1989) with linear
Gaussian or random nonlinear edges (n = 1000, noise s.d. 0.5). Missingness
is injected into 10 randomly chosen non-root variables at per-column rates
of 10%, 20% and 40% under MAR, MNAR and mixed mechanisms, plus a complete-
data control. PAIR-CI (fast variant) is compared with complete-case PC,
test-wise deletion and FZ-vote on the same datasets; the estimands are the
median SHD and skeleton precision, recall and F1 over 20 replicates.

    python experiments/alarm.py --n_obs 1000 --n_reps 20 --dgp nonlinear \
        --output results/alarm_nonlinear_results.csv

Feeds Table 3 and App. H (results/alarm_results.csv for --dgp linear).
MVPC, FZ-Rubin and MissDAG are run on the same data by mvpc_baseline.py
and new_baselines.py.
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
# ALARM network (Beinlich et al. 1989): 37 nodes, 46 edges
# ------------------------------------------------------------------

ALARM_NODES = [
    "HISTORY", "CVP", "PCWP", "HYPOVOLEMIA", "LVEDVOLUME", "LVFAILURE",
    "STROKEVOLUME", "ERRLOWOUTPUT", "HRBP", "HREKG", "ERRCAUTER", "HRSAT",
    "INSUFFANESTH", "ANAPHYLAXIS", "TPR", "EXPCO2", "KINKEDTUBE", "MINVOL",
    "FIO2", "PVSAT", "SAO2", "PAP", "PULMEMBOLUS", "SHUNT", "INTUBATION",
    "PRESS", "DISCONNECT", "MINVOLSET", "VENTMACH", "VENTTUBE", "VENTLUNG",
    "VENTALV", "ARTCO2", "CATECHOL", "HR", "CO", "BP",
]

ALARM_EDGES = [
    ("LVFAILURE", "HISTORY"),
    ("LVEDVOLUME", "CVP"),
    ("LVEDVOLUME", "PCWP"),
    ("HYPOVOLEMIA", "LVEDVOLUME"),
    ("LVFAILURE", "LVEDVOLUME"),
    ("HYPOVOLEMIA", "STROKEVOLUME"),
    ("LVFAILURE", "STROKEVOLUME"),
    ("ERRLOWOUTPUT", "HRBP"),
    ("HR", "HRBP"),
    ("ERRCAUTER", "HREKG"),
    ("HR", "HREKG"),
    ("ERRCAUTER", "HRSAT"),
    ("HR", "HRSAT"),
    ("ANAPHYLAXIS", "TPR"),
    ("ARTCO2", "EXPCO2"),
    ("VENTLUNG", "EXPCO2"),
    ("INTUBATION", "MINVOL"),
    ("VENTLUNG", "MINVOL"),
    ("FIO2", "PVSAT"),
    ("VENTALV", "PVSAT"),
    ("PVSAT", "SAO2"),
    ("SHUNT", "SAO2"),
    ("PULMEMBOLUS", "PAP"),
    ("INTUBATION", "SHUNT"),
    ("PULMEMBOLUS", "SHUNT"),
    ("INTUBATION", "PRESS"),
    ("KINKEDTUBE", "PRESS"),
    ("VENTTUBE", "PRESS"),
    ("MINVOLSET", "VENTMACH"),
    ("DISCONNECT", "VENTTUBE"),
    ("VENTMACH", "VENTTUBE"),
    ("INTUBATION", "VENTLUNG"),
    ("KINKEDTUBE", "VENTLUNG"),
    ("VENTTUBE", "VENTLUNG"),
    ("INTUBATION", "VENTALV"),
    ("VENTLUNG", "VENTALV"),
    ("VENTALV", "ARTCO2"),
    ("ARTCO2", "CATECHOL"),
    ("INSUFFANESTH", "CATECHOL"),
    ("SAO2", "CATECHOL"),
    ("TPR", "CATECHOL"),
    ("CATECHOL", "HR"),
    ("HR", "CO"),
    ("STROKEVOLUME", "CO"),
    ("CO", "BP"),
    ("TPR", "BP"),
]


def get_alarm_adjacency():
    p = len(ALARM_NODES)
    idx = {name: i for i, name in enumerate(ALARM_NODES)}
    adj = np.zeros((p, p), dtype=int)
    for parent, child in ALARM_EDGES:
        adj[idx[parent], idx[child]] = 1
    return adj


def _topo_sort(p, edges_idx):
    """Topological sort for a DAG given indexed edges."""
    from collections import deque
    children = {i: [] for i in range(p)}
    in_degree = np.zeros(p, dtype=int)
    for pa, ch in edges_idx:
        children[pa].append(ch)
        in_degree[ch] += 1
    queue = deque([i for i in range(p) if in_degree[i] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for ch in children[node]:
            in_degree[ch] -= 1
            if in_degree[ch] == 0:
                queue.append(ch)
    return order


def sample_network_data(nodes, edges, n, rng, dgp="linear", noise_scale=0.5):
    """Sample continuous data from a DAG.

    dgp="linear": X_j = sum_pa w * X_pa + noise
    dgp="nonlinear": X_j = sum_pa f(X_pa, w) + noise, with random nonlinear f
    """
    p = len(nodes)
    idx = {name: i for i, name in enumerate(nodes)}
    edges_idx = [(idx[pa], idx[ch]) for pa, ch in edges]

    # The four edge functions f_1..f_4 of App. E; the tanh term follows the
    # saturating nonlinearity of the `mim` model in Zheng et al. (2020).
    NL_FNS = [
        lambda x, w: w * x ** 2,
        lambda x, w: w * np.sin(2 * x),
        lambda x, w: w * np.abs(x),
        lambda x, w: w * np.tanh(1.5 * x),
    ]

    weights = {}
    fns = {}
    for pa_i, ch_i in edges_idx:
        w = rng.uniform(0.5, 1.0)
        if rng.rand() < 0.5:
            w = -w
        weights[(pa_i, ch_i)] = w
        if dgp == "nonlinear":
            fns[(pa_i, ch_i)] = NL_FNS[rng.randint(0, len(NL_FNS))]

    topo_order = _topo_sort(p, edges_idx)
    data = np.zeros((n, p))
    for j in topo_order:
        parents = [pa for (pa, ch) in edges_idx if ch == j]
        val = rng.randn(n) * noise_scale
        for pa in parents:
            if dgp == "nonlinear":
                val += fns[(pa, j)](data[:, pa], weights[(pa, j)])
            else:
                val += weights[(pa, j)] * data[:, pa]
        data[:, j] = val

    return data


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
    # On linear edges PAIR-CI swaps ExtraTrees for a ridge learner
    # (src/ci_oracle_linear.py); on nonlinear edges it uses ExtraTrees.
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


def run_alarm_experiment(n_obs=1000, n_reps=20, seed=42, alpha=0.05,
                         dgp="linear"):
    adj_true = get_alarm_adjacency()
    p = len(ALARM_NODES)
    n_true_edges = len(ALARM_EDGES)
    print(f"ALARM network: {p} nodes, {n_true_edges} true edges, dgp={dgp}")

    rng = np.random.RandomState(seed)
    all_results = []

    # Select columns for missingness injection (non-root nodes)
    idx = {name: i for i, name in enumerate(ALARM_NODES)}
    has_parent = [idx[child] for _, child in ALARM_EDGES]
    has_parent = sorted(set(has_parent))
    n_miss_cols = 10  # ~27% of 37 variables

    for rate in [0.1, 0.2, 0.4]:
        for mech in ["complete", "mar", "mnar", "mixed"]:
            if mech == "complete" and rate > 0.1:
                continue
            print(f"Rate={rate}, Mechanism={mech}...", end=" ", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                data = sample_network_data(ALARM_NODES, ALARM_EDGES,
                                           n_obs, rng, dgp=dgp)
                miss_cols = sorted(rng.choice(has_parent, n_miss_cols,
                                              replace=False))
                if mech == "complete":
                    data_miss = data.copy()
                else:
                    data_miss = inject_missingness(
                        data, mech, miss_cols, rate, rng)
                actual_miss = np.mean(np.isnan(data_miss))
                mr = run_all_methods(data_miss, alpha, rng.randint(0, 2**31),
                                     dgp=dgp)
                for mname, result in mr.items():
                    if result is None:
                        all_results.append(dict(
                            rate=rate, mechanism=mech, rep=rep, method=mname,
                            shd=np.nan, precision=np.nan, recall=np.nan,
                            f1=np.nan, runtime=np.nan, n_ci_tests=np.nan,
                            actual_miss_rate=actual_miss, failed=True))
                        continue
                    shd = structural_hamming_distance(adj_true,
                                                      result.adjacency_matrix)
                    em = edge_metrics(adj_true, result.adjacency_matrix)
                    all_results.append(dict(
                        rate=rate, mechanism=mech, rep=rep, method=mname,
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
    print("ALARM RESULTS")
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
    parser.add_argument("--dgp", type=str, default="linear",
                        choices=["linear", "nonlinear"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/alarm_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"ALARM experiment: {args.n_reps} reps, n={args.n_obs}, dgp={args.dgp}\n")
    t0 = time.time()
    df = run_alarm_experiment(n_obs=args.n_obs, n_reps=args.n_reps,
                              seed=args.seed, alpha=args.alpha,
                              dgp=args.dgp)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    analyze(df)
