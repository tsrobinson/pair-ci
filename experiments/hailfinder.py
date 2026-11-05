"""Graph recovery on the HAILFINDER network (p = 56, 66 edges).

Data are simulated from the HAILFINDER DAG (Abramson et al. 1996) with
random nonlinear edges (n = 1000, noise s.d. 0.5); edge functions and
weights follow alarm.py. Missingness is injected into 15 randomly chosen
non-root variables at per-column rates of 10%, 20% and 40% under MAR, MNAR
and mixed mechanisms, plus a complete-data control. PAIR-CI (fast variant)
is compared with complete-case PC, test-wise deletion and FZ-vote; the
estimands are the median SHD and skeleton precision, recall and F1 over 20
replicates. As released, each (rate, mechanism) cell ran as its own job:

    python experiments/hailfinder.py --n_obs 1000 --n_reps 20 \
        --dgp nonlinear --rate 0.2 --mechanism mnar \
        --output results/hailfinder_0.2_mnar.csv

analysis/combine_hailfinder.py merges the cells into
results/hailfinder_results.csv, which feeds Table 3 and App. I.
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
# HAILFINDER network (Abramson et al. 1996): 56 nodes, 66 edges
# ------------------------------------------------------------------

HAILFINDER_NODES = [
    "N0_7muVerMo", "SubjVertMo", "QGVertMotion", "CombVerMo",
    "AreaMeso_ALS", "SatContMoist", "RaoContMoist", "CombMoisture",
    "AreaMoDryAir", "VISCloudCov", "IRCloudCover", "CombClouds",
    "CldShadeOth", "AMInstabMt", "InsInMt", "WndHodograph",
    "OutflowFrMt", "MorningBound", "Boundaries", "CldShadeConv",
    "CompPlFcst", "CapChange", "LoLevMoistAd", "InsChange",
    "MountainFcst", "Date", "Scenario", "ScenRelAMCIN",
    "MorningCIN", "AMCINInScen", "CapInScen", "ScenRelAMIns",
    "LIfr12ZDENSd", "AMDewptCalPl", "AMInsWliScen", "InsSclInScen",
    "ScenRel3_4", "LatestCIN", "LLIW", "CurPropConv",
    "ScnRelPlFcst", "PlainsFcst", "N34StarFcst", "R5Fcst",
    "Dewpoints", "LowLLapse", "MeanRH", "MidLLapse",
    "MvmtFeatures", "RHRatio", "SfcWndShfDis", "SynForcng",
    "TempDis", "WindAloft", "WindFieldMt", "WindFieldPln",
]

HAILFINDER_EDGES = [
    ("N0_7muVerMo", "CombVerMo"),
    ("SubjVertMo", "CombVerMo"),
    ("QGVertMotion", "CombVerMo"),
    ("CombVerMo", "AreaMeso_ALS"),
    ("AreaMeso_ALS", "AreaMoDryAir"),
    ("AreaMeso_ALS", "CldShadeOth"),
    ("AreaMeso_ALS", "CompPlFcst"),
    ("SatContMoist", "CombMoisture"),
    ("RaoContMoist", "CombMoisture"),
    ("CombMoisture", "AreaMoDryAir"),
    ("AreaMoDryAir", "CldShadeOth"),
    ("VISCloudCov", "CombClouds"),
    ("IRCloudCover", "CombClouds"),
    ("CombClouds", "CldShadeOth"),
    ("CldShadeOth", "InsInMt"),
    ("CldShadeOth", "CompPlFcst"),
    ("AMInstabMt", "InsInMt"),
    ("InsInMt", "OutflowFrMt"),
    ("InsInMt", "CldShadeConv"),
    ("InsInMt", "MountainFcst"),
    ("WndHodograph", "OutflowFrMt"),
    ("WndHodograph", "Boundaries"),
    ("WndHodograph", "CldShadeConv"),
    ("OutflowFrMt", "Boundaries"),
    ("MorningBound", "Boundaries"),
    ("Boundaries", "CompPlFcst"),
    ("CldShadeConv", "CompPlFcst"),
    ("CompPlFcst", "CapChange"),
    ("CompPlFcst", "InsChange"),
    ("CapChange", "CapInScen"),
    ("LoLevMoistAd", "InsChange"),
    ("InsChange", "InsSclInScen"),
    ("MountainFcst", "R5Fcst"),
    ("Date", "Scenario"),
    ("Scenario", "ScenRelAMCIN"),
    ("Scenario", "ScenRelAMIns"),
    ("Scenario", "ScenRel3_4"),
    ("Scenario", "ScnRelPlFcst"),
    ("Scenario", "Dewpoints"),
    ("Scenario", "LowLLapse"),
    ("Scenario", "MeanRH"),
    ("Scenario", "MidLLapse"),
    ("Scenario", "MvmtFeatures"),
    ("Scenario", "RHRatio"),
    ("Scenario", "SfcWndShfDis"),
    ("Scenario", "SynForcng"),
    ("Scenario", "TempDis"),
    ("Scenario", "WindAloft"),
    ("Scenario", "WindFieldMt"),
    ("Scenario", "WindFieldPln"),
    ("ScenRelAMCIN", "AMCINInScen"),
    ("MorningCIN", "AMCINInScen"),
    ("AMCINInScen", "CapInScen"),
    ("CapInScen", "PlainsFcst"),
    ("ScenRelAMIns", "AMInsWliScen"),
    ("LIfr12ZDENSd", "AMInsWliScen"),
    ("AMDewptCalPl", "AMInsWliScen"),
    ("AMInsWliScen", "InsSclInScen"),
    ("InsSclInScen", "PlainsFcst"),
    ("ScenRel3_4", "N34StarFcst"),
    ("LatestCIN", "CurPropConv"),
    ("LLIW", "CurPropConv"),
    ("CurPropConv", "PlainsFcst"),
    ("ScnRelPlFcst", "PlainsFcst"),
    ("PlainsFcst", "N34StarFcst"),
    ("N34StarFcst", "R5Fcst"),
]


def get_hailfinder_adjacency():
    p = len(HAILFINDER_NODES)
    idx = {name: i for i, name in enumerate(HAILFINDER_NODES)}
    adj = np.zeros((p, p), dtype=int)
    for parent, child in HAILFINDER_EDGES:
        adj[idx[parent], idx[child]] = 1
    return adj


def _topo_sort(p, edges_idx):
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


def sample_network_data(nodes, edges, n, rng, dgp="nonlinear",
                        noise_scale=0.5):
    p = len(nodes)
    idx = {name: i for i, name in enumerate(nodes)}
    edges_idx = [(idx[pa], idx[ch]) for pa, ch in edges]

    # Same edge functions as alarm.py (f_1..f_4 of App. E).
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


def run_hailfinder_experiment(n_obs=1000, n_reps=20, seed=42, alpha=0.05,
                              dgp="nonlinear", rate_filter=None,
                              mech_filter=None):
    """Run every (rate, mechanism) cell, or one cell if rate_filter and
    mech_filter are both set.

    In a single-cell run, the skipped cells still draw each replicate's
    data, incomplete columns and method seed, but not its missingness mask.
    A per-cell job therefore reproduces a single-job run only for the 10%
    complete and 10% MAR cells; later cells get different, identically
    distributed datasets. The released results come from per-cell jobs,
    while new_baselines.py, mvpc_baseline.py and alpha_sweep_baselines.py
    replay the single-job stream, so on HAILFINDER the baselines see
    datasets matched to PAIR-CI's in distribution only.
    """
    adj_true = get_hailfinder_adjacency()
    p = len(HAILFINDER_NODES)
    n_true_edges = len(HAILFINDER_EDGES)
    if rate_filter is None and mech_filter is None:
        print(f"HAILFINDER network: {p} nodes, {n_true_edges} true edges, "
              f"dgp={dgp}")
    else:
        print(f"HAILFINDER cell run: rate={rate_filter}, mech={mech_filter}, "
              f"dgp={dgp}")

    rng = np.random.RandomState(seed)
    all_results = []

    # Select columns for missingness injection (non-root nodes)
    idx = {name: i for i, name in enumerate(HAILFINDER_NODES)}
    has_parent = sorted(set(idx[ch] for _, ch in HAILFINDER_EDGES))
    n_miss_cols = 15  # ~27% of 56 variables

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
                    _ = sample_network_data(
                        HAILFINDER_NODES, HAILFINDER_EDGES, n_obs, rng, dgp=dgp)
                    _ = sorted(rng.choice(has_parent, n_miss_cols,
                                          replace=False))
                    _ = rng.randint(0, 2**31)
                continue

            print(f"Rate={rate}, Mechanism={mech}...", end=" ", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                data = sample_network_data(HAILFINDER_NODES, HAILFINDER_EDGES,
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
    print("HAILFINDER RESULTS")
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
                        help="Run only this missingness mechanism (for "
                             "parallel per-cell jobs). If unset, runs all.")
    parser.add_argument("--output", type=str,
                        default="results/hailfinder_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.rate is not None and args.mechanism is not None:
        print(f"HAILFINDER per-cell: rate={args.rate}, mech={args.mechanism}, "
              f"reps={args.n_reps}, n={args.n_obs}, dgp={args.dgp}\n")
    else:
        print(f"HAILFINDER experiment: {args.n_reps} reps, n={args.n_obs}, "
              f"dgp={args.dgp}\n")
    t0 = time.time()
    df = run_hailfinder_experiment(n_obs=args.n_obs, n_reps=args.n_reps,
                                   seed=args.seed, alpha=args.alpha,
                                   dgp=args.dgp,
                                   rate_filter=args.rate,
                                   mech_filter=args.mechanism)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    if len(df) > 0:
        analyze(df)
