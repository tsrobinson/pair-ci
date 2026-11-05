"""Alpha sweeps of the tunable baselines on the PAIR-CI benchmark data.

Runs test-wise deletion, complete-case PC, FZ-vote and FZ-Rubin (and,
optionally, MVPC) at several significance levels, by default alpha in
{0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30}, recording SHD, skeleton
precision, recall, F1 and output edge counts (tp + fp) per alpha. The
datasets come from replaying the RNG streams of alpha_sensitivity.py
(--benchmark erdos: p = 10, nonlinear edges, MAR and MNAR) and alarm.py
(--benchmark alarm), so they are identical to those PAIR-CI was run on.
For --benchmark hailfinder the replay follows a single-job run of
hailfinder.py; the released PAIR-CI results come from per-cell jobs, whose
datasets differ (see hailfinder.py), so there the match is in distribution.

    python experiments/alpha_sweep_baselines.py --benchmark erdos \
        --n_graphs 10 --n_datasets 20 \
        --output results/alpha_sweep_baselines_p10.csv

    python experiments/alpha_sweep_baselines.py --benchmark alarm \
        --dgp nonlinear --n_reps 20 --rates 0.2 --mechs mar mnar mixed \
        --output results/alpha_sweep_baselines_alarm_nonlinear.csv

    python experiments/alpha_sweep_baselines.py --benchmark hailfinder \
        --dgp nonlinear --n_reps 20 --rates 0.2 --mechs mar mnar mixed \
        --alphas 0.01 0.05 0.10 0.20 \
        --output results/alpha_sweep_baselines_hailfinder.csv

Feeds App. T.1 (Figure pr_frontier, Table sparsity_matched) and App. T.2.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pc_integration import (
    run_pc_complete_case, run_pc_impute_then_test, run_pc_testwise_deletion,
    structural_hamming_distance, edge_metrics, dag_to_adjacency,
)
from baselines_new import run_pc_micd_pooled
from dgps import (
    generate_erdos_renyi_dag, inject_missingness, sample_from_dag_nonlinear,
)

DEFAULT_ALPHAS = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30]


def _method_registry(active):
    """Baseline runners by name, each called as fn(data, alpha, seed)."""
    reg = {
        "testwise": lambda d, a, s: run_pc_testwise_deletion(d, alpha=a),
        "complete_case": lambda d, a, s: run_pc_complete_case(d, alpha=a),
        "impute_fisherz": lambda d, a, s: run_pc_impute_then_test(
            d, alpha=a, n_imputations=5, random_state=s),
        "micd_pooled": lambda d, a, s: run_pc_micd_pooled(
            d, alpha=a, n_imputations=5, random_state=s),
    }
    if "mvpc" in active:
        from mvpc_baseline import run_pc_mvpc
        reg["mvpc"] = lambda d, a, s: run_pc_mvpc(d, alpha=a, timeout=1800)
    return {k: v for k, v in reg.items() if k in active}


def _record(rows, meta, mname, alpha, adj_true, result, err=None):
    rec = dict(meta, method=mname, alpha=alpha)
    if result is None:
        rec.update(dict(shd=np.nan, precision=np.nan, recall=np.nan,
                        f1=np.nan, tp=np.nan, fp=np.nan, fn=np.nan,
                        n_edges_out=np.nan, runtime=np.nan, failed=True))
        if err is not None:
            print(f"    {mname}@{alpha} failed: {err}", flush=True)
    else:
        shd = structural_hamming_distance(adj_true, result.adjacency_matrix)
        em = edge_metrics(adj_true, result.adjacency_matrix)
        rec.update(dict(shd=shd, precision=em["precision"],
                        recall=em["recall"], f1=em["f1"],
                        tp=em["tp"], fp=em["fp"], fn=em["fn"],
                        n_edges_out=em["tp"] + em["fp"],
                        runtime=result.runtime_seconds, failed=False))
    rows.append(rec)


def _sweep_methods(rows, meta, data_miss, adj_true, alphas, methods, base_seed):
    for alpha in alphas:
        for mname, fn in methods.items():
            try:
                res = fn(data_miss, alpha, base_seed)
                _record(rows, meta, mname, alpha, adj_true, res)
            except Exception as e:  # noqa: BLE001
                _record(rows, meta, mname, alpha, adj_true, None, err=e)


# ---- Erdos path: replays alpha_sensitivity.py ------------------------------

def run_erdos(alphas, methods, n_graphs=10, n_datasets=20, n_obs=1000, p=10,
              edge_prob=0.25, miss_rate=0.3, seed=42):
    # The draws below must stay in the order of
    # alpha_sensitivity.py::run_alpha_sensitivity, so that (graph, dataset,
    # mechanism) index the same data as the PAIR-CI sweep. Do not reorder.
    rng = np.random.RandomState(seed)
    rows = []
    mechanisms = ["mar", "mnar"]
    for g in range(n_graphs):
        print(f"Graph {g+1}/{n_graphs}", flush=True)
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
                base_seed = rng.randint(0, 2**31)
                meta = dict(graph=g, dataset=d, mechanism=mech, p=p,
                            n_obs=n_obs, dgp="nonlinear",
                            n_true_edges=len(edges))
                _sweep_methods(rows, meta, data_miss, adj_true, alphas,
                               methods, base_seed)
    return pd.DataFrame(rows)


# ---- Network paths: replay alarm.py / hailfinder.py streams ----------------

def run_network(benchmark, dgp, alphas, methods, n_reps=20, n_obs=1000,
                seed=42, rates=(0.2,), mechs=("mar", "mnar", "mixed"),
                output_path=None):
    # Walk every (rate, mechanism, rep) cell of the single-job loop in order,
    # drawing from the RNG exactly as alarm.py does, and run the methods only
    # on the requested cells (as in new_baselines.py::run_alarm).
    if benchmark == "alarm":
        from alarm import (ALARM_NODES as NODES, ALARM_EDGES as EDGES,
                           get_alarm_adjacency as get_adj,
                           sample_network_data)
        n_miss_cols = 10
    else:
        from hailfinder import (HAILFINDER_NODES as NODES,
                                HAILFINDER_EDGES as EDGES,
                                get_hailfinder_adjacency as get_adj,
                                sample_network_data)
        n_miss_cols = 15
    rng = np.random.RandomState(seed)
    adj_true = get_adj()
    p = len(NODES)
    idx = {name: i for i, name in enumerate(NODES)}
    has_parent = sorted({idx[child] for _, child in EDGES})
    run_rates = set(float(r) for r in rates)
    run_mechs = set(mechs)

    rows = []
    for rate in [0.1, 0.2, 0.4]:
        for mech in ["complete", "mar", "mnar", "mixed"]:
            if mech == "complete" and rate > 0.1:
                continue
            selected = (rate in run_rates) and (mech in run_mechs)
            if selected:
                print(f"Rate={rate}, Mechanism={mech}...", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                data = sample_network_data(NODES, EDGES, n_obs, rng, dgp=dgp)
                miss_cols = sorted(rng.choice(has_parent, n_miss_cols,
                                              replace=False))
                if mech == "complete":
                    data_miss = data.copy()
                else:
                    data_miss = inject_missingness(data, mech, miss_cols,
                                                   rate, rng)
                base_seed = rng.randint(0, 2**31)
                if not selected:
                    continue
                meta = dict(rate=rate, mechanism=mech, rep=rep, p=p,
                            n_obs=n_obs, dgp=dgp, n_true_edges=len(EDGES))
                _sweep_methods(rows, meta, data_miss, adj_true, alphas,
                               methods, base_seed)
            if selected:
                print(f"  done ({(time.time()-t0)/60:.1f} min)", flush=True)
                if output_path is not None and rows:
                    pd.DataFrame(rows).to_csv(output_path, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline alpha-sweep")
    parser.add_argument("--benchmark", type=str, default="erdos",
                        choices=["erdos", "alarm", "hailfinder"])
    parser.add_argument("--dgp", type=str, default="nonlinear",
                        choices=["linear", "nonlinear"])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=DEFAULT_ALPHAS)
    parser.add_argument("--methods", nargs="+",
                        default=["testwise", "complete_case",
                                 "impute_fisherz", "micd_pooled"],
                        choices=["testwise", "complete_case",
                                 "impute_fisherz", "micd_pooled", "mvpc"])
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--n_obs", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.2])
    parser.add_argument("--mechs", nargs="+",
                        default=["mar", "mnar", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    methods = _method_registry(set(args.methods))
    print(f"Methods: {sorted(methods)}; alphas: {args.alphas}")

    t0 = time.time()
    if args.benchmark == "erdos":
        df = run_erdos(args.alphas, methods, n_graphs=args.n_graphs,
                       n_datasets=args.n_datasets, n_obs=args.n_obs,
                       seed=args.seed)
    else:
        df = run_network(args.benchmark, args.dgp, args.alphas, methods,
                         n_reps=args.n_reps, n_obs=args.n_obs,
                         seed=args.seed, rates=args.rates, mechs=args.mechs,
                         output_path=args.output)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows)")

    ok = df[~df["failed"]]
    if len(ok):
        print("\nMedian (SHD, precision, recall, edges_out) by method x alpha:")
        summ = ok.groupby(["method", "alpha"])[
            ["shd", "precision", "recall", "n_edges_out"]].median().round(2)
        print(summ.to_string())
