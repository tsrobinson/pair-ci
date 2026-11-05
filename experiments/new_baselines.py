"""FZ-Rubin and MissDAG baselines on the graph-recovery benchmarks.

Runs PC with Fisher-Z pooled across five imputations by Rubin's rules
(FZ-Rubin, a port of the R package micd; method name micd_pooled) and a
Gaussian linear MissDAG (src/missdag_gaussian.py, not in the public
release; see NOTICE). The datasets are those of the other graph experiments,
obtained by replaying their RNG streams: the Erdos-Renyi grids of
graph_recovery_p{10,20,30}.py, ALARM (alarm.py), HAILFINDER (hailfinder.py;
FZ-Rubin only) and Sachs (sachs.py for the real data, sachs_nonlinear.py
for nonlinear edges). They are identical to PAIR-CI's except on HAILFINDER
and Sachs-nonlinear, where PAIR-CI ran as per-cell jobs and the match is in
distribution only (see hailfinder.py). Output CSVs share the schema of the
PAIR-CI scripts, so analysis/combine_*.py can concatenate them.

    python experiments/new_baselines.py --benchmark erdos \
        --p 20 --dgp linear --mechanism mar --n_graphs 10 --n_datasets 20 \
        --output results/newbase_p20_linear_mar.csv

    python experiments/new_baselines.py --benchmark alarm --dgp linear \
        --n_reps 20 --output results/newbase_alarm_linear.csv

Feeds the FZ-Rubin and MissDAG rows of Tables 2 and 3 and Apps. F, H, I
and J. mvpc_baseline.py reuses the replay loops below.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from baselines_new import run_pc_micd_pooled, run_missdag_gaussian
from pc_integration import (
    structural_hamming_distance, edge_metrics, dag_to_adjacency,
)
from dgps import (
    generate_erdos_renyi_dag, inject_missingness,
    sample_from_dag, sample_from_dag_nonlinear,
)


METHOD_SPECS = [
    ("micd_pooled", run_pc_micd_pooled,
     dict(alpha=0.05, n_imputations=5)),
    ("missdag", run_missdag_gaussian,
     dict(em_iter=30, lambda1=0.1, w_threshold=0.3)),
]


_ACTIVE_METHODS = set()  # populated from CLI in __main__

def run_methods(data_miss, method_seed):
    results = {}
    rng = np.random.RandomState(method_seed)
    for name, fn, kw in METHOD_SPECS:
        if _ACTIVE_METHODS and name not in _ACTIVE_METHODS:
            continue
        kw_local = dict(kw)
        kw_local["random_state"] = rng.randint(0, 2**31)
        try:
            results[name] = fn(data_miss, **kw_local)
        except Exception as e:  # noqa: BLE001
            print(f"    {name} failed: {e}")
            results[name] = None
    return results


def record_result(all_rows, *, g, d, mech, dgp, mname, result,
                  p, n_obs, n_true_edges, actual_miss, adj_true):
    base = dict(graph=g, dataset=d, mechanism=mech, dgp=dgp,
                method=mname, p=p, n_obs=n_obs,
                n_true_edges=n_true_edges,
                actual_miss_rate=actual_miss)
    if result is None:
        base.update(dict(shd=np.nan, precision=np.nan, recall=np.nan,
                         f1=np.nan, tp=np.nan, fp=np.nan, fn=np.nan,
                         runtime=np.nan, n_ci_tests=np.nan,
                         failed=True))
    else:
        shd = structural_hamming_distance(adj_true, result.adjacency_matrix)
        em = edge_metrics(adj_true, result.adjacency_matrix)
        base.update(dict(shd=shd, precision=em["precision"],
                         recall=em["recall"], f1=em["f1"],
                         tp=em["tp"], fp=em["fp"], fn=em["fn"],
                         runtime=result.runtime_seconds,
                         n_ci_tests=result.n_ci_tests,
                         failed=False))
    all_rows.append(base)


# -------- Erdos-Renyi p-sweep path (matches graph_recovery_p*.py) -----------


def run_p_sweep(mechanism, dgp, n_graphs, n_datasets, n_obs, p,
                edge_prob, miss_rate, n_miss_cols, seed):
    rng = np.random.RandomState(seed)
    sample_fn = sample_from_dag if dgp == "linear" else sample_from_dag_nonlinear
    rows = []
    for g in range(n_graphs):
        t_graph = time.time()
        print(f"Graph {g+1}/{n_graphs}, mechanism={mechanism}, dgp={dgp}")
        edges = generate_erdos_renyi_dag(p, edge_prob, rng)
        adj_true = dag_to_adjacency(p, edges)
        n_true_edges = len(edges)

        has_parent = [j for j in range(p) if any(True for (i, c) in edges if c == j)]
        if len(has_parent) >= n_miss_cols:
            miss_cols = sorted(rng.choice(has_parent, n_miss_cols, replace=False))
        else:
            miss_cols = sorted(rng.choice(range(p), n_miss_cols, replace=False))

        # Walk all four mechanisms, as graph_recovery_p*.py does, so that
        # graphs and datasets match theirs; only `mechanism` is run.
        for mech in ["mar", "mnar", "mixed", "complete"]:
            for d in range(n_datasets):
                data = sample_fn(edges, p, n_obs, rng)
                data_miss = inject_missingness(data, mech, miss_cols, miss_rate, rng)
                actual_miss = float(np.mean(np.isnan(data_miss)))
                method_seed = rng.randint(0, 2**31)
                if mech != mechanism:
                    continue

                t_ds = time.time()
                mr = run_methods(data_miss, method_seed)
                for mname, result in mr.items():
                    record_result(rows, g=g, d=d, mech=mech, dgp=dgp,
                                  mname=mname, result=result, p=p, n_obs=n_obs,
                                  n_true_edges=n_true_edges,
                                  actual_miss=actual_miss, adj_true=adj_true)
                print(f"  ds {d+1}/{n_datasets}: {time.time()-t_ds:.1f}s")
        print(f"  Graph {g+1} done in {(time.time()-t_graph)/60:.1f} min")
    return pd.DataFrame(rows)


# -------- ALARM path (matches experiments/alarm.py) -------------------------


def run_alarm(dgp, n_reps, n_obs, seed):
    from alarm import (ALARM_NODES, ALARM_EDGES, get_alarm_adjacency,
                       sample_network_data)
    rng = np.random.RandomState(seed)
    adj_true = get_alarm_adjacency()
    p = len(ALARM_NODES)
    n_true_edges = len(ALARM_EDGES)
    idx = {name: i for i, name in enumerate(ALARM_NODES)}
    has_parent = sorted({idx[child] for _, child in ALARM_EDGES})
    n_miss_cols = 10

    rows = []
    for rate in [0.1, 0.2, 0.4]:
        for mech in ["complete", "mar", "mnar", "mixed"]:
            if mech == "complete" and rate > 0.1:
                continue
            print(f"Rate={rate}, Mechanism={mech}...", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                data = sample_network_data(ALARM_NODES, ALARM_EDGES,
                                           n_obs, rng, dgp=dgp)
                miss_cols = sorted(rng.choice(has_parent, n_miss_cols, replace=False))
                if mech == "complete":
                    data_miss = data.copy()
                else:
                    data_miss = inject_missingness(data, mech, miss_cols, rate, rng)
                actual_miss = float(np.mean(np.isnan(data_miss)))
                mr = run_methods(data_miss, rng.randint(0, 2**31))
                for mname, result in mr.items():
                    rec = dict(rate=rate, mechanism=mech, rep=rep, method=mname,
                               dgp=dgp, p=p, n_obs=n_obs,
                               n_true_edges=n_true_edges,
                               actual_miss_rate=actual_miss)
                    if result is None:
                        rec.update(dict(shd=np.nan, precision=np.nan,
                                        recall=np.nan, f1=np.nan,
                                        tp=np.nan, fp=np.nan, fn=np.nan,
                                        runtime=np.nan, n_ci_tests=np.nan,
                                        failed=True))
                    else:
                        shd = structural_hamming_distance(adj_true,
                                                          result.adjacency_matrix)
                        em = edge_metrics(adj_true, result.adjacency_matrix)
                        rec.update(dict(shd=shd, precision=em["precision"],
                                        recall=em["recall"], f1=em["f1"],
                                        tp=em["tp"], fp=em["fp"], fn=em["fn"],
                                        runtime=result.runtime_seconds,
                                        n_ci_tests=result.n_ci_tests,
                                        failed=False))
                    rows.append(rec)
            print(f"  done ({(time.time()-t0)/60:.1f} min)")
    return pd.DataFrame(rows)


# -------- HAILFINDER path (matches experiments/hailfinder.py) ----------------


def run_hailfinder(dgp, n_reps, n_obs, seed, output_path=None,
                    rep_timeout=3600):
    """Run the baselines on HAILFINDER, replaying hailfinder.py's
    single-job RNG stream.

    rep_timeout is the cap in seconds per replicate and method (0 disables);
    runs that exceed it are recorded as failed (DNF). 3600 s is about 38%
    above PAIR-CI's median runtime per replicate at p = 56.
    """
    import signal as _signal

    class _ReplicateTimeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise _ReplicateTimeout()

    from hailfinder import (HAILFINDER_NODES, HAILFINDER_EDGES,
                             get_hailfinder_adjacency, sample_network_data)
    rng = np.random.RandomState(seed)
    adj_true = get_hailfinder_adjacency()
    p = len(HAILFINDER_NODES)
    n_true_edges = len(HAILFINDER_EDGES)
    idx = {name: i for i, name in enumerate(HAILFINDER_NODES)}
    has_parent = sorted({idx[child] for _, child in HAILFINDER_EDGES})
    n_miss_cols = 15

    rows = []
    for rate in [0.1, 0.2, 0.4]:
        for mech in ["complete", "mar", "mnar", "mixed"]:
            if mech == "complete" and rate > 0.1:
                continue
            print(f"Rate={rate}, Mechanism={mech}...", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                data = sample_network_data(HAILFINDER_NODES, HAILFINDER_EDGES,
                                           n_obs, rng, dgp=dgp)
                miss_cols = sorted(rng.choice(has_parent, n_miss_cols, replace=False))
                if mech == "complete":
                    data_miss = data.copy()
                else:
                    data_miss = inject_missingness(data, mech, miss_cols, rate, rng)
                actual_miss = float(np.mean(np.isnan(data_miss)))
                method_seed = rng.randint(0, 2**31)
                method_rng = np.random.RandomState(method_seed)
                # Run each method individually with per-replicate timeout
                for mname, fn, kw in METHOD_SPECS:
                    if _ACTIVE_METHODS and mname not in _ACTIVE_METHODS:
                        continue
                    rec = dict(rate=rate, mechanism=mech, rep=rep, method=mname,
                               dgp=dgp, p=p, n_obs=n_obs,
                               n_true_edges=n_true_edges,
                               actual_miss_rate=actual_miss)
                    kw_local = dict(kw)
                    # One seed per method from the replicate's method_rng,
                    # as in run_methods.
                    kw_local["random_state"] = method_rng.randint(0, 2**31)
                    old_handler = None
                    try:
                        if rep_timeout > 0:
                            old_handler = _signal.signal(
                                _signal.SIGALRM, _alarm_handler)
                            _signal.alarm(rep_timeout)
                        result = fn(data_miss, **kw_local)
                        if rep_timeout > 0:
                            _signal.alarm(0)
                            if old_handler is not None:
                                _signal.signal(_signal.SIGALRM, old_handler)
                        shd = structural_hamming_distance(
                            adj_true, result.adjacency_matrix)
                        em = edge_metrics(adj_true, result.adjacency_matrix)
                        rec.update(dict(shd=shd, precision=em["precision"],
                                        recall=em["recall"], f1=em["f1"],
                                        tp=em["tp"], fp=em["fp"], fn=em["fn"],
                                        runtime=result.runtime_seconds,
                                        n_ci_tests=result.n_ci_tests,
                                        failed=False))
                    except (_ReplicateTimeout, Exception) as e:
                        if rep_timeout > 0:
                            _signal.alarm(0)
                            if old_handler is not None:
                                _signal.signal(_signal.SIGALRM, old_handler)
                        is_timeout = isinstance(e, _ReplicateTimeout)
                        if is_timeout:
                            print(f"    {mname} DNF (>{rep_timeout}s)",
                                  flush=True)
                        else:
                            print(f"    {mname} failed: {e}", flush=True)
                        rec.update(dict(shd=np.nan, precision=np.nan,
                                        recall=np.nan, f1=np.nan,
                                        tp=np.nan, fp=np.nan, fn=np.nan,
                                        runtime=float(rep_timeout)
                                        if is_timeout else np.nan,
                                        n_ci_tests=np.nan, failed=True))
                    rows.append(rec)
            print(f"  done ({(time.time()-t0)/60:.1f} min)")
            # Save after each cell so a later hang loses nothing.
            if output_path is not None and rows:
                pd.DataFrame(rows).to_csv(output_path, index=False)
    return pd.DataFrame(rows)


# -------- Sachs path (matches experiments/sachs.py / sachs_nonlinear.py) -----


def run_sachs(dgp, n_reps, n_obs, seed, data_path=None):
    from sachs import SACHS_NODES, SACHS_EDGES, get_sachs_adjacency
    from alarm import sample_network_data
    rng = np.random.RandomState(seed)
    adj_true = get_sachs_adjacency()
    p = len(SACHS_NODES)
    n_true_edges = len(SACHS_EDGES)
    idx = {name: i for i, name in enumerate(SACHS_NODES)}
    has_parent = sorted({idx[child] for _, child in SACHS_EDGES})
    # Four incomplete columns, as in sachs.py and sachs_nonlinear.py.
    n_miss_cols = min(4, len(has_parent))

    # For --dgp real, bootstrap n = 853 rows per replicate as sachs.py does,
    # whatever --n_obs says.
    sachs_df_cache = None
    effective_n_obs = n_obs
    if dgp == "real" and data_path is not None:
        sachs_df_cache = pd.read_csv(data_path)
        effective_n_obs = len(sachs_df_cache)

    rows = []
    for rate in [0.1, 0.2, 0.4]:
        for mech in ["complete", "mar", "mnar", "mixed"]:
            if mech == "complete" and rate > 0.1:
                continue
            print(f"Rate={rate}, Mechanism={mech}...", flush=True)
            t0 = time.time()
            for rep in range(n_reps):
                # Real data: bootstrap resample. Otherwise: simulate.
                if dgp == "real" and sachs_df_cache is not None:
                    data = sachs_df_cache.values[
                        rng.choice(len(sachs_df_cache), effective_n_obs,
                                   replace=True)
                    ].astype(float)
                else:
                    data = sample_network_data(SACHS_NODES, SACHS_EDGES,
                                               n_obs, rng, dgp=dgp)
                miss_cols = sorted(rng.choice(has_parent, n_miss_cols, replace=False))
                if mech == "complete":
                    data_miss = data.copy()
                else:
                    data_miss = inject_missingness(data, mech, miss_cols, rate, rng)
                actual_miss = float(np.mean(np.isnan(data_miss)))
                mr = run_methods(data_miss, rng.randint(0, 2**31))
                for mname, result in mr.items():
                    rec = dict(rate=rate, mechanism=mech, rep=rep, method=mname,
                               dgp=dgp, p=p, n_obs=effective_n_obs,
                               n_true_edges=n_true_edges,
                               actual_miss_rate=actual_miss)
                    if result is None:
                        rec.update(dict(shd=np.nan, precision=np.nan,
                                        recall=np.nan, f1=np.nan,
                                        tp=np.nan, fp=np.nan, fn=np.nan,
                                        runtime=np.nan, n_ci_tests=np.nan,
                                        failed=True))
                    else:
                        shd = structural_hamming_distance(adj_true,
                                                          result.adjacency_matrix)
                        em = edge_metrics(adj_true, result.adjacency_matrix)
                        rec.update(dict(shd=shd, precision=em["precision"],
                                        recall=em["recall"], f1=em["f1"],
                                        tp=em["tp"], fp=em["fp"], fn=em["fn"],
                                        runtime=result.runtime_seconds,
                                        n_ci_tests=result.n_ci_tests,
                                        failed=False))
                    rows.append(rec)
            print(f"  done ({(time.time()-t0)/60:.1f} min)")
    return pd.DataFrame(rows)


# -------- CLI --------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="New baselines (micd, missDAG)")
    parser.add_argument("--benchmark", type=str, default="erdos",
                        choices=["erdos", "alarm", "hailfinder", "sachs"])
    parser.add_argument("--mechanism", type=str,
                        choices=["mar", "mnar", "mixed", "complete"])
    parser.add_argument("--dgp", type=str, required=True,
                        choices=["linear", "nonlinear", "real"])
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--edge_prob", type=float, default=None,
                        help="Default: 0.25 for p<=10, 0.2 for p>10")
    parser.add_argument("--miss_rate", type=float, default=0.3)
    parser.add_argument("--n_miss_cols", type=int, default=None,
                        help="Default: 3 for p=10, 6 for p=20, 8 for p=30")
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--n_obs", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=20,
                        help="ALARM only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rep_timeout", type=int, default=0,
                        help="Per-replicate timeout in seconds (0=disabled). "
                             "Methods exceeding this are marked DNF. "
                             "HAILFINDER default: 3600 (1h, ~38%% above our median).")
    parser.add_argument("--methods", nargs="+", default=["micd_pooled", "missdag"],
                        choices=["micd_pooled", "missdag"],
                        help="Which baselines to run (default: both)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Sachs real data CSV (for --benchmark sachs --dgp real)")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    _ACTIVE_METHODS.clear()
    _ACTIVE_METHODS.update(args.methods)
    print(f"Active methods: {sorted(_ACTIVE_METHODS)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.time()
    if args.benchmark == "erdos":
        if args.mechanism is None:
            raise SystemExit("--mechanism required for benchmark=erdos")
        edge_prob = args.edge_prob
        if edge_prob is None:
            edge_prob = 0.25 if args.p <= 10 else (0.15 if args.p >= 30 else 0.2)
        n_miss_cols = args.n_miss_cols
        if n_miss_cols is None:
            n_miss_cols = {10: 3, 20: 6, 30: 8}.get(args.p, max(3, args.p // 3))
        print(f"=== new baselines: p={args.p}, dgp={args.dgp}, "
              f"mechanism={args.mechanism} ===")
        df = run_p_sweep(
            mechanism=args.mechanism, dgp=args.dgp,
            n_graphs=args.n_graphs, n_datasets=args.n_datasets,
            n_obs=args.n_obs, p=args.p, edge_prob=edge_prob,
            miss_rate=args.miss_rate, n_miss_cols=n_miss_cols,
            seed=args.seed,
        )
    elif args.benchmark == "alarm":
        print(f"=== new baselines: ALARM, dgp={args.dgp} ===")
        df = run_alarm(dgp=args.dgp, n_reps=args.n_reps,
                       n_obs=args.n_obs, seed=args.seed)
    elif args.benchmark == "hailfinder":
        print(f"=== new baselines: HAILFINDER, dgp={args.dgp} ===")
        df = run_hailfinder(dgp=args.dgp, n_reps=args.n_reps,
                            n_obs=args.n_obs, seed=args.seed,
                            output_path=args.output,
                            rep_timeout=args.rep_timeout)
    elif args.benchmark == "sachs":
        print(f"=== new baselines: SACHS, dgp={args.dgp} ===")
        df = run_sachs(dgp=args.dgp, n_reps=args.n_reps,
                       n_obs=args.n_obs, seed=args.seed,
                       data_path=getattr(args, 'data_path', None))

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows)")
    if len(df) > 0 and "shd" in df.columns and not df["shd"].isna().all():
        ok = df[~df["failed"]]
        if len(ok):
            print("\nMedian SHD by method:")
            print(ok.groupby("method")["shd"].median().round(1).to_string())
