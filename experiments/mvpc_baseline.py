"""MVPC baseline (Tu et al. 2019) through causal-learn 0.1.3.7.

Runs missing-value PC: causal-learn's PC-stable search with test-wise-
deletion Fisher-Z and the permutation-based MV_Crtn_Fisher_Z correction,
the only correction released. Datasets come from the replay loops in
new_baselines.py, so they are identical to PAIR-CI's on the Erdos-Renyi
grids (graph_recovery_p{10,20,30}.py) and ALARM, and matched in
distribution on HAILFINDER (see hailfinder.py). Output CSVs share the
p-sweep schema, so analysis/combine_*.py can concatenate them.

    # Quick API and runtime check before submitting the grid:
    python experiments/mvpc_baseline.py --smoke

    python experiments/mvpc_baseline.py \
        --benchmark erdos --p 10 --dgp nonlinear --mechanism mnar \
        --n_graphs 10 --n_datasets 20 \
        --output results/mvpc_p10_nonlinear_mnar.csv

    python experiments/mvpc_baseline.py --benchmark alarm --dgp nonlinear \
        --n_reps 20 --output results/mvpc_alarm_nonlinear.csv
    python experiments/mvpc_baseline.py --benchmark hailfinder \
        --dgp nonlinear --n_reps 20 --rep_timeout 3600 \
        --output results/mvpc_hailfinder_nonlinear.csv

Feeds the MVPC rows of Tables 2 and 3 and Apps. F, H, I and T.1. Its
correction step uses NumPy's unseeded global generator, so results
reproduce in distribution, not bit for bit.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)  # for new_baselines and its alarm imports

import argparse
import signal as _signal
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from pc_integration import PCResult


class _MvpcTimeout(Exception):
    pass


def _causallearn_graph_to_adjacency(G, p):
    """Convert a causal-learn GeneralGraph endpoint matrix to our adjacency
    convention (adj[i,j]=1 for i->j; adj[i,j]=adj[j,i]=1 for undirected).

    causal-learn encoding: G[i,j]==-1 and G[j,i]==1  =>  i -> j;
    G[i,j]==G[j,i]==-1  =>  i -- j. Bidirected (1,1) should not occur under
    PC without latents; mapped to undirected defensively.
    """
    adj = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            gij, gji = G[i, j], G[j, i]
            if gij == 0 and gji == 0:
                continue
            if gij == -1 and gji == 1:
                adj[i, j] = 1
            elif gij == 1 and gji == -1:
                adj[j, i] = 1
            else:  # (-1,-1) undirected, or defensive (1,1)
                adj[i, j] = adj[j, i] = 1
    return adj


def run_pc_mvpc(data, alpha=0.05, random_state=None, timeout=1800,
                correction_name="MV_Crtn_Fisher_Z"):
    """MVPC (Tu et al. 2019) as implemented in causal-learn.

    ``random_state`` is accepted for registry compatibility and ignored.
    MVPC is not deterministic given the data: its correction step shuffles
    the predictor rows with NumPy's unseeded global generator
    (causallearn/utils/PCUtils/Helper.py::get_predictor_ws in 0.1.3.7).
    ``timeout`` (seconds) caps slow correction steps; 0 disables it, as on
    the HAILFINDER path, which sets its own per-replicate SIGALRM.
    """
    t0 = time.time()
    p = data.shape[1]
    from causallearn.search.ConstraintBased.PC import pc

    kwargs = dict(alpha=alpha, indep_test="mv_fisherz", stable=True,
                  mvpc=True, correction_name=correction_name,
                  show_progress=False, verbose=False)

    old_handler = None
    if timeout and timeout > 0:
        def _handler(signum, frame):
            raise _MvpcTimeout()
        old_handler = _signal.signal(_signal.SIGALRM, _handler)
        _signal.alarm(int(timeout))
    try:
        try:
            cg = pc(data, **kwargs)
        except TypeError:
            # Older causal-learn API takes the CIT function object.
            from causallearn.utils.cit import mv_fisherz
            kwargs["indep_test"] = mv_fisherz
            cg = pc(data, **kwargs)
    finally:
        if timeout and timeout > 0:
            _signal.alarm(0)
            if old_handler is not None:
                _signal.signal(_signal.SIGALRM, old_handler)

    adj = _causallearn_graph_to_adjacency(np.asarray(cg.G.graph), p)
    # causal-learn does not expose a test counter; n_ci_tests stays 0.
    return PCResult(adjacency_matrix=adj, sep_sets={},
                    runtime_seconds=time.time() - t0, method="mvpc")


def _install_method(alpha, timeout):
    """Make MVPC the only method in new_baselines' registry, so that its
    replay loops (see that module's docstring) run MVPC."""
    import new_baselines as nb
    nb.METHOD_SPECS[:] = [
        ("mvpc", run_pc_mvpc, dict(alpha=alpha, timeout=timeout)),
    ]
    nb._ACTIVE_METHODS.clear()
    nb._ACTIVE_METHODS.add("mvpc")
    return nb


def smoke_test():
    """Verify the causal-learn MVPC API and estimate per-replicate runtime."""
    from dgps import (generate_erdos_renyi_dag, inject_missingness,
                      sample_from_dag_nonlinear)
    from pc_integration import (dag_to_adjacency,
                                structural_hamming_distance, edge_metrics)
    rng = np.random.RandomState(0)
    p = 10
    edges = generate_erdos_renyi_dag(p, 0.25, rng)
    adj_true = dag_to_adjacency(p, edges)
    data = sample_from_dag_nonlinear(edges, p, 1000, rng)
    data_miss = inject_missingness(data, "mar", [1, 3, 5], 0.3, rng)
    print("Running MVPC on one p=10 MAR replicate...")
    t0 = time.time()
    res = run_pc_mvpc(data_miss, alpha=0.05, timeout=600)
    dt = time.time() - t0
    shd = structural_hamming_distance(adj_true, res.adjacency_matrix)
    em = edge_metrics(adj_true, res.adjacency_matrix)
    print(f"OK: {dt:.1f}s/replicate at p=10; SHD={shd}, "
          f"precision={em['precision']:.2f}, recall={em['recall']:.2f}")
    print(f"Rough grid estimate: p=10 cell (200 replicates) ~ "
          f"{dt*200/3600:.1f}h serial.")
    print("API check passed -- safe to submit the grid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVPC baseline (Tu et al. 2019)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick API/runtime check; run this first")
    parser.add_argument("--benchmark", type=str, default="erdos",
                        choices=["erdos", "alarm", "hailfinder"])
    parser.add_argument("--mechanism", type=str,
                        choices=["mar", "mnar", "mixed", "complete"])
    parser.add_argument("--dgp", type=str, default="nonlinear",
                        choices=["linear", "nonlinear"])
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--edge_prob", type=float, default=None)
    parser.add_argument("--miss_rate", type=float, default=0.3)
    parser.add_argument("--n_miss_cols", type=int, default=None)
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--n_obs", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=20, help="networks only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rep_timeout", type=int, default=1800,
                        help="Per-call timeout in seconds (0 disables)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        sys.exit(0)

    if args.output is None:
        raise SystemExit("--output required (unless --smoke)")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.time()
    if args.benchmark == "erdos":
        if args.mechanism is None:
            raise SystemExit("--mechanism required for benchmark=erdos")
        nb = _install_method(args.alpha, args.rep_timeout)
        edge_prob = args.edge_prob
        if edge_prob is None:
            edge_prob = 0.25 if args.p <= 10 else (0.15 if args.p >= 30 else 0.2)
        n_miss_cols = args.n_miss_cols
        if n_miss_cols is None:
            n_miss_cols = {10: 3, 20: 6, 30: 8}.get(args.p, max(3, args.p // 3))
        print(f"=== MVPC: p={args.p}, dgp={args.dgp}, mechanism={args.mechanism} ===")
        df = nb.run_p_sweep(
            mechanism=args.mechanism, dgp=args.dgp,
            n_graphs=args.n_graphs, n_datasets=args.n_datasets,
            n_obs=args.n_obs, p=args.p, edge_prob=edge_prob,
            miss_rate=args.miss_rate, n_miss_cols=n_miss_cols, seed=args.seed,
        )
    elif args.benchmark == "alarm":
        nb = _install_method(args.alpha, args.rep_timeout)
        print(f"=== MVPC: ALARM, dgp={args.dgp} ===")
        df = nb.run_alarm(dgp=args.dgp, n_reps=args.n_reps,
                          n_obs=args.n_obs, seed=args.seed)
    else:  # hailfinder: its loop has its own SIGALRM -- disable ours
        nb = _install_method(args.alpha, 0)
        print(f"=== MVPC: HAILFINDER, dgp={args.dgp} ===")
        df = nb.run_hailfinder(dgp=args.dgp, n_reps=args.n_reps,
                               n_obs=args.n_obs, seed=args.seed,
                               output_path=args.output,
                               rep_timeout=args.rep_timeout)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows)")
    ok = df[~df["failed"]] if "failed" in df else df
    if len(ok):
        print("\nMedian SHD:")
        print(ok.groupby("mechanism")["shd"].median().round(1).to_string())
