"""Search check: rerun the in-house baselines on one network's 20%-MNAR cell
under the simplified and the standard PC-stable adjacency search, on
identical data and seeds.

Estimand. Per replicate of the Table 3 cell (rate 0.2, MNAR, n = 1000,
20 replicates, the --seed 42 stream of experiments/{alarm,hailfinder}.py),
the SHD and skeleton precision, recall and F1 of complete-case PC, FZ-vote
(impute_fisherz) and test-wise deletion under

  (a) ``__orig``: the simplified search of the experiments
      (src/pc_integration.py::_pc_skeleton: conditioning sets from the
      lower-indexed endpoint's neighbourhood only, stopping at the first
      level at which no edge is removed); and
  (b) ``__std``: the standard PC-stable adjacency search (Kalisch and
      Buhlmann 2007, Alg. 1, with the level-frozen adjacency sets of
      Colombo and Maathuis 2014, Alg. 4.1; both ordered pairs; standard
      termination).

The data, the missingness draw and every method's seed are those of the
released run, so per-replicate differences between (a) and (b) isolate the
search. Arm (a) reproduces the per-replicate SHD in results/*_results.csv
exactly for test-wise deletion and complete case. FZ-vote's IterativeImputer
is not bit-identical across platforms and BLAS builds, so its arm (a)
matches the released medians to within 0.5 SHD but not replicate by
replicate (see the README). PAIR-CI is not rerun; analysis/
search_check_summary.py compares these rows with PAIR-CI's Table 3 rows.
Only the skeleton phase is swapped; orientation (_orient_edges) is
unchanged. The Fisher-Z tests are symmetric, so the standard search caches
decisions by unordered pair.

    python experiments/search_check_baselines.py --network alarm_nonlinear
    python experiments/search_check_baselines.py --network hailfinder

Writes results/search_check/rerun_alarm_nl.csv and rerun_hailf.csv by
default; feeds App. E ("PC algorithm with cached imputations").
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import pc_integration as pci  # noqa: E402

ORIG = pci._pc_skeleton
DEFAULT_OUT = {"alarm_nonlinear": "results/search_check/rerun_alarm_nl.csv",
               "hailfinder": "results/search_check/rerun_hailf.csv"}


def std_skeleton(p, ci_test, alpha, max_depth, verbose):
    """PC-stable adjacency search: Kalisch and Buhlmann Alg. 1 with the
    level-frozen adjacency sets of Colombo and Maathuis Alg. 4.1."""
    adj = np.ones((p, p), dtype=int) - np.eye(p, dtype=int); sep = {}; n = 0; l = 0; cache = {}
    max_d = max_depth if max_depth is not None else p - 2
    while l <= max_d:
        frozen = {i: [k for k in range(p) if adj[i, k]] for i in range(p)}
        if not any(len(frozen[i]) - 1 >= l for i in range(p)): break
        for i in range(p):
            for j in range(p):
                if i == j or not adj[i, j]: continue
                nb = [k for k in frozen[i] if k != j]
                if len(nb) < l: continue
                for S in itertools.combinations(nb, l):
                    key = (min(i, j), max(i, j), tuple(sorted(S)))
                    if key not in cache:
                        cache[key] = ci_test.test(i, j, S)[1]; n += 1
                    if cache[key] >= alpha:
                        adj[i, j] = adj[j, i] = 0; sep[(i, j)] = sep[(j, i)] = set(S); break
        l += 1
    return adj, sep, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True, choices=["alarm_nonlinear", "hailfinder"])
    ap.add_argument("--output", default=None,
                    help="CSV path (default: results/search_check/rerun_{alarm_nl,hailf}.csv)")
    a = ap.parse_args()
    out = a.output or os.path.join(ROOT, DEFAULT_OUT[a.network])
    net = a.network

    mod = __import__("hailfinder" if net == "hailfinder" else "alarm")
    from pc_integration import (run_pc_complete_case, run_pc_impute_then_test,
                                run_pc_testwise_deletion)
    calls = {"n": 0}
    # ALARM runs every cell; (0.2, mnar) is the 6th in loop order, so its
    # 20 replicates are calls 100-119. HAILFINDER runs that cell alone.
    TARGET = range(100, 120)

    def my_run_all(data_miss, alpha, seed, dgp="nonlinear"):
        k = calls["n"]; calls["n"] += 1
        if net == "alarm_nonlinear" and k not in TARGET: return {}
        # Skip PAIR-CI's draw so FZ-vote gets the seed it had in the run.
        rng = np.random.RandomState(seed); _ = rng.randint(0, 2**31)
        s_imp = rng.randint(0, 2**31); res = {}
        for tag, skel in (("orig", ORIG), ("std", std_skeleton)):
            pci._pc_skeleton = skel
            for name, fn, kw in (("complete_case", run_pc_complete_case, dict(alpha=alpha)),
                                 ("impute_fisherz", run_pc_impute_then_test, dict(alpha=alpha, n_imputations=5, random_state=s_imp)),
                                 ("testwise", run_pc_testwise_deletion, dict(alpha=alpha))):
                try: res[f"{name}__{tag}"] = fn(data_miss, **kw)
                except Exception as e: print("fail", name, tag, e, flush=True); res[f"{name}__{tag}"] = None
        pci._pc_skeleton = ORIG
        print(f"rep done ({time.strftime('%H:%M:%S')})", flush=True)
        return res

    mod.run_all_methods = my_run_all
    if net == "hailfinder":
        df = mod.run_hailfinder_experiment(n_obs=1000, n_reps=20, seed=42, alpha=0.05, dgp="nonlinear",
                                           rate_filter=0.2, mech_filter="mnar")
    else:
        df = mod.run_alarm_experiment(n_obs=1000, n_reps=20, seed=42, alpha=0.05, dgp="nonlinear")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
