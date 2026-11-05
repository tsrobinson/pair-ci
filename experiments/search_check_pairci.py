"""PAIR-CI under the standard PC-stable adjacency search, for one replicate
of the 20%-MNAR cell of ALARM-nonlinear or HAILFINDER. Optional; it has not
been run and its outputs are not in the paper.

The search check of App. E reruns only the baselines under PC-stable
(search_check_baselines.py) and compares them with PAIR-CI on the
simplified search. This script is the complementary arm, PAIR-CI (fast
variant, as in Table 3) on the standard search, shipped so that the
comparison can be completed.

Estimand: per-replicate SHD and skeleton precision, recall and F1 of
PAIR-CI when the in-house search (pc_integration._pc_skeleton: one
endpoint's neighbourhood, early stop) is replaced by Kalisch and Buhlmann
(2007) Alg. 1 with the level-frozen adjacency sets of Colombo and Maathuis
(2014) Alg. 4.1. The data, the missingness draw and PAIR-CI's seed are
those of the released run (the --seed 42 stream of
experiments/{alarm,hailfinder}.py), so per-replicate differences from
results/*_results.csv isolate the search. PAIR-CI's test is asymmetric in
its two variables, so ordered triples are cached separately.

The standard search issues many more CI tests than the simplified one;
allow up to 12 h per replicate on 8 cores (slurm/submit_search_check.sh
pairci submits 2 networks x 20 replicates as 40 jobs).

    python experiments/search_check_pairci.py --network alarm_nonlinear --rep 0

Writes results/search_check/pairci_stdsearch_{network}_rep{NN}.csv.
"""
import argparse, itertools, os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import pc_integration as pci  # noqa: E402


def std_skeleton(p, ci_test, alpha, max_depth, verbose):
    """PC-stable adjacency search, caching p-values by ordered (i, j, S)."""
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
                    key = (i, j, S)
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
    ap.add_argument("--rep", type=int, required=True, help="replicate index 0-19")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "search_check"),
                    help="output directory (default: results/search_check/)")
    a = ap.parse_args()
    mod = __import__("hailfinder" if a.network == "hailfinder" else "alarm")
    mod._pc_skeleton = std_skeleton; pci._pc_skeleton = std_skeleton
    state = {"k": 0, "row": None}
    # ALARM runs every cell; (0.2, mnar) is the 6th, so replicate r is call
    # 100 + r. HAILFINDER runs that cell alone.
    target = (100 + a.rep) if a.network == "alarm_nonlinear" else a.rep

    def one(data_miss, alpha, seed, dgp="nonlinear"):
        k = state["k"]; state["k"] += 1
        if k != target: return {}
        rng = np.random.RandomState(seed)
        print(f'starting rep {a.rep} (call {k}, PAIR-CI seed from {seed}); miss share {np.isnan(data_miss).mean():.4f}', flush=True)
        t0 = time.time()
        res = mod.run_pc_fast(data_miss, alpha=alpha, random_state=rng.randint(0, 2**31), ci_test_cls=None)
        print(f"rep {a.rep}: {time.time()-t0:.0f}s, {res.n_ci_tests} tests", flush=True)
        return {"ci_oracle_fast__std": res}

    mod.run_all_methods = one
    if a.network == "hailfinder":
        df = mod.run_hailfinder_experiment(n_obs=1000, n_reps=20, seed=42, alpha=0.05, dgp="nonlinear", rate_filter=0.2, mech_filter="mnar")
    else:
        df = mod.run_alarm_experiment(n_obs=1000, n_reps=20, seed=42, alpha=0.05, dgp="nonlinear")
    df = df[df["rep"] == a.rep]
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"pairci_stdsearch_{a.network}_rep{a.rep:02d}.csv"), index=False)


if __name__ == "__main__":
    main()
