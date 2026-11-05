"""MVPC (Tu et al. 2019) on the p = 10 rate and sample-size sweep.

Replays the random stream of experiments/robustness.py (10 Erdos-Renyi
graphs with nonlinear edges, 20 datasets per graph, rates 10/30/50%,
n in {500, 1000, 2000}, MAR and MNAR) and runs MVPC on every dataset.
Test-wise deletion is run alongside as a check that the datasets coincide:
its SHD must equal the released value dataset by dataset. Estimands as in
Appendix F: median SHD and skeleton F1 per (mechanism, rate, n).

    python experiments/mvpc_robustness.py --output results/mvpc_robustness.csv

Feeds the MVPC rows of the Appendix F sweep tables.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)


def _methods(data_miss, alpha, n_imputations, n_folds, random_state):
    from pc_integration import run_pc_testwise_deletion
    from mvpc_baseline import run_pc_mvpc
    out = {}
    for name, fn in (("mvpc", lambda d: run_pc_mvpc(d, alpha=alpha, timeout=1800)),
                     ("testwise", lambda d: run_pc_testwise_deletion(d, alpha=alpha))):
        try:
            out[name] = fn(data_miss)
        except Exception as e:  # noqa: BLE001
            print(f"    {name} failed: {e}")
            out[name] = None
    return out


def _one(cfg):
    import robustness
    robustness.run_all_methods = _methods
    mech, rate, n_obs, seed = cfg
    return robustness.run_robustness(mechanism=mech, miss_rate=rate, n_obs=n_obs,
                                     n_graphs=10, n_datasets=20, n_folds=10, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=6)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    import robustness
    cfgs = [(m, r, n, a.seed) for m in robustness.MECHANISMS for r in robustness.MISS_RATES
            for n in robustness.N_OBS_GRID]
    with Pool(a.n_jobs) as pool:
        parts = pool.map(_one, cfgs)
    df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"wrote {a.output}: {len(df)} rows from {len(cfgs)} configurations")


if __name__ == "__main__":
    main()
