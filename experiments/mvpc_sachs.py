"""MVPC (Tu et al. 2019) on the Sachs benchmarks, on the paper's datasets.

Replays the random streams of experiments/sachs.py (the real flow-cytometry
data, bootstrap resamples of the 853 rows) and experiments/sachs_nonlinear.py
(nonlinear data simulated on the Sachs structure, one job per (rate,
mechanism) cell, as in the released run), and runs MVPC on each replicate.
Test-wise deletion is run alongside as a check that the datasets coincide:
its SHD must equal the released value replicate by replicate. Estimands as
in Appendix J: median SHD and skeleton precision, recall and F1.

    python experiments/mvpc_sachs.py --dgp real --output results/mvpc_sachs_real.csv
    python experiments/mvpc_sachs.py --dgp nonlinear --output results/mvpc_sachs_nonlinear.csv

Feeds the MVPC rows of the Sachs tables in Appendix J.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from pc_integration import run_pc_testwise_deletion  # noqa: E402
from mvpc_baseline import run_pc_mvpc  # noqa: E402


def _methods(data_miss, alpha, seed, **_):
    # The outer loop has already drawn this replicate's seed from the shared
    # stream; neither method consumes it, so the stream is unchanged.
    out = {}
    for name, fn in (("mvpc", lambda d: run_pc_mvpc(d, alpha=alpha, timeout=1800)),
                     ("testwise", lambda d: run_pc_testwise_deletion(d, alpha=alpha))):
        try:
            out[name] = fn(data_miss)
        except Exception as e:  # noqa: BLE001
            print(f"    {name} failed: {e}")
            out[name] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dgp", choices=["real", "nonlinear"], required=True)
    ap.add_argument("--data", default="data/sachs_data.csv")
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    if a.dgp == "real":
        import sachs as mod
        mod.run_all_methods = _methods
        df = mod.run_sachs_experiment(a.data, n_reps=a.n_reps, seed=a.seed, alpha=0.05)
    else:
        import sachs_nonlinear as mod
        mod.run_all_methods = _methods
        cells = [(r, m) for r in (0.1, 0.2, 0.4) for m in ("complete", "mar", "mnar", "mixed")
                 if not (m == "complete" and r > 0.1)]
        df = pd.concat([mod.run_sachs_synthetic(n_obs=1000, n_reps=a.n_reps, seed=a.seed, alpha=0.05,
                                                dgp="nonlinear", rate_filter=r, mech_filter=m)
                        for r, m in cells], ignore_index=True)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"wrote {a.output}: {len(df)} rows")


if __name__ == "__main__":
    main()
