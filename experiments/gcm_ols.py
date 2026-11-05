"""GCM with a correctly specified linear learner, on complete data.

GCM's guarantee (Shah and Peters 2020) needs the product of the two
regressions' errors to vanish faster than n^(-1/2). OLS meets this on the
linear-Gaussian DGP, so GCM-OLS should be calibrated there, and is
misspecified on the post-nonlinear one. The test statistic and 5-fold
cross-fitting are those of calibration.py's gcm_test, with LinearRegression
in place of the 50-tree forest. Grid: the three calibration DGPs, n in
{500, 1000, 2000, 5000}, |X| in {2, 5, 10} (cells with |X| >= n/50
skipped), signal 0 and 0.3, 300 replicates per cell. The estimands are
the rejection rates at alpha = 0.05.

    python experiments/gcm_ols.py --n_jobs 8 \
        --output results/gcm_ols_audit.csv

Each task seeds its own generator from (seed, config index, replicate), so
results do not depend on --n_jobs. Feeds App. R.1 (Table gcm_ols).
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
from itertools import product
from scipy import stats
from joblib import Parallel, delayed

from dgps import DGP_FUNCTIONS


def gcm_ols_pval(X, Y, Z):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_predict
    n = X.shape[0]
    res_y = Y - cross_val_predict(LinearRegression(), X, Y, cv=5)
    res_z = Z - cross_val_predict(LinearRegression(), X, Z, cv=5)
    prod = res_y * res_z
    T_n = np.sqrt(n) * np.mean(prod) / np.std(prod)
    return abs(T_n), 2 * (1 - stats.norm.cdf(abs(T_n)))


def run_one(task):
    dgp_name, n, dim_x, signal, seed_material = task
    ss = np.random.SeedSequence(seed_material)
    rng = np.random.RandomState(ss.generate_state(1)[0] % (2**31))
    X, Y, Z = DGP_FUNCTIONS[dgp_name](n, dim_x, signal, rng)
    try:
        stat, pval = gcm_ols_pval(X, Y, Z)
        return dict(dgp=dgp_name, n=n, dim_x=dim_x, signal=signal,
                    learner="ols", statistic=stat, p_value=pval,
                    rejected=pval < 0.05, failed=False)
    except Exception:
        return dict(dgp=dgp_name, n=n, dim_x=dim_x, signal=signal,
                    learner="ols", statistic=np.nan, p_value=np.nan,
                    rejected=False, failed=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n_reps", type=int, default=300)
    ap.add_argument("--dgps", nargs="+",
                    default=["linear_gaussian", "post_nonlinear",
                             "latent_confounder"])
    ap.add_argument("--sample_sizes", nargs="+", type=int,
                    default=[500, 1000, 2000, 5000])
    ap.add_argument("--dim_xs", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--signals", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--output", type=str, default="results/gcm_ols_audit.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_reps = 5
        args.sample_sizes = [300]
        args.dim_xs = [2]
        args.output = "results/gcm_ols_smoke.csv"

    configs = [(d, n, dx, s) for d, n, dx, s in
               product(args.dgps, args.sample_sizes, args.dim_xs,
                       args.signals) if dx < n / 50]
    tasks = [(d, n, dx, s, [args.seed, ci, rep])
             for ci, (d, n, dx, s) in enumerate(configs)
             for rep in range(args.n_reps)]
    print(f"{len(configs)} configs x {args.n_reps} reps = {len(tasks)} tests")

    t0 = time.time()
    rows = Parallel(n_jobs=args.n_jobs)(delayed(run_one)(t) for t in tasks)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {args.output} [{(time.time() - t0) / 60:.1f} min]")

    ok = df[~df["failed"]]
    null = ok[ok["signal"] == 0]
    print("\nGCM-OLS FPR by DGP (complete data, signal=0):")
    print(null.groupby(["dgp", "n"])["rejected"].mean()
          .unstack("n").round(3).to_string())
    alt = ok[ok["signal"] > 0]
    if len(alt):
        print("\nGCM-OLS power (signal=0.3):")
        print(alt.groupby(["dgp", "n"])["rejected"].mean()
              .unstack("n").round(3).to_string())


if __name__ == "__main__":
    main()
