"""Complete-data audit of GCM and KCI configurations.

Re-runs both baselines on complete data, on the calibration DGPs (linear
Gaussian, post-nonlinear, latent confounder) with |X| in {2, 5, 10} and
signal 0 and 0.3, to check that their Table 1 false positive rates are not
artefacts of one configuration. The estimands are the rejection rates at
alpha = 0.05.

GCM (n in {500, 1000, 2000, 5000}, 400 replicates per cell): random-forest
regressions with 50, 100 or 400 trees, residuals by 5-fold cross-fitting
or out-of-bag.
KCI (causal-learn; n in {500, 1000}, 200 replicates per cell):
  kci_default          raw data, empirical width, gamma null (Table 1)
  kci_std              standardised, empirical width, gamma null
  kci_median           standardised, median-heuristic width, gamma null
  kci_median_specnull  standardised, median-heuristic width, simulated
                       (spectral) null; the tuned KCI of Table 1

    python experiments/kci_gcm_audit.py --n_jobs 16 \
        --output results/kci_gcm_audit.csv

Feeds App. R.1 (Table kci_gcm_audit) and Table 1's tuned-KCI entry.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

from dgps import DGP_FUNCTIONS


# ---- GCM variants ----------------------------------------------------------

def gcm_variant(X, Y, Z, n_trees, residuals, rng):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    n = X.shape[0]

    def _fit_residuals(target, seed):
        rf = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=5, n_jobs=1,
            random_state=seed,
            oob_score=(residuals == "oob"), bootstrap=True,
        )
        if residuals == "cv5":
            pred = cross_val_predict(rf, X, target, cv=5)
        else:  # out-of-bag residuals
            rf.fit(X, target)
            pred = rf.oob_prediction_
        return target - pred

    res_y = _fit_residuals(Y, rng.randint(0, 2**31))
    res_z = _fit_residuals(Z, rng.randint(0, 2**31))
    prod = res_y * res_z
    T_n = np.sqrt(n) * np.mean(prod) / np.std(prod)
    pval = 2 * (1 - stats.norm.cdf(abs(T_n)))
    return abs(T_n), pval


# ---- KCI variants ----------------------------------------------------------

def kci_variant(X, Y, Z, est_width, approx, standardise):
    from causallearn.utils.cit import CIT
    data = np.column_stack([Z, Y, X])
    if standardise:
        sd = data.std(axis=0)
        sd[sd == 0] = 1.0
        data = (data - data.mean(axis=0)) / sd
    kwargs = {}
    if est_width is not None:
        kwargs["est_width"] = est_width
    if approx is not None:
        kwargs["approx"] = approx
    cit = CIT(data, "kci", **kwargs)
    pval = cit(0, 1, list(range(2, 2 + X.shape[1])))
    return np.nan, float(pval)


GCM_CONFIGS = [
    ("gcm_50_cv5", dict(n_trees=50, residuals="cv5")),      # as in Table 1
    ("gcm_100_cv5", dict(n_trees=100, residuals="cv5")),
    ("gcm_400_cv5", dict(n_trees=400, residuals="cv5")),
    ("gcm_50_oob", dict(n_trees=50, residuals="oob")),
    ("gcm_100_oob", dict(n_trees=100, residuals="oob")),
    ("gcm_400_oob", dict(n_trees=400, residuals="oob")),
]

KCI_CONFIGS = [
    # kci_default is the configuration of calibration.py and Table 1.
    ("kci_default", dict(est_width=None, approx=None, standardise=False)),
    ("kci_std", dict(est_width="empirical", approx=True, standardise=True)),
    ("kci_median", dict(est_width="median", approx=True, standardise=True)),
    ("kci_median_specnull", dict(est_width="median", approx=False,
                                 standardise=True)),
]


def run_one(family, config_name, cfg, dgp_name, n, dim_x, signal, rep, seed):
    rng = np.random.RandomState(seed)
    X, Y, Z = DGP_FUNCTIONS[dgp_name](n, dim_x, signal, rng)
    t0 = time.time()
    try:
        if family == "gcm":
            stat, pval = gcm_variant(X, Y, Z, rng=rng, **cfg)
        else:
            stat, pval = kci_variant(X, Y, Z, **cfg)
        failed = not np.isfinite(pval)
    except Exception:
        stat, pval, failed = np.nan, np.nan, True
    return dict(family=family, config=config_name, dgp=dgp_name, n=n,
                dim_x=dim_x, signal=signal, rep=rep,
                statistic=stat, p_value=pval,
                rejected=(pval < 0.05) if not failed else False,
                failed=failed, runtime=time.time() - t0)


def build_jobs(args):
    # One seed per job, drawn in a fixed order from the master generator, so
    # results do not depend on --n_jobs or --batch.
    jobs = []
    master = np.random.RandomState(args.seed)
    dgps = ["linear_gaussian", "post_nonlinear", "latent_confounder"]
    gcm_cfgs = [(n, c) for n, c in GCM_CONFIGS
                if not args.gcm_configs or n in args.gcm_configs]
    kci_cfgs = [(n, c) for n, c in KCI_CONFIGS
                if not args.kci_configs or n in args.kci_configs]
    for dgp in dgps:
        for dim_x in args.dims:
            for signal in [0.0, 0.3]:
                for name, cfg in gcm_cfgs:
                    for n in args.gcm_ns:
                        for rep in range(args.gcm_reps):
                            jobs.append(("gcm", name, cfg, dgp, n, dim_x,
                                         signal, rep,
                                         master.randint(0, 2**31)))
                for name, cfg in kci_cfgs:
                    for n in args.kci_ns:
                        for rep in range(args.kci_reps):
                            jobs.append(("kci", name, cfg, dgp, n, dim_x,
                                         signal, rep,
                                         master.randint(0, 2**31)))
    return jobs


def analyze(df):
    ok = df[~df.failed]
    print("\n" + "=" * 70)
    print("FPR at signal=0 (nominal 0.05), by config x dgp (pooled over n, |X|):")
    null = ok[ok.signal == 0]
    print(null.groupby(["family", "config", "dgp"]).rejected.mean()
          .unstack("dgp").round(3).to_string())
    print("\nFPR by config x n (pooled over dgp, |X|):")
    print(null.groupby(["family", "config", "n"]).rejected.mean()
          .unstack("n").round(3).to_string())
    print("\nPower at signal=0.3 by config (pooled):")
    print(ok[ok.signal == 0.3].groupby(["family", "config"]).rejected.mean()
          .round(3).to_string())
    print("\nMedian runtime (s) by config:")
    print(ok.groupby(["family", "config"]).runtime.median().round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KCI/GCM audit (complete data)")
    parser.add_argument("--gcm_reps", type=int, default=400)
    parser.add_argument("--kci_reps", type=int, default=200)
    parser.add_argument("--gcm_ns", type=int, nargs="+",
                        default=[500, 1000, 2000, 5000])
    parser.add_argument("--kci_ns", type=int, nargs="+", default=[500, 1000])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--gcm_configs", nargs="+", default=None,
                        help="Subset of GCM config names (default: all)")
    parser.add_argument("--kci_configs", nargs="+", default=None,
                        help="Subset of KCI config names (default: all)")
    parser.add_argument("--batch", type=int, default=4000,
                        help="Jobs per batch; CSV saved after each batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str,
                        default="results/kci_gcm_audit.csv")
    args = parser.parse_args()

    if args.quick:
        args.gcm_reps, args.kci_reps = 10, 5
        args.gcm_ns, args.kci_ns = [500], [500]
        args.dims = [2]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    jobs = build_jobs(args)
    print(f"{len(jobs)} test evaluations queued "
          f"({args.n_jobs} workers, batches of {args.batch})", flush=True)
    t0 = time.time()
    rows = []
    for start in range(0, len(jobs), args.batch):
        chunk = jobs[start:start + args.batch]
        rows.extend(Parallel(n_jobs=args.n_jobs)(
            delayed(run_one)(*j) for j in chunk))
        pd.DataFrame(rows).to_csv(args.output, index=False)
        done = start + len(chunk)
        el = (time.time() - t0) / 3600
        eta = el / done * (len(jobs) - done)
        print(f"  {done}/{len(jobs)} done ({el:.1f}h elapsed, "
              f"~{eta:.1f}h remaining); saved", flush=True)
    print(f"Total: {(time.time()-t0)/60:.1f} min")
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows)")
    analyze(df)
