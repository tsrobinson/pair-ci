"""Dependence-valid pooling of per-imputation p-values for GCM and KCI.

Supplementary diagnostic; not reported in the paper. Fisher's method, which
calibration.py uses to pool imputed KCI's M per-imputation p-values,
assumes independence, although the M imputations share their data. This
script asks whether the imputed pipelines' false positive rates under
missingness survive pooling rules that are valid under dependence. For each
replicate it records all M p-values of GCM, KCI and, if the installed
causal-learn provides it, RCIT, and pools them five ways:

    fisher          -2 sum log p ~ chi2_{2M}    (assumes independence)
    twice_median    min(1, 2 median p)          (valid under any dependence)
    twice_mean      min(1, 2 mean p)            (valid under any dependence)
    bonferroni_min  min(1, M min p)             (valid under any dependence)
    licht_rubin     Rubin's rules on z_m = Phi^{-1}(p_m), within variance 1

Grid: the calibration DGPs, n in {500, 1000, 2000}, |X| in {2, 5, 10}
(cells with |X| >= n/50 skipped), signal 0, MAR and MNAR at 30%, M = 5,
100 replicates. Every method uses M MICE draws (posterior sampling) on
[X_miss, Z, Y], as calibration.py's imputed KCI does; calibration.py's
imputed GCM instead uses a single draw. The estimand is the rejection rate
at alpha = 0.05 per method and rule.

    python experiments/pooling_validity.py --n_jobs 16 \
        --output results/pooling_validity.csv

Per-task seeds come from SeedSequence([seed, config index, replicate]), so
results do not depend on --n_jobs, but the datasets differ from those of
calibration.py.
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
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from joblib import Parallel, delayed

from dgps import DGP_FUNCTIONS, inject_missingness_calibration


def pool_pvals(pvals):
    p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1.0)
    M = len(p)
    out = {}
    chi2 = -2.0 * np.sum(np.log(p))
    out["fisher"] = 1.0 - stats.chi2.cdf(chi2, 2 * M)
    out["twice_median"] = min(1.0, 2.0 * float(np.median(p)))
    out["twice_mean"] = min(1.0, 2.0 * float(np.mean(p)))
    out["bonferroni_min"] = min(1.0, M * float(np.min(p)))
    z = stats.norm.ppf(np.clip(p, 1e-15, 1 - 1e-15))
    zbar = float(np.mean(z))
    B = float(np.var(z, ddof=1)) if M > 1 else 0.0
    T = 1.0 + (1.0 + 1.0 / M) * B  # W = 1 on the probit scale
    out["licht_rubin"] = float(stats.norm.cdf(zbar / np.sqrt(T)))
    return out


def gcm_pval(X, Y, Z):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    n = X.shape[0]
    rf_y = RandomForestRegressor(n_estimators=50, min_samples_leaf=5,
                                 n_jobs=1, random_state=42)
    rf_z = RandomForestRegressor(n_estimators=50, min_samples_leaf=5,
                                 n_jobs=1, random_state=43)
    res_y = Y - cross_val_predict(rf_y, X, Y, cv=5)
    res_z = Z - cross_val_predict(rf_z, X, Z, cv=5)
    prod = res_y * res_z
    T_n = np.sqrt(n) * np.mean(prod) / np.std(prod)
    return 2 * (1 - stats.norm.cdf(abs(T_n)))


def kernel_pval(data, z_idx, y_idx, cond_set, kind):
    """kind is 'kci' or 'rcit'; causal-learn is imported lazily."""
    from causallearn.utils.cit import CIT
    cit = CIT(data, kind)
    return cit(z_idx, y_idx, list(cond_set))


def run_one(task):
    (dgp_name, n, dim_x, signal, mechanism, miss_rate, n_imp,
     methods, seed_material) = task
    ss = np.random.SeedSequence(seed_material)
    rng = np.random.RandomState(ss.generate_state(1)[0] % (2**31))

    X, Y, Z = DGP_FUNCTIONS[dgp_name](n, dim_x, signal, rng)
    X_miss = inject_missingness_calibration(X, mechanism, miss_rate, rng)

    # As in calibration.py's kci_imputed_test: Z and Y are complete and
    # serve as predictors in each imputation.
    data_full = np.column_stack([X_miss, Z, Y])
    z_idx, y_idx = dim_x, dim_x + 1
    cond_set = tuple(range(dim_x))

    imputed = []
    for _ in range(n_imp):
        imp = IterativeImputer(max_iter=10,
                               random_state=rng.randint(0, 2**31),
                               sample_posterior=True)
        imputed.append(imp.fit_transform(data_full))

    base = dict(dgp=dgp_name, n=n, dim_x=dim_x, signal=signal,
                mechanism=mechanism, miss_rate=miss_rate, M=n_imp)
    rows = []
    for method in methods:
        pvals, failed = [], 0
        for d in imputed:
            try:
                if method == "gcm":
                    pv = gcm_pval(d[:, :dim_x], d[:, y_idx], d[:, z_idx])
                else:
                    pv = kernel_pval(d, z_idx, y_idx, cond_set, method)
                if pv is not None and np.isfinite(pv):
                    pvals.append(float(pv))
                else:
                    failed += 1
            except Exception:
                failed += 1
        if not pvals:
            rows.append({**base, "method": method, "pool_rule": "none",
                         "p_value": np.nan, "n_pvals": 0, "n_failed": failed,
                         "pvals": "", "failed": True})
            continue
        pooled = pool_pvals(pvals)
        pv_str = ";".join(f"{p:.6g}" for p in pvals)
        for rule, pv in pooled.items():
            rows.append({**base, "method": method, "pool_rule": rule,
                         "p_value": pv, "n_pvals": len(pvals),
                         "n_failed": failed, "pvals": pv_str,
                         "failed": False})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n_reps", type=int, default=100)
    ap.add_argument("--n_imp", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["gcm", "kci", "rcit"])
    ap.add_argument("--dgps", nargs="+",
                    default=["linear_gaussian", "post_nonlinear",
                             "latent_confounder"])
    ap.add_argument("--sample_sizes", nargs="+", type=int,
                    default=[500, 1000, 2000])
    ap.add_argument("--dim_xs", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--signals", nargs="+", type=float, default=[0.0])
    ap.add_argument("--mechanisms", nargs="+", default=["mar", "mnar"])
    ap.add_argument("--miss_rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--output", type=str,
                    default="results/pooling_validity.csv")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid to verify the pipeline end to end")
    args = ap.parse_args()

    if args.smoke:
        args.n_reps = 2
        args.sample_sizes = [300]
        args.dim_xs = [2]
        args.dgps = ["linear_gaussian"]
        args.output = "results/pooling_validity_smoke.csv"

    # Probe for RCIT, which causal-learn 0.1.3.7 may lack.
    methods = list(args.methods)
    if "rcit" in methods:
        try:
            probe = np.random.RandomState(0).normal(size=(60, 3))
            kernel_pval(probe, 0, 1, (2,), "rcit")
            print("RCIT: available in this causal-learn install.")
        except Exception as e:
            print(f"RCIT: NOT available ({type(e).__name__}: {e}) -- skipping.")
            methods = [m for m in methods if m != "rcit"]

    configs = [(d, n, dx, s, m) for d, n, dx, s, m in
               product(args.dgps, args.sample_sizes, args.dim_xs,
                       args.signals, args.mechanisms)
               if dx < n / 50]
    tasks = []
    for ci, (d, n, dx, s, m) in enumerate(configs):
        for rep in range(args.n_reps):
            tasks.append((d, n, dx, s, m, args.miss_rate, args.n_imp,
                          methods, [args.seed, ci, rep]))
    print(f"{len(configs)} configs x {args.n_reps} reps = {len(tasks)} tasks; "
          f"methods={methods}, n_jobs={args.n_jobs}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    t0 = time.time()
    all_rows, CHUNK = [], 400
    for lo in range(0, len(tasks), CHUNK):
        chunk = tasks[lo:lo + CHUNK]
        got = Parallel(n_jobs=args.n_jobs)(delayed(run_one)(t) for t in chunk)
        for rows in got:
            all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(args.output, index=False)
        print(f"  {min(lo + CHUNK, len(tasks))}/{len(tasks)} tasks "
              f"[{(time.time() - t0) / 60:.1f} min] -> {args.output}")

    df = pd.DataFrame(all_rows)
    ok = df[~df["failed"]]
    null = ok[ok["signal"] == 0]
    if len(null):
        print("\nFPR by method x pool_rule x mechanism (signal=0, alpha=0.05):")
        null = null.assign(rej=null["p_value"] < 0.05)
        print(null.groupby(["method", "pool_rule", "mechanism"])["rej"]
              .mean().unstack("mechanism").round(3).to_string())


if __name__ == "__main__":
    main()
