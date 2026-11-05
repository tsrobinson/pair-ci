"""Direct accuracy of the MICE imputer where the true values are known.

--benchmark calibration: the DGPs of calibration.py, n in {500, 1000, 2000,
5000}, |X| in {2, 5, 10}, signal 0 and 0.3, MAR and MNAR at per-column
rates 0.1, 0.3 and 0.5, 100 replicates per cell. X is imputed M = 5 times
by MICE with posterior draws, with Y and Z as auxiliary predictors, as in
the experiments.
--benchmark erdos: the p = 10 nonlinear graph setting (MAR, MNAR, mixed)
with full-matrix MICE, as in the cached PC pipeline. Its output is not used
in the paper.

Per cell and draw (and for the across-draw mean imputation, m = -1): R^2 on
the missing entries, NRMSE, and the correlations corr(delta, Y) and
corr(delta, Z) of the imputation error delta = imputed - true with the test
variables, the empirical analogues of the kappa factors (App. C).

    python experiments/imputer_quality.py --benchmark calibration \
        --n_jobs 16 --output results/imputer_quality_calibration.csv

Each cell seeds its own generator from a master stream drawn in a fixed
order, so results do not depend on --n_jobs. Feeds App. S.3 (Table
imputer_quality).
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
from joblib import Parallel, delayed
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from dgps import (
    DGP_FUNCTIONS, inject_missingness_calibration, impute_datasets,
    generate_erdos_renyi_dag, inject_missingness, sample_from_dag_nonlinear,
)


def _entry_metrics(X_true, X_imp_list, miss_mask, Y, Z):
    """Metrics over all missing entries, per draw and for the mean draw.

    `r2` and `nrmse` are relative to the variance of the true missing
    values, a tail-selected subpopulation under quantile-driven
    missingness, which makes them a demanding benchmark; `nrmse_marginal`
    is relative to the column's full marginal s.d. Per-draw rows
    (m = 0..M-1) include the posterior noise that proper imputation adds
    (per-entry MSE about twice the conditional variance); the m = -1 row
    scores the across-M mean imputation, the point-prediction measure.
    """
    out = []
    X_mean_imp = np.mean(np.stack(X_imp_list), axis=0)
    for m, X_imp in list(enumerate(X_imp_list)) + [(-1, X_mean_imp)]:
        deltas, trues, y_al, z_al, marg_vars, cnts = [], [], [], [], [], []
        for c in range(X_true.shape[1]):
            rows = miss_mask[:, c]
            if rows.sum() == 0:
                continue
            deltas.append(X_imp[rows, c] - X_true[rows, c])
            trues.append(X_true[rows, c])
            y_al.append(Y[rows])
            z_al.append(Z[rows] if Z is not None else Y[rows] * np.nan)
            marg_vars.append(np.var(X_true[:, c]))
            cnts.append(int(rows.sum()))
        if not deltas:
            continue
        delta = np.concatenate(deltas)
        true = np.concatenate(trues)
        yv = np.concatenate(y_al)
        zv = np.concatenate(z_al)
        var_true = np.var(true)
        var_marg = float(np.average(marg_vars, weights=cnts))
        mse = np.mean(delta ** 2)
        r2 = 1.0 - mse / var_true if var_true > 0 else np.nan
        nrmse = np.sqrt(mse) / np.sqrt(var_true) if var_true > 0 else np.nan
        nrmse_marg = np.sqrt(mse) / np.sqrt(var_marg) if var_marg > 0 else np.nan
        cy = np.corrcoef(delta, yv)[0, 1] if len(delta) > 2 else np.nan
        cz = (np.corrcoef(delta, zv)[0, 1]
              if (Z is not None and len(delta) > 2) else np.nan)
        out.append(dict(m=m, r2=r2, nrmse=nrmse, nrmse_marginal=nrmse_marg,
                        corr_delta_y=cy, corr_delta_z=cz,
                        n_missing_entries=int(len(delta))))
    return out


def run_calibration_cell(dgp_name, n, dim_x, signal, mechanism, rate, rep,
                         seed, n_imp=5):
    rng = np.random.RandomState(seed)
    X, Y, Z = DGP_FUNCTIONS[dgp_name](n, dim_x, signal, rng)
    X_miss = inject_missingness_calibration(X, mechanism, rate, rng)
    miss_mask = np.isnan(X_miss)
    if not miss_mask.any():
        return []
    # Y and Z are auxiliary predictors in the imputation model.
    X_imps = impute_datasets(X_miss, rng, n_imp=n_imp,
                             aux=np.column_stack([Y, Z]))
    rows = []
    for r in _entry_metrics(X, X_imps, miss_mask, Y, Z):
        rows.append(dict(benchmark="calibration", dgp=dgp_name, n=n,
                         dim_x=dim_x, signal=signal, mechanism=mechanism,
                         miss_rate=rate, rep=rep, **r))
    return rows


def run_erdos_cell(mechanism, rep, seed, p=10, n_obs=1000, edge_prob=0.25,
                   rate=0.3, n_imp=5):
    rng = np.random.RandomState(seed)
    edges = generate_erdos_renyi_dag(p, edge_prob, rng)
    has_parent = [j for j in range(p) if any(True for (i, c) in edges if c == j)]
    if len(has_parent) >= 3:
        miss_cols = sorted(rng.choice(has_parent, 3, replace=False))
    else:
        miss_cols = sorted(rng.choice(range(p), 3, replace=False))
    data = sample_from_dag_nonlinear(edges, p, n_obs, rng)
    data_miss = inject_missingness(data, mechanism, miss_cols, rate, rng)
    miss_mask = np.isnan(data_miss)
    if not miss_mask.any():
        return []
    # Full-matrix MICE, as in the cached PC pipeline.
    imps = []
    for _ in range(n_imp):
        imp = IterativeImputer(max_iter=10,
                               random_state=rng.randint(0, 2**31),
                               sample_posterior=True)
        imps.append(imp.fit_transform(data_miss))
    rows = []
    for r in _entry_metrics(data, imps, miss_mask, Y=data[:, 0], Z=None):
        r.pop("corr_delta_y", None)
        r.pop("corr_delta_z", None)
        rows.append(dict(benchmark="erdos", dgp="nonlinear", n=n_obs,
                         dim_x=p, signal=np.nan, mechanism=mechanism,
                         miss_rate=rate, rep=rep, **r))
    return rows


def analyze(df):
    print("\n" + "=" * 70)
    mean_imp = df[df.m == -1]
    draws = df[df.m >= 0]
    print("Median R^2 of the MEAN imputation (m=-1) by mechanism x rate:")
    print(mean_imp.groupby(["benchmark", "mechanism", "miss_rate"]).r2
          .median().round(3).to_string())
    print("\nMedian per-draw NRMSE (vs marginal SD) by mechanism x rate:")
    print(draws.groupby(["benchmark", "mechanism", "miss_rate"])
          .nrmse_marginal.median().round(3).to_string())
    df = draws  # per-draw rows for the remaining breakdowns
    print("\nMedian R^2 by dgp x mechanism (calibration benchmark):")
    cal = df[df.benchmark == "calibration"]
    if len(cal):
        print(cal.groupby(["dgp", "mechanism"]).r2.median()
              .unstack("mechanism").round(3).to_string())
        print("\nMean corr(delta, Y) / corr(delta, Z) by mechanism (signal=0):")
        c0 = cal[cal.signal == 0]
        print(c0.groupby("mechanism")[["corr_delta_y", "corr_delta_z"]]
              .mean().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imputer quality measurement")
    parser.add_argument("--benchmark", type=str, default="calibration",
                        choices=["calibration", "erdos"])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.quick:
        args.reps = 5

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    master = np.random.RandomState(args.seed)
    jobs = []
    if args.benchmark == "calibration":
        dgps = ["linear_gaussian", "post_nonlinear", "latent_confounder"]
        ns = [500, 1000] if args.quick else [500, 1000, 2000, 5000]
        dims = [2] if args.quick else [2, 5, 10]
        for dgp in dgps:
            for n in ns:
                for dim_x in dims:
                    for signal in [0.0, 0.3]:
                        for mech in ["mar", "mnar"]:
                            for rate in [0.1, 0.3, 0.5]:
                                for rep in range(args.reps):
                                    jobs.append(delayed(run_calibration_cell)(
                                        dgp, n, dim_x, signal, mech, rate,
                                        rep, master.randint(0, 2**31)))
    else:
        for mech in ["mar", "mnar", "mixed"]:
            for rep in range(args.reps):
                jobs.append(delayed(run_erdos_cell)(
                    mech, rep, master.randint(0, 2**31)))

    print(f"{len(jobs)} cells queued ({args.n_jobs} workers)", flush=True)
    t0 = time.time()
    nested = Parallel(n_jobs=args.n_jobs, verbose=5)(jobs)
    rows = [r for cell in nested for r in cell]
    print(f"Total: {(time.time()-t0)/60:.1f} min")
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows)")
    analyze(df)
