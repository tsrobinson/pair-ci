"""Estimate the kappa factors of each adversarial cell (Appendix N).

For each (dgp, rate, imputer) cell of experiments/adversarial_kappa.py,
re-simulates the data and cached imputation, without the CI test, and
estimates the factors of Definition 9 (Appendix C), with
delta = X_true - X_imputed on the missing entries:

    kappa_imp = sqrt(mean(delta^2)),  kappa_Y = |corr(Y, delta)|,
    kappa_Z   = |corr(Z, delta)|

These are unconditional analogues of the conditional correlations in the
definition. Each replicate pools M imputations and the masked columns;
the cell value is the mean over replicates. kappa_tot (kappa_imp
unsquared) and kappa_yz = kappa_Y kappa_Z are stored too. The factors are
joined to each cell's false positive rate and Clopper-Pearson 95% interval
from results/adversarial_kappa_results.csv and written to
results/kappa_trace.csv, which plot_kappa_trace.py turns into Figure 4.
About 5 minutes on a laptop: python analysis/compute_kappa_trace.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "experiments"))
sys.path.insert(0, os.path.join(THIS, "..", "src"))

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from adversarial_kappa import DGPS, inject_mnar


def fit_imputer(data, imputer_type, rng):
    """Impute [Z | Y | X_miss] once, with the settings of adversarial_kappa.py.

    The imputer sees Y and Z, as the cached imputer does.
    """
    if imputer_type == "linear":
        from sklearn.linear_model import BayesianRidge
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            sample_posterior=True,
            random_state=rng.randint(0, 2**31),
        )
    elif imputer_type == "rf":
        imp = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=50, n_jobs=-1,
                random_state=rng.randint(0, 2**31),
            ),
            max_iter=10,
            random_state=rng.randint(0, 2**31),
        )
    else:
        raise ValueError(imputer_type)
    return imp.fit_transform(data)


def kappa_factors_one_rep(dgp_name, rate, imputer_type, rng,
                          n=500, n_imputations=5):
    """Return (kappa_imp, kappa_Y, kappa_Z) for one replicate.

    Columns with fewer than 5 missing entries are skipped; returns NaNs if
    none remain.
    """
    dgp_fn = DGPS[dgp_name]
    X, Y, Z, miss_cols = dgp_fn(n, rng)
    X_miss = inject_mnar(X, miss_cols, rate, rng)

    data = np.column_stack([Z, Y, X_miss])

    imp_errs = []   # X_true - X_imputed at the missing entries
    Y_aligned = []  # Y at those rows, repeated per column and imputation
    Z_aligned = []
    for m in range(n_imputations):
        completed = fit_imputer(data, imputer_type, rng)
        # Columns 2 onward in `data` correspond to X
        X_imputed = completed[:, 2:]
        for col in miss_cols:
            mask = np.isnan(X_miss[:, col])
            if mask.sum() < 5:
                continue
            err = X[mask, col] - X_imputed[mask, col]
            imp_errs.append(err)
            Y_aligned.append(Y[mask])
            Z_aligned.append(Z[mask])

    if len(imp_errs) == 0:
        return np.nan, np.nan, np.nan

    err_all = np.concatenate(imp_errs)
    Y_all = np.concatenate(Y_aligned)
    Z_all = np.concatenate(Z_aligned)

    kappa_imp = np.sqrt(np.mean(err_all ** 2))

    if err_all.std() < 1e-9:
        kappa_Y = 0.0
        kappa_Z = 0.0
    else:
        kappa_Y = abs(np.corrcoef(Y_all, err_all)[0, 1])
        kappa_Z = abs(np.corrcoef(Z_all, err_all)[0, 1])
        if np.isnan(kappa_Y):
            kappa_Y = 0.0
        if np.isnan(kappa_Z):
            kappa_Z = 0.0

    return kappa_imp, kappa_Y, kappa_Z


def aggregate_fpr_with_ci(fpr_df, group_cols):
    """Compute mean FPR + Clopper-Pearson 95% CI per group."""
    rows = []
    for keys, sub in fpr_df.groupby(group_cols):
        ok = sub[~sub.failed]
        if len(ok) == 0:
            continue
        n_rep = len(ok)
        n_rej = ok.rejected.sum()
        fpr = n_rej / n_rep
        if n_rej == 0:
            lo, hi = 0.0, stats.beta.ppf(0.975, 1, n_rep)
        elif n_rej == n_rep:
            lo, hi = stats.beta.ppf(0.025, n_rep, 1), 1.0
        else:
            lo = stats.beta.ppf(0.025, n_rej, n_rep - n_rej + 1)
            hi = stats.beta.ppf(0.975, n_rej + 1, n_rep - n_rej)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row.update(dict(fpr=fpr, fpr_lo=lo, fpr_hi=hi, n_reps=n_rep))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps_kappa", type=int, default=30,
                        help="Reps per cell for kappa estimation.")
    parser.add_argument("--n_imputations", type=int, default=5)
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fpr_csv", type=str,
                        default="results/adversarial_kappa_results.csv")
    parser.add_argument("--output", type=str,
                        default="results/kappa_trace.csv")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(THIS, ".."))
    fpr_path = os.path.join(base, args.fpr_csv)
    out_path = os.path.join(base, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading existing FPR data from {fpr_path}")
    fpr_df = pd.read_csv(fpr_path)
    print(f"  {len(fpr_df)} rows, {fpr_df.dgp.nunique()} DGPs")

    fpr_agg = aggregate_fpr_with_ci(
        fpr_df, ["dgp", "rate", "cond_strategy", "imputer"])
    print(f"FPR aggregated: {len(fpr_agg)} cells")

    # kappa is computed per (dgp, rate, imputer): the conditioning strategy
    # affects the test, not the imputation.
    cells = (fpr_df.groupby(["dgp", "rate", "imputer"])
             .size().reset_index().drop(columns=0))
    print(f"\nComputing kappa for {len(cells)} (dgp, rate, imputer) cells, "
          f"{args.n_reps_kappa} reps each...")

    kappa_rows = []
    # One RandomState per cell, seeded from the master stream in cell order.
    rng_master = np.random.RandomState(args.seed)
    for i, row in cells.iterrows():
        dgp_name, rate, imputer = row.dgp, row.rate, row.imputer
        print(f"  [{i+1:2d}/{len(cells)}] {dgp_name:<20s} "
              f"rate={rate} imputer={imputer:<6s}", end=" ", flush=True)
        rng = np.random.RandomState(rng_master.randint(0, 2**31))
        ki_list, ky_list, kz_list = [], [], []
        for rep in range(args.n_reps_kappa):
            ki, ky, kz = kappa_factors_one_rep(
                dgp_name, rate, imputer, rng,
                n=args.n_obs, n_imputations=args.n_imputations)
            if not np.isnan(ki):
                ki_list.append(ki)
                ky_list.append(ky)
                kz_list.append(kz)
        if len(ki_list) == 0:
            continue
        kappa_imp = float(np.mean(ki_list))
        kappa_Y = float(np.mean(ky_list))
        kappa_Z = float(np.mean(kz_list))
        kappa_tot = kappa_imp * kappa_Y * kappa_Z
        kappa_yz = kappa_Y * kappa_Z
        print(f"k_imp={kappa_imp:.3f} k_Y={kappa_Y:.3f} k_Z={kappa_Z:.3f} "
              f"k_tot={kappa_tot:.4f}")
        kappa_rows.append(dict(
            dgp=dgp_name, rate=rate, imputer=imputer,
            kappa_imp=kappa_imp, kappa_Y=kappa_Y, kappa_Z=kappa_Z,
            kappa_tot=kappa_tot, kappa_yz=kappa_yz,
        ))
    kappa_df = pd.DataFrame(kappa_rows)

    merged = fpr_agg.merge(kappa_df, on=["dgp", "rate", "imputer"], how="left")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged)} rows to {out_path}")
    print()
    print(merged[["dgp", "rate", "imputer", "cond_strategy",
                  "kappa_imp", "kappa_Y", "kappa_Z", "kappa_yz",
                  "fpr", "fpr_lo", "fpr_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
