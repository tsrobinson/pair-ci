"""Standalone calibration and power of PAIR-CI and the baseline CI tests.

Tests H0: Z indep Y | X on three DGPs (linear Gaussian, post-nonlinear,
latent confounder; src/dgps.py) with signal in {0, 0.3, 0.6, 1.0},
n in {500, 1000, 2000, 5000} and |X| in {2, 5, 10} (cells with
|X| >= n/50 are skipped), under complete data, MAR and MNAR at a 30%
per-column rate. The estimand is the rejection rate at alpha = 0.05:
the false positive rate at signal 0 and power otherwise. Methods are
PAIR-CI (random forests, M = 5, K = 5, cached imputation), Fisher's Z
(complete case, single imputation, Rubin-pooled), GCM and KCI (each
complete case and imputed). One row per replicate and method. As run, KCI
was a separate call at n = 1000, signals 0 and 0.6, 50 replicates:

    python experiments/calibration.py --n_reps 100 \
        --methods ours fisher_z gcm fisher_z_rubin \
        --output results/ci_calibration_fast.csv

Feeds Table 1, Table calibration_by_dgp (App. A) and Apps. D, L and R.2.
The released MAR rows at |X| = 2 come from a rerun with --seed 4242 (KCI
4243) after the generator fix described in the README.
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

from itertools import product
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from dgps import (
    DGP_FUNCTIONS, inject_missingness_calibration, impute_datasets,
)
from ci_oracle import GeneralCITest



def fisher_z_test(X, Y, Z):
    n, p = X.shape
    try:
        from numpy.linalg import lstsq
        if p > 0:
            # Intercept in the residual regressions, matching the Fisher-Z
            # df of n - |S| - 3 used below.
            Xi = np.column_stack([np.ones(n), X])
            res_y = Y - Xi @ lstsq(Xi, Y, rcond=None)[0]
            res_z = Z - Xi @ lstsq(Xi, Z, rcond=None)[0]
        else:
            res_y, res_z = Y - Y.mean(), Z - Z.mean()
        r, _ = stats.pearsonr(res_y, res_z)
        if not np.isfinite(r):
            return 0.0, 1.0
        one_plus = max(1.0 + r, 1e-15)
        one_minus = max(1.0 - r, 1e-15)
        z_stat = 0.5 * (np.log(one_plus) - np.log(one_minus))
        se = 1.0 / np.sqrt(max(n - p - 3, 1))
        t_stat = abs(z_stat) / se
        pval = 2 * (1 - stats.norm.cdf(t_stat))
        return t_stat, pval
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0


def gcm_test(X, Y, Z):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    n = X.shape[0]
    try:
        rf_y = RandomForestRegressor(
            n_estimators=50, min_samples_leaf=5, n_jobs=-1, random_state=42
        )
        rf_z = RandomForestRegressor(
            n_estimators=50, min_samples_leaf=5, n_jobs=-1, random_state=43
        )
        y_hat = cross_val_predict(rf_y, X, Y, cv=5)
        z_hat = cross_val_predict(rf_z, X, Z, cv=5)
        res_y = Y - y_hat
        res_z = Z - z_hat
        prod = res_y * res_z
        T_n = np.sqrt(n) * np.mean(prod) / np.std(prod)
        pval = 2 * (1 - stats.norm.cdf(abs(T_n)))
        return abs(T_n), pval
    except Exception:
        return 0.0, 1.0


def kci_test(data, z_idx, y_idx, cond_set):
    from causallearn.utils.cit import CIT
    cit = CIT(data, "kci")
    return cit(z_idx, y_idx, list(cond_set))


def kci_imputed_test(data_miss, z_idx, y_idx, cond_set, rng, n_imp=5):
    """KCI on n_imp MICE draws, p-values pooled by Fisher's method.

    Fisher's method treats the draws as independent, which they are not
    (App. R.1).
    """
    datasets = []
    for _ in range(n_imp):
        imp = IterativeImputer(
            max_iter=10,
            random_state=rng.randint(0, 2**31),
            sample_posterior=True,
        )
        datasets.append(imp.fit_transform(data_miss))

    pvals = []
    for d in datasets:
        try:
            pv = kci_test(d, z_idx, y_idx, cond_set)
            if pv is not None and not np.isnan(pv):
                pvals.append(pv)
        except Exception:
            continue

    if not pvals:
        return np.nan, np.nan

    chi2 = -2 * np.sum(np.log(np.clip(pvals, 1e-300, 1)))
    combined_p = 1 - stats.chi2.cdf(chi2, 2 * len(pvals))
    return chi2, combined_p


def kci_complete_case_test(data_miss, z_idx, y_idx, cond_set):
    cols = list(cond_set) + [z_idx, y_idx]
    mask = ~np.any(np.isnan(data_miss[:, cols]), axis=1)
    if mask.sum() < 50:
        return np.nan, np.nan
    pv = kci_test(data_miss[mask], z_idx, y_idx, cond_set)
    if pv is None or np.isnan(pv):
        return np.nan, np.nan
    return 0.0, pv


def our_test(data_miss, z_idx, y_idx, cond_set, rng,
             n_imputations=5, n_folds=5):
    ci = GeneralCITest(
        data_miss,
        n_imputations=n_imputations,
        n_folds=n_folds,
        imputation_strategy="cached",
        random_state=rng.randint(0, 2**31),
    )
    stat, pval = ci.test(x_idx=z_idx, y_idx=y_idx, cond_set=cond_set)
    return stat, pval



def run_single_rep(dgp_name, n, dim_x, signal, mechanism, miss_rate,
                   alpha, methods, rng):
    dgp_fn = DGP_FUNCTIONS[dgp_name]
    X, Y, Z = dgp_fn(n, dim_x, signal, rng)

    X_miss = inject_missingness_calibration(X, mechanism, miss_rate, rng)
    has_missing = np.any(np.isnan(X_miss))

    mask_cc = ~np.any(np.isnan(X_miss), axis=1)
    X_cc, Y_cc, Z_cc = X_miss[mask_cc], Y[mask_cc], Z[mask_cc]
    n_complete = mask_cc.sum()

    if has_missing:
        imp = IterativeImputer(
            max_iter=10,
            random_state=rng.randint(0, 2**31),
            sample_posterior=True,
        )
        # One posterior draw, with Y and Z as auxiliary predictors as in
        # Witte et al. (2022) and Foraita et al. (2020). This single
        # imputation feeds both FZ-single and imputed GCM.
        full = np.column_stack([X_miss, Y, Z])
        imputed_full = imp.fit_transform(full)
        X_imp = imputed_full[:, :X_miss.shape[1]]
    else:
        X_imp = X_miss.copy()

    results = []
    base = dict(
        dgp=dgp_name, n=n, dim_x=dim_x, signal=signal,
        mechanism=mechanism, miss_rate=miss_rate,
        n_complete=n_complete,
    )

    # --- Fisher-Z ---
    if "fisher_z" in methods:
        if n_complete > dim_x + 10:
            stat, pval = fisher_z_test(X_cc, Y_cc, Z_cc)
        else:
            stat, pval = 0.0, 1.0
        results.append({
            **base, "method": "Fisher-Z", "variant": "Complete Case",
            "statistic": stat, "p_value": pval,
            "rejected": pval < alpha, "failed": False,
        })
        stat, pval = fisher_z_test(X_imp, Y, Z)
        results.append({
            **base, "method": "Fisher-Z", "variant": "Imputed",
            "statistic": stat, "p_value": pval,
            "rejected": pval < alpha, "failed": False,
        })

    # --- GCM ---
    if "gcm" in methods:
        if n_complete > dim_x + 10:
            stat, pval = gcm_test(X_cc, Y_cc, Z_cc)
        else:
            stat, pval = 0.0, 1.0
        results.append({
            **base, "method": "GCM", "variant": "Complete Case",
            "statistic": stat, "p_value": pval,
            "rejected": pval < alpha, "failed": False,
        })
        stat, pval = gcm_test(X_imp, Y, Z)
        results.append({
            **base, "method": "GCM", "variant": "Imputed",
            "statistic": stat, "p_value": pval,
            "rejected": pval < alpha, "failed": False,
        })

    # --- KCI ---
    if "kci" in methods:
        data_full = np.column_stack([X_miss, Z, Y])
        z_idx = dim_x
        y_idx = dim_x + 1
        cond_set = tuple(range(dim_x))

        try:
            stat, pval = kci_complete_case_test(
                data_full, z_idx, y_idx, cond_set)
            failed = np.isnan(pval) if isinstance(pval, float) else False
        except Exception:
            stat, pval, failed = np.nan, np.nan, True
        results.append({
            **base, "method": "KCI", "variant": "Complete Case",
            "statistic": stat if not failed else np.nan,
            "p_value": pval if not failed else np.nan,
            "rejected": pval < alpha if not failed else False,
            "failed": failed,
        })

        try:
            stat, pval = kci_imputed_test(
                data_full, z_idx, y_idx, cond_set,
                np.random.RandomState(rng.randint(0, 2**31)),
            )
            failed = np.isnan(pval) if isinstance(pval, float) else False
        except Exception:
            stat, pval, failed = np.nan, np.nan, True
        results.append({
            **base, "method": "KCI", "variant": "Imputed",
            "statistic": stat if not failed else np.nan,
            "p_value": pval if not failed else np.nan,
            "rejected": pval < alpha if not failed else False,
            "failed": failed,
        })

    # --- Ours ---
    if "ours" in methods:
        data_ours = np.column_stack([Z, Y, X_miss])
        z_idx_ours = 0
        y_idx_ours = 1
        cond_set_ours = tuple(range(2, 2 + dim_x))

        try:
            stat, pval = our_test(
                data_ours, z_idx_ours, y_idx_ours, cond_set_ours,
                np.random.RandomState(rng.randint(0, 2**31)),
            )
            results.append({
                **base, "method": "Ours", "variant": "Integrated",
                "statistic": stat, "p_value": pval,
                "rejected": pval < alpha, "failed": False,
            })
        except Exception:
            results.append({
                **base, "method": "Ours", "variant": "Integrated",
                "statistic": np.nan, "p_value": np.nan,
                "rejected": False, "failed": True,
            })

    # --- Fisher-Z with Rubin's Rules (Witte et al. 2022) ---
    if "fisher_z_rubin" in methods:
        # Draw the Rubin seed, then restore the main stream, so that
        # including this method leaves every other method's draws (and
        # all later replicates) unchanged.
        _rng_snapshot = rng.get_state()
        rubin_rng = np.random.RandomState(rng.randint(0, 2**31))
        rng.set_state(_rng_snapshot)

        M = 5
        if has_missing:
            # Y and Z are auxiliary predictors in the imputation model
            # for X (Witte et al. 2022, Table 3 and Sec. 4.3).
            X_imps = impute_datasets(
                X_miss, rubin_rng, n_imp=M,
                aux=np.column_stack([Y, Z]),
            )
        else:
            X_imps = [X_miss.copy()] * M

        z_vals = []
        n_eff = X_miss.shape[0]
        p_cond = X_miss.shape[1]
        dof_within = max(n_eff - p_cond - 3, 1)

        for X_m in X_imps:
            try:
                if p_cond > 0:
                    # Intercept included, as in fisher_z_test.
                    Xi = np.column_stack([np.ones(n_eff), X_m])
                    res_y = Y - Xi @ np.linalg.lstsq(Xi, Y, rcond=None)[0]
                    res_z = Z - Xi @ np.linalg.lstsq(Xi, Z, rcond=None)[0]
                else:
                    res_y, res_z = Y - Y.mean(), Z - Z.mean()
                r_m, _ = stats.pearsonr(res_y, res_z)
                if not np.isfinite(r_m):
                    continue
                one_plus = max(1.0 + r_m, 1e-15)
                one_minus = max(1.0 - r_m, 1e-15)
                z_vals.append(0.5 * (np.log(one_plus) - np.log(one_minus)))
            except (np.linalg.LinAlgError, ValueError):
                continue

        if len(z_vals) >= 2:
            z_arr = np.array(z_vals)
            M_eff = len(z_arr)
            z_bar = np.mean(z_arr)
            W_bar = 1.0 / dof_within          # within-imputation variance
            B = np.var(z_arr, ddof=1)          # between-imputation variance
            T = W_bar + (1 + 1.0 / M_eff) * B  # Rubin's total variance

            # Rubin's (1987) degrees of freedom, without the Barnard-Rubin
            # small-sample correction, as in Witte et al. (2022, App. A):
            # nu = (M-1) [1 + W / ((1 + 1/M) B)]^2 = (M-1) / gamma^2,
            # with gamma = (1 + 1/M) B / T.
            if T > 0 and B > 0:
                gamma = (1 + 1.0 / M_eff) * B / T
                nu = (M_eff - 1) / (gamma ** 2)
            else:
                nu = dof_within

            t_stat = abs(z_bar) / np.sqrt(max(T, 1e-15))
            pval = 2 * (1 - stats.t.cdf(t_stat, max(nu, 1)))
        elif len(z_vals) == 1:
            # Only one draw gave a finite correlation: plain Fisher-Z.
            z_bar = z_vals[0]
            t_stat = abs(z_bar) * np.sqrt(dof_within)
            pval = 2 * (1 - stats.norm.cdf(t_stat))
        else:
            t_stat, pval = 0.0, 1.0

        results.append({
            **base, "method": "Fisher-Z", "variant": "Rubin's Rules",
            "statistic": t_stat, "p_value": pval,
            "rejected": pval < alpha, "failed": False,
        })

    return results



def run_experiment(sample_sizes, dim_xs, signals, mechanisms, dgps,
                   methods, miss_rate, alpha, n_reps, seed):
    rng = np.random.RandomState(seed)

    configs = []
    for dgp, n, dx, sig, mech in product(dgps, sample_sizes, dim_xs,
                                          signals, mechanisms):
        if dx >= n / 50:
            continue
        configs.append((dgp, n, dx, sig, mech))

    print(f"Methods: {methods}")
    print(f"DGPs: {dgps}")
    print(f"Configurations: {len(configs)}")
    print(f"Reps per config: {n_reps}")
    print(f"Total runs: {len(configs) * n_reps}")
    print()

    all_results = []
    t_start = time.time()

    for i, (dgp, n, dx, sig, mech) in enumerate(configs):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = (time.time() - t_start) / 60
            print(f"Config {i+1}/{len(configs)} "
                  f"({dgp}, n={n}, dx={dx}, sig={sig}, {mech}) "
                  f"[{elapsed:.1f} min]")

        for rep in range(n_reps):
            results = run_single_rep(
                dgp, n, dx, sig, mech, miss_rate, alpha, methods, rng
            )
            for r in results:
                r["rep"] = rep
            all_results.extend(results)

    return pd.DataFrame(all_results)



def analyze(df, alpha=0.05):
    df_ok = df[~df["failed"]].copy()
    df_ok["label"] = df_ok["method"] + " (" + df_ok["variant"] + ")"

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    null = df_ok[df_ok["signal"] == 0]
    print("\n--- FPR under null (signal=0, target: 5%) ---\n")
    fpr_table = (null.groupby(["label", "mechanism"])["rejected"]
                 .mean().unstack("mechanism"))
    if "complete" in fpr_table.columns:
        col_order = [c for c in ["complete", "mar", "mnar"]
                     if c in fpr_table.columns]
        fpr_table = fpr_table[col_order]
    print(fpr_table.round(3).to_string())

    print("\n\n--- FPR by DGP (signal=0) ---")
    for label in sorted(null["label"].unique()):
        sub = null[null["label"] == label]
        print(f"\n{label}:")
        tbl = (sub.groupby(["dgp", "mechanism"])["rejected"]
               .mean().unstack("mechanism"))
        if "complete" in tbl.columns:
            col_order = [c for c in ["complete", "mar", "mnar"]
                         if c in tbl.columns]
            tbl = tbl[col_order]
        print(tbl.round(3).to_string())

    alt = df_ok[df_ok["signal"] > 0]
    if len(alt) > 0:
        print("\n\n--- Power (signal > 0, by signal strength) ---\n")
        for sig in sorted(alt["signal"].unique()):
            sub = alt[alt["signal"] == sig]
            print(f"\nSignal = {sig}:")
            pwr = (sub.groupby(["label", "mechanism"])["rejected"]
                   .mean().unstack("mechanism"))
            if "complete" in pwr.columns:
                col_order = [c for c in ["complete", "mar", "mnar"]
                             if c in pwr.columns]
                pwr = pwr[col_order]
            print(pwr.round(3).to_string())

    print("\n\n" + "=" * 70)
    print("KEY SUMMARY FOR PAPER (Table 1 format)")
    print("=" * 70)
    print("\nAverage FPR across DGPs and sample sizes (signal=0):\n")
    summary = (null.groupby(["label", "mechanism"])["rejected"]
               .mean().unstack("mechanism"))
    if "complete" in summary.columns:
        col_order = [c for c in ["complete", "mar", "mnar"]
                     if c in summary.columns]
        summary = summary[col_order]
    print(summary.round(3).to_string())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified CI calibration experiment")
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--methods", nargs="+",
                        default=["ours", "fisher_z", "gcm", "kci",
                                 "fisher_z_rubin"])
    parser.add_argument("--dgps", nargs="+",
                        default=["linear_gaussian", "post_nonlinear",
                                 "latent_confounder"])
    parser.add_argument("--sample_sizes", nargs="+", type=int,
                        default=[500, 1000, 2000, 5000])
    parser.add_argument("--dim_xs", nargs="+", type=int,
                        default=[2, 5, 10])
    parser.add_argument("--signals", nargs="+", type=float,
                        default=[0, 0.3, 0.6, 1.0])
    parser.add_argument("--mechanisms", nargs="+",
                        default=["complete", "mar", "mnar"])
    parser.add_argument("--miss_rate", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/ci_calibration_results.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("UNIFIED CI CALIBRATION EXPERIMENT")
    print("=" * 70)
    print(f"Output: {args.output}\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    t0 = time.time()
    df = run_experiment(
        sample_sizes=args.sample_sizes, dim_xs=args.dim_xs,
        signals=args.signals, mechanisms=args.mechanisms,
        dgps=args.dgps, methods=args.methods,
        miss_rate=args.miss_rate, alpha=args.alpha,
        n_reps=args.n_reps, seed=args.seed,
    )
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    analyze(df, args.alpha)
