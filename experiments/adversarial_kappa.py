"""Adversarial MNAR DGPs that stress PAIR-CI's kappa cancellation.

Each DGP is built to make all three factors of the kappa decomposition
(App. C) large at once: imputation error, the correlation of Y with the
unrecovered part of X, and the dependence of Z on it. All cells are
nulls; the estimand is the false positive rate at alpha = 0.05, with
exact Clopper-Pearson 95% intervals. Missingness is MNAR self-masking
with logistic steepness 5 on the columns each DGP marks as incomplete,
and PAIR-CI uses cached imputation throughout.

DGPs:
  hub              X_0 with 10 children (loading 0.5); Y and Z linear in X_0,
                   which is incomplete.
  chain            X_0 -> X_1 -> X_2 -> X_3; X_2 incomplete and the common
                   parent of Y and Z.
  dense_block      5 variables, 3 incomplete; correlation 0.8 within each
                   block and 0.2 across, so the observed block is a poor
                   predictor of the incomplete one.
  weak_hub         hub with loading 0.2, so the children barely inform X_0.
  branch_separator Y depends only on incomplete A, Z only on incomplete B,
                   A indep B; each has an observed proxy (loading 0.5).
  hub_nonlinear    hub with Y = sin(X_0), Z = X_0^2.
Conditioning sets: "full" (all of X) or "missing" (incomplete columns
only). Imputers: "linear" (MICE with Bayesian ridge, the paper's default)
or "rf" (MICE with random forests).

As run (100 replicates per cell, 48 cells; the default --n_reps is 200):
    python -u experiments/adversarial_kappa.py --n_reps 100 \
        --sample_sizes 500 --rates 0.3 0.5

Writes results/adversarial_kappa_results.csv, which feeds App. N and,
through analysis/compute_kappa_trace.py, Figure kappa_fpr_trace.
Runtime is roughly 15-20 hours on 16 cores.
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
from sklearn.ensemble import RandomForestRegressor

from ci_oracle import GeneralCITest


class RFImputerCITest(GeneralCITest):
    """PAIR-CI with random-forest MICE in place of Bayesian-ridge MICE.

    IterativeImputer cannot draw from the posterior with a random forest,
    so the variation across imputations comes only from the forests'
    bootstrap and seeds. The draws are therefore not proper in the sense
    of (A3); use this class for the robustness check only.
    """

    def _impute_full_dataset(self):
        imputed = []
        for m in range(self.n_imputations):
            imp = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=50, n_jobs=-1,
                    random_state=self.rng.randint(0, 2**31),
                ),
                max_iter=10,
                random_state=self.rng.randint(0, 2**31),
                # No sample_posterior: unsupported with a random forest.
            )
            imputed.append(imp.fit_transform(self.data))
        return imputed


IMPUTER_CLASSES = {
    "linear": GeneralCITest,
    "rf":     RFImputerCITest,
}


# ---------- Adversarial DGPs ----------
# Each DGP returns (X, Y, Z, miss_cols) where miss_cols is a list of column
# indices to inject missingness on.

def dgp_hub(n, rng, n_children=10, alpha=0.5):
    hub = rng.randn(n)
    children = np.zeros((n, n_children))
    for j in range(n_children):
        children[:, j] = alpha * hub + np.sqrt(1 - alpha**2) * rng.randn(n)
    X = np.column_stack([hub, children])
    Y = hub + 0.5 * rng.randn(n)
    Z = hub + 0.5 * rng.randn(n)
    return X, Y, Z, [0]


def dgp_chain(n, rng):
    X0 = rng.randn(n)
    X1 = 0.8 * X0 + 0.6 * rng.randn(n)
    X2 = 0.8 * X1 + 0.6 * rng.randn(n)
    X3 = 0.8 * X2 + 0.6 * rng.randn(n)
    X = np.column_stack([X0, X1, X2, X3])
    Y = X2 + 0.5 * rng.randn(n)
    Z = X2 + 0.5 * rng.randn(n)
    return X, Y, Z, [2]


def dgp_dense_block(n, rng, dim_x=5, n_missing=3):
    """Two Gaussian blocks: the first n_missing columns (incomplete) and the
    rest (observed), correlation 0.8 within each block and 0.2 across.

    The weak cross-block correlation keeps the observed columns from
    imputing the incomplete ones well, which is the point of the design.
    """
    # Build block covariance: [missing block | weak cross | observed block]
    rho_within = 0.8
    rho_cross = 0.2
    Sigma = np.zeros((dim_x, dim_x))
    for i in range(dim_x):
        for j in range(dim_x):
            if i == j:
                Sigma[i, j] = 1.0
            elif i < n_missing and j < n_missing:
                Sigma[i, j] = rho_within
            elif i >= n_missing and j >= n_missing:
                Sigma[i, j] = rho_within
            else:
                Sigma[i, j] = rho_cross
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(n, dim_x) @ L.T
    Y = X[:, 0] + 0.5 * rng.randn(n)
    Z = X[:, 0] + 0.5 * rng.randn(n)
    return X, Y, Z, list(range(n_missing))


def dgp_weak_hub(n, rng, n_children=10):
    return dgp_hub(n, rng, n_children=n_children, alpha=0.2)


def dgp_branch_separator(n, rng):
    """Y depends only on A, Z only on B, with A indep B; both incomplete.

    Structure: A -> {A_proxy, Y} and B -> {B_proxy, Z}, the proxies
    observed with loading 0.5. Z indep Y holds for every conditioning set,
    so unlike the other DGPs there is no shared unrecovered component; the
    question is whether imperfect imputation of A and B from their proxies
    creates a spurious Y-Z dependence.
    """
    A = rng.randn(n)
    B = rng.randn(n)
    A_proxy = 0.5 * A + np.sqrt(0.75) * rng.randn(n)
    B_proxy = 0.5 * B + np.sqrt(0.75) * rng.randn(n)
    Y = A + 0.5 * rng.randn(n)
    Z = B + 0.5 * rng.randn(n)
    # X columns: [A, B, A_proxy, B_proxy]
    X = np.column_stack([A, B, A_proxy, B_proxy])
    return X, Y, Z, [0, 1]  # A and B missing


def dgp_hub_nonlinear(n, rng, n_children=10, alpha=0.5):
    """Hub with nonlinear Y and Z dependence on the missing variable."""
    hub = rng.randn(n)
    children = np.zeros((n, n_children))
    for j in range(n_children):
        children[:, j] = alpha * hub + np.sqrt(1 - alpha**2) * rng.randn(n)
    X = np.column_stack([hub, children])
    Y = np.sin(hub) + 0.3 * rng.randn(n)
    Z = hub ** 2 - 1.0 + 0.3 * rng.randn(n)  # centred
    return X, Y, Z, [0]


DGPS = {
    "hub":              dgp_hub,
    "chain":            dgp_chain,
    "dense_block":      dgp_dense_block,
    "weak_hub":         dgp_weak_hub,
    "branch_separator": dgp_branch_separator,
    "hub_nonlinear":    dgp_hub_nonlinear,
}


# ---------- Missingness injection ----------

def inject_mnar(X, miss_cols, rate, rng, steepness=5):
    X_miss = X.copy()
    n = X_miss.shape[0]
    for col in miss_cols:
        v = X_miss[:, col]
        logit = v - np.quantile(v, 1 - rate)
        prob = 1 / (1 + np.exp(-steepness * logit))
        X_miss[rng.rand(n) < prob, col] = np.nan
    return X_miss


# ---------- Experiment ----------

def _run_imputer_on_data(data, z_idx, y_idx, cond_set, imputer_type,
                         test_seed, alpha=0.05):
    cls = IMPUTER_CLASSES[imputer_type]
    try:
        ci = cls(
            data, n_imputations=5, n_folds=5,
            imputation_strategy="cached",
            random_state=test_seed,
        )
        stat, pval = ci.test(x_idx=z_idx, y_idx=y_idx, cond_set=cond_set)
        return dict(statistic=stat, p_value=pval,
                    rejected=pval < alpha, failed=False)
    except Exception:
        return dict(statistic=np.nan, p_value=np.nan,
                    rejected=False, failed=True)


def run_experiment(args):
    rng = np.random.RandomState(args.seed)
    all_results = []
    # Imputers are looped inside each replicate, so within a
    # (dgp, n, rate, cond) cell every imputer sees the same (X, mask).
    # The per-replicate pairing behind delta_rf_minus_lin relies on this.
    outer_configs = list(product(args.dgps, args.sample_sizes, args.rates,
                                 args.cond_strategies))
    print(f"DGPs: {args.dgps}")
    print(f"Sample sizes: {args.sample_sizes}")
    print(f"Rates: {args.rates}")
    print(f"Cond strategies: {args.cond_strategies}")
    print(f"Imputers: {args.imputers}")
    print(f"Outer configs: {len(outer_configs)}; reps each: {args.n_reps}")
    print(f"Total tests: {len(outer_configs) * args.n_reps * len(args.imputers)}")
    print()

    t_start = time.time()
    for i, (dgp_name, n, rate, cond_s) in enumerate(outer_configs):
        elapsed = (time.time() - t_start) / 60
        print(f"[{elapsed:6.1f} min] Config {i+1}/{len(outer_configs)}: "
              f"{dgp_name}, n={n}, rate={rate}, cond={cond_s}",
              flush=True)
        dgp_fn = DGPS[dgp_name]
        for rep in range(args.n_reps):
            X, Y, Z, miss_cols = dgp_fn(n, rng)
            X_miss = inject_mnar(X, miss_cols, rate, rng)
            data = np.column_stack([Z, Y, X_miss])
            z_idx, y_idx = 0, 1
            p_x = X_miss.shape[1]

            if cond_s == "full":
                cond_set = tuple(range(2, 2 + p_x))
            elif cond_s == "missing":
                cond_set = tuple(2 + c for c in miss_cols)
            else:
                raise ValueError(f"Unknown cond_strategy: {cond_s}")

            # One test seed per imputer, drawn from the main stream after
            # data generation: identical (data, cond_set), distinct seeds.
            imputer_seeds = [rng.randint(0, 2**31) for _ in args.imputers]
            for imp_t, test_seed in zip(args.imputers, imputer_seeds):
                res = _run_imputer_on_data(data, z_idx, y_idx, cond_set,
                                           imp_t, test_seed, args.alpha)
                res.update(dict(
                    dgp=dgp_name, n=n, rate=rate,
                    cond_strategy=cond_s, imputer=imp_t,
                    rep=rep,
                ))
                all_results.append(res)
    return pd.DataFrame(all_results)


def clopper_pearson(k, n, conf=0.95):
    """Exact two-sided Clopper-Pearson CI for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def analyze(df, alpha=0.05):
    df = df[~df["failed"]].copy()
    print("\n" + "=" * 70)
    print("ADVERSARIAL KAPPA RESULTS")
    print("=" * 70)
    print(f"\nTarget: FPR <= alpha = {alpha}. 95% Clopper-Pearson CIs in brackets.\n")

    worst_fpr_per_imputer = {}
    for (dgp, cond_s, imp_t), sub in df.groupby(
            ["dgp", "cond_strategy", "imputer"]):
        print(f"--- DGP={dgp}, cond={cond_s}, imputer={imp_t} ---")
        for (n, rate), cell in sub.groupby(["n", "rate"]):
            k = int(cell["rejected"].sum())
            nn = int(len(cell))
            fpr = k / nn
            lo, hi = clopper_pearson(k, nn)
            flag = "  " if hi <= alpha + 0.01 else "* " if fpr <= alpha + 0.05 else "**"
            print(f"  {flag}n={n}, rate={rate}: "
                  f"FPR = {fpr:.3f}  [{lo:.3f}, {hi:.3f}]  "
                  f"({k}/{nn} reps)")
            cur = worst_fpr_per_imputer.get(imp_t, (0.0, None))
            if fpr > cur[0]:
                worst_fpr_per_imputer[imp_t] = (fpr, (dgp, cond_s, n, rate, lo, hi))
        print()

    print("=" * 70)
    print("VERDICT (per imputer)")
    print("=" * 70)
    for imp_t, (worst_fpr, cell) in worst_fpr_per_imputer.items():
        if cell is None:
            continue
        dgp, cond_s, n, rate, lo, hi = cell
        print(f"\nImputer = {imp_t}:")
        print(f"  Worst-case point FPR: {worst_fpr:.3f} at "
              f"({dgp}, cond={cond_s}, n={n}, rate={rate})")
        print(f"  95% CI: [{lo:.3f}, {hi:.3f}]")
        if hi <= 0.055:
            print("  PAIRED CANCELLATION ROBUST: upper CI <= 5.5% everywhere.")
        elif hi <= 0.10:
            print("  MODEST INFLATION: some cells have upper CI up to 10%.")
        else:
            print(f"  SUBSTANTIAL INFLATION: upper CI exceeds 10% in at least one cell.")

    # Linear vs RF imputer comparison
    if set(df["imputer"].unique()) >= {"linear", "rf"}:
        print("\nLinear vs RF imputer comparison (same DGP, cond_strategy, n, rate):")
        pivot = (df.groupby(["dgp", "cond_strategy", "n", "rate",
                             "imputer"])["rejected"]
                 .mean().unstack("imputer"))
        if {"linear", "rf"} <= set(pivot.columns):
            pivot = pivot[["linear", "rf"]]
            pivot["delta_rf_minus_lin"] = (pivot["rf"] - pivot["linear"]).round(3)
            print(pivot.round(3).to_string())
            print("\n(Negative delta_rf_minus_lin means RF imputer reduces FPR.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=200)
    parser.add_argument("--dgps", nargs="+", default=list(DGPS.keys()))
    parser.add_argument("--sample_sizes", nargs="+", type=int,
                        default=[500, 1000, 2000])
    parser.add_argument("--rates", nargs="+", type=float,
                        default=[0.3, 0.5])
    parser.add_argument("--cond_strategies", nargs="+",
                        default=["full", "missing"],
                        choices=["full", "missing"])
    parser.add_argument("--imputers", nargs="+",
                        default=["linear", "rf"],
                        choices=["linear", "rf"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/adversarial_kappa_results.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("ADVERSARIAL KAPPA EXPERIMENT")
    print("=" * 70)
    print(f"Output: {args.output}\n")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    t0 = time.time()
    df = run_experiment(args)
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    analyze(df, args.alpha)
