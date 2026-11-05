"""PAIR-CI calibration when the imputer is deliberately poor.

Runs PAIR-CI (general variant: M = 5, K = 10, 200 trees) on imputations
supplied from outside, under three strategies:
  mice    MICE with posterior draws (IterativeImputer), the default;
  mean    column-mean imputation, so the M datasets are identical (B = 0);
  sample  draws from each column's observed marginal ("marginal" in the
          paper), ignoring the other variables.
DGPs are linear Gaussian and post-nonlinear, n = 500, |X| = 5, 30% per
column under complete data, MAR and MNAR, signal 0 and 0.6, 200 replicates.
The estimands are the rejection rates at alpha = 0.05.

    python experiments/imputation_degradation.py --n_reps 200 --n_obs 500 \
        --dim_x 5 --seed 42 --output results/imputation_degradation.csv

On complete data only MICE is run and its rows are copied to the other two
strategies, so the run makes 5,600 test evaluations. Feeds App. Q (Table
degradation_combined).
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np
import pandas as pd
import time
import argparse

from ci_oracle import GeneralCITest
from dgps import (dgp_linear_gaussian, dgp_post_nonlinear,
                  inject_missingness_calibration)


# ── Imputation strategies ────────────────────────────────────────────

def impute_mice(data, M, rng):
    """Proper multiple imputation via MICE (IterativeImputer)."""
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    datasets = []
    for _ in range(M):
        imp = IterativeImputer(max_iter=10,
                               random_state=rng.randint(0, 2**31),
                               sample_posterior=True)
        datasets.append(imp.fit_transform(data))
    return datasets


def impute_mean(data, M, rng):
    """Column-mean imputation. Deterministic — all M copies identical."""
    completed = data.copy()
    for col in range(data.shape[1]):
        mask = np.isnan(data[:, col])
        if mask.any():
            completed[mask, col] = np.nanmean(data[:, col])
    return [completed.copy() for _ in range(M)]


def impute_sample(data, M, rng):
    """Marginal sampling: replace each missing value with a random
    draw from the observed values of that column."""
    datasets = []
    for _ in range(M):
        completed = data.copy()
        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])
            if mask.any():
                observed = data[~mask, col]
                completed[mask, col] = rng.choice(observed,
                                                  size=mask.sum(),
                                                  replace=True)
        datasets.append(completed)
    return datasets


STRATEGIES = {
    "mice": impute_mice,
    "mean": impute_mean,
    "sample": impute_sample,
}


# ── CI test with pre-computed imputations ────────────────────────────

class PreimputedCITest(GeneralCITest):
    """GeneralCITest that accepts pre-computed imputed datasets,
    bypassing the built-in imputation step."""

    def __init__(self, data, imputed_datasets, **kwargs):
        # Construct as per-query so the parent does not impute, then
        # switch to cached and supply the datasets.
        kwargs["imputation_strategy"] = "per_query"
        super().__init__(data, **kwargs)
        self.imputation_strategy = "cached"
        self._cached_imputations = imputed_datasets


# ── Main simulation ──────────────────────────────────────────────────

DGP_FNS = {
    "linear_gaussian": dgp_linear_gaussian,
    "post_nonlinear": dgp_post_nonlinear,
}


def run_simulation(n_reps=200, n_obs=500, signals=(0, 0.6),
                   dim_x=5, M=5, K=10, n_trees=200,
                   alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    results = []
    mechanisms = ["complete", "mar", "mnar"]

    configs = [(dgp, sig, mech)
               for dgp in DGP_FNS
               for sig in signals
               for mech in mechanisms]

    total_configs = len(configs)
    t0 = time.time()

    for cfg_i, (dgp_name, signal, mechanism) in enumerate(configs):
        dgp_fn = DGP_FNS[dgp_name]
        cond = tuple(range(2, 2 + dim_x))

        for rep in range(n_reps):
            X, Y, Z = dgp_fn(n_obs, dim_x, signal, rng)
            data = np.column_stack([Z, Y, X])

            if mechanism != "complete":
                X_miss = inject_missingness_calibration(
                    data[:, 2:], mechanism, 0.3, rng)
                data[:, 2:] = X_miss

            # Every strategy uses this seed for both its imputation draws and
            # the test, so the strategies are paired on data and seeds.
            rs = rng.randint(0, 2**31)

            for strat_name, strat_fn in STRATEGIES.items():
                # With nothing to impute the strategies coincide: run MICE.
                if mechanism == "complete" and strat_name != "mice":
                    continue

                imp_rng = np.random.RandomState(rs)
                imputed = strat_fn(data, M, imp_rng)

                ci = PreimputedCITest(
                    data, imputed,
                    n_imputations=M, n_folds=K, n_trees=n_trees,
                    random_state=rs)

                t_stat, p_val = ci.test(0, 1, cond)

                results.append({
                    "dgp": dgp_name, "n_obs": n_obs, "signal": signal,
                    "mechanism": mechanism, "rep": rep,
                    "strategy": strat_name,
                    "t_stat": t_stat, "p_value": p_val,
                    "reject": p_val < alpha,
                })

        elapsed = (time.time() - t0) / 60
        print(f"  [{elapsed:6.1f} min] Config {cfg_i+1}/{total_configs}: "
              f"{dgp_name} sig={signal} {mechanism}")

    # Copy the complete-data MICE rows to mean and sample.
    df = pd.DataFrame(results)
    complete_mice = df[(df["mechanism"] == "complete") &
                       (df["strategy"] == "mice")].copy()
    for strat in ["mean", "sample"]:
        dup = complete_mice.copy()
        dup["strategy"] = strat
        df = pd.concat([df, dup], ignore_index=True)

    return df


# ── Analysis ─────────────────────────────────────────────────────────

def analyze(df, alpha=0.05):
    print("\n" + "=" * 70)
    print("IMPUTATION DEGRADATION EXPERIMENT")
    print("=" * 70)

    # --- FPR ---
    null = df[df["signal"] == 0]
    if len(null) > 0:
        print("\n--- FPR under null (signal=0, target: 5%) ---\n")
        pivot = (null.groupby(["strategy", "mechanism"])["reject"]
                 .mean().unstack("mechanism"))
        for col in ["complete", "mar", "mnar"]:
            if col in pivot.columns:
                pivot[col] = pivot[col].map(lambda x: f"{x:.3f}")
        strat_order = [s for s in ["mice", "mean", "sample"]
                       if s in pivot.index]
        pivot = pivot.loc[strat_order]
        print(pivot.to_string())

        print("\n--- FPR by DGP (signal=0) ---\n")
        for strat in strat_order:
            sub = null[null["strategy"] == strat]
            pivot2 = (sub.groupby(["dgp", "mechanism"])["reject"]
                      .mean().unstack("mechanism"))
            for col in pivot2.columns:
                pivot2[col] = pivot2[col].map(lambda x: f"{x:.3f}")
            print(f"{strat.upper()}:")
            print(pivot2.to_string())
            print()

    # --- Power ---
    alt = df[df["signal"] > 0]
    if len(alt) > 0:
        print("--- Power (signal > 0) ---\n")
        for sig in sorted(alt["signal"].unique()):
            sub = alt[alt["signal"] == sig]
            pivot3 = (sub.groupby(["strategy", "mechanism"])["reject"]
                      .mean().unstack("mechanism"))
            for col in pivot3.columns:
                pivot3[col] = pivot3[col].map(lambda x: f"{x:.3f}")
            strat_order = [s for s in ["mice", "mean", "sample"]
                           if s in pivot3.index]
            pivot3 = pivot3.loc[strat_order]
            print(f"Signal = {sig}:")
            print(pivot3.to_string())
            print()

    # --- Key message ---
    print("=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    if len(null) > 0:
        max_fpr = null.groupby(["strategy", "mechanism"])["reject"].mean()
        worst = max_fpr.max()
        worst_idx = max_fpr.idxmax()
        print(f"\n  Worst-case FPR across all strategies: "
              f"{worst:.3f} ({worst_idx[0]}, {worst_idx[1]})")
        print(f"  Target: 0.050")
        if worst <= alpha:
            print(f"  --> ALL strategies remain calibrated (FPR <= {alpha})")
            print(f"      Calibration comes from the permutation design,")
            print(f"      not the imputation quality.")
        else:
            print(f"  --> Some strategies exceed nominal alpha.")
            print(f"      Check which strategies and mechanisms are affected.")
    print()


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imputation degradation experiment")
    parser.add_argument("--n_reps", type=int, default=200,
                        help="Reps per config (default: 200)")
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--dim_x", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/imputation_degradation.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("IMPUTATION DEGRADATION EXPERIMENT")
    print("=" * 70)
    print(f"\nStrategies: {list(STRATEGIES.keys())}")
    print(f"DGPs: {list(DGP_FNS.keys())}")
    print(f"n_obs: {args.n_obs}, dim_x: {args.dim_x}")
    print(f"Signals: [0, 0.6]")
    print(f"Mechanisms: [complete, mar, mnar]")
    print(f"Reps: {args.n_reps}")
    n_evals = (2 * 2 * 2 * args.n_reps * 3  # mar/mnar: all strategies
               + 2 * 2 * args.n_reps)        # complete: mice only
    print(f"Total CI test evaluations: ~{n_evals}")
    print()

    t0 = time.time()
    df = run_simulation(n_reps=args.n_reps, n_obs=args.n_obs,
                        dim_x=args.dim_x, seed=args.seed)
    total_min = (time.time() - t0) / 60
    print(f"\nTotal time: {total_min:.1f} min")

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    analyze(df)
