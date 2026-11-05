"""Standalone calibration and power of PAIR-CI under different base learners.

Holds the rest of PAIR-CI fixed (M = 5, K = 5, cached imputation, paired
permutation, Bayle within-imputation variance, Rubin's rules,
Barnard-Rubin df) and swaps the learner: random forest (the default),
ExtraTrees, LightGBM (default and tuned) and XGBoost; a HistGradientBoosting
variant is defined but was not run. DGPs are linear Gaussian and
post-nonlinear with |X| = 5, n in {500, 1000}, signal in {0, 0.3, 0.6, 1.0},
under complete data, MAR and MNAR (30% per column), 50 replicates per cell.
The estimands are the rejection rates at alpha = 0.05 and runtime per test.

    python experiments/learner_comparison.py --n_reps 50 \
        --learners rf extra_trees lgbm lgbm_tuned xgboost --M 5 --K 5 \
        --dgps linear_gaussian post_nonlinear --sample_sizes 500 1000 \
        --dim_x 5 --signals 0 0.3 0.6 1.0 --mechanisms complete mar mnar \
        --output results/learner_comparison_results.csv

Feeds App. O and App. T.2. Learner consistency (A5) is established for
random forests (Scornet et al. 2015), not for the boosting learners. Takes
about 8-12 hours on 16 cores.
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

from dgps import DGP_FUNCTIONS, inject_missingness_calibration
from ci_oracle import GeneralCITest

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class HGBTCITest(GeneralCITest):
    """PAIR-CI with scikit-learn's HistGradientBoosting as base learner.

    Only the per-fold learner hooks are overridden; the rest of the test is
    inherited. Defined but not run for the paper.
    """

    def _fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
        if len(np.unique(y_tr)) < 2:
            return 0.0, float("nan")
        # HGBT has no max_features option.
        cf = HistGradientBoostingClassifier(
            max_iter=100, min_samples_leaf=self.min_samples_leaf,
            learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
            random_state=self.rng.randint(0, 2**31),
        )
        cp = HistGradientBoostingClassifier(
            max_iter=100, min_samples_leaf=self.min_samples_leaf,
            learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
            random_state=self.rng.randint(0, 2**31),
        )
        cf.fit(Xf_tr, y_tr)
        cp.fit(Xp_tr, y_tr)
        eps = 1e-15
        pf = np.clip(cf.predict_proba(Xf_te), eps, 1 - eps)
        pp = np.clip(cp.predict_proba(Xp_te), eps, 1 - eps)
        individual = self._bce_individual(y_te, pp, cp.classes_) \
                     - self._bce_individual(y_te, pf, cf.classes_)
        return individual.mean(), np.var(individual, ddof=1)

    def _fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        rf = HistGradientBoostingRegressor(
            max_iter=100, min_samples_leaf=self.min_samples_leaf,
            learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
            random_state=self.rng.randint(0, 2**31),
        )
        rp = HistGradientBoostingRegressor(
            max_iter=100, min_samples_leaf=self.min_samples_leaf,
            learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
            random_state=self.rng.randint(0, 2**31),
        )
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)
        individual = (y_te - rp.predict(Xp_te)) ** 2 \
                     - (y_te - rf.predict(Xf_te)) ** 2
        return individual.mean(), np.var(individual, ddof=1)


class LGBMCITest(GeneralCITest):
    """PAIR-CI with LightGBM (100 trees, learning rate 0.1) as base learner.

    Registered only if lightgbm is installed.
    """

    def _fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
        if len(np.unique(y_tr)) < 2:
            return 0.0, float("nan")
        cf = LGBMClassifier(
            n_estimators=100, min_child_samples=self.min_samples_leaf,
            learning_rate=0.1, n_jobs=-1, verbosity=-1,
            random_state=self.rng.randint(0, 2**31),
        )
        cp = LGBMClassifier(
            n_estimators=100, min_child_samples=self.min_samples_leaf,
            learning_rate=0.1, n_jobs=-1, verbosity=-1,
            random_state=self.rng.randint(0, 2**31),
        )
        cf.fit(Xf_tr, y_tr)
        cp.fit(Xp_tr, y_tr)
        eps = 1e-15
        pf = np.clip(cf.predict_proba(Xf_te), eps, 1 - eps)
        pp = np.clip(cp.predict_proba(Xp_te), eps, 1 - eps)
        individual = self._bce_individual(y_te, pp, cp.classes_) \
                     - self._bce_individual(y_te, pf, cf.classes_)
        return individual.mean(), np.var(individual, ddof=1)

    def _fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        rf = LGBMRegressor(
            n_estimators=100, min_child_samples=self.min_samples_leaf,
            learning_rate=0.1, n_jobs=-1, verbosity=-1,
            random_state=self.rng.randint(0, 2**31),
        )
        rp = LGBMRegressor(
            n_estimators=100, min_child_samples=self.min_samples_leaf,
            learning_rate=0.1, n_jobs=-1, verbosity=-1,
            random_state=self.rng.randint(0, 2**31),
        )
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)
        individual = (y_te - rp.predict(Xp_te)) ** 2 \
                     - (y_te - rf.predict(Xf_te)) ** 2
        return individual.mean(), np.var(individual, ddof=1)


class ExtraTreesCITest(GeneralCITest):
    """PAIR-CI with ExtraTrees (Geurts et al. 2006) as base learner, in the
    general-variant harness (no early stopping)."""

    def _fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        nf = Xf_tr.shape[1]
        mf = self._get_max_features(nf)
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
        if len(np.unique(y_tr)) < 2:
            return 0.0, float("nan")
        cf = ExtraTreesClassifier(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31), n_jobs=-1,
        )
        cp = ExtraTreesClassifier(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31), n_jobs=-1,
        )
        cf.fit(Xf_tr, y_tr)
        cp.fit(Xp_tr, y_tr)
        eps = 1e-15
        pf = np.clip(cf.predict_proba(Xf_te), eps, 1 - eps)
        pp = np.clip(cp.predict_proba(Xp_te), eps, 1 - eps)
        individual = self._bce_individual(y_te, pp, cp.classes_) \
                     - self._bce_individual(y_te, pf, cf.classes_)
        return individual.mean(), np.var(individual, ddof=1)

    def _fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        nf = Xf_tr.shape[1]
        mf = self._get_max_features(nf)
        rf = ExtraTreesRegressor(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31), n_jobs=-1,
        )
        rp = ExtraTreesRegressor(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31), n_jobs=-1,
        )
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)
        individual = (y_te - rp.predict(Xp_te)) ** 2 \
                     - (y_te - rf.predict(Xf_te)) ** 2
        return individual.mean(), np.var(individual, ddof=1)


class LGBMTunedCITest(GeneralCITest):
    """LightGBM with less regularisation (200 trees, learning rate 0.05,
    min_child_samples 2, 63 leaves) and row subsampling (0.8), to approach
    the variance reduction of bagging."""

    def _fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
        if len(np.unique(y_tr)) < 2:
            return 0.0, float("nan")
        kw = dict(
            n_estimators=200, learning_rate=0.05,
            min_child_samples=2, num_leaves=63,
            subsample=0.8, subsample_freq=5,
            n_jobs=-1, verbosity=-1,
        )
        cf = LGBMClassifier(random_state=self.rng.randint(0, 2**31), **kw)
        cp = LGBMClassifier(random_state=self.rng.randint(0, 2**31), **kw)
        cf.fit(Xf_tr, y_tr)
        cp.fit(Xp_tr, y_tr)
        eps = 1e-15
        pf = np.clip(cf.predict_proba(Xf_te), eps, 1 - eps)
        pp = np.clip(cp.predict_proba(Xp_te), eps, 1 - eps)
        individual = self._bce_individual(y_te, pp, cp.classes_) \
                     - self._bce_individual(y_te, pf, cf.classes_)
        return individual.mean(), np.var(individual, ddof=1)

    def _fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        kw = dict(
            n_estimators=200, learning_rate=0.05,
            min_child_samples=2, num_leaves=63,
            subsample=0.8, subsample_freq=5,
            n_jobs=-1, verbosity=-1,
        )
        rf = LGBMRegressor(random_state=self.rng.randint(0, 2**31), **kw)
        rp = LGBMRegressor(random_state=self.rng.randint(0, 2**31), **kw)
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)
        individual = (y_te - rp.predict(Xp_te)) ** 2 \
                     - (y_te - rf.predict(Xf_te)) ** 2
        return individual.mean(), np.var(individual, ddof=1)


class XGBoostCITest(GeneralCITest):
    """PAIR-CI with XGBoost (200 trees, depth 6, learning rate 0.1) as base
    learner. Registered only if xgboost is installed."""

    def _fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
        if len(np.unique(y_tr)) < 2:
            return 0.0, float("nan")
        kw = dict(
            n_estimators=200, learning_rate=0.1,
            max_depth=6, n_jobs=-1, verbosity=0,
            eval_metric="logloss",
        )
        cf = XGBClassifier(random_state=self.rng.randint(0, 2**31), **kw)
        cp = XGBClassifier(random_state=self.rng.randint(0, 2**31), **kw)
        cf.fit(Xf_tr, y_tr)
        cp.fit(Xp_tr, y_tr)
        eps = 1e-15
        pf = np.clip(cf.predict_proba(Xf_te), eps, 1 - eps)
        pp = np.clip(cp.predict_proba(Xp_te), eps, 1 - eps)
        individual = self._bce_individual(y_te, pp, cp.classes_) \
                     - self._bce_individual(y_te, pf, cf.classes_)
        return individual.mean(), np.var(individual, ddof=1)

    def _fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        kw = dict(
            n_estimators=200, learning_rate=0.1,
            max_depth=6, n_jobs=-1, verbosity=0,
        )
        rf = XGBRegressor(random_state=self.rng.randint(0, 2**31), **kw)
        rp = XGBRegressor(random_state=self.rng.randint(0, 2**31), **kw)
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)
        individual = (y_te - rp.predict(Xp_te)) ** 2 \
                     - (y_te - rf.predict(Xf_te)) ** 2
        return individual.mean(), np.var(individual, ddof=1)


LEARNER_CLASSES = {
    "rf":          GeneralCITest,
    "extra_trees": ExtraTreesCITest,
    "hgbt":        HGBTCITest,
}
if HAS_LIGHTGBM:
    LEARNER_CLASSES["lgbm"] = LGBMCITest
    LEARNER_CLASSES["lgbm_tuned"] = LGBMTunedCITest
if HAS_XGBOOST:
    LEARNER_CLASSES["xgboost"] = XGBoostCITest


def run_single_test(data_miss, z_idx, y_idx, cond_set, learner, M, K, rng):
    cls = LEARNER_CLASSES[learner]
    ci = cls(
        data_miss,
        n_imputations=M,
        n_folds=K,
        imputation_strategy="cached",
        random_state=rng.randint(0, 2**31),
    )
    stat, pval = ci.test(x_idx=z_idx, y_idx=y_idx, cond_set=cond_set)
    return stat, pval


def run_experiment(args):
    rng = np.random.RandomState(args.seed)

    # Drop learners whose package is not installed.
    learners = [lr for lr in args.learners if lr in LEARNER_CLASSES]
    missing = [lr for lr in args.learners if lr not in LEARNER_CLASSES]
    if missing:
        print(f"WARNING: Requested learners not available, skipping: {missing}")
    if not learners:
        print("ERROR: No available learners to run. Exiting.")
        sys.exit(1)

    M = args.M
    K = args.K
    dgps = args.dgps
    sample_sizes = args.sample_sizes
    dim_x = args.dim_x
    signals = args.signals
    mechanisms = args.mechanisms
    miss_rate = args.miss_rate
    alpha = args.alpha
    n_reps = args.n_reps

    # Learners are looped inside each replicate, so within a
    # (dgp, n, signal, mechanism) cell every learner sees the same
    # (data, mask) and the comparison is paired.
    outer_configs = list(product(dgps, sample_sizes, signals, mechanisms))
    print(f"Learners: {learners}")
    print(f"M (fixed): {M}, K (fixed): {K}")
    print(f"DGPs: {dgps}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"dim_x: {dim_x}")
    print(f"Signals: {signals}")
    print(f"Mechanisms: {mechanisms}")
    print(f"Outer configs: {len(outer_configs)}")
    print(f"Reps per config: {n_reps}")
    print(f"Total tests: {len(outer_configs) * n_reps * len(learners)}")
    print()

    all_results = []
    t_start = time.time()

    for i, (dgp, n, signal, mechanism) in enumerate(outer_configs):
        elapsed = (time.time() - t_start) / 60
        print(f"[{elapsed:6.1f} min] Config {i+1}/{len(outer_configs)}: "
              f"{dgp}, n={n}, sig={signal}, {mechanism}")

        dgp_fn = DGP_FUNCTIONS[dgp]
        for rep in range(n_reps):
            X, Y, Z = dgp_fn(n, dim_x, signal, rng)
            X_miss = inject_missingness_calibration(X, mechanism, miss_rate, rng)
            data = np.column_stack([Z, Y, X_miss])
            z_idx = 0
            y_idx = 1
            cond_set = tuple(range(2, 2 + dim_x))
            # One test seed per learner, drawn from the main stream after the
            # data: identical data, distinct seeds.
            learner_seeds = [rng.randint(0, 2**31) for _ in learners]

            for learner, learner_seed in zip(learners, learner_seeds):
                learner_rng = np.random.RandomState(learner_seed)
                try:
                    t0 = time.time()
                    stat, pval = run_single_test(data, z_idx, y_idx, cond_set,
                                                 learner, M, K, learner_rng)
                    runtime = time.time() - t0
                    all_results.append(dict(
                        dgp=dgp, n=n, dim_x=dim_x, signal=signal,
                        mechanism=mechanism, learner=learner, M=M, K=K,
                        statistic=stat, p_value=pval,
                        rejected=pval < alpha, runtime=runtime,
                        failed=False, rep=rep,
                    ))
                except Exception:
                    all_results.append(dict(
                        dgp=dgp, n=n, dim_x=dim_x, signal=signal,
                        mechanism=mechanism, learner=learner, M=M, K=K,
                        statistic=np.nan, p_value=np.nan,
                        rejected=False, runtime=np.nan,
                        failed=True, rep=rep,
                    ))

    return pd.DataFrame(all_results)


def analyze(df, alpha=0.05):
    df = df[~df["failed"]].copy()

    print("\n" + "=" * 70)
    print("LEARNER-COMPARISON RESULTS")
    print("=" * 70)

    # MNAR calibration check: flag learners with FPR above 0.055.
    null = df[df["signal"] == 0]
    print("\n*** GO/NO-GO: MNAR calibration (FPR must be <= 0.05 to include) ***\n")
    mnar_fpr = null[null["mechanism"] == "mnar"].groupby("learner")["rejected"].mean()
    for learner, fpr in mnar_fpr.items():
        status = "PASS" if fpr <= 0.055 else "FAIL"
        print(f"  {learner:8s}: MNAR FPR = {fpr:.3f}  [{status}]")
    if (mnar_fpr > 0.055).any():
        print("\n  WARNING: At least one learner breached nominal calibration under MNAR.")
        print("  That learner MUST NOT be included in the paper without further work.")
    else:
        print("\n  All learners maintain MNAR calibration. Power comparison is meaningful.")

    # FPR at signal=0 (full breakdown)
    print("\n--- FPR under null (signal=0, target: 5%) ---\n")
    fpr = null.groupby(["learner", "mechanism"])["rejected"].mean().unstack("mechanism")
    col_order = [c for c in ["complete", "mar", "mnar"] if c in fpr.columns]
    print(fpr[col_order].round(3).to_string())

    # Power at each signal level
    for sig in sorted(df[df["signal"] > 0]["signal"].unique()):
        alt = df[df["signal"] == sig]
        print(f"\n--- Power at signal={sig} ---\n")
        pwr = alt.groupby(["learner", "mechanism"])["rejected"].mean().unstack("mechanism")
        col_order = [c for c in ["complete", "mar", "mnar"] if c in pwr.columns]
        print(pwr[col_order].round(3).to_string())

    # Breakdown by DGP at signal=0.3 (the weak-signal case)
    weak = df[df["signal"] == 0.3]
    if len(weak) > 0:
        print("\n--- Power at signal=0.3 by DGP ---\n")
        for dgp in sorted(weak["dgp"].unique()):
            sub = weak[weak["dgp"] == dgp]
            print(f"\n{dgp}:")
            pwr = sub.groupby(["learner", "mechanism"])["rejected"].mean().unstack("mechanism")
            col_order = [c for c in ["complete", "mar", "mnar"] if c in pwr.columns]
            print(pwr[col_order].round(3).to_string())

    # Runtime comparison
    print("\n--- Median runtime per test (seconds) ---\n")
    rt = df.groupby(["learner", "mechanism"])["runtime"].median().unstack("mechanism")
    col_order = [c for c in ["complete", "mar", "mnar"] if c in rt.columns]
    print(rt[col_order].round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learner comparison: RF vs HGBT for base classifier")
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--learners", nargs="+",
                        default=["rf", "extra_trees", "lgbm", "lgbm_tuned", "xgboost"])
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--dgps", nargs="+",
                        default=["linear_gaussian", "post_nonlinear"])
    parser.add_argument("--sample_sizes", nargs="+", type=int,
                        default=[500, 1000])
    parser.add_argument("--dim_x", type=int, default=5)
    parser.add_argument("--signals", nargs="+", type=float,
                        default=[0, 0.3, 0.6, 1.0])
    parser.add_argument("--mechanisms", nargs="+",
                        default=["complete", "mar", "mnar"])
    parser.add_argument("--miss_rate", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="results/learner_comparison_results.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("LEARNER-COMPARISON EXPERIMENT (RF vs HGBT)")
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
