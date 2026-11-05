"""Fast (ExtraTrees) variant of PAIR-CI, used inside PC on the larger graphs.

FastCITest is GeneralCITest (ci_oracle.py) with ExtraTrees learners,
K = 5 folds by default, and early stopping: once `early_stop_after`
imputations are done, the rest are skipped if |t| exceeds
`early_stop_threshold` (calibration checked in Appendix P). The null,
assumptions, variance and pooling are those of GeneralCITest. Running this
file compares runtime and null rejections of the two variants.
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from scipy import stats
from typing import Optional
from conditional_permutation import conditional_permute_batched


class FastCITest:
    """PAIR-CI with ExtraTrees and early stopping; see GeneralCITest."""

    def __init__(
        self,
        data: np.ndarray,
        n_imputations: int = 5,
        n_folds: int = 5,
        n_trees: int = 100,
        min_samples_leaf: int = 5,
        max_subsample: int = 2000,
        imputation_strategy: str = "cached",
        early_stop: bool = True,
        early_stop_after: int = 2,
        early_stop_threshold: float = 4.0,
        use_conditional_permutation: bool = True,
        variance_estimator: str = "bayle",
        random_state: Optional[int] = None,
    ):
        if isinstance(data, pd.DataFrame):
            data = data.values.astype(float)
        self.data = data
        self.n, self.p = data.shape
        self.n_imputations = n_imputations
        self.n_folds = n_folds
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.max_subsample = max_subsample
        self.imputation_strategy = imputation_strategy
        self.early_stop = early_stop
        self.early_stop_after = early_stop_after
        self.early_stop_threshold = early_stop_threshold
        self.use_conditional_permutation = use_conditional_permutation
        self.variance_estimator = variance_estimator
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self._cached_imputations = None
        if imputation_strategy == "cached":
            self._cached_imputations = self._impute_full_dataset()

    def __call__(self, x_idx, y_idx, cond_set=()):
        return self.test(x_idx, y_idx, cond_set)

    def test(self, x_idx, y_idx, cond_set=()):
        cond_set = tuple(cond_set)
        target_is_discrete = self._is_discrete(x_idx)

        if self.imputation_strategy == "cached":
            imputed_datasets = self._cached_imputations
        else:
            imputed_datasets = self._impute_per_query(x_idx, y_idx, cond_set)

        fold_diffs, fold_vars = self._compare_models_early_stop(
            imputed_datasets, x_idx, y_idx, cond_set, target_is_discrete
        )

        return self._compute_statistic(fold_diffs, fold_vars)

    # ------------------------------------------------------------------
    # Imputation (as in GeneralCITest)
    # ------------------------------------------------------------------

    def _impute_full_dataset(self):
        imputed = []
        for m in range(self.n_imputations):
            imp = IterativeImputer(
                max_iter=10, random_state=self.rng.randint(0, 2**31),
                sample_posterior=True,
            )
            imputed.append(imp.fit_transform(self.data))
        return imputed

    def _impute_per_query(self, x_idx, y_idx, cond_set):
        all_cols = list(range(self.p))
        exclude = {x_idx, y_idx}
        impute_cols = [c for c in all_cols if c not in exclude]
        imputed = []
        for m in range(self.n_imputations):
            imp = IterativeImputer(
                max_iter=10, random_state=self.rng.randint(0, 2**31),
                sample_posterior=True,
            )
            imputed_subset = imp.fit_transform(self.data[:, impute_cols])
            full = np.empty((self.n, self.p))
            j = 0
            for c in range(self.p):
                if c in exclude:
                    full[:, c] = self.data[:, c]
                else:
                    full[:, c] = imputed_subset[:, j]
                    j += 1
            for ec in sorted(exclude):
                mask_ec = np.isnan(full[:, ec])
                if mask_ec.all():
                    full[:, ec] = 0.0
                elif mask_ec.any():
                    imp_ec = IterativeImputer(
                        max_iter=10, random_state=self.rng.randint(0, 2**31),
                        sample_posterior=True,
                    )
                    full_tmp = imp_ec.fit_transform(full.copy())
                    full[:, ec] = full_tmp[:, ec]
            imputed.append(full)
        return imputed

    # ------------------------------------------------------------------
    # Model comparison with early stopping
    # ------------------------------------------------------------------

    def _compare_models_early_stop(self, imputed_datasets, x_idx, y_idx,
                                    cond_set, target_is_discrete):
        M = len(imputed_datasets)
        K = self.n_folds
        diffs = np.zeros((M, K))
        fold_vars = np.zeros((M, K))

        for m, imp_data in enumerate(imputed_datasets):
            if imp_data.shape[0] > self.max_subsample:
                idx = self.rng.choice(
                    imp_data.shape[0], self.max_subsample, replace=False
                )
                imp_data = imp_data[idx]

            n = imp_data.shape[0]
            target = imp_data[:, x_idx]
            y_col = imp_data[:, y_idx]
            X_cond = (imp_data[:, list(cond_set)]
                      if len(cond_set) > 0
                      else np.empty((n, 0)))

            kf = KFold(n_splits=K, shuffle=True,
                       random_state=self.rng.randint(0, 2**31))

            for ki, (tr, te) in enumerate(kf.split(imp_data)):
                if X_cond.shape[1] > 0:
                    Xf_tr = np.column_stack([X_cond[tr], y_col[tr]])
                    Xf_te = np.column_stack([X_cond[te], y_col[te]])
                else:
                    Xf_tr = y_col[tr].reshape(-1, 1)
                    Xf_te = y_col[te].reshape(-1, 1)

                if self.use_conditional_permutation and X_cond.shape[1] > 0:
                    plac_tr = conditional_permute_batched(y_col[tr], X_cond[tr], rng=self.rng)
                    plac_te = conditional_permute_batched(y_col[te], X_cond[te], rng=self.rng)
                else:
                    plac_tr = self.rng.permutation(y_col[tr])
                    plac_te = self.rng.permutation(y_col[te])

                if X_cond.shape[1] > 0:
                    Xp_tr = np.column_stack([X_cond[tr], plac_tr])
                    Xp_te = np.column_stack([X_cond[te], plac_te])
                else:
                    Xp_tr = plac_tr.reshape(-1, 1)
                    Xp_te = plac_te.reshape(-1, 1)

                if target_is_discrete:
                    diffs[m, ki], fold_vars[m, ki] = self._fold_cls(
                        Xf_tr, Xf_te, Xp_tr, Xp_te, target[tr], target[te]
                    )
                else:
                    diffs[m, ki], fold_vars[m, ki] = self._fold_reg(
                        Xf_tr, Xf_te, Xp_tr, Xp_te, target[tr], target[te]
                    )

            # Stops on |t|, so a strongly negative statistic also ends early;
            # the statistic is then pooled over the m + 1 imputations done.
            if (self.early_stop and m + 1 >= self.early_stop_after
                    and m + 1 < M):
                partial = diffs[:m + 1, :]
                partial_fv = fold_vars[:m + 1, :]
                t_partial, _ = self._compute_statistic(partial, partial_fv)

                if abs(t_partial) > self.early_stop_threshold:
                    return diffs[:m + 1, :], fold_vars[:m + 1, :]

        return diffs, fold_vars

    # ------------------------------------------------------------------
    # Classification / Regression (using ExtraTrees)
    # ------------------------------------------------------------------

    def _fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        nf = Xf_tr.shape[1]
        mf = self._get_max_features(nf)
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
        if len(np.unique(y_tr)) < 2:
            # Single-class fold: skip via NaN variance (see GeneralCITest).
            return 0.0, np.nan

        cf = ExtraTreesClassifier(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31),
            n_jobs=-1,
        )
        cp = ExtraTreesClassifier(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31),
            n_jobs=-1,
        )
        cf.fit(Xf_tr, y_tr)
        cp.fit(Xp_tr, y_tr)

        eps = 1e-15
        pf = np.clip(cf.predict_proba(Xf_te), eps, 1 - eps)
        pp = np.clip(cp.predict_proba(Xp_te), eps, 1 - eps)
        loss_p = self._bce_individual(y_te, pp, cp.classes_)
        loss_f = self._bce_individual(y_te, pf, cf.classes_)
        d = loss_p - loss_f
        return d.mean(), d.var(ddof=1)

    def _fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        nf = Xf_tr.shape[1]
        mf = self._get_max_features(nf)

        rf = ExtraTreesRegressor(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31),
            n_jobs=-1,
        )
        rp = ExtraTreesRegressor(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31),
            n_jobs=-1,
        )
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)

        d = (y_te - rp.predict(Xp_te)) ** 2 - (y_te - rf.predict(Xf_te)) ** 2
        return d.mean(), d.var(ddof=1)

    # ------------------------------------------------------------------
    # Pooling (as GeneralCITest, without the last_components_ snapshot)
    # ------------------------------------------------------------------

    def _compute_statistic(self, diffs, fold_vars=None):
        M, K = diffs.shape
        n = min(self.n, self.max_subsample)
        n_k = n // K
        mu_m = diffs.mean(axis=1)
        mu_hat = mu_m.mean()
        U_m = np.zeros(M)
        for m in range(M):
            if self.variance_estimator == "bayle" and fold_vars is not None:
                fv = fold_vars[m, :]
                if np.all(np.isnan(fv)):
                    U_m[m] = 0.0
                else:
                    U_m[m] = np.nanmean(fv) / n
            else:
                s2 = np.var(diffs[m, :], ddof=1)
                U_m[m] = (1.0 / K + n_k / (n - n_k)) * s2
        W_bar = U_m.mean()
        B = np.var(mu_m, ddof=1) if M > 1 else 0.0
        T = W_bar + (1.0 + 1.0 / M) * B
        if T <= 0 or np.isnan(T):
            return 0.0, 0.5
        t_stat = mu_hat / np.sqrt(T)
        if M > 1 and B > 0 and W_bar > 0:
            r = (1.0 + 1.0 / M) * B / W_bar
            df_m = (M - 1) * (1.0 + 1.0 / r) ** 2
            # nu_com = n - K: the Bayle et al. (2020, Thm 4) variance pools
            # per-observation deviations within folds, not K fold means.
            # K - 1 would make the Barnard-Rubin df far too small at small K.
            nu_com = max(n - K, 1)
            gamma = (1.0 + 1.0 / M) * B / T
            df_obs = (1.0 - gamma) * (nu_com * (nu_com + 1)) / (nu_com + 3)
            df = (df_m * df_obs) / (df_m + df_obs)
            df = max(df, 1.0)
        else:
            df = max(n - K, 1.0)
        p_value = 1.0 - stats.t.cdf(t_stat, df=df)
        return t_stat, p_value

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_discrete(self, col_idx):
        vals = self.data[:, col_idx]
        vals = vals[~np.isnan(vals)]
        return len(np.unique(vals)) <= 20

    def _get_max_features(self, n_features):
        if n_features < 12:
            return None
        elif n_features <= 80:
            return 12
        else:
            return int(np.sqrt(n_features))

    @staticmethod
    def _bce_individual(y_true, proba, classes):
        eps = 1e-15
        n = len(y_true)
        c2i = {c: i for i, c in enumerate(classes)}
        losses = np.zeros(n)
        for i in range(n):
            c = y_true[i]
            p = proba[i, c2i[c]] if c in c2i else eps
            losses[i] = -np.log(np.clip(p, eps, 1.0 - eps))
        return losses


# ------------------------------------------------------------------
# Benchmarking: FastCITest vs GeneralCITest
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from ci_oracle import GeneralCITest

    print("=== FastCITest vs GeneralCITest Benchmark ===\n")
    np.random.seed(42)
    n = 1000

    Z = np.random.randn(n, 5)
    X = Z @ np.random.randn(5) + np.random.randn(n) * 0.5
    Y = 0.5 * X + np.random.randn(n) * 0.5
    data = np.column_stack([X, Y, Z])

    t0 = time.time()
    ci_std = GeneralCITest(data, n_imputations=5, n_folds=5, random_state=42)
    s1, p1 = ci_std.test(0, 1, (2, 3, 4, 5, 6))
    t_std = time.time() - t0

    t0 = time.time()
    ci_fast = FastCITest(data, random_state=42)
    s2, p2 = ci_fast.test(0, 1, (2, 3, 4, 5, 6))
    t_fast = time.time() - t0

    print(f"Standard: t={s1:.3f}, p={p1:.4f}, time={t_std:.2f}s")
    print(f"Fast:     t={s2:.3f}, p={p2:.4f}, time={t_fast:.2f}s")
    print(f"Speedup:  {t_std/t_fast:.1f}x")

    print("\nCalibration check (10 reps under null):")
    rng = np.random.RandomState(123)
    rej_std, rej_fast = 0, 0
    n_reps = 10
    for _ in range(n_reps):
        Z = rng.randn(n, 3)
        X = Z @ rng.randn(3) + rng.randn(n) * 0.5
        Y = rng.randn(n)  # independent
        data = np.column_stack([X, Y, Z])
        _, p_s = GeneralCITest(data, n_imputations=5, n_folds=5,
                                random_state=rng.randint(0, 2**31)).test(0, 1, (2, 3, 4))
        _, p_f = FastCITest(data,
                            random_state=rng.randint(0, 2**31)).test(0, 1, (2, 3, 4))
        if p_s < 0.05: rej_std += 1
        if p_f < 0.05: rej_fast += 1

    print(f"  Standard rejections: {rej_std}/{n_reps}")
    print(f"  Fast rejections:     {rej_fast}/{n_reps}")

    # Extrapolates the single-test runtimes above to a p = 20 PC run;
    # 190 = 20 choose 2, the number of depth-0 tests.
    n_ci_tests_p20 = 190
    print(f"\nEstimated PC runtime (p=20, ~{n_ci_tests_p20} CI tests):")
    print(f"  Standard (M=5,K=5,100 trees): ~{n_ci_tests_p20 * t_std / 60:.0f} min")
    print(f"  Fast (M=3,K=3,50 trees):      ~{n_ci_tests_p20 * t_fast / 60:.0f} min")
