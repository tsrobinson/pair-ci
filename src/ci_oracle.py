"""PAIR-CI conditional-independence test, general (random-forest) variant.

GeneralCITest tests X_i indep X_j | X_S on incompletely observed data by
comparing held-out losses of a learner given (X_S, X_j) and given
(X_S, placebo of X_j), pooled over cross-validation folds and multiple
imputations (Section 3; defaults in Appendix E). Running this file performs
a few null and alternative sanity checks.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from scipy import stats
from typing import Optional
from conditional_permutation import conditional_permute_batched
import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.*")

_HAS_CITEST = False
try:
    from citest import CIMissTest
    from citest.data import Dataset
    _HAS_CITEST = True
except ImportError:
    pass


class GeneralCITest:
    """PAIR-CI test of H0: X[x_idx] indep X[y_idx] | X[cond_set].

    In each of M imputed datasets and K folds, random forests predict the
    response X[x_idx] from (X_S, X[y_idx]) and from (X_S, placebo), where
    the placebo is a local permutation of X[y_idx] given X_S. The statistic
    averages the held-out loss difference (placebo minus full); within-
    imputation variance is that of Bayle et al. (2020), pooled across
    imputations by Rubin's rules with Barnard-Rubin (1999) df. The test is
    one-sided: large values mean X[y_idx] improves prediction.

    Size is asymptotically at most alpha for the null given the imputed
    X_S under (A1'), (A2)-(A4), as M grows. For the null given the true
    X_S it also needs a consistent imputer (A1), in practice MAR; under
    MNAR calibration is measured rather than proven (Appendix N).

    imputation_strategy="cached" imputes the full data M times once and
    reuses the imputations for every query; any other value re-imputes per
    query, leaving the two test variables out of the conditioning-set
    imputation. variance_estimator="bayle" or, for anything else, the
    Nadeau-Bengio corrected variance (Appendix K). All randomness, including
    the imputers, folds, placebos and forests, is drawn in sequence from
    one RandomState seeded with random_state, so results depend on the
    order of queries.
    """

    def __init__(self, data, n_imputations=5, n_folds=10, n_trees=100,
                 min_samples_leaf=5, max_subsample=2000,
                 imputation_strategy="cached",
                 use_conditional_permutation=True,
                 variance_estimator="bayle",
                 random_state=None):
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
        """Return (t statistic, one-sided p-value)."""
        cond_set = tuple(cond_set)
        target_is_discrete = self._is_discrete(x_idx)
        if self.imputation_strategy == "cached":
            imputed_datasets = self._cached_imputations
        else:
            imputed_datasets = self._impute_per_query(x_idx, y_idx, cond_set)
        fold_diffs, fold_vars = self._compare_models(
            imputed_datasets, x_idx, y_idx, cond_set, target_is_discrete
        )
        return self._compute_statistic(fold_diffs, fold_vars)

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
        data_subset = self.data[:, impute_cols]
        imputed = []
        for m in range(self.n_imputations):
            imp = IterativeImputer(
                max_iter=10, random_state=self.rng.randint(0, 2**31),
                sample_posterior=True,
            )
            imputed_subset = imp.fit_transform(data_subset)
            full = np.empty((self.n, self.p))
            j = 0
            for c in range(self.p):
                if c in exclude:
                    full[:, c] = self.data[:, c]
                else:
                    full[:, c] = imputed_subset[:, j]
                    j += 1
            # The test variables are imputed afterwards, from the completed
            # remaining columns.
            for ec in sorted(exclude):
                mask_ec = np.isnan(full[:, ec])
                if mask_ec.all():
                    # Nothing observed to fit: fill with 0 (the mean if the
                    # data are standardised), a null-signal placeholder for
                    # extreme-missingness cells.
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

    def _compare_models(self, imputed_datasets, x_idx, y_idx, cond_set,
                        target_is_discrete):
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
            X_cond = imp_data[:, list(cond_set)] if len(cond_set) > 0 else np.empty((n, 0))
            kf = KFold(n_splits=K, shuffle=True,
                       random_state=self.rng.randint(0, 2**31))
            for k, (tr, te) in enumerate(kf.split(imp_data)):
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
                    mean_d, var_d = self._fold_cls(
                        Xf_tr, Xf_te, Xp_tr, Xp_te, target[tr], target[te]
                    )
                else:
                    mean_d, var_d = self._fold_reg(
                        Xf_tr, Xf_te, Xp_tr, Xp_te, target[tr], target[te]
                    )
                diffs[m, k] = mean_d
                fold_vars[m, k] = var_d
        return diffs, fold_vars

    def _fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        nf = Xf_tr.shape[1]
        mf = self._get_max_features(nf)
        y_tr, y_te = y_tr.astype(int), y_te.astype(int)
        if len(np.unique(y_tr)) < 2:
            # Single-class training fold: a zero variance would bias the
            # fold-averaged Bayle variance down, so return NaN (skipped by
            # nanmean in _compute_statistic) with a neutral difference of 0.
            return 0.0, np.nan
        cf = RandomForestClassifier(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31),n_jobs=-1,
        )
        cp = RandomForestClassifier(
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
        rf = RandomForestRegressor(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31), n_jobs=-1,
        )
        rp = RandomForestRegressor(
            n_estimators=self.n_trees, min_samples_leaf=self.min_samples_leaf,
            max_features=mf, random_state=self.rng.randint(0, 2**31),n_jobs=-1,
        )
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)
        individual = (y_te - rp.predict(Xp_te)) ** 2 \
                     - (y_te - rf.predict(Xf_te)) ** 2
        return individual.mean(), np.var(individual, ddof=1)

    def _compute_statistic(self, diffs, fold_vars=None):
        """Pool (M, K) fold differences by Rubin's rules into (t, p).

        A non-positive or NaN total variance returns (0.0, 0.5).
        """
        M, K = diffs.shape
        n = min(self.n, self.max_subsample)
        n_k = n // K
        mu_m = diffs.mean(axis=1)
        mu_hat = mu_m.mean()
        U_m = np.zeros(M)
        for m in range(M):
            if self.variance_estimator == "bayle" and fold_vars is not None:
                # nanmean skips single-class folds that returned NaN variance.
                fv = fold_vars[m, :]
                if np.all(np.isnan(fv)):
                    U_m[m] = 0.0
                else:
                    U_m[m] = np.nanmean(fv) / n
            else:
                # Nadeau and Bengio (1999) corrected resampled variance.
                s2 = np.var(diffs[m, :], ddof=1)
                U_m[m] = (1.0 / K + n_k / (n - n_k)) * s2
        W_bar = U_m.mean()
        B = np.var(mu_m, ddof=1) if M > 1 else 0.0
        T = W_bar + (1.0 + 1.0 / M) * B
        if T <= 0 or np.isnan(T):
            self.last_components_ = dict(
                mu_hat=mu_hat, W_bar=W_bar, B=B, T=T, df=np.nan,
                M=M, K=K, n_eff=n, degenerate=True,
            )
            return 0.0, 0.5
        t_stat = mu_hat / np.sqrt(T)
        if M > 1 and B > 0 and W_bar > 0:
            r = (1.0 + 1.0 / M) * B / W_bar
            df_m = (M - 1) * (1.0 + 1.0 / r) ** 2
            # Barnard and Rubin (1999) df. Complete-data df nu_com = n - K:
            # the Bayle et al. (2020, Thms 1 and 4) variance pools per-
            # observation deviations within each fold (n/K - 1 df per fold).
            # K - 1 would treat the K fold means as the units instead.
            nu_com = max(n - K, 1)
            gamma = (1.0 + 1.0 / M) * B / T
            df_obs = (1.0 - gamma) * (nu_com * (nu_com + 1)) / (nu_com + 3)
            df = (df_m * df_obs) / (df_m + df_obs)
            df = max(df, 1.0)
        else:
            df = max(n - K, 1.0)
        p_value = 1.0 - stats.t.cdf(t_stat, df=df)
        # Variance components of the last test, read by
        # experiments/variance_decomposition.py and placebo_diagnostics.py.
        self.last_components_ = dict(
            mu_hat=mu_hat, W_bar=W_bar, B=B, T=T, df=df,
            M=M, K=K, n_eff=n, degenerate=False,
        )
        return t_stat, p_value

    def _is_discrete(self, col_idx):
        """At most 20 distinct observed values: classify with log loss."""
        vals = self.data[:, col_idx]
        vals = vals[~np.isnan(vals)]
        return len(np.unique(vals)) <= 20

    def _get_max_features(self, n_features):
        """No feature bagging below 12 features, so the candidate is
        available at every split (Appendix E)."""
        if n_features < 12:
            return None
        elif n_features <= 80:
            return 12
        else:
            return int(np.sqrt(n_features))

    @staticmethod
    def _bce(y_true, proba, classes):
        eps = 1e-15
        n = len(y_true)
        c2i = {c: i for i, c in enumerate(classes)}
        loss = 0.0
        for i in range(n):
            c = y_true[i]
            p = proba[i, c2i[c]] if c in c2i else eps
            loss -= np.log(np.clip(p, eps, 1.0 - eps))
        return loss / n

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


class CIIRATestWrapper:
    """Wrapper around the optional citest package; used only by the
    consistency check when this file is run."""

    def __init__(self):
        if not _HAS_CITEST:
            raise ImportError("citest not found. pip install git+https://github.com/MIDASverse/citest.git")

    @staticmethod
    def run_ciira_test(data, y_col, expl_vars, **kwargs):
        dataset = Dataset()
        dataset.make(data, y=y_col, expl_vars=expl_vars)
        test = CIMissTest(dataset, **kwargs)
        test.run()
        return test


class CausalLearnCIWrapper:
    """GeneralCITest behind a causal-learn style call returning a p-value."""

    def __init__(self, data, **kwargs):
        self.ci_test = GeneralCITest(data, **kwargs)

    def causal_learn_test(self, data, x, y, conditioning_set):
        _, p_value = self.ci_test.test(x, y, tuple(conditioning_set))
        return p_value


if __name__ == "__main__":
    print("=== CI Oracle Validation ===\n")
    np.random.seed(42)
    n = 1000

    print("Test 1: X perp Y | Z (should NOT reject, p > 0.05)")
    Z = np.random.randn(n, 2)
    X = Z @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5
    Y = np.random.randn(n)
    data_indep = np.column_stack([X, Y, Z])
    ci = GeneralCITest(data_indep, n_imputations=5, n_folds=5, random_state=42)
    s, p = ci.test(x_idx=0, y_idx=1, cond_set=(2, 3))
    print(f"  t_CI = {s:.3f}, p = {p:.3f}\n")

    print("Test 2: X not perp Y | Z (should reject, p < 0.05)")
    Z = np.random.randn(n, 2)
    X = Z @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5
    Y = 0.8 * X + np.random.randn(n) * 0.3
    data_dep = np.column_stack([X, Y, Z])
    ci2 = GeneralCITest(data_dep, n_imputations=5, n_folds=5, random_state=42)
    s2, p2 = ci2.test(x_idx=0, y_idx=1, cond_set=(2, 3))
    print(f"  t_CI = {s2:.3f}, p = {p2:.3f}\n")

    print("Test 3: X perp Y | Z with MNAR missingness (should NOT reject, p > 0.05)")
    data_miss = data_indep.copy()
    data_miss[data_miss[:, 2] > 0.5, 2] = np.nan
    ci3 = GeneralCITest(data_miss, n_imputations=5, n_folds=5,
                        imputation_strategy="cached", random_state=42)
    s3, p3 = ci3.test(x_idx=0, y_idx=1, cond_set=(2, 3))
    print(f"  t_CI = {s3:.3f}, p = {p3:.3f}\n")

    print("Test 4: Continuous X not perp Y | Z (regression, should reject, p < 0.05)")
    Z = np.random.randn(n, 2)
    X = Z @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5
    Y = 0.5 * X + np.random.randn(n) * 0.5
    data_cont = np.column_stack([X, Y, Z])
    ci4 = GeneralCITest(data_cont, n_imputations=5, n_folds=5, random_state=42)
    s4, p4 = ci4.test(x_idx=0, y_idx=1, cond_set=(2, 3))
    print(f"  t_CI = {s4:.3f}, p = {p4:.3f}\n")

    if _HAS_CITEST:
        print("Test 5: Consistency check against citest package")
        df = pd.DataFrame(data_indep, columns=["X", "Y", "Z1", "Z2"])
        df_miss = df.copy()
        df_miss.loc[df_miss["Z1"] > 0.5, "Z1"] = np.nan
        try:
            w = CIIRATestWrapper()
            result = w.run_ciira_test(
                df_miss, y_col="Y", expl_vars=["X", "Z1", "Z2"],
                classifier_args={"n_estimators": 20},
            )
            result.summary()
        except Exception as e:
            print(f"  citest comparison failed: {e}")
    else:
        print("Test 5: citest not installed -- skipping consistency check")

    print("\n=== Validation Complete ===")
