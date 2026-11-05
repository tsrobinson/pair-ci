"""Linear-learner variants of GeneralCITest and FastCITest.

The forests are replaced by Ridge(alpha=1e-6), i.e. OLS guarded against
near-singular designs, and by L2 logistic regression for discrete
responses. Used on the linear DGPs, so that PAIR-CI uses the same model
class as the linear baselines (Fisher-Z, MissDAG's linear NOTEARS) and the
comparison isolates the test rather than the learner. Imputation, placebos,
variance and pooling are inherited unchanged.
"""
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

from ci_oracle import GeneralCITest
from ci_oracle_fast import FastCITest


def _linear_fold_cls(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
    y_tr, y_te = y_tr.astype(int), y_te.astype(int)
    if len(np.unique(y_tr)) < 2:
        # Single-class fold: NaN variance, skipped by _compute_statistic.
        return 0.0, np.nan
    cf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=200,
        random_state=self.rng.randint(0, 2**31),
    )
    cp = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=200,
        random_state=self.rng.randint(0, 2**31),
    )
    cf.fit(Xf_tr, y_tr)
    cp.fit(Xp_tr, y_tr)
    eps = 1e-15
    pf = np.clip(cf.predict_proba(Xf_te), eps, 1 - eps)
    pp = np.clip(cp.predict_proba(Xp_te), eps, 1 - eps)
    individual = (self._bce_individual(y_te, pp, cp.classes_)
                  - self._bce_individual(y_te, pf, cf.classes_))
    return individual.mean(), np.var(individual, ddof=1)


def _linear_fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
    # Near-OLS fit, matching the residualisation behind Fisher-Z's partial
    # correlation; a closed-form solve, unlike BayesianRidge's iterative
    # tuning.
    rf = Ridge(alpha=1e-6)
    rp = Ridge(alpha=1e-6)
    rf.fit(Xf_tr, y_tr)
    rp.fit(Xp_tr, y_tr)
    individual = ((y_te - rp.predict(Xp_te)) ** 2
                  - (y_te - rf.predict(Xf_te)) ** 2)
    return individual.mean(), np.var(individual, ddof=1)


class LinearLearnerCITest(GeneralCITest):
    _fold_cls = _linear_fold_cls
    _fold_reg = _linear_fold_reg


class LinearLearnerFastCITest(FastCITest):
    _fold_cls = _linear_fold_cls
    _fold_reg = _linear_fold_reg
