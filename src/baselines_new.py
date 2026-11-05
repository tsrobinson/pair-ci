# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file contains a Python port of micd::gaussMItest (Foraita & Witte,
# R package micd, GPL (>= 3); method of Witte et al. 2022) and is therefore
# distributed under the GNU General Public License, version 3 or later
# (see LICENSES/GPL-3.0.txt). The rest of this package is MIT-licensed
# (see LICENSE and NOTICE).
"""PC with MI-pooled Fisher-Z (FZ-Rubin), and the MissDAG entry point.

run_pc_micd_pooled runs the PC search of pc_integration with a port of
micd::gaussMItest (Witte et al., 2022): the Fisher-Z statistic of the
partial correlation in each of M MICE imputations, pooled by Rubin's
rules. It is valid when the variables are jointly Gaussian and the
imputation model is correct (MAR). run_missdag_gaussian is re-exported
from missdag_gaussian, which is not redistributed (see NOTICE); without
it, a stub raises ImportError. Both return a PCResult.
"""
from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np

from pc_integration import PCResult, _pc_skeleton, _orient_edges


# =============================================================================
# micd: multiply-imputed pooled Fisher-Z CI test
# =============================================================================


def _partial_corr_from_cov(cov: np.ndarray, i: int, j: int, S: tuple) -> float:
    """Partial correlation r_{ij.S} from a covariance matrix.

    As pcalg::pcorOrder: r = -P_01 / sqrt(P_00 P_11), where P is the
    (pseudo-)inverse of the covariance submatrix on (i, j, S).
    """
    idx = [i, j] + list(S)
    sub = cov[np.ix_(idx, idx)]
    try:
        # Pseudo-inverse, as in pcalg::pcorOrder, tolerates singular blocks.
        prec = np.linalg.pinv(sub)
        denom = np.sqrt(prec[0, 0] * prec[1, 1])
        if denom <= 0:
            return 0.0
        r = -prec[0, 1] / denom
    except np.linalg.LinAlgError:
        return 0.0
    # Keeps the Fisher transform finite.
    return float(np.clip(r, -0.999999, 0.999999))


def _pool_fisher_z(z_values: np.ndarray, n: int, k: int) -> float:
    """Pool M Fisher-Z statistics by Rubin's rules; return a two-sided p.

    As micd::gaussMItest, with k = |S| and n the sample size:
        W  = 1 / (n - k - 3),  B = var(z, ddof=1),  T = W + (1 + 1/M) B
        df = (M - 1) (1 + (W / B) M / (M + 1))^2
        p  = 2 P(t_df > |mean(z)| / sqrt(T))
    This is Rubin's (1987) df, without the Barnard-Rubin small-sample
    correction used by PAIR-CI, to match micd.
    """
    from scipy import stats as sp_stats

    M = len(z_values)
    avg = float(np.mean(z_values))
    W = 1.0 / max(n - k - 3, 1)
    if M > 1:
        B = float(np.sum((z_values - avg) ** 2) / (M - 1))
    else:
        B = 0.0
    TV = W + (1.0 + 1.0 / M) * B
    if TV <= 0:
        return 1.0
    ts = avg / np.sqrt(TV)
    if B > 0 and M > 1:
        df = (M - 1) * (1 + (W / B) * (M / (M + 1))) ** 2
    else:
        # No between-imputation variance: normal reference, as in micd.
        df = 1e9
    pval = 2.0 * sp_stats.t.sf(abs(ts), df=df)
    return float(pval)


class _MicdCITest:
    """MI-pooled Fisher-Z test; caches the M imputed covariance matrices."""

    def __init__(self, data: np.ndarray, n_imputations: int = 5,
                 random_state: Optional[int] = None):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self.n = data.shape[0]
        rng = np.random.RandomState(random_state)
        self._covs = []
        for _ in range(n_imputations):
            imp = IterativeImputer(
                max_iter=10,
                random_state=rng.randint(0, 2**31),
                sample_posterior=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_imp = imp.fit_transform(data)
            # Divisor n - 1, as R's cov().
            self._covs.append(np.cov(data_imp.T, bias=False))
        self.n_tests = 0

    def test(self, i: int, j: int, cond_set):
        self.n_tests += 1
        S = tuple(cond_set)
        z_values = np.empty(len(self._covs))
        for m, cov in enumerate(self._covs):
            r = _partial_corr_from_cov(cov, i, j, S)
            one_minus_r = max(1.0 - r, 1e-15)
            one_plus_r = max(1.0 + r, 1e-15)
            z_values[m] = 0.5 * (np.log(one_plus_r) - np.log(one_minus_r))
        pval = _pool_fisher_z(z_values, self.n, len(S))
        # The mean z is returned for logging only.
        stat = float(np.mean(z_values))
        return stat, pval


def run_pc_micd_pooled(data, alpha=0.05, n_imputations=5, max_depth=None,
                       random_state=None, verbose=False):
    """PC with a micd-style MI-pooled Fisher-Z CI test (Witte et al. 2022)."""
    t_start = time.time()
    p = data.shape[1]
    ci = _MicdCITest(data, n_imputations=n_imputations, random_state=random_state)
    adj, sep_sets, _ = _pc_skeleton(p, ci, alpha, max_depth, verbose)
    adj = _orient_edges(adj, sep_sets, p, verbose)
    return PCResult(
        adjacency_matrix=adj,
        sep_sets=sep_sets,
        n_ci_tests=ci.n_tests,
        runtime_seconds=time.time() - t_start,
        method="micd_pooled",
    )


try:  # MissDAG-derived module; absent from the public release (see NOTICE).
    from missdag_gaussian import run_missdag_gaussian  # noqa: F401
except ImportError:  # pragma: no cover
    def run_missdag_gaussian(*args, **kwargs):
        raise ImportError(
            "The MissDAG baseline (src/missdag_gaussian.py) is derived from "
            "https://github.com/ErdunGAO/MissDAG, which carries no licence, "
            "and is not redistributed. Obtain MissDAG from its authors; see "
            "README.md (Licences) for how the paper's MissDAG runs were made.")
