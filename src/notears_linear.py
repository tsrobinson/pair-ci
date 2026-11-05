# SPDX-License-Identifier: Apache-2.0
#
# Linear NOTEARS solver, modified from https://github.com/xunzheng/notears
# (Zheng, Aragam, Ravikumar & Xing 2018), Copyright the NOTEARS authors,
# licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt).
# Changes from the original: rewritten in covariance form (the loss takes an
# empirical covariance rather than a data matrix), equal-variance squared
# loss only, condensed into a single function, and called from the MissDAG
# M-step of this package.
"""Linear NOTEARS (Zheng et al. 2018), covariance form."""
from __future__ import annotations

import numpy as np


def _notears_linear(cov_emp: np.ndarray, lambda1: float = 0.1,
                    max_iter: int = 100, h_tol: float = 1e-8,
                    rho_max: float = 1e16) -> np.ndarray:
    """Linear NOTEARS in covariance form, by augmented Lagrangian.

    Solves min_W 0.5 tr((I-W)^T S (I-W)) + lambda1 ||W||_1 s.t. h(W) = 0,
    with S = cov_emp and h(W) = tr(exp(W o W)) - d. Returns the d x d
    weighted adjacency (W[i, j] != 0 for i -> j), unthresholded.
    """
    import scipy.linalg as slin
    import scipy.optimize as sopt

    d = cov_emp.shape[0]

    def _loss(W):
        I = np.eye(d)
        loss = 0.5 * np.trace((I - W).T @ cov_emp @ (I - W))
        G = -cov_emp @ (I - W)
        return loss, G

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G = E.T * W * 2
        return h, G

    # W = W+ - W-, both non-negative, so the L1 penalty is linear and
    # L-BFGS-B can handle it through bounds.
    def _adj(w):
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G = G_loss + (rho * h + alpha) * G_h
        g = np.concatenate((G + lambda1, -G + lambda1), axis=None)
        return obj, g

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    return _adj(w_est)

