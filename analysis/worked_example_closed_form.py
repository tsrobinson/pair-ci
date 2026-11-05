"""Worked example for Proposition 1 (Appendix U): closed-form induced
partial correlation rho(Z, Y | Xtilde) under probit self-masking + linear
imputation, checked against Monte Carlo.

SCM: X ~ N(0,1); Y = b X + e_Y; Z = c X + e_Z (e's independent Gaussians).
Under H0 (signal 0), Z is independent of Y given X. Missingness: X is
missing with probability Phi(s (X - t)) -- probit self-masking in the upper
tail; t set to the (1 - rho_miss) quantile. Imputation: least-squares linear
regression of X on (1, Y, Z) fitted on observed cases (the Bayesian-ridge
conditional mean; posterior-draw noise can be added via --draw_noise).

The completed column Xtilde equals X on observed rows and the imputed linear
index on missing rows. The induced partial correlation of (Y, Z) given
Xtilde is assembled exactly:

  (i)  selection moments of X on each subpopulation (1-D Gaussian integrals
       with inverse-Mills closed forms, evaluated here by quadrature);
  (ii) all (X, Y, Z) subpopulation moments follow analytically because
       selection acts on X only, so (Y,Z) | X is untouched;
  (iii) imputer coefficients from observed-subpopulation moments;
  (iv) pooled second moments of (Xtilde, Y, Z) as a two-component mixture,
       then the standard partial-correlation formula.

No simulation enters (i)-(iv). The MC check simulates the full pipeline.

Usage: python analysis/worked_example_closed_form.py
"""

import argparse
import numpy as np
from scipy import stats
from scipy.integrate import quad


def selection_moments(s, t, missing):
    """E[1], E[X], E[X^2] on the observed (R=1) or missing (R=0) subpop.
    Weight: P(R=0|x) = Phi(s(x-t)); observed weight = 1 - that."""
    def w(x):
        p_miss = stats.norm.cdf(s * (x - t))
        return p_miss if missing else 1.0 - p_miss

    def integrand(x, k):
        return (x ** k) * w(x) * stats.norm.pdf(x)

    m0 = quad(integrand, -12, 12, args=(0,), limit=200)[0]
    m1 = quad(integrand, -12, 12, args=(1,), limit=200)[0]
    m2 = quad(integrand, -12, 12, args=(2,), limit=200)[0]
    return m0, m1 / m0, m2 / m0  # P(subpop), E[X|subpop], E[X^2|subpop]


def subpop_xyz_moments(EX, EX2, b, c, sy2, sz2):
    """First/second moments of (X, Y, Z) on a subpopulation defined by
    selection on X alone (so E[Y|X]=bX etc. is unchanged)."""
    m = dict(X=EX, Y=b * EX, Z=c * EX)
    mm = dict(XX=EX2, XY=b * EX2, XZ=c * EX2,
              YY=b * b * EX2 + sy2, ZZ=c * c * EX2 + sz2,
              YZ=b * c * EX2)
    return m, mm


def closed_form(b=0.8, c=0.8, sy=0.6, sz=0.6, s=5.0, rho_miss=0.3,
                draw_noise=False):
    """Return (rho(Z, Y | Xtilde), imputer details) from steps (i)-(iv)."""
    sy2, sz2 = sy * sy, sz * sz
    t = stats.norm.ppf(1.0 - rho_miss)

    P_o, EX_o, EX2_o = selection_moments(s, t, missing=False)
    P_m, EX_m, EX2_m = selection_moments(s, t, missing=True)
    m_o, mm_o = subpop_xyz_moments(EX_o, EX2_o, b, c, sy2, sz2)
    m_m, mm_m = subpop_xyz_moments(EX_m, EX2_m, b, c, sy2, sz2)

    # (iii) imputer on observed subpop: X ~ (1, Y, Z), least squares.
    cov_o = np.array([
        [mm_o['YY'] - m_o['Y'] ** 2, mm_o['YZ'] - m_o['Y'] * m_o['Z']],
        [mm_o['YZ'] - m_o['Y'] * m_o['Z'], mm_o['ZZ'] - m_o['Z'] ** 2],
    ])
    cxv = np.array([mm_o['XY'] - m_o['X'] * m_o['Y'],
                    mm_o['XZ'] - m_o['X'] * m_o['Z']])
    beta = np.linalg.solve(cov_o, cxv)          # (beta_Y, beta_Z)
    alpha = m_o['X'] - beta @ np.array([m_o['Y'], m_o['Z']])
    # residual variance of the fit on the observed subpop (for draw noise)
    var_x_o = mm_o['XX'] - m_o['X'] ** 2
    v_res = var_x_o - beta @ cxv

    # (iv) pooled moments of (Xt, Y, Z): Xt = X on observed, alpha +
    # beta.(Y,Z) [+ noise] on missing.
    bY, bZ = beta
    m_m_t = alpha + bY * m_m['Y'] + bZ * m_m['Z']
    mm_m_tt = (alpha ** 2 + bY ** 2 * mm_m['YY'] + bZ ** 2 * mm_m['ZZ']
               + 2 * alpha * bY * m_m['Y'] + 2 * alpha * bZ * m_m['Z']
               + 2 * bY * bZ * mm_m['YZ'])
    if draw_noise:
        mm_m_tt += v_res
    mm_m_tY = alpha * m_m['Y'] + bY * mm_m['YY'] + bZ * mm_m['YZ']
    mm_m_tZ = alpha * m_m['Z'] + bY * mm_m['YZ'] + bZ * mm_m['ZZ']

    def pool(a_o, a_m):
        return P_o * a_o + P_m * a_m

    E = dict(
        T=pool(m_o['X'], m_m_t), Y=pool(m_o['Y'], m_m['Y']),
        Z=pool(m_o['Z'], m_m['Z']),
        TT=pool(mm_o['XX'], mm_m_tt), YY=pool(mm_o['YY'], mm_m['YY']),
        ZZ=pool(mm_o['ZZ'], mm_m['ZZ']),
        TY=pool(mm_o['XY'], mm_m_tY), TZ=pool(mm_o['XZ'], mm_m_tZ),
        YZ=pool(mm_o['YZ'], mm_m['YZ']),
    )
    vT = E['TT'] - E['T'] ** 2
    vY = E['YY'] - E['Y'] ** 2
    vZ = E['ZZ'] - E['Z'] ** 2
    r_TY = (E['TY'] - E['T'] * E['Y']) / np.sqrt(vT * vY)
    r_TZ = (E['TZ'] - E['T'] * E['Z']) / np.sqrt(vT * vZ)
    r_YZ = (E['YZ'] - E['Y'] * E['Z']) / np.sqrt(vY * vZ)
    rho = (r_YZ - r_TY * r_TZ) / np.sqrt((1 - r_TY ** 2) * (1 - r_TZ ** 2))
    return rho, dict(P_miss=P_m, beta_Y=bY, beta_Z=bZ, alpha=alpha,
                     v_res=v_res)


def monte_carlo(b=0.8, c=0.8, sy=0.6, sz=0.6, s=5.0, rho_miss=0.3,
                n=200000, reps=20, draw_noise=False, seed=0):
    """Mean over reps of the sample partial correlation, and its Monte Carlo
    standard error."""
    rng = np.random.RandomState(seed)
    t = stats.norm.ppf(1.0 - rho_miss)
    out = []
    for _ in range(reps):
        X = rng.randn(n)
        Y = b * X + rng.randn(n) * sy
        Z = c * X + rng.randn(n) * sz
        miss = rng.rand(n) < stats.norm.cdf(s * (X - t))
        obs = ~miss
        A = np.column_stack([np.ones(obs.sum()), Y[obs], Z[obs]])
        coef, res, _, _ = np.linalg.lstsq(A, X[obs], rcond=None)
        Xt = X.copy()
        pred = coef[0] + coef[1] * Y[miss] + coef[2] * Z[miss]
        if draw_noise:
            v = res[0] / obs.sum() if len(res) else 0.0
            pred = pred + rng.randn(miss.sum()) * np.sqrt(v)
        Xt[miss] = pred
        B = np.column_stack([np.ones(n), Xt])
        ry = Y - B @ np.linalg.lstsq(B, Y, rcond=None)[0]
        rz = Z - B @ np.linalg.lstsq(B, Z, rcond=None)[0]
        out.append(np.corrcoef(ry, rz)[0, 1])
    return float(np.mean(out)), float(np.std(out) / np.sqrt(reps))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steepness", type=float, default=5.0)
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--draw_noise", action="store_true")
    a = ap.parse_args()

    rho_cf, info = closed_form(s=a.steepness, rho_miss=a.rate,
                               draw_noise=a.draw_noise)
    rho_mc, se = monte_carlo(s=a.steepness, rho_miss=a.rate,
                             draw_noise=a.draw_noise)
    print(f"probit self-masking, steepness={a.steepness}, target rate={a.rate}"
          f", draw_noise={a.draw_noise}")
    print(f"  realised P(miss)      : {info['P_miss']:.4f}")
    print(f"  imputer (a, bY, bZ)   : {info['alpha']:.4f}, "
          f"{info['beta_Y']:.4f}, {info['beta_Z']:.4f}  (v_res {info['v_res']:.4f})")
    print(f"  closed-form rho(Z,Y|Xt): {rho_cf:.5f}")
    print(f"  Monte Carlo            : {rho_mc:.5f} +/- {se:.5f}")
    ok = abs(rho_cf - rho_mc) < 4 * se + 1e-3
    print(f"  AGREEMENT: {'PASS' if ok else 'FAIL'}")
    # The rate sweep gives the 10/30/50% values quoted in Appendix U.
    print("\n  steepness sweep (rate 0.3, no draw noise):")
    for s_ in [1, 3, 5, 10]:
        r_, _ = closed_form(s=s_, rho_miss=0.3)
        print(f"    s={s_:>4}: rho = {r_:+.5f}")
    print("  rate sweep (steepness 5):")
    for rm in [0.1, 0.3, 0.5]:
        r_, _ = closed_form(s=5.0, rho_miss=rm)
        print(f"    rate={rm}: rho = {r_:+.5f}")
