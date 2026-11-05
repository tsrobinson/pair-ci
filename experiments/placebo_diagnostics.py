"""Quality of the local-permutation placebo at |X| = 10.

Supplementary diagnostic; not reported in the paper. In
results/ci_calibration_results.csv, PAIR-CI's complete-data false positive
rate at |X| = 10 on the linear-Gaussian DGP rises with n (0.07, 0.09, 0.16
at n = 1000, 2000, 5000) with a positive mean null t-statistic (0.47 to
0.80), and is 0 on the latent-confounder DGP, where Y and Z depend on
disjoint parts of X. The hypothesis tested is that the kNN local
permutation (src/conditional_permutation.py) degrades with dimension, so
the placebo U carries less X-information than Y and the full model wins
for reasons unrelated to Y.

Arms, all on complete data at signal 0:
  deployed      the kNN placebo with the default k = max(2, n^(2/(d+2)))
  knn_k8/32/128 the kNN placebo with k fixed at 8, 32 or 128
  oracle        U drawn from the true P(Y | X) using the replicate's own
                coefficients; at signal 0, Y is itself such a draw, so this
                arm is exactly symmetric
  marginal      unconditional permutation
PAIR-CI is run as in calibration.py (M = 5, K = 5) at n in {1000, 2000,
5000}, 150 replicates. Each row logs mu_hat, W_bar, B and T, so that mean
and variance effects can be separated.

    python experiments/placebo_diagnostics.py --dgp linear_gaussian \
        --output results/placebo_diag_linear_gaussian.csv
    python experiments/placebo_diagnostics.py --smoke   # about 1 minute
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import ci_oracle
from conditional_permutation import conditional_permute_batched as _knn_placebo


# ---------------------------------------------------------------------------
# DGPs. These reproduce src/dgps.py at signal = 0 (same draw order) and
# also return the replicate's E[Y | X] and noise s.d., which the oracle
# placebo needs. Keep in sync with dgps.py.
# ---------------------------------------------------------------------------

def gen_linear_gaussian(n, dim_x, rng):
    X = rng.randn(n, dim_x)
    beta_z = rng.randn(dim_x) * 0.5
    Z = X @ beta_z + rng.randn(n) * 0.5
    beta_y = rng.randn(dim_x) * 0.5
    Y = X @ beta_y + rng.randn(n) * 0.5
    return X, Y, Z, (lambda Xc: Xc @ beta_y), 0.5


def gen_post_nonlinear(n, dim_x, rng):
    X = rng.randn(n, dim_x)
    beta = rng.randn(dim_x) * 0.5
    Z = np.tanh(X @ beta + rng.randn(n) * 0.3)
    gamma = rng.randn(dim_x) * 0.3
    Y = X @ gamma + rng.randn(n) * 0.5
    return X, Y, Z, (lambda Xc: Xc @ gamma), 0.5


def gen_latent_confounder(n, dim_x, rng):
    assert dim_x > 2, "oracle mean uses X[:, 2]; run at dim_x > 2"
    X = rng.randn(n, dim_x)
    L = rng.randn(n)
    Z = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * L + rng.randn(n) * 0.3
    Y = np.cos(X[:, 2]) + rng.randn(n) * 0.5
    return X, Y, Z, (lambda Xc: np.cos(Xc[:, 2])), 0.5


GENERATORS = {
    "linear_gaussian": gen_linear_gaussian,
    "post_nonlinear": gen_post_nonlinear,
    "latent_confounder": gen_latent_confounder,
}

ARMS = ["deployed", "knn_k8", "knn_k32", "knn_k128", "oracle", "marginal"]


def make_placebo_fn(arm, mu_fn, noise_sd):
    """Return a drop-in replacement for conditional_permute_batched."""
    if arm.startswith("knn_k"):
        k_forced = int(arm[len("knn_k"):])

        def forced_k(Y, X, k=None, rng=None):
            return _knn_placebo(Y, X, k=k_forced, rng=rng)
        return forced_k
    if arm == "oracle":
        def oracle(Y, X, k=None, rng=None):
            if rng is None:
                rng = np.random.RandomState()
            return mu_fn(X) + noise_sd * rng.randn(len(Y))
        return oracle
    raise ValueError(f"no patched placebo for arm {arm}")


def run_one(arm, dgp_name, n, dim_x, rep, data_seed, test_seed):
    """One PAIR-CI test under one placebo arm. The dataset and the test
    seed do not depend on the arm, so arms are paired within a replicate."""
    data_rng = np.random.RandomState(data_seed)
    X, Y, Z, mu_fn, sd = GENERATORS[dgp_name](n, dim_x, data_rng)
    data = np.column_stack([Z, Y, X])
    cond_set = tuple(range(2, 2 + dim_x))

    original = ci_oracle.conditional_permute_batched
    try:
        if arm not in ("deployed", "marginal"):
            ci_oracle.conditional_permute_batched = make_placebo_fn(
                arm, mu_fn, sd)
        t0 = time.time()
        # As calibration.py::our_test: M = 5, K = 5, cached imputation
        # (a no-op on complete data), other settings at their defaults.
        ci = ci_oracle.GeneralCITest(
            data, n_imputations=5, n_folds=5,
            imputation_strategy="cached",
            use_conditional_permutation=(arm != "marginal"),
            random_state=test_seed,
        )
        stat, pval = ci.test(x_idx=0, y_idx=1, cond_set=cond_set)
        runtime = time.time() - t0
        comp = getattr(ci, "last_components_", {})
        return dict(
            arm=arm, dgp=dgp_name, n=n, dim_x=dim_x, rep=rep,
            statistic=stat, p_value=pval, rejected=pval < 0.05,
            mu_hat=comp.get("mu_hat", np.nan),
            W_bar=comp.get("W_bar", np.nan),
            B=comp.get("B", np.nan),
            T=comp.get("T", np.nan),
            df=comp.get("df", np.nan),
            degenerate=comp.get("degenerate", False),
            runtime=runtime, failed=False,
        )
    except Exception as e:
        return dict(
            arm=arm, dgp=dgp_name, n=n, dim_x=dim_x, rep=rep,
            statistic=np.nan, p_value=np.nan, rejected=False,
            mu_hat=np.nan, W_bar=np.nan, B=np.nan, T=np.nan, df=np.nan,
            degenerate=False, runtime=np.nan, failed=True,
        )
    finally:
        ci_oracle.conditional_permute_batched = original


def main():
    ap = argparse.ArgumentParser(description="Placebo-quality diagnostics")
    ap.add_argument("--dgp", type=str, default=None,
                    choices=list(GENERATORS), help="run a single DGP "
                    "(default: all three)")
    ap.add_argument("--ns", nargs="+", type=int, default=[1000, 2000, 5000])
    ap.add_argument("--dim_x", type=int, default=10)
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output", type=str,
                    default="results/placebo_diagnostics.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dgps = [args.dgp] if args.dgp else list(GENERATORS)
    ns, reps = args.ns, args.reps
    if args.smoke:
        dgps, ns, reps = ["linear_gaussian"], [400], 2

    total = len(dgps) * len(ns) * reps * len(args.arms)
    print(f"placebo_diagnostics: dgps={dgps} ns={ns} dim_x={args.dim_x} "
          f"reps={reps} arms={args.arms} -> {total} tests")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows, done, t_start = [], 0, time.time()
    for dgp_name in dgps:
        for n in ns:
            for rep in range(reps):
                # Seeds fixed per replicate and shared by all arms. hash() of
                # a str is salted per process unless PYTHONHASHSEED is set,
                # so without it the datasets differ from run to run.
                data_seed = (args.seed * 1_000_003 + hash((dgp_name, n))
                             % 999_983 + rep) % (2 ** 31)
                test_seed = (data_seed * 7 + 13) % (2 ** 31)
                for arm in args.arms:
                    rows.append(run_one(arm, dgp_name, n, args.dim_x,
                                        rep, data_seed, test_seed))
                    done += 1
                if rep % 10 == 0 or (args.smoke and rep == reps - 1):
                    el = (time.time() - t_start) / 60
                    print(f"  {dgp_name} n={n} rep={rep + 1}/{reps} "
                          f"[{done}/{total} tests, {el:.1f} min]",
                          flush=True)
                    pd.DataFrame(rows).to_csv(args.output, index=False)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows -> {args.output} "
          f"({(time.time() - t_start) / 60:.1f} min)")

    df = pd.DataFrame(rows)
    ok = df[~df.failed]
    print("\nFPR / mean(t) by arm x n:")
    print(ok.groupby(["dgp", "arm", "n"])
            .agg(fpr=("rejected", "mean"), mean_t=("statistic", "mean"),
                 mean_mu=("mu_hat", "mean"))
            .round(3).to_string())


if __name__ == "__main__":
    main()
