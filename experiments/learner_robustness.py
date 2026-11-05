"""PAIR-CI calibration under weak and misspecified learners.

Under the internal null the full and placebo models are exchangeable
whatever the learner (Proposition 2), so a weak or misspecified learner
should cost power rather than size. Learners (regression only, so
continuous Z): rf100, the default 100-tree forest; rf25_shallow, 25 trees
of depth 3; rf10_stump, 10 stumps; ridge (alpha = 1), misspecified on the
post-nonlinear DGP; knn, fixed k = 10, outside the stability assumption
(A4). The rest is the general harness with M = 5, K = 5 and cached
imputation. DGPs are linear Gaussian and post-nonlinear, |X| = 5, n in
{1000, 2000}, signal in {0, 0.3, 0.6}, under complete data, MAR and MNAR
(30% per column), 200 replicates per cell: 36,000 tests. The estimands
are the rejection rates at alpha = 0.05.

    python experiments/learner_robustness.py \
        --output results/learner_robustness.csv

Each learner runs on its own datasets (the learner is part of the cell),
so comparisons across learners are unpaired. Results are saved after
each cell. Feeds App. S.2 (Table learner_robustness).
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

from dgps import DGP_FUNCTIONS, inject_missingness_calibration
from ci_oracle import GeneralCITest


def _factories(rng):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    def rs():
        return rng.randint(0, 2**31)
    return {
        "rf100": lambda: RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, random_state=rs(), n_jobs=-1),
        "rf25_shallow": lambda: RandomForestRegressor(
            n_estimators=25, max_depth=3, min_samples_leaf=5,
            random_state=rs(), n_jobs=-1),
        "rf10_stump": lambda: RandomForestRegressor(
            n_estimators=10, max_depth=1, random_state=rs(), n_jobs=-1),
        "ridge": lambda: Ridge(alpha=1.0),
        "knn": lambda: KNeighborsRegressor(n_neighbors=10),
    }


class LearnerCITest(GeneralCITest):
    """GeneralCITest with a pluggable regression learner.

    Only the regression hook is replaced; a discrete Z would still use the
    default forest classifier.
    """

    def __init__(self, data, learner_factory, **kwargs):
        self._factory = learner_factory
        super().__init__(data, **kwargs)

    def _fold_reg(self, Xf_tr, Xf_te, Xp_tr, Xp_te, y_tr, y_te):
        rf = self._factory()
        rp = self._factory()
        rf.fit(Xf_tr, y_tr)
        rp.fit(Xp_tr, y_tr)
        individual = (y_te - rp.predict(Xp_te)) ** 2 \
                     - (y_te - rf.predict(Xf_te)) ** 2
        return individual.mean(), np.var(individual, ddof=1)


def main(args):
    rng = np.random.RandomState(args.seed)
    rows = []
    cells = [(dgp, mech, n, sig, ln)
             for dgp in args.dgps
             for mech in ["complete", "mar", "mnar"]
             for n in args.ns
             for sig in args.signals
             for ln in args.learners]
    t0 = time.time()
    for i, (dgp, mech, n, sig, lname) in enumerate(cells, 1):
        tc = time.time()
        for rep in range(args.reps):
            X, Y, Z = DGP_FUNCTIONS[dgp](n, args.dim_x, sig, rng)
            X_miss = inject_missingness_calibration(X, mech, args.rate, rng)
            data = np.column_stack([Z, Y, X_miss])
            try:
                fac = _factories(rng)[lname]
                ci = LearnerCITest(data, fac, n_imputations=5, n_folds=5,
                                   imputation_strategy="cached",
                                   random_state=rng.randint(0, 2**31))
                stat, pval = ci.test(x_idx=0, y_idx=1,
                                     cond_set=tuple(range(2, 2 + args.dim_x)))
                rows.append(dict(learner=lname, dgp=dgp, mechanism=mech, n=n,
                                 signal=sig, dim_x=args.dim_x, rep=rep,
                                 statistic=stat, p_value=pval,
                                 rejected=pval < 0.05, failed=False))
            except Exception:
                rows.append(dict(learner=lname, dgp=dgp, mechanism=mech, n=n,
                                 signal=sig, dim_x=args.dim_x, rep=rep,
                                 statistic=np.nan, p_value=np.nan,
                                 rejected=False, failed=True))
        print(f"[{i}/{len(cells)}] {lname}/{dgp}/{mech}/n={n}/s={sig}: "
              f"{(time.time()-tc)/60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Total: {(time.time()-t0)/3600:.1f} h")
    return pd.DataFrame(rows)


def analyze(df):
    ok = df[~df.failed]
    print("\n=== FPR at signal=0 by learner x dgp x mechanism (nominal 0.05) ===")
    print(ok[ok.signal == 0].groupby(["learner", "dgp", "mechanism"]).rejected
          .mean().unstack("mechanism").round(3).to_string())
    print("\n=== Power by learner x signal (pooled) ===")
    print(ok[ok.signal > 0].groupby(["learner", "signal"]).rejected.mean()
          .unstack("signal").round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learner-robustness calibration")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--dim_x", type=int, default=5)
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--dgps", nargs="+",
                        default=["linear_gaussian", "post_nonlinear"])
    parser.add_argument("--ns", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--signals", type=float, nargs="+",
                        default=[0.0, 0.3, 0.6])
    parser.add_argument("--learners", nargs="+",
                        default=["rf100", "rf25_shallow", "rf10_stump",
                                 "ridge", "knn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str,
                        default="results/learner_robustness.csv")
    args = parser.parse_args()
    if args.quick:
        args.reps = 10
        args.ns = [1000]
        args.signals = [0.0, 0.3]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = main(args)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows)")
    analyze(df)
