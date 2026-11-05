"""PAIR-CI calibration with missingness on the test variables Z and Y.

The standalone experiments (Section 5.1) put missingness only in the
conditioning set X, but inside PC an incomplete variable is often Z or Y.
Under cached imputation every column is imputed, and two rules are compared:
  imputed_labels   the deployed rule: the loss is evaluated on imputed Z,
                   with imputed Y as a predictor. A possible anti-conservative
                   channel: imputed Z depends partly on Y (an imputer
                   predictor), and only the full model sees Y.
  labels_observed  evaluate only rows where both Z and Y are observed,
                   keeping the imputed conditioning set.
Missingness at 30% (steepness-3 logistic, as in src/dgps.py) on Z, Y, both,
or Z plus two X columns ("xz", the usual situation inside PC), either MAR
(driven by the last X column, always observed) or MNAR self-masking. DGPs
are linear Gaussian and post-nonlinear, n = 1000, |X| = 5, signal 0 and
0.3, 250 replicates per cell. The estimands are the rejection rates at
alpha = 0.05. The rule is part of the cell, so the two rules see different
datasets.

    python experiments/test_variable_missingness.py \
        --output results/test_variable_missingness.csv

Feeds Section 3.2 and App. T.3 (Table testvar_missingness).
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

from dgps import DGP_FUNCTIONS
from ci_oracle import GeneralCITest


class LabelsObservedCITest(GeneralCITest):
    """labels_observed rule: score the paired loss comparison only on rows
    where both test variables are observed in the original data. The
    conditioning set still comes from the cached imputation; IterativeImputer
    leaves observed entries unchanged, so the retained rows carry the true
    Z and Y."""

    def test(self, x_idx, y_idx, cond_set=()):
        self._row_mask = (~np.isnan(self.data[:, x_idx])
                          & ~np.isnan(self.data[:, y_idx]))
        try:
            return super().test(x_idx, y_idx, cond_set)
        finally:
            self._row_mask = None

    def _compare_models(self, imputed_datasets, x_idx, y_idx, cond_set,
                        target_is_discrete):
        filtered = [d[self._row_mask] for d in imputed_datasets]
        return super()._compare_models(filtered, x_idx, y_idx, cond_set,
                                       target_is_discrete)

    def _compute_statistic(self, diffs, fold_vars=None):
        # Variance scaling must use the retained row count, not self.n.
        n_orig = self.n
        if self._row_mask is not None:
            self.n = int(self._row_mask.sum())
        try:
            return super()._compute_statistic(diffs, fold_vars)
        finally:
            self.n = n_orig


def _inject(col_values, style, rate, rng, driver=None):
    """Steepness-3 logistic missingness on one column, as in dgps.py.
    style 'mar' is driven by an observed driver column; 'mnar' self-masks."""
    v = driver if style == "mar" else col_values
    logit = v - np.quantile(v, 1 - rate)
    prob = 1 / (1 + np.exp(-3 * logit))
    out = col_values.copy()
    out[rng.rand(len(out)) < prob] = np.nan
    return out


def make_dataset(dgp_name, n, dim_x, signal, style, miss_on, rate, rng):
    X, Y, Z = DGP_FUNCTIONS[dgp_name](n, dim_x, signal, rng)
    driver = X[:, -1]  # last X column stays observed in every placement
    Zm, Ym, Xm = Z.copy(), Y.copy(), X.copy()
    if miss_on in ("z", "both", "xz"):
        Zm = _inject(Z, style, rate, rng, driver)
    if miss_on in ("y", "both"):
        Ym = _inject(Y, style, rate, rng, driver)
    if miss_on == "xz":
        for c in range(min(2, dim_x)):
            Xm[:, c] = _inject(X[:, c], style, rate, rng, driver)
    return np.column_stack([Zm, Ym, Xm])


def run_grid(args):
    rng = np.random.RandomState(args.seed)
    rows = []
    cells = [(dgp, style, miss_on, rule, signal)
             for dgp in args.dgps
             for style in ["mar", "mnar"]
             for miss_on in args.placements
             for rule in ["imputed_labels", "labels_observed"]
             for signal in args.signals]
    for i, (dgp, style, miss_on, rule, signal) in enumerate(cells, 1):
        tc = time.time()
        for rep in range(args.reps):
            data = make_dataset(dgp, args.n_obs, args.dim_x, signal, style,
                                miss_on, args.rate, rng)
            cls = (LabelsObservedCITest if rule == "labels_observed"
                   else GeneralCITest)
            try:
                ci = cls(data, n_imputations=5, n_folds=5,
                         imputation_strategy="cached",
                         random_state=rng.randint(0, 2**31))
                stat, pval = ci.test(
                    x_idx=0, y_idx=1,
                    cond_set=tuple(range(2, 2 + args.dim_x)))
                rows.append(dict(dgp=dgp, style=style, miss_on=miss_on,
                                 rule=rule, signal=signal, n=args.n_obs,
                                 dim_x=args.dim_x, rate=args.rate, rep=rep,
                                 statistic=stat, p_value=pval,
                                 rejected=pval < 0.05, failed=False))
            except Exception:
                rows.append(dict(dgp=dgp, style=style, miss_on=miss_on,
                                 rule=rule, signal=signal, n=args.n_obs,
                                 dim_x=args.dim_x, rate=args.rate, rep=rep,
                                 statistic=np.nan, p_value=np.nan,
                                 rejected=False, failed=True))
        print(f"[{i}/{len(cells)}] {dgp}/{style}/{miss_on}/{rule}/s={signal}: "
              f"{(time.time()-tc)/60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
    return pd.DataFrame(rows)


def analyze(df):
    ok = df[~df.failed]
    print("\n" + "=" * 70)
    print("FPR at signal=0 (nominal 0.05) by placement x style x rule:")
    null = ok[ok.signal == 0]
    print(null.groupby(["miss_on", "style", "rule"]).rejected
          .agg(["mean", "count"]).round(3).to_string())
    print("\nPower at signal=0.3:")
    alt = ok[ok.signal == 0.3]
    if len(alt):
        print(alt.groupby(["miss_on", "style", "rule"]).rejected.mean()
              .round(3).to_string())
    print("\nDecision rule: adopt whichever rule keeps FPR <= 0.07 across all")
    print("placements; if imputed_labels exceeds it anywhere, labels_observed")
    print("becomes the documented default and affected experiments re-run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-variable missingness")
    parser.add_argument("--reps", type=int, default=250)
    parser.add_argument("--n_obs", type=int, default=1000)
    parser.add_argument("--dim_x", type=int, default=5)
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--dgps", nargs="+",
                        default=["linear_gaussian", "post_nonlinear"])
    parser.add_argument("--placements", nargs="+",
                        default=["z", "y", "both", "xz"])
    parser.add_argument("--signals", type=float, nargs="+",
                        default=[0.0, 0.3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str,
                        default="results/test_variable_missingness.csv")
    args = parser.parse_args()

    if args.quick:
        args.reps = 10
        args.placements = ["z"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    df = run_grid(args)
    print(f"Total: {(time.time()-t0)/3600:.1f} h")
    print(f"Saved to {args.output}  ({len(df)} rows)")
    analyze(df)
