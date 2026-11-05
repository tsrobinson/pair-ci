"""Does Rubin's combined variance T over-cover the variance of mu_hat?

For each replicate, one dataset is generated and PAIR-CI (GeneralCITest,
random forests, M = 5, K = 5, cached imputation) is run R = 4 times on it
with fresh internal seeds (imputations, fold splits, placebo draws, forest
seeds). Logging (mu_hat, W_bar, B, T, df) per run separates
  Var_MC(mu_hat)  the sampling variance of mu_hat across datasets, using
                  the first run per dataset;
  sigma2_draw     the variance of mu_hat across runs with the data fixed;
  mean(T)         the variance the test uses.
Theorem 4(i) predicts mean(T) > Var_MC(mu_hat) under the internal null.
The working hypothesis is that the excess is about sigma2_draw, since T
counts draw noise at (1 + 1/M) B while it enters Var(mu_hat) at about B/M.
Complete-data cells isolate the draw noise that does not come from
imputation.

Grid: linear-Gaussian and post-nonlinear DGPs, complete data, MAR and MNAR
(30% per column), n in {500, 1000}, |X| in {2, 5}, signal 0; 250 datasets
per cell, so 24 cells. The variance components are read from
GeneralCITest.last_components_.

    python experiments/variance_decomposition.py \
        --output results/variance_decomposition.csv

One sequential generator serves every cell. In the released CSV the four
MAR cells at |X| = 2 predate the MAR generator fix and are labelled
mnar_1col; a rerun produces genuine MAR cells there and is not bit-identical
downstream of them (see the README). Feeds App. R.3 (Table variance_ratio).
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
from scipy import stats

from dgps import DGP_FUNCTIONS, inject_missingness_calibration
from ci_oracle import GeneralCITest


def run_cell(dgp_name, mechanism, n, dim_x, signal, rep, n_reimp, rng,
             miss_rate=0.3, M=5, K=5):
    X, Y, Z = DGP_FUNCTIONS[dgp_name](n, dim_x, signal, rng)
    X_miss = inject_missingness_calibration(X, mechanism, miss_rate, rng)
    data = np.column_stack([Z, Y, X_miss])
    cond_set = tuple(range(2, 2 + dim_x))
    rows = []
    for r in range(n_reimp):
        try:
            ci = GeneralCITest(data, n_imputations=M, n_folds=K,
                               imputation_strategy="cached",
                               random_state=rng.randint(0, 2**31))
            t_stat, pval = ci.test(x_idx=0, y_idx=1, cond_set=cond_set)
            comp = getattr(ci, "last_components_", {})
            rows.append(dict(
                dgp=dgp_name, mechanism=mechanism, n=n, dim_x=dim_x,
                signal=signal, rep=rep, run=r,
                t_stat=t_stat, p_value=pval, rejected=pval < 0.05,
                mu_hat=comp.get("mu_hat", np.nan),
                W_bar=comp.get("W_bar", np.nan),
                B=comp.get("B", np.nan),
                T=comp.get("T", np.nan),
                df=comp.get("df", np.nan),
                degenerate=comp.get("degenerate", False),
                failed=False))
        except Exception:
            rows.append(dict(dgp=dgp_name, mechanism=mechanism, n=n,
                             dim_x=dim_x, signal=signal, rep=rep, run=r,
                             t_stat=np.nan, p_value=np.nan, rejected=False,
                             mu_hat=np.nan, W_bar=np.nan, B=np.nan,
                             T=np.nan, df=np.nan, degenerate=False,
                             failed=True))
    return rows


def analyze(df):
    ok = df[~df.failed & ~df.degenerate]
    print("\n" + "=" * 78)
    print("VARIANCE DECOMPOSITION (per cell; signal = 0)")
    print("=" * 78)
    header = (f"{'cell':<38} {'FPR':>5} {'FPR_z':>6} {'VarMC':>9} "
              f"{'meanT':>9} {'ratio':>6} {'s2draw':>9} {'excess':>9}")
    print(header)
    for key, cell in ok[ok.signal == 0].groupby(
            ["dgp", "mechanism", "n", "dim_x"]):
        first = cell[cell.run == 0]
        if len(first) < 20:
            continue
        var_mc = first.mu_hat.var(ddof=1)
        mean_t = cell["T"].mean()  # cell.T would transpose the frame
        # Draw variance: variance of mu_hat across runs, data held fixed.
        s2_draw = cell.groupby("rep").mu_hat.var(ddof=1).mean()
        fpr = first.rejected.mean()
        fpr_z = (first.t_stat > stats.norm.ppf(0.95)).mean()
        ratio = mean_t / var_mc if var_mc > 0 else np.nan
        excess = mean_t - var_mc
        name = f"{key[0][:14]}/{key[1]}/n={key[2]}/d={key[3]}"
        print(f"{name:<38} {fpr:>5.3f} {fpr_z:>6.3f} {var_mc:>9.2e} "
              f"{mean_t:>9.2e} {ratio:>6.2f} {s2_draw:>9.2e} {excess:>9.2e}")
    print("\nReading guide: ratio > 1 quantifies conservatism of T against the")
    print("true Var(mu_hat); the prediction is excess ~ s2draw. FPR_z shows the")
    print("rejection rate a N(0,1) critical value would give (isolates the df/")
    print("critical-value contribution from the variance-overshoot contribution).")
    print("\nMean components by mechanism (signal = 0):")
    print(ok[ok.signal == 0].groupby("mechanism")[["W_bar", "B", "T", "df"]]
          .mean().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAIR-CI variance decomposition")
    parser.add_argument("--reps", type=int, default=250,
                        help="datasets per cell")
    parser.add_argument("--n_reimp", type=int, default=4,
                        help="repeated runs per dataset (fresh draws)")
    parser.add_argument("--dgps", nargs="+",
                        default=["linear_gaussian", "post_nonlinear"])
    parser.add_argument("--mechanisms", nargs="+",
                        default=["complete", "mar", "mnar"])
    parser.add_argument("--ns", type=int, nargs="+", default=[500, 1000])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--signals", type=float, nargs="+", default=[0.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str,
                        default="results/variance_decomposition.csv")
    args = parser.parse_args()

    if args.quick:
        args.reps, args.n_reimp = 10, 2
        args.ns, args.dims = [500], [2]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rng = np.random.RandomState(args.seed)
    all_rows = []
    t0 = time.time()
    n_cells = (len(args.dgps) * len(args.mechanisms) * len(args.ns)
               * len(args.dims) * len(args.signals))
    i_cell = 0
    for dgp in args.dgps:
        for mech in args.mechanisms:
            for n in args.ns:
                for dim_x in args.dims:
                    for signal in args.signals:
                        i_cell += 1
                        tc = time.time()
                        for rep in range(args.reps):
                            all_rows.extend(run_cell(
                                dgp, mech, n, dim_x, signal, rep,
                                args.n_reimp, rng))
                        print(f"[{i_cell}/{n_cells}] {dgp}/{mech}/n={n}/"
                              f"d={dim_x}/s={signal}: "
                              f"{(time.time()-tc)/60:.1f} min", flush=True)
                        # Save after each cell.
                        pd.DataFrame(all_rows).to_csv(args.output, index=False)
    print(f"Total: {(time.time()-t0)/3600:.1f} h")
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}  ({len(df)} rows)")
    analyze(df)
