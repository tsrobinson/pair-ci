"""K = 5 versus K = 10 folds on the worst calibration cell.

False positive rate of PAIR-CI (general variant, M = 5, cached imputation)
with K = 5 and with K = 10 on the linear-Gaussian DGP with |X| = 10, MNAR at
30%, n = 1000 and signal 0, 200 replicates each. Both K values draw from one
sequential generator, so they see different datasets.

A one-off check behind the choice of K = 10 for the general variant, kept
for provenance. It has no CLI, prints only, and is not run by
reproduce_all.sh; the paper's sensitivity to K comes from sensitivity.py.

    python experiments/k_compare.py
"""

import os, sys
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from dgps import DGP_FUNCTIONS, inject_missingness_calibration
from ci_oracle import GeneralCITest

rng = np.random.RandomState(42)
dgp_fn = DGP_FUNCTIONS["linear_gaussian"]

for K in [5, 10]:
    rejects = 0
    n_reps = 200
    for rep in range(n_reps):
        X, Y, Z = dgp_fn(1000, 10, 0.0, rng)
        X_miss = inject_missingness_calibration(X, "mnar", 0.3, rng)
        data = np.column_stack([Z, Y, X_miss])
        ci = GeneralCITest(data, n_imputations=5, n_folds=K,
                           imputation_strategy="cached",
                           random_state=rng.randint(0, 2**31))
        stat, pval = ci.test(x_idx=0, y_idx=1, cond_set=tuple(range(2, 12)))
        if pval < 0.05:
            rejects += 1
        if (rep + 1) % 50 == 0:
            print(f"K={K}: {rep+1}/{n_reps} done, running FPR={rejects/(rep+1):.3f}",
                  flush=True)
    print(f"K={K}: FPR = {rejects/n_reps:.3f} ({rejects}/{n_reps})", flush=True)
