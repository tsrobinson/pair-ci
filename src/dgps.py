"""Data-generating processes and missingness mechanisms.

The calibration DGPs return (X, Y, Z) with Z indep Y | X exactly when
signal == 0; they and inject_missingness_calibration feed the standalone
CI-test experiments (Section 5). The DAG samplers draw linear-Gaussian or
nonlinear additive-noise data on a random DAG for the graph-recovery
experiments, with the edge functions listed in Appendix E.
"""
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer



def dgp_linear_gaussian(n, dim_x, signal, rng):
    X = rng.randn(n, dim_x)
    beta_z = rng.randn(dim_x) * 0.5
    Z = X @ beta_z + rng.randn(n) * 0.5
    beta_y = rng.randn(dim_x) * 0.5
    Y = signal * Z + X @ beta_y + rng.randn(n) * 0.5
    return X, Y, Z


def dgp_post_nonlinear(n, dim_x, signal, rng):
    X = rng.randn(n, dim_x)
    beta = rng.randn(dim_x) * 0.5
    Z = np.tanh(X @ beta + rng.randn(n) * 0.3)
    gamma = rng.randn(dim_x) * 0.3
    Y = signal * np.sin(2 * Z) + X @ gamma + rng.randn(n) * 0.5
    return X, Y, Z


def dgp_latent_confounder(n, dim_x, signal, rng):
    X = rng.randn(n, dim_x)
    L = rng.randn(n)
    Z = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * L + rng.randn(n) * 0.3
    if dim_x > 2:
        Y = signal * L + np.cos(X[:, 2]) + rng.randn(n) * 0.5
    else:
        Y = signal * L + rng.randn(n) * 0.5
    return X, Y, Z


DGP_FUNCTIONS = {
    "linear_gaussian": dgp_linear_gaussian,
    "post_nonlinear": dgp_post_nonlinear,
    "latent_confounder": dgp_latent_confounder,
}


def inject_missingness_calibration(X, mechanism, rate, rng):
    """Mask up to two columns of the conditioning set X.

    P(missing) = expit(3 (v - q)), with q the (1 - rate) quantile of the
    driver v: under "mar" the first fully observed column of X, under
    "mnar" the masked column itself.
    """
    X_miss = X.copy()
    n, dim_x = X_miss.shape

    if mechanism == "complete":
        return X_miss

    n_miss_cols = min(2, dim_x)
    if mechanism == "mar":
        # MAR needs a fully observed driver in X, so at dim_x = 2 only one
        # column is masked (Appendix V describes the correction).
        n_miss_cols = min(n_miss_cols, dim_x - 1)

    miss_cols = list(range(n_miss_cols))
    X_ref = X.copy()  # masks are computed from the complete values
    for col in miss_cols:
        if mechanism == "mar":
            drivers = [c for c in range(dim_x) if c not in miss_cols]
            driver = X_ref[:, drivers[0]]
            logit = driver - np.quantile(driver, 1 - rate)
        elif mechanism == "mnar":
            logit = X_ref[:, col] - np.quantile(X_ref[:, col], 1 - rate)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        prob = 1 / (1 + np.exp(-3 * logit))  # steepness = 3
        X_miss[rng.rand(n) < prob, col] = np.nan

    return X_miss


def impute_datasets(X_miss, rng, n_imp=5, aux=None):
    """Return n_imp MICE imputations (posterior draws) of X_miss.

    If aux (an n x k array of fully observed columns) is given, the
    imputation model conditions on it, as in the pipelines of Foraita et
    al. (2020) and Witte et al. (2022), where the test variables (Y, Z)
    are auxiliary predictors when imputing X. Only the X columns are
    returned.
    """
    datasets = []
    p_x = X_miss.shape[1]
    for _ in range(n_imp):
        imp = IterativeImputer(
            max_iter=10,
            random_state=rng.randint(0, 2**31),
            sample_posterior=True,
        )
        if aux is None:
            datasets.append(imp.fit_transform(X_miss))
        else:
            full = np.column_stack([X_miss, aux])
            imputed_full = imp.fit_transform(full)
            datasets.append(imputed_full[:, :p_x])
    return datasets



def generate_erdos_renyi_dag(p, edge_prob, rng):
    """Random DAG on 0..p-1: each edge i -> j, i < j, with prob edge_prob."""
    edges = set()
    for i in range(p):
        for j in range(i + 1, p):
            if rng.rand() < edge_prob:
                edges.add((i, j))
    return edges


def sample_from_dag(edges, p, n, rng, noise_scale=0.5):
    """Linear-Gaussian SEM; weights uniform on +-[0.5, 1]."""
    weights = {}
    for i, j in edges:
        w = rng.uniform(0.5, 1.0)
        if rng.rand() < 0.5:
            w = -w
        weights[(i, j)] = w
    data = np.zeros((n, p))
    for j in range(p):
        parents = [i for (i, c) in edges if c == j]
        val = rng.randn(n) * noise_scale
        for pa in parents:
            val += weights[(pa, j)] * data[:, pa]
        data[:, j] = val
    return data


def sample_from_dag_nonlinear(edges, p, n, rng, noise_scale=0.5):
    """Additive-noise SEM; each edge gets one of four functions at random.

    The functions match those in experiments/alarm.py and hailfinder.py;
    tanh follows the `mim` benchmark of Zheng et al. (2020).
    """
    edge_fns = [
        lambda x, w: w * x ** 2,                     # quadratic
        lambda x, w: w * np.sin(2 * x),              # sinusoidal
        lambda x, w: w * np.abs(x),                  # absolute value
        lambda x, w: w * np.tanh(1.5 * x),           # saturating (tanh)
    ]

    weights = {}
    fns = {}
    for i, j in edges:
        w = rng.uniform(0.5, 1.0)
        if rng.rand() < 0.5:
            w = -w
        weights[(i, j)] = w
        fns[(i, j)] = edge_fns[rng.randint(0, len(edge_fns))]

    data = np.zeros((n, p))
    for j in range(p):
        parents = [i for (i, c) in edges if c == j]
        val = rng.randn(n) * noise_scale
        for pa in parents:
            val += fns[(pa, j)](data[:, pa], weights[(pa, j)])
        data[:, j] = val

    return data


def inject_missingness(data, mechanism, miss_cols, rate, rng):
    """Mask miss_cols as in inject_missingness_calibration.

    The MAR driver is the first column outside miss_cols; "mixed"
    alternates MAR and MNAR over miss_cols.
    """
    data = data.copy()
    n = data.shape[0]
    if mechanism == "complete":
        return data
    for idx, col in enumerate(miss_cols):
        if mechanism == "mar" or (mechanism == "mixed" and idx % 2 == 0):
            drivers = [c for c in range(data.shape[1]) if c not in miss_cols]
            driver = drivers[0] if drivers else 0
            logit = data[:, driver] - np.quantile(data[:, driver], 1 - rate)
        elif mechanism == "mnar" or (mechanism == "mixed" and idx % 2 == 1):
            logit = data[:, col] - np.quantile(data[:, col], 1 - rate)
        else:
            raise ValueError(f"Unknown: {mechanism}")
        prob = 1 / (1 + np.exp(-3 * logit))
        data[rng.rand(n) < prob, col] = np.nan
    return data



def analyze_kill_switch(df):
    """Print median SHD by mechanism and method, and a one-sided
    Mann-Whitney test of ci_oracle_cached against impute_fisherz."""
    import pandas as pd
    from scipy.stats import mannwhitneyu

    print(f"\n{'='*70}\nKILL-SWITCH RESULTS\n{'='*70}")
    dv = df[~df["failed"]].copy()
    pivot = dv.groupby(["mechanism", "method"])["shd"].median().unstack("method")
    print(f"\nMedian SHD:\n{pivot.round(1).to_string()}")

    print(f"\n{'-'*70}\nKEY COMPARISON: ci_oracle_cached vs impute_fisherz\n{'-'*70}")
    for mech in ["mar", "mnar", "mixed", "complete"]:
        ours = dv[(dv["mechanism"] == mech) & (dv["method"] == "ci_oracle_cached")]["shd"]
        base = dv[(dv["mechanism"] == mech) & (dv["method"] == "impute_fisherz")]["shd"]
        if len(ours) == 0 or len(base) == 0:
            continue
        imp = (base.median() - ours.median()) / base.median() * 100 if base.median() > 0 else 0
        try:
            _, up = mannwhitneyu(ours, base, alternative="less")
        except Exception:
            up = np.nan
        print(f"  {mech.upper():8s}: Ours={ours.median():.1f} Base={base.median():.1f} "
              f"Imp={imp:+.1f}% p={up:.4f}")
