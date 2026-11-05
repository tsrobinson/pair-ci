"""Local (nearest-neighbor) permutation used to build the PAIR-CI placebo.

Each Y_i is replaced by the Y value of a randomly chosen one of the k
nearest neighbors of X_i, which approximately preserves the law of Y given
X while breaking any further dependence (Runge, 2018; Appendix E). Draws
are made independently per row, so the result is a local resampling with
replacement rather than a permutation of Y. Running this file compares
conditional and unconditional placebos on a toy example.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional


def conditional_permute(
    Y: np.ndarray,
    X: np.ndarray,
    k: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Loop version of conditional_permute_batched (unused)."""
    if rng is None:
        rng = np.random.RandomState()

    n = len(Y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    d = X.shape[1]

    # k = n^{2/(d+2)}: the number of points within the optimal kernel
    # bandwidth n^{-1/(d+2)} for a Lipschitz density (Appendix E).
    if k is None:
        k = max(2, int(np.floor(n ** (2.0 / (d + 2)))))

    k = min(k, n - 1)

    # Too few points for a neighborhood: unconditional permutation.
    if k < 2:
        return rng.permutation(Y)

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # The neighbor set includes i itself; if i is drawn, redraw from the
    # others.
    U = np.empty(n)
    for i in range(n):
        neighbors = indices[i]
        j = rng.choice(neighbors)
        if j == i and k > 1:
            others = neighbors[neighbors != i]
            if len(others) > 0:
                j = rng.choice(others)
        U[i] = Y[j]

    return U


def conditional_permute_batched(
    Y: np.ndarray,
    X: np.ndarray,
    k: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Placebo for Y given X by local resampling among the k nearest
    neighbors in X; k defaults to max(2, floor(n^{2/(d+2)})), d = dim X.

    Uses rng.randint once for all rows, so it consumes the RandomState
    stream differently from conditional_permute.
    """
    if rng is None:
        rng = np.random.RandomState()

    n = len(Y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]

    if k is None:
        k = max(2, int(np.floor(n ** (2.0 / (d + 2)))))
    k = min(k, n - 1)

    if k < 2:
        return rng.permutation(Y)

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    random_col = rng.randint(0, k, size=n)
    chosen_neighbors = indices[np.arange(n), random_col]

    # If a row drew itself, take the next neighbor in its list instead.
    self_mask = chosen_neighbors == np.arange(n)
    if self_mask.any():
        alt_col = (random_col[self_mask] + 1) % k
        chosen_neighbors[self_mask] = indices[
            np.where(self_mask)[0], alt_col
        ]

    return Y[chosen_neighbors]


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Conditional Permutation Validation ===\n")

    rng = np.random.RandomState(42)
    n = 1000

    X = rng.randn(n, 2)
    Y = 0.8 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3

    # An unconditional permutation breaks the Y-X correlation; a conditional
    # one should preserve it.
    U_uncond = rng.permutation(Y)
    corr_uncond = np.corrcoef(X[:, 0], U_uncond)[0, 1]

    U_cond = conditional_permute_batched(Y, X, rng=rng)
    corr_cond = np.corrcoef(X[:, 0], U_cond)[0, 1]
    corr_orig = np.corrcoef(X[:, 0], Y)[0, 1]

    print(f"Correlation of X[:,0] with:")
    print(f"  Original Y:                  {corr_orig:.3f}")
    print(f"  Unconditional permutation:   {corr_uncond:.3f}  (should be ~0)")
    print(f"  Conditional permutation:     {corr_cond:.3f}  (should be ~{corr_orig:.3f})")

    print("\n--- CI Test comparison ---\n")

    from ci_oracle import GeneralCITest

    Z = rng.randn(n, 2)
    X_var = Z @ np.array([1.0, 0.5]) + rng.randn(n) * 0.5
    Y_indep = rng.randn(n)  # Independent of X given Z
    data = np.column_stack([X_var, Y_indep, Z])

    ci_uncond = GeneralCITest(data, n_imputations=5, n_folds=5,
                              use_conditional_permutation=False, random_state=42)
    s_u, p_u = ci_uncond.test(0, 1, (2, 3))
    print(f"Unconditional permutation (H0 true): t={s_u:.3f}, p={p_u:.3f}")

    ci_cond = GeneralCITest(data, n_imputations=5, n_folds=5,
                            use_conditional_permutation=True, random_state=42)
    s_c, p_c = ci_cond.test(0, 1, (2, 3))
    print(f"Conditional permutation (H0 true):   t={s_c:.3f}, p={p_c:.3f}")

    Y_dep = 0.8 * X_var + rng.randn(n) * 0.3
    data_alt = np.column_stack([X_var, Y_dep, Z])

    ci_uncond2 = GeneralCITest(data_alt, n_imputations=5, n_folds=5,
                               use_conditional_permutation=False, random_state=42)
    s_u2, p_u2 = ci_uncond2.test(0, 1, (2, 3))
    print(f"\nUnconditional permutation (H1 true): t={s_u2:.3f}, p={p_u2:.3f}")

    ci_cond2 = GeneralCITest(data_alt, n_imputations=5, n_folds=5,
                             use_conditional_permutation=True, random_state=42)
    s_c2, p_c2 = ci_cond2.test(0, 1, (2, 3))
    print(f"Conditional permutation (H1 true):   t={s_c2:.3f}, p={p_c2:.3f}")

    print("\n=== Validation Complete ===")
