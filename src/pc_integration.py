"""PC algorithm with the PAIR-CI oracle and the Fisher-Z baselines.

run_pc_with_ci_oracle runs PC with PAIR-CI on imputations cached once per
dataset (Algorithm 1, Appendix E); run_pc_complete_case,
run_pc_impute_then_test (FZ-vote) and run_pc_testwise_deletion use Fisher-Z.
All share one simplified skeleton search: conditioning sets are drawn from
the neighbours of one endpoint only, edges are removed as soon as a test
accepts (so the result depends on variable order; not PC-stable), and the
search stops at the first depth with no removal. The orientation step's
fourth rule differs from Meek's R4 (see _orient_edges). Also provides SHD
and skeleton metrics.
"""
import numpy as np
import warnings
from typing import Optional
from dataclasses import dataclass
from itertools import combinations

from ci_oracle import GeneralCITest


@dataclass
class PCResult:
    """adjacency_matrix[i, j] = 1 for i -> j; both entries for i - j."""
    adjacency_matrix: np.ndarray
    sep_sets: dict
    n_ci_tests: int = 0
    runtime_seconds: float = 0.0
    method: str = ""


def run_pc_with_ci_oracle(data, alpha=0.05, n_imputations=5, n_folds=10,
                          n_trees=100, imputation_strategy="cached",
                          max_depth=None, random_state=None, verbose=False,
                          ci_test_cls=None, method_label=None):
    """PC with a PAIR-CI oracle (GeneralCITest unless ci_test_cls is given).

    The oracle is built once, so with imputation_strategy="cached" all
    tests share the same M imputations. Tests reject when p < alpha.
    """
    import time
    t_start = time.time()
    p = data.shape[1]
    cls = ci_test_cls if ci_test_cls is not None else GeneralCITest
    ci_test = cls(
        data, n_imputations=n_imputations, n_folds=n_folds,
        n_trees=n_trees, imputation_strategy=imputation_strategy,
        random_state=random_state,
    )
    adj, sep_sets, n_tests = _pc_skeleton(p, ci_test, alpha, max_depth, verbose)
    adj_oriented = _orient_edges(adj, sep_sets, p, verbose)
    label = method_label or ("ci_oracle_" + imputation_strategy)
    return PCResult(
        adjacency_matrix=adj_oriented, sep_sets=sep_sets,
        n_ci_tests=n_tests, runtime_seconds=time.time() - t_start,
        method=label,
    )


def _pc_skeleton(p, ci_test, alpha, max_depth, verbose):
    """Simplified PC skeleton search (see the module docstring).

    For each pair i < j, conditioning sets are subsets of adj(i) minus j
    only, and an edge is removed at the first set with p >= alpha.
    """
    adj = np.ones((p, p), dtype=int) - np.eye(p, dtype=int)
    sep_sets = {}
    n_tests = 0
    depth = 0
    max_d = max_depth if max_depth is not None else p - 2
    while depth <= max_d:
        if verbose:
            print(f"  Depth {depth}: {adj.sum() // 2} edges remaining")
        any_removed = False
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] == 0:
                    continue
                nbrs = [k for k in range(p) if adj[i, k] == 1 and k != j]
                if len(nbrs) < depth:
                    continue
                found = False
                for cond_set in combinations(nbrs, depth):
                    _, pval = ci_test.test(i, j, cond_set)
                    n_tests += 1
                    if pval >= alpha:
                        adj[i, j] = adj[j, i] = 0
                        sep_sets[(i, j)] = set(cond_set)
                        sep_sets[(j, i)] = set(cond_set)
                        found = True
                        any_removed = True
                        if verbose:
                            print(f"    Removed {i}-{j} | {cond_set} (p={pval:.4f})")
                        break
        if not any_removed:
            break
        depth += 1
    return adj, sep_sets, n_tests


def _orient_edges(adj, sep_sets, p, verbose):
    """Orient v-structures, then apply orientation rules until none fires.

    On conflicting v-structures, the first found (by collider index) wins.
    """
    oriented = adj.copy()
    for k in range(p):
        parents = [i for i in range(p) if oriented[i, k] == 1 and i != k]
        for a in range(len(parents)):
            for b in range(a + 1, len(parents)):
                i, j = parents[a], parents[b]
                if oriented[i, j] == 1 or oriented[j, i] == 1:
                    continue
                sep = sep_sets.get((i, j), sep_sets.get((j, i), None))
                if sep is not None and k not in sep:
                    oriented[k, i] = oriented[k, j] = 0
                    if verbose:
                        print(f"    V-structure: {i} -> {k} <- {j}")

    def _directed(a, b):
        return oriented[a, b] == 1 and oriented[b, a] == 0

    def _undirected(a, b):
        return oriented[a, b] == 1 and oriented[b, a] == 1

    # Each pass visits every undirected edge in both directions, in index
    # order, and may orient several; the loop ends after a pass with no
    # change. max_iter is only a safeguard.
    max_iter = max(100, p * p)
    changed = True
    itr = 0
    while changed and itr < max_iter:
        changed = False
        itr += 1
        for i in range(p):
            for j in range(p):
                if i == j or not _undirected(i, j):
                    continue
                fired = False
                # Meek R1: k -> i, i - j undirected, k,j non-adjacent => i -> j
                for k in range(p):
                    if k == i or k == j:
                        continue
                    if (_directed(k, i) and oriented[k, j] == 0
                            and oriented[j, k] == 0):
                        oriented[j, i] = 0
                        changed = True
                        fired = True
                        break
                if fired:
                    continue
                # Meek R2: i -> k -> j with i - j undirected => i -> j
                for k in range(p):
                    if k == i or k == j:
                        continue
                    if _directed(i, k) and _directed(k, j):
                        oriented[j, i] = 0
                        changed = True
                        fired = True
                        break
                if fired:
                    continue
                # Meek R3: two undirected neighbours of i both directed into
                # j, non-adjacent to each other, i-j undirected => i -> j
                nbr_dir = [k for k in range(p)
                           if k != i and k != j and _undirected(i, k)
                           and _directed(k, j)]
                if len(nbr_dir) >= 2:
                    for a in range(len(nbr_dir)):
                        for b in range(a + 1, len(nbr_dir)):
                            k1, k2 = nbr_dir[a], nbr_dir[b]
                            if (oriented[k1, k2] == 0
                                    and oriented[k2, k1] == 0):
                                oriented[j, i] = 0
                                changed = True
                                fired = True
                                break
                        if fired:
                            break
                if fired:
                    continue
                # Fourth rule as implemented: i - k, k -> l -> j, i and l
                # non-adjacent, i - j => i -> j. This is not Meek's R4,
                # which requires i and l adjacent and k and j non-adjacent;
                # under the condition here R1 applied to (j, i) gives
                # j -> i, and whichever is visited first wins.
                for k in range(p):
                    if k == i or k == j or not _undirected(i, k):
                        continue
                    for l in range(p):
                        if l == i or l == j or l == k:
                            continue
                        if (_directed(k, l) and _directed(l, j)
                                and oriented[i, l] == 0
                                and oriented[l, i] == 0):
                            oriented[j, i] = 0
                            changed = True
                            fired = True
                            break
                    if fired:
                        break
    return oriented


def _fisherz_test_closure(data: np.ndarray):
    """Fisher-Z CI test on a complete data matrix.

    Partial correlation by least-squares residualisation on (1, X_S);
    two-sided p-value with standard error 1 / sqrt(n - |S| - 3). Valid for
    jointly Gaussian data. Same computation as TestwiseCITest below,
    without row deletion.
    """
    from scipy import stats as sp_stats
    n = data.shape[0]

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.n_tests = 0

    def test(i: int, j: int, cond_set):
        ctx.n_tests += 1
        try:
            if len(cond_set) == 0:
                r, _ = sp_stats.pearsonr(data[:, i], data[:, j])
            else:
                Z = np.column_stack([
                    np.ones(n), data[:, list(cond_set)]
                ])
                coef_i = np.linalg.lstsq(Z, data[:, i], rcond=None)[0]
                coef_j = np.linalg.lstsq(Z, data[:, j], rcond=None)[0]
                res_i = data[:, i] - Z @ coef_i
                res_j = data[:, j] - Z @ coef_j
                r, _ = sp_stats.pearsonr(res_i, res_j)
            if not np.isfinite(r):
                return 0.0, 1.0
            one_minus = max(1.0 - r, 1e-15)
            one_plus = max(1.0 + r, 1e-15)
            z = 0.5 * (np.log(one_plus) - np.log(one_minus))
            se = 1.0 / np.sqrt(n - len(cond_set) - 3)
            z_stat = abs(z) / se
            pval = 2.0 * (1.0 - sp_stats.norm.cdf(z_stat))
            return z_stat, pval
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0

    class _CITest:
        def __init__(self):
            pass

        def test(self_inner, i, j, cond_set):
            return test(i, j, cond_set)

        @property
        def n_tests(self_inner):
            return ctx.n_tests

    return _CITest()


def run_pc_complete_case(data, alpha=0.05, max_depth=None, verbose=False):
    """Complete-case PC: drop rows with any NaN, run PC + Fisher-Z.

    Fewer than 30 complete rows returns an empty graph with a warning.
    """
    import time
    t_start = time.time()
    mask = ~np.any(np.isnan(data), axis=1)
    data_complete = data[mask]
    p = data.shape[1]
    if data_complete.shape[0] < 30:
        warnings.warn(f"Only {data_complete.shape[0]} complete cases.")
        return PCResult(
            adjacency_matrix=np.zeros((p, p), dtype=int),
            sep_sets={}, method="complete_case",
            runtime_seconds=time.time() - t_start,
        )
    ci = _fisherz_test_closure(data_complete)
    adj, sep_sets, _ = _pc_skeleton(p, ci, alpha, max_depth, verbose)
    adj = _orient_edges(adj, sep_sets, p, verbose)
    return PCResult(
        adjacency_matrix=adj, sep_sets=sep_sets,
        n_ci_tests=ci.n_tests,
        runtime_seconds=time.time() - t_start,
        method="complete_case",
    )


def run_pc_impute_then_test(data, alpha=0.05, n_imputations=5,
                             max_depth=None, verbose=False, random_state=None):
    """Impute-then-test PC (FZ-vote): run PC + Fisher-Z on each of M
    imputed datasets and keep each oriented entry of the adjacency matrix
    that appears in more than M/2 of the M graphs. sep_sets is empty.
    """
    import time
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    t_start = time.time()
    rng = np.random.RandomState(random_state)
    p = data.shape[1]
    edge_votes = np.zeros((p, p), dtype=float)
    for _ in range(n_imputations):
        imp = IterativeImputer(
            max_iter=10,
            random_state=rng.randint(0, 2**31),
            sample_posterior=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_imp = imp.fit_transform(data)
        ci = _fisherz_test_closure(data_imp)
        adj_m, sep_m, _ = _pc_skeleton(p, ci, alpha, max_depth, verbose)
        adj_m = _orient_edges(adj_m, sep_m, p, verbose)
        edge_votes += adj_m
    adj_final = (edge_votes > n_imputations / 2).astype(int)
    return PCResult(
        adjacency_matrix=adj_final, sep_sets={},
        runtime_seconds=time.time() - t_start,
        method="impute_fisherz",
    )


def run_pc_testwise_deletion(data, alpha=0.05, max_depth=None, verbose=False):
    """Test-wise deletion PC: each Fisher-Z test uses the rows complete on
    its own variables. A test with fewer than 30 such rows returns p = 1,
    so the edge is removed."""
    import time
    from scipy import stats as sp_stats
    t_start = time.time()
    p = data.shape[1]

    class TestwiseCITest:
        def __init__(self, data):
            self.data = data
            self.n_tests = 0
        def test(self, x_idx, y_idx, cond_set):
            self.n_tests += 1
            cols = [x_idx, y_idx] + list(cond_set)
            mask = ~np.any(np.isnan(self.data[:, cols]), axis=1)
            d = self.data[mask]
            if d.shape[0] < 30:
                return 0.0, 1.0
            try:
                from scipy.stats import pearsonr
                if len(cond_set) == 0:
                    r, _ = pearsonr(d[:, x_idx], d[:, y_idx])
                else:
                    # The intercept is what the n - |S| - 3 in the
                    # Fisher-Z standard error assumes.
                    Z = np.column_stack([
                        np.ones(d.shape[0]), d[:, list(cond_set)]
                    ])
                    res_x = d[:, x_idx] - Z @ np.linalg.lstsq(Z, d[:, x_idx], rcond=None)[0]
                    res_y = d[:, y_idx] - Z @ np.linalg.lstsq(Z, d[:, y_idx], rcond=None)[0]
                    r, _ = pearsonr(res_x, res_y)
                if not np.isfinite(r):
                    return 0.0, 1.0
                one_minus = max(1.0 - r, 1e-15)
                one_plus = max(1.0 + r, 1e-15)
                z = 0.5 * (np.log(one_plus) - np.log(one_minus))
                se = 1.0 / np.sqrt(d.shape[0] - len(cond_set) - 3)
                z_stat = abs(z) / se
                pval = 2 * (1 - sp_stats.norm.cdf(z_stat))
                return z_stat, pval
            except (np.linalg.LinAlgError, ValueError):
                return 0.0, 1.0

    tw = TestwiseCITest(data)
    adj, ss, nt = _pc_skeleton(p, tw, alpha, max_depth, verbose)
    adj = _orient_edges(adj, ss, p, verbose)
    return PCResult(adjacency_matrix=adj, sep_sets=ss, n_ci_tests=tw.n_tests,
                    runtime_seconds=time.time() - t_start, method="testwise_deletion")


def structural_hamming_distance(adj_true, adj_est):
    """Missing or extra adjacencies plus adjacencies whose marks differ;
    each pair counts at most once."""
    p = adj_true.shape[0]
    shd = 0
    for i in range(p):
        for j in range(i + 1, p):
            te = adj_true[i, j] or adj_true[j, i]
            ee = adj_est[i, j] or adj_est[j, i]
            if te and not ee:
                shd += 1
            elif not te and ee:
                shd += 1
            elif te and ee:
                if adj_true[i, j] != adj_est[i, j] or adj_true[j, i] != adj_est[j, i]:
                    shd += 1
    return shd


def edge_metrics(adj_true, adj_est):
    """Skeleton precision, recall and F1 (orientation ignored)."""
    p = adj_true.shape[0]
    tp = fp = fn = 0
    for i in range(p):
        for j in range(i + 1, p):
            te = adj_true[i, j] or adj_true[j, i]
            ee = adj_est[i, j] or adj_est[j, i]
            if te and ee: tp += 1
            elif not te and ee: fp += 1
            elif te and not ee: fn += 1
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
    return {"precision": pr, "recall": re, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def dag_to_adjacency(n_nodes, edges):
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in edges:
        adj[i, j] = 1
    return adj
