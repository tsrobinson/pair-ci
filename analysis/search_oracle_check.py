"""Perfect-oracle checks of the in-house PC adjacency search
(src/pc_integration.py::_pc_skeleton).

With a perfect d-separation oracle any search that is complete for the PC
skeleton returns the true skeleton exactly.  The in-house search departs from
Kalisch & Buhlmann (2007, Alg. 1) in two ways -- conditioning sets are drawn
from the lower-indexed endpoint's neighbourhood only, and it stops at the
first level at which no edge is removed -- so it can keep spurious adjacencies
even under a perfect oracle.  Three parts:

  examples  The two DAGs of part (i) of the remark on the scope of the
            argument in Appendix A ("Skeleton False-Edge Control at a
            Fixed Level"):  u -> w <- v, w -> t, v -> t keeps
            u - t under a one-endpoint search; c -> a <- d, c -> b <- d,
            c -> d returns the complete graph under the early-stop rule.
            Deterministic; this is the oracle check the paper reports.
  er        Estimand: P(returned skeleton != true skeleton) and the mean number
            of spurious adjacencies, over Erdos-Renyi DAGs drawn by
            src/dgps.generate_erdos_renyi_dag (edge i -> j only for i < j, so
            the lower index is the ancestor side), at the experiments'
            (p, edge_prob); also with a random node relabelling.  Binomial /
            SD-over-sqrt(N) Monte Carlo SEs are printed.  Supplementary: not
            reported in the paper.
  networks  The same on the true ALARM and HAILFINDER DAGs with the node
            indexing used in the experiments (column order = *_NODES list),
            240 s cap per network.  Supplementary: not reported in the paper.

Built-in controls: a standard K&B search with the same oracle must return the
true skeleton on every DAG, and the oracle must never delete a true edge
(asserted).  Requires networkx >= 3.3 (nx.is_d_separator).

Usage:
    python analysis/search_oracle_check.py                 # examples + er + networks
    python analysis/search_oracle_check.py --part examples
    python analysis/search_oracle_check.py --smoke         # examples + a 20-DAG er run
"""
import argparse
import itertools
import os
import signal
import sys
import time

import numpy as np
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from pc_integration import _pc_skeleton  # noqa: E402
from dgps import generate_erdos_renyi_dag  # noqa: E402


class Oracle:
    """Perfect d-separation oracle with the ci_test.test(i, j, S) interface:
    p-value 1 if i and j are d-separated by S in G, else 0."""
    def __init__(s, G): s.G = G; s.n = 0
    def test(s, i, j, cond):
        s.n += 1
        return 0.0, (1.0 if nx.is_d_separator(s.G, {i}, {j}, set(cond)) else 0.0)


def standard_pc(p, orc):
    """Kalisch & Buhlmann (2007) Alg. 1 adjacency search (control arm)."""
    adj = np.ones((p, p), int) - np.eye(p, dtype=int); l = 0
    while True:
        if not any(adj[i].sum() - 1 >= l for i in range(p) for j in range(p) if adj[i, j]): break
        for i in range(p):
            for j in range(p):
                if i == j or not adj[i, j]: continue
                nb = [k for k in range(p) if adj[i, k] and k != j]
                if len(nb) < l: continue
                for S in itertools.combinations(nb, l):
                    if orc.test(i, j, S)[1] >= 0.05:
                        adj[i, j] = adj[j, i] = 0; break
        l += 1
    return adj


def _skel(p, E):
    """Symmetric skeleton adjacency matrix of the edge list E."""
    true = np.zeros((p, p), int)
    for a, b in E: true[a, b] = true[b, a] = 1
    return true


def part_examples():
    print("== Remark (i) counterexamples (perfect oracle) ==")
    # Example 1: one-endpoint rule.  u=0, v=1, w=2, t=3 (u has the lower index,
    # so the in-house search conditions on subsets of adj(u) for the pair u-t).
    E1 = [(0, 2), (1, 2), (2, 3), (1, 3)]
    # Example 2: early-stop rule.  c=0, d=1, a=2, b=3.
    E2 = [(0, 2), (1, 2), (0, 3), (1, 3), (0, 1)]
    ok = True
    for name, E, pair, expect in (
            ("u->w<-v, w->t, v->t", E1, (0, 3), "in-house keeps u-t; standard PC recovers the truth"),
            ("c->a<-d, c->b<-d, c->d", E2, (2, 3), "in-house returns the complete graph; standard PC recovers the truth")):
        p = 4; G = nx.DiGraph(); G.add_nodes_from(range(p)); G.add_edges_from(E)
        true = _skel(p, E)
        a_in, _, _ = _pc_skeleton(p, Oracle(G), 0.05, None, False)
        a_std = standard_pc(p, Oracle(G))
        kept = bool(a_in[pair])
        complete = bool((a_in + np.eye(p, dtype=int) == 1).all())
        std_exact = bool((a_std == true).all())
        if E is E1:
            res = kept and std_exact
            print(f"  {name}: in-house keeps the spurious pair: {kept}; standard PC exact: {std_exact}  -> {'PASS' if res else 'FAIL'} ({expect})")
        else:
            res = complete and std_exact
            print(f"  {name}: in-house returns complete graph: {complete}; standard PC exact: {std_exact}  -> {'PASS' if res else 'FAIL'} ({expect})")
        ok &= res
    return ok


def run_er(p, ep, ndag, seed, permute=False):
    """Run both searches on ndag random DAGs; True if the control is exact."""
    rng = np.random.RandomState(seed); bad_in = bad_std = 0; spur = []
    for _ in range(ndag):
        E = generate_erdos_renyi_dag(p, ep, rng)
        if permute:
            perm = rng.permutation(p); E = {(perm[a], perm[b]) for a, b in E}
        G = nx.DiGraph(); G.add_nodes_from(range(p)); G.add_edges_from(E)
        true = _skel(p, E)
        orc = Oracle(G)
        a_in, _, _ = _pc_skeleton(p, orc, 0.05, None, False)
        a_std = standard_pc(p, orc)
        bad_std += int((a_std != true).any())
        bad_in += int((a_in != true).any())
        spur.append(int(((a_in == 1) & (true == 0)).sum() // 2))
        assert ((a_in == 0) & (true == 1)).sum() == 0  # oracle never deletes a true edge
    f = bad_in / ndag
    sd = np.std(spur, ddof=1) if ndag > 1 else float("nan")
    print(f"  p={p} ep={ep} permute={permute} N={ndag} (seed {seed}): in-house wrong={f:.3f} (MCSE {np.sqrt(f*(1-f)/ndag):.3f}), "
          f"mean spurious={np.mean(spur):.2f} (MCSE {sd/np.sqrt(ndag):.2f}); standard-PC wrong={bad_std/ndag:.3f}", flush=True)
    return bad_std == 0


def part_er(smoke=False):
    print("== Erdos-Renyi DAGs (perfect oracle) ==")
    if smoke:
        return run_er(10, 0.25, 20, 7)
    ok = run_er(10, 0.25, 400, 7)
    ok &= run_er(10, 0.25, 400, 8, permute=True)
    ok &= run_er(20, 0.20, 25, 9)
    return ok


def part_networks(cap=240):
    print("== ALARM / HAILFINDER true DAGs (perfect oracle) ==")
    import alarm, hailfinder

    def handler(*a): raise TimeoutError
    signal.signal(signal.SIGALRM, handler)
    for name, nodes, edges in (("ALARM", alarm.ALARM_NODES, alarm.ALARM_EDGES),
                               ("HAILFINDER", hailfinder.HAILFINDER_NODES, hailfinder.HAILFINDER_EDGES)):
        p = len(nodes); idx = {v: i for i, v in enumerate(nodes)}
        E = [(idx[a], idx[b]) for a, b in edges]
        G = nx.DiGraph(); G.add_nodes_from(range(p)); G.add_edges_from(E)
        true = _skel(p, E)
        frac_low_to_high = np.mean([a < b for a, b in E])
        orc = Oracle(G); t = time.time(); signal.alarm(cap)
        try:
            adj, _, _ = _pc_skeleton(p, orc, 0.05, None, False)
            spur = int(((adj == 1) & (true == 0)).sum() // 2); miss = int(((adj == 0) & (true == 1)).sum() // 2)
            print(f"  {name}: p={p}, {len(E)} edges, share of edges low->high index {frac_low_to_high:.2f}; "
                  f"in-house search with perfect oracle: spurious={spur}, missing={miss}, tests={orc.n}, {time.time()-t:.0f}s", flush=True)
        except TimeoutError:
            print(f"  {name}: timed out after {cap}s ({orc.n} tests)", flush=True)
        signal.alarm(0)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["examples", "er", "networks", "all"], default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="examples + one 20-DAG p=10 ER run (seconds)")
    a = ap.parse_args()
    ok = True
    if a.smoke:
        ok &= part_examples(); ok &= part_er(smoke=True)
    else:
        if a.part in ("examples", "all"): ok &= part_examples()
        if a.part in ("er", "all"): ok &= part_er()
        if a.part in ("networks", "all"): part_networks()
    print("ALL CONTROLS PASS" if ok else "A CONTROL FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
