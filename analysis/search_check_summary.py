"""Numbers quoted in Appendix "Implementation Details" (paragraph "PC algorithm
with cached imputations") for the search check, recomputed from

  results/search_check/rerun_{alarm_nl,hailf}.csv   (experiments/search_check_baselines.py)
  results/{alarm_nonlinear,hailfinder}_results.csv  (PAIR-CI, Table 3, simplified search)
  results/mvpc_{alarm_nonlinear,hailfinder_nonlinear}.csv (MVPC, causal-learn PC-stable)

Cell: 20% MNAR, n = 1,000, 20 replicates (Table 3).  Estimands, per network:

  * median-SHD reduction of test-wise deletion and FZ-vote when the
    simplified search is replaced by the PC-stable search: median(SHD, simplified)
    - median(SHD, PC-stable) over the 20 replicates (the paper's 7 / 7.5 on
    ALARM-nonlinear and 24 / 18.5 on HAILFINDER);
  * spurious adjacencies (skeleton false positives) per replicate under each
    search, back-derived as fp = tp (1 - precision) / precision with
    tp = recall x |E| (|E| = 46 ALARM, 66 HAILFINDER; the CSVs log P/R only);
    the paper's "reduces the baselines' spurious adjacencies in every replicate"
    is checked for test-wise deletion and FZ-vote (complete case is reported
    too; on HAILFINDER it returns the empty graph under both searches);
  * PAIR-CI (simplified search) against each PC-stable baseline and MVPC:
    median SHD, median skeleton F1, and the number of paired replicates in
    which PAIR-CI's SHD is lower.  Pairing is by replicate index; datasets are
    bit-identical across methods except MVPC on HAILFINDER (matched in
    distribution; one MVPC replicate exceeded the 1-hour cap, so 19 pairs).

These are descriptive counts over the 20 replicates of one cell per
network; no inference is attempted.

Exit status is non-zero if any number quoted in the appendix is not
reproduced.  Usage:  python analysis/search_check_summary.py [--results-dir results]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETS = {
    "ALARM (nonlinear)": dict(rerun="search_check/rerun_alarm_nl.csv", base="alarm_nonlinear_results.csv",
                              mvpc="mvpc_alarm_nonlinear.csv", n_true=46,
                              posted={"testwise": 7.0, "impute_fisherz": 7.5}),
    "HAILFINDER": dict(rerun="search_check/rerun_hailf.csv", base="hailfinder_results.csv",
                       mvpc="mvpc_hailfinder_nonlinear.csv", n_true=66,
                       posted={"testwise": 24.0, "impute_fisherz": 18.5}),
}
LABEL = {"testwise": "test-wise deletion", "impute_fisherz": "FZ-vote", "complete_case": "complete case"}


def _spurious(d, n_true):
    """Skeleton false positives per replicate, from recall and precision."""
    tp = d["recall"].to_numpy() * n_true
    prec = d["precision"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fp = np.where(prec > 0, tp * (1 - prec) / prec, 0.0)
    return np.round(fp, 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    R = a.results_dir
    fails = []

    def check(desc, ok):
        print(f"    [{'ok' if ok else 'FAIL'}] {desc}")
        if not ok:
            fails.append(desc)

    for net, c in NETS.items():
        r = pd.read_csv(os.path.join(R, c["rerun"]))
        b = pd.read_csv(os.path.join(R, c["base"]))
        m = pd.read_csv(os.path.join(R, c["mvpc"]))
        b = b[(b["rate"] == 0.2) & (b["mechanism"] == "mnar")]
        m = m[(m["rate"] == 0.2) & (m["mechanism"] == "mnar")]
        n_fail_mvpc = int(m["failed"].astype(bool).sum())
        m = m[~m["failed"].astype(bool)]
        assert not r["failed"].astype(bool).any() and not b["failed"].astype(bool).any()
        print(f"== {net}, 20% MNAR ({r['rep'].nunique()} replicates; MVPC failed/DNF: {n_fail_mvpc})")

        print("  Baselines, simplified search -> PC-stable search:")
        for meth in ("testwise", "impute_fisherz", "complete_case"):
            o = r[r["method"] == f"{meth}__orig"].set_index("rep").sort_index()
            s = r[r["method"] == f"{meth}__std"].set_index("rep").sort_index()
            p = b[b["method"] == meth].set_index("rep").sort_index()
            red = o["shd"].median() - s["shd"].median()
            fo, fs = _spurious(o, c["n_true"]), _spurious(s, c["n_true"])
            n_red = int((fs < fo).sum())
            same = int((o["shd"] == p["shd"]).sum())
            print(f"    {LABEL[meth]:<19} median SHD {o['shd'].median():6.1f} -> {s['shd'].median():6.1f} "
                  f"(reduction {red:4.1f}); median F1 {o['f1'].median():.3f} -> {s['f1'].median():.3f}; "
                  f"median spurious {np.median(fo):5.1f} -> {np.median(fs):5.1f}, reduced in {n_red}/{len(o)}; "
                  f"simplified arm = paper CSV in {same}/{len(o)} reps (paper median {p['shd'].median():.1f})")
            if meth in c["posted"]:
                check(f"{net}: {LABEL[meth]} median SHD reduction = {c['posted'][meth]}", red == c["posted"][meth])
                check(f"{net}: {LABEL[meth]} spurious adjacencies reduced in every replicate", n_red == len(o))

        pc = b[b["method"] == "ci_oracle_fast"].set_index("rep").sort_index()
        print(f"  PAIR-CI (simplified search): median SHD {pc['shd'].median():.1f}, median F1 {pc['f1'].median():.3f}")
        comps = [(f"{LABEL[k]} (PC-stable)", r[r["method"] == f"{k}__std"].set_index("rep")) for k in ("testwise", "impute_fisherz")]
        comps.append(("MVPC (PC-stable + correction)", m.set_index("rep")))
        for name, d in comps:
            j = pc[["shd", "f1"]].join(d[["shd", "f1"]], lsuffix="_pc", rsuffix="_b", how="inner")
            lower = int((j["shd_pc"] < j["shd_b"]).sum())
            print(f"    vs {name:<30}: median SHD {j['shd_pc'].median():.1f} vs {d['shd'].median():.1f}; "
                  f"median F1 {pc['f1'].median():.3f} vs {d['f1'].median():.3f}; PAIR-CI lower SHD in {lower}/{len(j)} pairs")
            check(f"{net}: PAIR-CI lower median SHD and higher median F1 than {name}",
                  pc["shd"].median() < d["shd"].median() and pc["f1"].median() > d["f1"].median())
            if not name.startswith("MVPC"):
                check(f"{net}: PAIR-CI lower SHD in each of the 20 paired replicates vs {name}", lower == 20 == len(j))
            elif lower != len(j):
                print(f"    NOTE: vs MVPC the per-replicate count is {lower}/{len(j)}, not 20/20"
                      + (" (datasets matched in distribution only)" if net == "HAILFINDER" else ""))
        print()

    print("ALL APPENDIX SEARCH-CHECK NUMBERS REPRODUCED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
