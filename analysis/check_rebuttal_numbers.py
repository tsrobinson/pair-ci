"""
Check that numbers quoted in the paper's text and in the review discussion
are reproduced by the result CSVs and the loaders that build the tables.

Each check is (label, computed value, quoted value, tolerance); the
tolerance is the rounding of the quoted value, usually half a unit in the
last printed digit, and check_le tests one-sided claims. In the output,
"posted" is the quoted value. Exit status is non-zero if any check fails,
so the script can gate a table rebuild:

    python analysis/check_rebuttal_numbers.py --results-dir results/

Bootstrap intervals depend on the seed and are checked to within 0.03.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tables import load_csv  # noqa: E402
from tables_revision import (load_er, load_network, cell_stats, _calib, _fpr,  # noqa: E402
                             variance_ratio_table, sparsity_match)

CHECKS: list[tuple[str, float, float, float]] = []


def check(desc: str, got: float, posted: float, tol: float = 0.0005):
    """Passes iff |got - posted| <= tol."""
    CHECKS.append((desc, float(got), float(posted), tol))


def check_le(desc: str, got: float, bound: float):
    """One-sided claim ("at most", "calibrated"): passes iff got <= bound."""
    CHECKS.append((desc + " (<=)", float(got), float(bound), float("inf") if got <= bound else -1.0))


def run(results_dir: Path):
    # ── Standalone calibration: Table 1 and Appendix R ──
    c = _calib(results_dir)
    c0 = c[c["signal"] == 0]
    ours = c0[c0["method"] == "Ours"]
    for mech, v in (("complete", 0.018), ("mar", 0.021), ("mnar", 0.036)):
        check(f"Table 1 PAIR-CI {mech}", _fpr(ours, mechanism=mech), v)
    for meth, var, mech, v in (("Fisher-Z", "Rubin's Rules", "mar", 0.058), ("Fisher-Z", "Imputed", "mar", 0.103),
                               ("GCM", "Imputed", "mar", 0.254), ("KCI", "Imputed", "mar", 0.329),
                               ("Fisher-Z", "Rubin's Rules", "mnar", 0.282), ("KCI", "Imputed", "mnar", 0.447),
                               ("KCI", "Imputed", "complete", 0.278)):
        check(f"Table 1 {meth}/{var} {mech}", _fpr(c0, method=meth, variant=var, mechanism=mech), v)
    for meth, vals in (("Fisher-Z", (0.050, 0.049, 0.050)), ("GCM", (0.215, 0.192, 0.191)), ("KCI", (0.147, 0.136, 0.158))):
        for mech, v in zip(("complete", "mar", "mnar"), vals):
            check(f"Deletion row {meth} {mech}", _fpr(c0, method=meth, variant="Complete Case", mechanism=mech), v)
    for mech, vals in (("complete", (0.007, 0.016, 0.022, 0.026)), ("mar", (0.000, 0.021, 0.027, 0.029)),
                       ("mnar", (0.018, 0.037, 0.049, 0.034))):
        for n, v in zip((500, 1000, 2000, 5000), vals):
            check(f"PAIR-CI FPR {mech} n={n}", _fpr(ours, mechanism=mech, n=n), v)
    check("PAIR-CI max mech x |X| cell", ours.groupby(["mechanism", "dim_x"])["rejected"].mean().max(), 0.067)
    check("PAIR-CI max mech x n x |X| cell", ours.groupby(["mechanism", "n", "dim_x"])["rejected"].mean().max(), 0.077)
    o3 = c[(c["method"] == "Ours") & (c["signal"] == 0.3) & (c["mechanism"] == "mnar")]
    for n, v in ((500, 0.28), (1000, 0.54), (2000, 0.75)):
        check(f"PAIR-CI power 0.3 MNAR n={n}", _fpr(o3, n=n), v, 0.005)
    gcm_cc = c0[(c0["method"] == "GCM") & (c0["variant"] == "Complete Case") & (c0["mechanism"] == "complete")]
    for n, v in ((500, 0.152), (1000, 0.218), (2000, 0.211), (5000, 0.260)):
        check(f"GCM complete-data FPR n={n}", _fpr(gcm_cc, n=n), v)
    for dgp, v in (("linear_gaussian", 0.329), ("post_nonlinear", 0.264), ("latent_confounder", 0.054)):
        check(f"GCM complete-data FPR {dgp}", _fpr(gcm_cc, dgp=dgp), v)
    kci_cc = c0[(c0["method"] == "KCI") & (c0["variant"] == "Complete Case") & (c0["mechanism"] == "complete")]
    check("KCI complete-case min over DGP", kci_cc.groupby("dgp")["rejected"].mean().min(), 0.127)
    check("KCI complete-case max over DGP", kci_cc.groupby("dgp")["rejected"].mean().max(), 0.173)
    fzr_mar = c0[(c0["method"] == "Fisher-Z") & (c0["variant"] == "Rubin's Rules") & (c0["mechanism"] == "mar")]
    for dgp, v in (("linear_gaussian", 0.055), ("post_nonlinear", 0.060), ("latent_confounder", 0.060)):
        check(f"FZ-Rubin MAR {dgp} (corrected generator)", _fpr(fzr_mar, dgp=dgp), v)
    lat = c[(c["method"] == "Ours") & (c["dgp"] == "latent_confounder") & (c["mechanism"] == "mnar")]
    for sig, v in ((0.0, 0.000), (0.3, 0.754), (0.6, 0.982), (1.0, 0.995)):
        check(f"PAIR-CI latent-MNAR signal {sig}", _fpr(lat, signal=sig), v)

    # ── KCI/GCM configuration audit and GCM-OLS (Appendix R) ──
    a = load_csv(results_dir, "kci_gcm_audit.csv")
    a = a[~a["failed"].astype(bool)]
    a0, a3 = a[a["signal"] == 0], a[a["signal"] == 0.3]
    check("audit KCI default pooled", _fpr(a0, config="kci_default"), 0.144)
    check("audit KCI tuned (spectral) pooled", _fpr(a0, config="kci_median_specnull"), 0.072)
    check("audit KCI tuned (gamma) pooled", _fpr(a0, config="kci_median"), 0.078)
    check("audit KCI tuned n=500", _fpr(a0, config="kci_median_specnull", n=500), 0.082)
    check("audit KCI tuned n=1000", _fpr(a0, config="kci_median_specnull", n=1000), 0.062)
    check("audit KCI tuned |X|=10", _fpr(a0, config="kci_median_specnull", dim_x=10), 0.102)
    check("audit KCI original regime n=1000 |X|<=5", _fpr(a0[a0["dim_x"] <= 5], config="kci_median_specnull", n=1000), 0.044)
    check("audit KCI tuned power n=500", _fpr(a3, config="kci_median_specnull", n=500), 0.98, 0.005)
    check("audit KCI tuned power n=1000", _fpr(a3, config="kci_median_specnull", n=1000), 1.00, 0.005)
    g = a0[a0["family"] == "gcm"]
    cfg = g.groupby("config")["rejected"].mean()
    check("audit GCM configs min", cfg.min(), 0.21, 0.005)
    check("audit GCM configs max", cfg.max(), 0.23, 0.005)
    lin = g[g["dgp"] == "linear_gaussian"].groupby("config")["rejected"].mean()
    check("audit GCM linear min", lin.min(), 0.33, 0.005)
    check("audit GCM linear max", lin.max(), 0.36, 0.005)
    check("audit GCM n=500 (pooled configs)", _fpr(g, n=500), 0.197)
    check("audit GCM n=5000", _fpr(g, n=5000), 0.248)
    check("audit GCM |X|=2", _fpr(g, dim_x=2), 0.071)
    check("audit GCM |X|=10", _fpr(g, dim_x=10), 0.423)
    o = load_csv(results_dir, "gcm_ols_audit.csv")
    o = o[~o["failed"].astype(bool)]
    check("GCM-OLS linear", _fpr(o, signal=0.0, dgp="linear_gaussian"), 0.048)
    cells = o[o["signal"] == 0].groupby(["dgp", "n", "dim_x"])["rejected"].mean()
    check("GCM-OLS cells min", cells.min(), 0.030)
    check("GCM-OLS cells max", cells.max(), 0.070)
    check("GCM-OLS tests", len(o), 19800, 0)

    # ── Conservatism of the combined variance estimator (Appendix R) ──
    R = variance_ratio_table(results_dir)
    check("T-ratio raw cells > 1", (R["ratio_raw"] > 1).sum(), 21, 0)
    check("T-ratio raw max", R["ratio_raw"].max(), 5.6, 0.05)
    check("T-ratio robust min", R["ratio_robust"].min(), 3.0, 0.05)
    check("T-ratio robust max", R["ratio_robust"].max(), 6.1, 0.05)
    check("T-ratio robust all > 1", (R["ratio_robust"] > 1).sum(), 24, 0)
    worst = R.loc[R["fpr_t"].idxmax()]
    check("above-nominal cell FPR", worst["fpr_t"], 0.072)
    check("above-nominal cell raw ratio (~0.8)", worst["ratio_raw"], 0.72, 0.05)
    check("FPR at raw ratio ~3.5", R[(R["ratio_raw"] > 3.4) & (R["ratio_raw"] < 3.6)]["fpr_t"].mean(), 0.004, 0.001)

    # ── Networks incl. MVPC (Table 3, Apps H-I); imputer quality (App S) ──
    for net, key, shd, f1 in (("alarm_linear", "ours", 22.5, 0.89), ("alarm_linear", "mvpc", 20.0, 0.93),
                              ("alarm_linear", "testwise", 34.0, 0.84),
                              ("alarm_nonlinear", "ours", 33.0, 0.80), ("alarm_nonlinear", "mvpc", 40.0, 0.65),
                              ("alarm_nonlinear", "testwise", 48.0, 0.62),
                              ("hailfinder", "ours", 64.0, 0.59), ("hailfinder", "mvpc", 72.0, 0.57),
                              ("hailfinder", "testwise", 103.5, 0.49)):
        d, _ = load_network(results_dir, net)
        st = cell_stats(d[(d["method_key"] == key) & (d["rate"] == 0.2) & (d["mechanism"] == "mnar")])
        check(f"{net} {key} MNAR 20% SHD", st["shd"], shd, 0.05)
        check(f"{net} {key} MNAR 20% F1", st["f1"], f1, 0.005)
    d, failed = load_network(results_dir, "hailfinder")
    for key, prec, rec, edges in (("ours", 0.73, 0.48, 42), ("testwise", 0.43, 0.58, 88.5),
                                  ("impute_fisherz", 0.38, 0.61, 108.5), ("complete_case", 0.00, 0.00, 0)):
        st = cell_stats(d[(d["method_key"] == key) & (d["rate"] == 0.2) & (d["mechanism"] == "mnar")])
        check(f"HAILFINDER {key} precision", st["prec"], prec, 0.005)
        check(f"HAILFINDER {key} recall", st["rec"], rec, 0.005)
        check(f"HAILFINDER {key} edges", st["edges"], edges, 0.05)
    check("HAILFINDER FZ-vote MNAR SHD", cell_stats(d[(d["method_key"] == "impute_fisherz") & (d["rate"] == 0.2) & (d["mechanism"] == "mnar")])["shd"], 122.0, 0.05)
    check("HAILFINDER CC MNAR SHD", cell_stats(d[(d["method_key"] == "complete_case") & (d["rate"] == 0.2) & (d["mechanism"] == "mnar")])["shd"], 66.0, 0.05)
    mv40 = cell_stats(d[(d["method_key"] == "mvpc") & (d["rate"] == 0.4) & (d["mechanism"] == "mnar")])
    check("HAILFINDER MVPC 40% F1", mv40["f1"], 0.57, 0.005)
    check("HAILFINDER MVPC 40% SHD", mv40["shd"], 69, 0.05)
    check("HAILFINDER MVPC DNF in MNAR 20%", len(failed[(failed["method_key"] == "mvpc") & (failed["rate"] == 0.2) & (failed["mechanism"] == "mnar")]), 1, 0)
    dn, _ = load_network(results_dir, "alarm_nonlinear")
    dl, _ = load_network(results_dir, "alarm_linear")
    check("ALARM nonlinear PAIR-CI precision (MNAR)", cell_stats(dn[(dn["method_key"] == "ours") & (dn["rate"] == 0.2) & (dn["mechanism"] == "mnar")])["prec"], 0.94, 0.005)
    mv_prec = [cell_stats(dn[(dn["method_key"] == "mvpc") & (dn["rate"] == 0.2) & (dn["mechanism"] == m)])["prec"] for m in ("mar", "mnar", "mixed")]
    check("ALARM nonlinear MVPC precision, min over mechanisms", min(mv_prec), 0.71, 0.005)
    check("ALARM nonlinear MVPC precision, max over mechanisms", max(mv_prec), 0.76, 0.005)
    mv_lin = [cell_stats(dl[(dl["method_key"] == "mvpc") & (dl["rate"] == 0.2) & (dl["mechanism"] == m)])["prec"] for m in ("mar", "mnar", "mixed")]
    check("ALARM linear MVPC precision (median over mechanisms)", float(np.median(mv_lin)), 0.95, 0.01)
    q = load_csv(results_dir, "imputer_quality_calibration.csv")
    mi, dr = q[q["m"] == -1], q[q["m"] >= 0]
    check("imputer R2 MAR (mean imputation)", mi[mi["mechanism"] == "mar"]["r2"].median(), 0.19, 0.005)
    dgpmed = mi[mi["mechanism"] == "mnar"].groupby("dgp")["r2"].median()
    check("imputer R2 MNAR per-DGP min", dgpmed.min(), -5.6, 0.05)
    check("imputer R2 MNAR per-DGP max", dgpmed.max(), -1.5, 0.05)
    lm = dr[(dr["dgp"] == "latent_confounder") & (dr["mechanism"] == "mnar")]
    check("latent-MNAR corr(delta,Z)", lm["corr_delta_z"].median(), -0.44, 0.005)
    check("latent-MNAR corr(delta,Y)", abs(lm["corr_delta_y"].median()), 0.00, 0.005)
    check("max |corr(delta,Y)| over cells", dr.groupby(["dgp", "mechanism", "n", "dim_x"])["corr_delta_y"].median().abs().max(), 0.011, 0.001)
    kci_lat = c0[(c0["method"] == "KCI") & (c0["variant"] == "Imputed") & (c0["mechanism"] == "mnar") & (c0["dgp"] == "latent_confounder")]
    check("KCI imputed latent-MNAR FPR", kci_lat["rejected"].mean(), 0.320)

    # ── Sparsity matching and p = 10 paired contrasts (Appendix T) ──
    S = sparsity_match(results_dir)
    an = S[S["cell"].str.startswith("ALARM (nonlinear)")].set_index(["method", "matched"])
    check("ALARM-nl PAIR-CI edges", an.loc[("ours", True), "edges"], 33.5, 0.05)
    for key, v in (("testwise", 0.68), ("micd_pooled", 0.68), ("complete_case", 0.49)):
        check(f"ALARM-nl sparsity-matched F1 {key}", an.loc[(key, True), "f1"], v, 0.005)
    check("ALARM-nl best swept F1", an.xs(False, level="matched").drop("mvpc")["f1"].max(), 0.68, 0.005)
    check("ALARM-nl MVPC native F1", an.loc[("mvpc", False), "f1"], 0.65, 0.005)
    hf = S[S["cell"].str.startswith("HAILFINDER")].set_index(["method", "matched"])
    check("HAILFINDER best swept F1 (test-wise)", hf.loc[("testwise", False), "f1"], 0.54, 0.005)
    check("HAILFINDER test-wise min edges", hf.loc[("testwise", False), "edges"], 71, 0.5)
    check("HAILFINDER MVPC edges", hf.loc[("mvpc", False), "edges"], 63, 0.5)
    al = load_csv(results_dir, "alpha_sensitivity_results.csv")
    al = al[(~al["failed"].astype(bool)) & (al["mechanism"] == "mnar") & (al["alpha"] == 0.05)]
    check("p=10 PAIR-CI precision MNAR", al["precision"].median(), 1.00, 0.005)
    check("p=10 PAIR-CI recall MNAR", al["recall"].median(), 0.56, 0.005)
    for dgp, v in (("linear", -0.74), ("nonlinear", -0.75)):
        df = load_er(results_dir, 10, dgp)
        df = df[df["mechanism"].isin(["mar", "mnar", "mixed"])]
        piv = df.pivot_table(index=["graph", "dataset", "mechanism"], columns="method_key", values="shd")
        check(f"p=10 {dgp} mean paired SHD diff vs test-wise", (piv["ours"] - piv["testwise"]).mean(), v, 0.005)
    bs = load_csv(results_dir, "paired_bootstrap_contrasts.csv", required=False)
    if bs is not None:
        for dgp, lo, hi in (("linear", -1.05, -0.42), ("nonlinear", -1.23, -0.25)):
            r = bs[(bs["cell"] == f"ER {dgp} p=10") & (bs["metric"] == "shd") & (bs["baseline"] == "testwise")].iloc[0]
            check(f"p=10 {dgp} bootstrap CI lower (seed-dependent)", r["ci_lo"], lo, 0.03)
            check(f"p=10 {dgp} bootstrap CI upper (seed-dependent)", r["ci_hi"], hi, 0.03)

    # ── Learner robustness (Appendix S) and the kappa fit (Appendix N) ──
    l = load_csv(results_dir, "learner_robustness.csv")
    l = l[~l["failed"].astype(bool)]
    l0, l3 = l[l["signal"] == 0], l[l["signal"] == 0.3]
    check("learner tests", len(l), 36000, 0)
    check("ridge FPR complete", _fpr(l0, learner="ridge", mechanism="complete"), 0.000)
    check("ridge FPR MAR", _fpr(l0, learner="ridge", mechanism="mar"), 0.002, 0.0015)
    check("ridge FPR MNAR", _fpr(l0, learner="ridge", mechanism="mnar"), 0.080, 0.0015)
    check("stump FPR pooled", _fpr(l0, learner="rf10_stump"), 0.022)
    st_pw = l3[l3["learner"] == "rf10_stump"].groupby("dgp")["rejected"].mean()
    check("stump power min", st_pw.min(), 0.07, 0.005)
    check("stump power max", st_pw.max(), 0.12, 0.005)
    check("shallow RF FPR linear-Gaussian", _fpr(l0, learner="rf25_shallow", dgp="linear_gaussian"), 0.078)
    check("kNN FPR linear-Gaussian", _fpr(l0, learner="knn", dgp="linear_gaussian"), 0.108)
    kt = load_csv(results_dir, "kappa_trace.csv")
    prod = kt["kappa_imp"] ** 2 * kt["kappa_Y"].abs() * kt["kappa_Z"].abs()
    check("kappa product Spearman", stats.spearmanr(prod, kt["fpr"])[0], 0.89, 0.005)
    check("kappa cells", len(kt), 48, 0)
    infl = kt["fpr"] > 0.05
    check("kappa product median calibrated", prod[~infl].median(), 0.002, 0.0005)
    check("kappa product median inflated", prod[infl].median(), 0.18, 0.005)
    x, y = np.log(prod[infl]), np.log(kt.loc[infl, "fpr"] - 0.05)
    res = stats.linregress(x, y)
    check("kappa log-log slope", res.slope, 0.69, 0.005)
    check("kappa log-log slope SE", res.stderr, 0.15, 0.005)

    # ── Test-variable missingness (Appendix T) and the placebo diagnostic ──
    t = load_csv(results_dir, "test_variable_missingness.csv")
    t = t[(~t["failed"].astype(bool)) & (t["signal"] == 0) & (t["rule"] == "imputed_labels")]
    cells = t.groupby(["dgp", "style", "miss_on"])["rejected"].mean()
    check("test-var missingness cells", len(cells), 16, 0)
    check("test-var missingness min", cells.min(), 0.000)
    check("test-var missingness max", cells.max(), 0.068)
    p = pd.concat([load_csv(results_dir, f"placebo_diag_{d}.csv") for d in ("linear_gaussian", "post_nonlinear", "latent_confounder")])
    p = p[~p["failed"].astype(bool)]
    check("placebo tests", len(p), 8100, 0)
    check("placebo oracle FPR (all cells)", p[p["arm"] == "oracle"]["rejected"].max(), 0.000)
    lg = p[p["dgp"] == "linear_gaussian"]
    for n, v in ((1000, 0.080), (2000, 0.187), (5000, 0.100)):
        check(f"placebo deployed linear-Gaussian n={n}", _fpr(lg, arm="deployed", n=n), v)
    check("placebo linear-Gaussian |X|=10 max deployed FPR", lg[lg["arm"] == "deployed"].groupby("n")["rejected"].mean().max(), 0.187)

    # ── Further numbers quoted in the text ──
    check("GCM audit test evaluations", len(a[a["family"] == "gcm"]), 172800, 0)
    lat_by_mech = c0[(c0["method"] == "Ours") & (c0["dgp"] == "latent_confounder")].groupby("mechanism")["rejected"].mean()
    check("PAIR-CI latent-confounder FPR min over mechanisms", lat_by_mech.min(), 0.000)
    check("PAIR-CI latent-confounder FPR max over mechanisms", lat_by_mech.max(), 0.001)
    t4 = c0[(c0["method"] == "Ours") & (c0["mechanism"].isin(["mar", "mnar"]))].groupby(["dgp", "mechanism"])["rejected"].mean()
    check("Table 4 PAIR-CI max cell (0.070)", t4.max(), 0.070)
    d, _ = load_network(results_dir, "hailfinder")
    hf20 = d[(d["rate"] == 0.2) & (d["mechanism"].isin(["mar", "mnar", "mixed"]))]
    par = hf20[hf20["method_key"].isin(["testwise", "impute_fisherz", "micd_pooled"])].groupby(["method_key", "mechanism"])
    check("HAILFINDER parametric baselines F1 min (0.45)", par["f1"].median().min(), 0.45, 0.005)
    check("HAILFINDER parametric baselines F1 max (0.49)", par["f1"].median().max(), 0.49, 0.005)
    # "89-112 edges ... 57-64% false" are ranges over MAR/MNAR/mixed at 20%.
    check("HAILFINDER 20% parametric edges min over mechs (89)", par["edges"].median().min(), 89, 0.5)
    check("HAILFINDER 20% parametric edges max over mechs (112)", par["edges"].median().max(), 112, 0.5)
    check("HAILFINDER 20% parametric false-edge share min (57%)", (1 - par["precision"].median()).min(), 0.57, 0.005)
    check("HAILFINDER 20% parametric false-edge share max (64%)", (1 - par["precision"].median()).max(), 0.64, 0.005)
    dn, _ = load_network(results_dir, "alarm_nonlinear")
    an20 = dn[(dn["rate"] == 0.2) & (dn["mechanism"].isin(["mar", "mnar", "mixed"]))]
    o = an20[an20["method_key"] == "ours"].groupby("mechanism")
    check("ALARM-nl PAIR-CI F1 min over mechs (0.79)", o["f1"].median().min(), 0.79, 0.005)
    check("ALARM-nl PAIR-CI F1 max over mechs (0.80)", o["f1"].median().max(), 0.80, 0.005)
    check("ALARM-nl PAIR-CI recall min (0.67)", o["recall"].median().min(), 0.67, 0.005)
    check("ALARM-nl PAIR-CI recall max (0.70)", o["recall"].median().max(), 0.70, 0.005)
    # 0.49-0.65 is the range over the four baselines other than MVPC; MVPC
    # reaches 0.66-0.69 under MAR and mixed, so the paper quotes 0.49-0.69.
    b4 = an20[an20["method_key"].isin(["complete_case", "testwise", "impute_fisherz", "micd_pooled"])].groupby(["method_key", "mechanism"])
    check("ALARM-nl four non-MVPC baselines F1 min (0.49)", b4["f1"].median().min(), 0.49, 0.005)
    check("ALARM-nl four non-MVPC baselines F1 max (0.65)", b4["f1"].median().max(), 0.65, 0.005)
    b = an20[an20["method_key"].isin(["complete_case", "testwise", "impute_fisherz", "micd_pooled", "mvpc"])].groupby(["method_key", "mechanism"])
    check("ALARM-nl all baselines incl. MVPC F1 max (0.69 -- paper's range)", b["f1"].median().max(), 0.69, 0.005)
    check_le("ALARM-nl every baseline recall <= PAIR-CI's min recall", b["recall"].median().max(), o["recall"].median().min() + 0.005)
    er30 = load_er(results_dir, 30, "linear")
    mv30 = er30[(er30["method_key"] == "mvpc") & (er30["mechanism"].isin(["mar", "mnar", "mixed"]))].groupby("mechanism")["shd"].median()
    check("ER linear p=30 MVPC SHD min under missingness (48)", mv30.min(), 48, 0.5)
    check("ER linear p=30 MVPC SHD max under missingness (49)", mv30.max(), 49, 0.5)
    o30 = er30[er30["method_key"] == "ours"].groupby("mechanism")["shd"].median()
    check("ER linear p=30 PAIR-CI SHD (53, median of mechanism medians)", o30.median(), 53, 0.5)
    mv_all = pd.concat([load_csv(results_dir, f"mvpc_p{p}_{dgp}_{m}.csv") for p in (10, 20, 30) for dgp in ("linear", "nonlinear") for m in ("mar", "mnar", "mixed", "complete")]
                       + [load_csv(results_dir, f) for f in ("mvpc_alarm_linear.csv", "mvpc_alarm_nonlinear.csv", "mvpc_hailfinder_nonlinear.csv")])
    check_le("MVPC DNF rate <= 1%", mv_all["failed"].astype(bool).mean(), 0.01)
    cc_rows = c0[(c0["mechanism"] != "complete") & (c0["dim_x"] >= 5) & (c0["method"] == "Fisher-Z") & (c0["variant"] == "Complete Case")]
    check("deletion retains roughly half the rows", (cc_rows["n_complete"] / cc_rows["n"]).mean(), 0.5, 0.03)
    ms = load_csv(results_dir, "m_sensitivity_results.csv")
    ms = ms[(~ms["failed"].astype(bool)) & (ms["signal"] == 0.3)]
    pw = ms.groupby(["M", "mechanism"])["rejected"].mean().unstack("M")
    check_le("M: 3->10 moves power by at most 6pp", (pw[10] - pw[3]).abs().max(), 0.06)
    lc = load_csv(results_dir, "learner_comparison_results.csv")
    lc = lc[~lc["failed"].astype(bool)]
    check("ExtraTrees FPR under MNAR 'slightly hot' (0.09)", _fpr(lc, signal=0.0, learner="extra_trees", mechanism="mnar"), 0.09, 0.005)
    for lr in ("rf", "lgbm", "lgbm_tuned", "xgboost"):
        check_le(f"{lr} calibrated standalone (max FPR over mechanisms)", lc[(lc["signal"] == 0) & (lc["learner"] == lr)].groupby("mechanism")["rejected"].mean().max(), 0.06)
    check_le("ridge full power where the signal is linear (>= 0.99)", 1 - _fpr(l3, learner="ridge", dgp="linear_gaussian"), 0.01)
    se = load_csv(results_dir, "sensitivity_results.csv")
    se = se[(~se["failed"].astype(bool)) & se["config"].str.startswith("M=")]
    cfg = se.groupby("config")["shd"].median()
    check("E.1 grid configurations (11)", len(cfg), 11, 0)
    check_le("E.1 grid SHD spread within +-2", cfg.max() - cfg.min(), 2)
    al_all = load_csv(results_dir, "alpha_sensitivity_results.csv")
    al_all = al_all[~al_all["failed"].astype(bool)]
    f1a = al_all.groupby(["mechanism", "alpha"])["f1"].median().unstack("alpha")
    check_le("alpha sweep: no rank reversal in F1 ordering across mechanisms", float((f1a.loc["mar"].rank() - f1a.loc["mnar"].rank()).abs().max()), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    run(Path(args.results_dir))
    n_fail = 0
    for desc, got, posted, tol in CHECKS:
        ok = abs(got - posted) <= tol + 1e-12
        n_fail += not ok
        flag = "ok  " if ok else "FAIL"
        print(f"  {flag}  {desc:<52} got {got:>9.4f}  posted {posted:>8.4f}  tol {tol:g}")
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
