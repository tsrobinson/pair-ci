"""
Table builders for the camera-ready paper, collected in REVISION_TABLES.

build_tables.py merges these into its registry, so

    python analysis/build_tables.py --output-dir tables/

builds every table. A builder named like one in build_tables.py
(calibration, scaling, realworld, alarm_full, alarm_full_nonlinear,
hailfinder_full, hailfinder_f1, shd_linear, shd_nonlinear) replaces it
with a layout that includes MVPC. check_rebuttal_numbers.py checks the
numbers quoted in the text against the same loaders.

Conventions, as in build_tables.py: medians per cell for graph metrics;
mean rejection rate at signal 0 for FPR; failed replicates dropped (counts
reported in the notes where they matter). Edge counts are tp + fp when
logged, otherwise back-derived from precision, recall and the true edge
count (some ALARM and HAILFINDER result files log only precision and
recall).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from build_tables import (
    load_csv, load_p_sweep, load_newbase_p_sweep, ours_row,
    compute_skeleton_shd, fmt_int, fmt_dec1, fmt_frac, red, bold,
    gray_dagger, bold_lowest,
)

REVISION_TABLES: dict[str, callable] = {}


def register(name: str):
    def decorator(fn):
        REVISION_TABLES[name] = fn
        return fn
    return decorator


# ── Shared loaders ───────────────────────────────────────────────────

NETWORKS = {
    "alarm_linear": dict(base="alarm_results.csv", nb="newbase_alarm_linear.csv",
                         mvpc="mvpc_alarm_linear.csv", n_true=46,
                         label="ALARM (linear)"),
    "alarm_nonlinear": dict(base="alarm_nonlinear_results.csv",
                            nb="newbase_alarm_nonlinear.csv",
                            mvpc="mvpc_alarm_nonlinear.csv", n_true=46,
                            label="ALARM (nonlinear)"),
    "hailfinder": dict(base="hailfinder_results.csv",
                       nb="newbase_hailfinder_nonlinear.csv",
                       mvpc="mvpc_hailfinder_nonlinear.csv", n_true=66,
                       label="HAILFINDER"),
}

CONSTRAINT_METHODS = [
    ("PAIR-CI", "ours"),
    ("Complete case", "complete_case"),
    ("Test-wise", "testwise"),
    ("FZ-vote", "impute_fisherz"),
    (r"FZ-Rubin (\citeauthor{witte2022multiple})", "micd_pooled"),
    (r"MVPC$^\ddagger$ (\citeauthor{tu2019causal})", "mvpc"),
]
SCORE_METHODS = [(r"MissDAG$^\dagger$ (\citeauthor{gao2022missdag})", "missdag")]
SHORT_LABELS = {"ours": "P-CI", "complete_case": "CC", "testwise": "TW",
                "impute_fisherz": "FZ-v", "micd_pooled": "FZ-R",
                "mvpc": r"MVPC$^\ddagger$", "missdag": r"MissDAG$^\dagger$"}
MECH_LABELS = {"complete": "Complete", "mar": "MAR", "mnar": "MNAR", "mixed": "Mixed"}


def _norm_method(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method_key"] = df["method"].where(
        ~df["method"].isin(["ci_oracle_fast", "ci_oracle_cached"]), "ours")
    return df


def edges_out(sub: pd.DataFrame, n_true: float | None = None) -> pd.Series:
    """Median-able output edge count per replicate: tp + fp, else back-derived."""
    if "n_edges_out" in sub.columns and sub["n_edges_out"].notna().all():
        return sub["n_edges_out"].astype(float)
    out = pd.Series(np.nan, index=sub.index, dtype=float)
    if {"tp", "fp"}.issubset(sub.columns):
        out = (sub["tp"] + sub["fp"]).astype(float)
    nan = out.isna()
    if nan.any():
        edges = sub["n_true_edges"] if "n_true_edges" in sub.columns else pd.Series(n_true, index=sub.index)
        edges = edges.fillna(n_true) if n_true is not None else edges
        tp = sub.loc[nan, "recall"] * edges.loc[nan]
        fp = tp * (1 / sub.loc[nan, "precision"].clip(1e-10) - 1)
        out.loc[nan] = tp + fp
    return out


def load_network(results_dir: Path, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All methods for one network. Returns (clean rows, failed rows)."""
    spec = NETWORKS[key]
    parts = []
    base = load_csv(results_dir, spec["base"])
    if base is not None:
        parts.append(base)
    for name in (spec["nb"], spec["mvpc"]):
        d = load_csv(results_dir, name, required=False)
        if d is not None:
            parts.append(d)
    df = _norm_method(pd.concat(parts, ignore_index=True, sort=False))
    if "n_true_edges" not in df.columns:
        df["n_true_edges"] = spec["n_true"]
    df["n_true_edges"] = df["n_true_edges"].fillna(spec["n_true"])
    df["edges"] = edges_out(df, spec["n_true"])
    df["skel"] = compute_skeleton_shd(df, n_true_edges=spec["n_true"])
    failed = df[df["failed"].astype(bool)]
    return df[~df["failed"].astype(bool)], failed


def load_er(results_dir: Path, p: int, dgp: str) -> pd.DataFrame | None:
    """All methods for one Erdős–Rényi (p, dgp) cell, failed rows dropped."""
    parts = []
    base = load_p_sweep(results_dir, p, dgp)
    if base is not None:
        parts.append(base)
    nb = load_newbase_p_sweep(results_dir, p, dgp)
    if nb is not None:
        parts.append(nb)
    for mech in ("mar", "mnar", "mixed", "complete"):
        d = load_csv(results_dir, f"mvpc_p{p}_{dgp}_{mech}.csv", required=False)
        if d is not None:
            parts.append(d[~d["failed"].astype(bool)])
    if not parts:
        return None
    df = _norm_method(pd.concat(parts, ignore_index=True, sort=False))
    df["edges"] = edges_out(df)
    df["skel"] = compute_skeleton_shd(df)
    return df


def cell_stats(sub: pd.DataFrame) -> dict:
    """Medians of the graph metrics over the rows of one cell, and the IQR
    of SHD."""
    if len(sub) == 0:
        return dict(shd=np.nan, skel=np.nan, prec=np.nan, rec=np.nan, f1=np.nan,
                    edges=np.nan, n=0, shd_iqr=np.nan)
    q1, q3 = sub["shd"].quantile([0.25, 0.75])
    return dict(shd=sub["shd"].median(), skel=sub["skel"].median(),
                prec=sub["precision"].median(), rec=sub["recall"].median(),
                f1=sub["f1"].median(), edges=sub["edges"].median(), n=len(sub),
                shd_iqr=q3 - q1)


def _fmt_metric(metric: str, v: float) -> str:
    if pd.isna(v):
        return "--"
    if metric in ("shd", "edges"):
        return fmt_dec1(v) if not float(v).is_integer() else str(int(v))
    if metric == "skel":
        return fmt_int(v)
    return fmt_frac(v, 2)


def _calib(results_dir: Path) -> pd.DataFrame:
    df = load_csv(results_dir, "ci_calibration_results.csv")
    return df[~df["failed"].astype(bool)]


def _fpr(df: pd.DataFrame, **sel) -> float:
    """Mean of `rejected` over the rows matching column == value for each
    keyword; NaN if none match."""
    sub = df
    for k, v in sel.items():
        sub = sub[sub[k] == v]
    return sub["rejected"].mean() if len(sub) else np.nan


# ── Table 1 and its disaggregations ─────────────────────────────────

@register("calibration")
def build_calibration(results_dir: Path) -> str:
    """Table 1, Panel A: FPR at signal 0, pooled over DGP, n and |X|.

    The Complete column is filled for PAIR-CI and the complete-case
    variants only: the imputed variants are defined under missingness, and
    on complete data the M imputations coincide, so Fisher pooling of
    identical p-values inflates KCI. Tuned KCI (standardised inputs,
    median-heuristic width, spectral null) comes from the complete-data
    audit grid (n <= 1,000), so its MAR and MNAR cells are empty; the
    caption explains its dagger.
    """
    df = _calib(results_dir)
    df = df[df["signal"] == 0]
    audit = load_csv(results_dir, "kci_gcm_audit.csv", required=False)

    def cell(method, variant, mech, is_ours=False):
        v = _fpr(df, method=method, variant=variant, mechanism=mech)
        s = red(fmt_frac(v), v > 0.05)
        return bold(s, is_ours and mech == "mnar")

    rows = [
        (r"PAIR-CI", r"integrated (MICE, $M{=}5$)", "Ours", "Integrated", True, True),
        (r"Fisher-$Z$", "complete-case", "Fisher-Z", "Complete Case", True, False),
        ("", "single imputation", "Fisher-Z", "Imputed", False, False),
        ("", "Rubin's rules", "Fisher-Z", "Rubin's Rules", False, False),
        ("GCM (RF)", "complete-case", "GCM", "Complete Case", True, False),
        ("", "imputed", "GCM", "Imputed", False, False),
        ("KCI (default)", "complete-case", "KCI", "Complete Case", True, False),
        ("", "imputed", "KCI", "Imputed", False, False),
    ]
    lines = [
        r"\begin{tabular}{llccc}",
        r"    \toprule",
        r"    Test & Variant & Complete & MAR & MNAR \\",
        r"    \midrule",
    ]
    prev_test = None
    for test, variant, m, v, has_complete, is_ours in rows:
        if test and prev_test is not None:
            lines.append(r"    \midrule")
        prev_test = test or prev_test
        comp = cell(m, v, "complete") if has_complete else "--"
        lines.append(f"    {test:<14} & {variant:<28} & {comp} & "
                     f"{cell(m, v, 'mar', is_ours)} & {cell(m, v, 'mnar', is_ours)} \\\\")
    if audit is not None:
        a = audit[(~audit["failed"].astype(bool)) & (audit["signal"] == 0)]
        v = a[a["config"] == "kci_median_specnull"]["rejected"].mean()
        lines.append(f"    KCI (tuned)$^\\dagger$ & complete-case & {red(fmt_frac(v), v > 0.05)} & -- & -- \\\\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("calibration_by_n")
def build_calibration_by_n(results_dir: Path) -> str:
    """PAIR-CI FPR by mechanism x n, pooled over DGP and |X| (not in the
    paper; Table 1, Panel B covers it)."""
    df = _calib(results_dir)
    o = df[(df["method"] == "Ours") & (df["signal"] == 0)]
    ns = (500, 1000, 2000, 5000)
    lines = [r"\begin{tabular}{lcccc}", r"    \toprule",
             r"    Mechanism & $n{=}500$ & $n{=}1{,}000$ & $n{=}2{,}000$ & $n{=}5{,}000$ \\",
             r"    \midrule"]
    for mech in ("complete", "mar", "mnar"):
        cells = [red(fmt_frac(_fpr(o, mechanism=mech, n=n)), _fpr(o, mechanism=mech, n=n) > 0.05) for n in ns]
        lines.append(f"    {MECH_LABELS[mech]:<9} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("calibration_by_dimx")
def build_calibration_by_dimx(results_dir: Path) -> str:
    """PAIR-CI FPR by mechanism x |X|, pooled over DGP and n (not in the
    paper)."""
    df = _calib(results_dir)
    o = df[(df["method"] == "Ours") & (df["signal"] == 0)]
    lines = [r"\begin{tabular}{lccc}", r"    \toprule",
             r"    Mechanism & $|\Xb|{=}2$ & $|\Xb|{=}5$ & $|\Xb|{=}10$ \\", r"    \midrule"]
    for mech in ("complete", "mar", "mnar"):
        cells = []
        for d in (2, 5, 10):
            v = _fpr(o, mechanism=mech, dim_x=d)
            cells.append(red(fmt_frac(v), v > 0.05))
        lines.append(f"    {MECH_LABELS[mech]:<9} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("calibration_full")
def build_calibration_full(results_dir: Path) -> str:
    """Appendix R: PAIR-CI FPR in every mechanism x n x |X| cell, pooled
    over the three DGPs (300 tests per cell; n=500 x |X|=10 is not in the
    grid). Cells above the nominal level are in red."""
    df = _calib(results_dir)
    o = df[(df["method"] == "Ours") & (df["signal"] == 0)]
    lines = [r"\begin{tabular}{llccc}", r"    \toprule",
             r"    Mechanism & $n$ & $|\Xb|{=}2$ & $|\Xb|{=}5$ & $|\Xb|{=}10$ \\", r"    \midrule"]
    for mi, mech in enumerate(("complete", "mar", "mnar")):
        for ni, n in enumerate((500, 1000, 2000, 5000)):
            cells = []
            for d in (2, 5, 10):
                v = _fpr(o, mechanism=mech, n=n, dim_x=d)
                cells.append("--" if pd.isna(v) else red(fmt_frac(v), v > 0.05))
            lab = MECH_LABELS[mech] if ni == 0 else ""
            lines.append(f"    {lab:<9} & {n:,} & " + " & ".join(cells) + r" \\")
        if mi < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("calibration_full_by_dgp")
def build_calibration_full_by_dgp(results_dir: Path) -> str:
    """The same grid split by DGP (100 tests per cell), showing the
    linear-Gaussian |X|=10 cells examined by the placebo diagnostic (not in
    the paper)."""
    df = _calib(results_dir)
    o = df[(df["method"] == "Ours") & (df["signal"] == 0)]
    dgps = [("Lin.\\ Gaussian", "linear_gaussian"), ("Post-nonlinear", "post_nonlinear"),
            ("Latent conf.", "latent_confounder")]
    lines = [r"\begin{tabular}{ll" + "ccc" * 3 + "}", r"    \toprule",
             r"     & & \multicolumn{3}{c}{Complete} & \multicolumn{3}{c}{MAR} & \multicolumn{3}{c}{MNAR} \\",
             r"    \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
             r"    DGP & $n$ & $|\Xb|{=}2$ & 5 & 10 & $|\Xb|{=}2$ & 5 & 10 & $|\Xb|{=}2$ & 5 & 10 \\",
             r"    \midrule"]
    for di, (dlab, dkey) in enumerate(dgps):
        for ni, n in enumerate((500, 1000, 2000, 5000)):
            cells = []
            for mech in ("complete", "mar", "mnar"):
                for d in (2, 5, 10):
                    v = _fpr(o, dgp=dkey, mechanism=mech, n=n, dim_x=d)
                    cells.append("--" if pd.isna(v) else red(fmt_frac(v, 2), v > 0.05))
            lab = dlab if ni == 0 else ""
            lines.append(f"    {lab:<16} & {n:,} & " + " & ".join(cells) + r" \\")
        if di < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _calibration_power(results_dir: Path, power_mechs=("complete", "mar", "mnar")) -> str:
    df = _calib(results_dir)
    o = df[df["method"] == "Ours"]
    ns = (500, 1000, 2000, 5000)
    lines = [r"\begin{tabular}{llcccc}", r"    \toprule",
             r"     & Mechanism & $n{=}500$ & $n{=}1{,}000$ & $n{=}2{,}000$ & $n{=}5{,}000$ \\",
             r"    \midrule"]
    for si, (sig, lab, mechs) in enumerate(((0.0, "FPR (signal 0)", ("complete", "mar", "mnar")),
                                            (0.3, "Power (signal 0.3)", power_mechs))):
        for mi, mech in enumerate(mechs):
            cells = []
            for n in ns:
                v = _fpr(o, signal=sig, mechanism=mech, n=n)
                s = fmt_frac(v)
                cells.append(red(s, v > 0.05) if sig == 0 else s)
            first = lab if mi == 0 else ""
            lines.append(f"    {first:<20} & {MECH_LABELS[mech]:<9} & " + " & ".join(cells) + r" \\")
        if si == 0:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("calibration_power")
def build_calibration_power(results_dir: Path) -> str:
    """PAIR-CI FPR (signal 0) beside power (signal 0.3) by mechanism x n,
    pooled over DGP and |X|, with power for all three mechanisms (not in
    the paper; Table 1 uses the compact version)."""
    return _calibration_power(results_dir)


@register("calibration_power_compact")
def build_calibration_power_compact(results_dir: Path) -> str:
    """Table 1, Panel B: FPR by mechanism x n, and power at signal 0.3
    under MNAR only."""
    return _calibration_power(results_dir, power_mechs=("mnar",))


# ── Baseline audits ─────────────────────────────────────────────────

AUDIT_CONFIGS = [
    ("gcm_50_cv5", "GCM, RF-50, cross-fitted (Table 1)"),
    ("gcm_100_cv5", "GCM, RF-100, cross-fitted"),
    ("gcm_400_cv5", "GCM, RF-400, cross-fitted"),
    ("gcm_50_oob", "GCM, RF-50, honest OOB"),
    ("gcm_100_oob", "GCM, RF-100, honest OOB"),
    ("gcm_400_oob", "GCM, RF-400, honest OOB"),
    ("kci_default", "KCI, default (Table 1)"),
    ("kci_std", "KCI, standardised inputs"),
    ("kci_median", "KCI, + median-heuristic width"),
    ("kci_median_specnull", "KCI, + spectral null (tuned)"),
]


@register("kci_gcm_audit")
def build_kci_gcm_audit(results_dir: Path) -> str:
    """Appendix R: complete-data FPR for GCM and KCI
    configurations, pooled and disaggregated by DGP, n and |X|; power at
    signal 0.3.  GCM: 400 reps/cell over n in {500..5000}; KCI: 200 reps/cell
    over n in {500, 1000} (O(n^3) cost)."""
    a = load_csv(results_dir, "kci_gcm_audit.csv")
    if a is None:
        return "% MISSING: kci_gcm_audit.csv"
    a = a[~a["failed"].astype(bool)]
    a0, a3 = a[a["signal"] == 0], a[a["signal"] == 0.3]
    dgps = ("linear_gaussian", "post_nonlinear", "latent_confounder")
    lines = [r"\begin{tabular}{l" + "c" * 12 + "}", r"    \toprule",
             r"     & & \multicolumn{3}{c}{DGP} & \multicolumn{4}{c}{$n$} & \multicolumn{3}{c}{$|\Xb|$} & \\",
             r"    \cmidrule(lr){3-5} \cmidrule(lr){6-9} \cmidrule(lr){10-12}",
             r"    Configuration & Pooled & Lin.\ G. & Post-nl & Latent & 500 & 1{,}000 & 2{,}000 & 5{,}000 & 2 & 5 & 10 & Power \\",
             r"    \midrule"]
    for i, (cfg, label) in enumerate(AUDIT_CONFIGS):
        s0 = a0[a0["config"] == cfg]
        if len(s0) == 0:
            continue
        vals = [s0["rejected"].mean()]
        vals += [_fpr(s0, dgp=d) for d in dgps]
        vals += [_fpr(s0, n=n) for n in (500, 1000, 2000, 5000)]
        vals += [_fpr(s0, dim_x=d) for d in (2, 5, 10)]
        cells = ["--" if pd.isna(v) else red(fmt_frac(v), v > 0.05) for v in vals]
        pw = a3[a3["config"] == cfg]["rejected"].mean()
        cells.append(fmt_frac(pw))
        if i == 6:
            lines.append(r"    \midrule")
        lines.append(f"    {label:<36} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("gcm_ols")
def build_gcm_ols(results_dir: Path) -> str:
    """Appendix R: GCM with a correctly specified OLS learner,
    complete data, FPR at signal 0 per DGP x n x |X| (300 reps/cell) and
    power at signal 0.3 per DGP."""
    g = load_csv(results_dir, "gcm_ols_audit.csv")
    if g is None:
        return "% MISSING: gcm_ols_audit.csv"
    g = g[~g["failed"].astype(bool)]
    g0, g3 = g[g["signal"] == 0], g[g["signal"] == 0.3]
    dgps = [("Linear Gaussian", "linear_gaussian"), ("Post-nonlinear", "post_nonlinear"),
            ("Latent confounder", "latent_confounder")]
    lines = [r"\begin{tabular}{llcccc}", r"    \toprule",
             r"    DGP & $n$ & $|\Xb|{=}2$ & $|\Xb|{=}5$ & $|\Xb|{=}10$ & Pooled FPR / power \\",
             r"    \midrule"]
    for di, (dlab, dkey) in enumerate(dgps):
        for ni, n in enumerate((500, 1000, 2000, 5000)):
            cells = []
            for d in (2, 5, 10):
                v = _fpr(g0, dgp=dkey, n=n, dim_x=d)
                cells.append("--" if pd.isna(v) else red(fmt_frac(v), v > 0.05))
            tail = f"{fmt_frac(_fpr(g0, dgp=dkey))} / {fmt_frac(_fpr(g3, dgp=dkey))}" if ni == 0 else ""
            lab = dlab if ni == 0 else ""
            lines.append(f"    {lab:<18} & {n:,} & " + " & ".join(cells) + f" & {tail} \\\\")
        if di < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Assumptions in practice (Appendix S) ────────────────────────────

LEARNERS = [
    ("rf100", "RF, 100 trees, leaf 5 (default)"),
    ("rf25_shallow", "RF, 25 trees, depth 3"),
    ("rf10_stump", "RF, 10 stumps (depth 1)"),
    ("ridge", "Ridge regression (linear)"),
    ("knn", r"$k$NN regression, $k{=}10$ (fixed)"),
]


@register("learner_robustness")
def build_learner_robustness(results_dir: Path) -> str:
    """Appendix S: FPR and power at signal 0.3 by learner.
    Grid: 2 DGPs x 3 mechanisms x n in {1000, 2000} x |X|=5 x 3 signals."""
    l = load_csv(results_dir, "learner_robustness.csv")
    if l is None:
        return "% MISSING: learner_robustness.csv"
    l = l[~l["failed"].astype(bool)]
    l0, l3 = l[l["signal"] == 0], l[l["signal"] == 0.3]
    lines = [r"\begin{tabular}{lcccccccc}", r"    \toprule",
             r"     & \multicolumn{3}{c}{FPR by mechanism} & \multicolumn{2}{c}{FPR by DGP} & \multicolumn{3}{c}{Power (0.3) by mechanism} \\",
             r"    \cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}",
             r"    Learner & Complete & MAR & MNAR & Lin.\ G. & Post-nl & Complete & MAR & MNAR \\",
             r"    \midrule"]
    for key, label in LEARNERS:
        s0, s3 = l0[l0["learner"] == key], l3[l3["learner"] == key]
        if len(s0) == 0:
            continue
        vals = [_fpr(s0, mechanism=m) for m in ("complete", "mar", "mnar")]
        vals += [_fpr(s0, dgp=d) for d in ("linear_gaussian", "post_nonlinear")]
        cells = [red(fmt_frac(v), v > 0.05) for v in vals]
        cells += [fmt_frac(_fpr(s3, mechanism=m), 2) for m in ("complete", "mar", "mnar")]
        lines.append(f"    {label:<36} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("imputer_quality")
def build_imputer_quality(results_dir: Path) -> str:
    """Appendix S: imputer accuracy per DGP x mechanism x rate x n
    on the calibration grid, pooled over |X| and signal.  R^2 is for the
    across-M mean imputation on the missing entries (benchmark: the missing
    subpopulation's own variance); NRMSE is per posterior draw against the
    column's marginal SD; corr(delta, Y) and
    corr(delta, Z) are per-draw residual correlations of the imputation error
    with the test variables (the kappa factors)."""
    q = load_csv(results_dir, "imputer_quality_calibration.csv")
    if q is None:
        return "% MISSING: imputer_quality_calibration.csv"
    mi, dr = q[q["m"] == -1], q[q["m"] >= 0]
    dgps = [("Lin.\\ Gaussian", "linear_gaussian"), ("Post-nonlinear", "post_nonlinear"),
            ("Latent conf.", "latent_confounder")]
    ns = (500, 1000, 2000, 5000)
    lines = [r"\begin{tabular}{lll" + "cccc" * 2 + "cc}", r"    \toprule",
             r"     & & & \multicolumn{4}{c}{Mean-imputation $R^2$} & \multicolumn{4}{c}{Per-draw NRMSE} & & \\",
             r"    \cmidrule(lr){4-7} \cmidrule(lr){8-11}",
             r"    DGP & Mech. & Rate & 500 & 1{,}000 & 2{,}000 & 5{,}000 & 500 & 1{,}000 & 2{,}000 & 5{,}000 & $\mathrm{corr}(\delta,Y)$ & $\mathrm{corr}(\delta,Z)$ \\",
             r"    \midrule"]
    for di, (dlab, dkey) in enumerate(dgps):
        for mi_, mech in enumerate(("mar", "mnar")):
            for ri, rate in enumerate((0.1, 0.3, 0.5)):
                cells = []
                for n in ns:
                    s = mi[(mi["dgp"] == dkey) & (mi["mechanism"] == mech) & (mi["miss_rate"] == rate) & (mi["n"] == n)]
                    cells.append("--" if len(s) == 0 else f"{s['r2'].median():.2f}")
                for n in ns:
                    s = dr[(dr["dgp"] == dkey) & (dr["mechanism"] == mech) & (dr["miss_rate"] == rate) & (dr["n"] == n)]
                    cells.append("--" if len(s) == 0 else f"{s['nrmse_marginal'].median():.2f}")
                s = dr[(dr["dgp"] == dkey) & (dr["mechanism"] == mech) & (dr["miss_rate"] == rate)]
                cells += [f"{s['corr_delta_y'].median():+.3f}", f"{s['corr_delta_z'].median():+.3f}"]
                lab = dlab if (mi_ == 0 and ri == 0) else ""
                mlab = MECH_LABELS[mech] if ri == 0 else ""
                lines.append(f"    {lab:<16} & {mlab:<5} & {int(rate*100)}\\% & " + " & ".join(cells) + r" \\")
        if di < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Conservatism of the combined variance (Appendix R) ──────────────

def variance_ratio_table(results_dir: Path) -> pd.DataFrame:
    """Per calibration cell: mean(T) over all runs vs the Monte-Carlo variance
    of mu_hat over independent datasets (run 0), and a robust version
    (median T over all runs / MAD^2 of mu_hat over all runs); FPR under the
    Barnard-Rubin t and under N(0,1) critical values (run 0)."""
    v = load_csv(results_dir, "variance_decomposition.csv")
    v = v[~v["failed"].astype(bool)]
    rows = []
    for (dgp, mech, n, d), g in v.groupby(["dgp", "mechanism", "n", "dim_x"]):
        g0 = g[g["run"] == 0]
        mad = 1.4826 * np.median(np.abs(g["mu_hat"] - g["mu_hat"].median()))
        rows.append(dict(
            dgp=dgp, mechanism=mech, n=n, dim_x=d,
            ratio_raw=g["T"].mean() / g0["mu_hat"].var(ddof=1),
            ratio_robust=g["T"].median() / mad ** 2,
            fpr_t=g0["rejected"].mean(),
            fpr_normal=(g0["t_stat"] > stats.norm.ppf(0.95)).mean(),
            n_datasets=len(g0), n_runs=g["run"].nunique(),
        ))
    return pd.DataFrame(rows)


@register("variance_ratio")
def build_variance_ratio(results_dir: Path) -> str:
    """Appendix R: over-coverage of T per cell, raw and robust. Cells run
    with the MAR generator's |X|=2 defect (Appendix V) are labelled
    'MNAR (1 col.)', the one-column self-masking they actually ran."""
    R = variance_ratio_table(results_dir)
    labels = {"complete": "Complete", "mar": "MAR", "mnar": "MNAR", "mnar_1col": "MNAR (1 col.)"}
    dgps = [("Linear Gaussian", "linear_gaussian"), ("Post-nonlinear", "post_nonlinear")]
    lines = [r"\begin{tabular}{llcccccc}", r"    \toprule",
             r"    DGP & Mechanism & $n$ & $|\Xb|$ & $\bar T/\widehat{\mathrm{Var}}(\hat\mu)$ & robust & FPR ($t_\nu$) & FPR ($N(0,1)$) \\",
             r"    \midrule"]
    for di, (dlab, dkey) in enumerate(dgps):
        sub = R[R["dgp"] == dkey].sort_values(["mechanism", "n", "dim_x"])
        for i, r in enumerate(sub.itertuples()):
            raw = f"{r.ratio_raw:.2f}"
            raw = red(raw, r.ratio_raw < 1)
            lines.append(f"    {dlab if i == 0 else '':<16} & {labels[r.mechanism]:<14} & {r.n:,} & {r.dim_x} & {raw} & {r.ratio_robust:.2f} & {red(fmt_frac(r.fpr_t), r.fpr_t > 0.05)} & {fmt_frac(r.fpr_normal)} \\\\")
        if di == 0:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Placebo diagnostic ──────────────────────────────────────────────

PLACEBO_ARMS = [("deployed", r"deployed $k$NN ($k_{\mathrm{nn}} = \lfloor n^{2/(d+2)} \rfloor$)"),
                ("knn_k8", r"$k$NN, $k{=}8$"), ("knn_k32", r"$k$NN, $k{=}32$"),
                ("knn_k128", r"$k$NN, $k{=}128$"),
                ("marginal", "unconditional permutation"),
                ("oracle", r"oracle draw from $P(Y \mid \Xb)$")]


@register("placebo_diag")
def build_placebo_diag(results_dir: Path) -> str:
    """Placebo diagnostic: complete data, |X|=10, 150 paired replicates per
    cell; FPR and mean null statistic by placebo generator (not in the
    paper)."""
    parts = [load_csv(results_dir, f"placebo_diag_{d}.csv", required=False)
             for d in ("linear_gaussian", "post_nonlinear", "latent_confounder")]
    parts = [p for p in parts if p is not None]
    if not parts:
        return "% MISSING: placebo_diag_*.csv"
    p = pd.concat(parts, ignore_index=True)
    p = p[~p["failed"].astype(bool)]
    dgps = [("Linear Gaussian", "linear_gaussian"), ("Post-nonlinear", "post_nonlinear"),
            ("Latent confounder", "latent_confounder")]
    lines = [r"\begin{tabular}{llcccccc}", r"    \toprule",
             r"     & & \multicolumn{3}{c}{FPR} & \multicolumn{3}{c}{Mean null $t_{\mathrm{CI}}$} \\",
             r"    \cmidrule(lr){3-5} \cmidrule(lr){6-8}",
             r"    DGP & Placebo generator & $n{=}1{,}000$ & $2{,}000$ & $5{,}000$ & $n{=}1{,}000$ & $2{,}000$ & $5{,}000$ \\",
             r"    \midrule"]
    for di, (dlab, dkey) in enumerate(dgps):
        for ai, (arm, alab) in enumerate(PLACEBO_ARMS):
            s = p[(p["dgp"] == dkey) & (p["arm"] == arm)]
            if len(s) == 0:
                continue
            f = [_fpr(s, n=n) for n in (1000, 2000, 5000)]
            t = [s[s["n"] == n]["statistic"].mean() for n in (1000, 2000, 5000)]
            cells = [red(fmt_frac(v), v > 0.05) for v in f] + [f"{v:+.2f}" for v in t]
            lines.append(f"    {dlab if ai == 0 else '':<18} & {alab:<52} & " + " & ".join(cells) + r" \\")
        if di < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Test-variable missingness (Appendix T) ──────────────────────────

@register("testvar_missingness")
def build_testvar_missingness(results_dir: Path) -> str:
    """Appendix T: missingness on the test variables themselves (30%,
    n=1000, |X|=5), FPR and power at signal 0.3 under the imputed-labels
    rule (deployed) and the labels-observed alternative."""
    t = load_csv(results_dir, "test_variable_missingness.csv")
    if t is None:
        return "% MISSING: test_variable_missingness.csv"
    t = t[~t["failed"].astype(bool)]
    on_labels = {"z": "$Z$", "y": "$Y$", "both": "$Y$ and $Z$", "xz": "$\\Xb$ and $Z$"}
    dgps = [("Linear Gaussian", "linear_gaussian"), ("Post-nonlinear", "post_nonlinear")]
    lines = [r"\begin{tabular}{lllcccc}", r"    \toprule",
             r"     & & & \multicolumn{2}{c}{Imputed labels (deployed)} & \multicolumn{2}{c}{Labels observed} \\",
             r"    \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
             r"    DGP & Mechanism & Missing on & FPR & Power (0.3) & FPR & Power (0.3) \\",
             r"    \midrule"]
    for di, (dlab, dkey) in enumerate(dgps):
        i = 0
        for style in ("mar", "mnar"):
            for on in ("z", "y", "both", "xz"):
                cells = []
                for rule in ("imputed_labels", "labels_observed"):
                    s = t[(t["dgp"] == dkey) & (t["style"] == style) & (t["miss_on"] == on) & (t["rule"] == rule)]
                    f = _fpr(s, signal=0.0)
                    cells += [red(fmt_frac(f), f > 0.05), fmt_frac(_fpr(s, signal=0.3), 2)]
                lines.append(f"    {dlab if i == 0 else '':<16} & {MECH_LABELS[style] if on == 'z' else '':<5} & {on_labels[on]:<14} & " + " & ".join(cells) + r" \\")
                i += 1
        if di == 0:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Sensitivity summary (Appendix T) ────────────────────────────────

@register("sensitivity_summary")
def build_sensitivity_summary(results_dir: Path) -> str:
    """Appendix T: one table over M, K, learner, alpha and the joint grid
    of Appendix E.1, each row pointing to the appendix with the full
    result."""
    rows = []
    ms = load_csv(results_dir, "m_sensitivity_results.csv", required=False)
    if ms is not None:
        ms = ms[~ms["failed"].astype(bool)]
        for mech in ("complete", "mar", "mnar"):
            vals = [ _fpr(ms, signal=0.3, M=M, mechanism=mech) for M in (3, 5, 10, 20)]
            rows.append((r"$M$ (imputations)", "3 / 5 / 10 / 20", f"power at 0.3, {MECH_LABELS[mech]}",
                         " / ".join(fmt_frac(v, 2) for v in vals), r"App.~\ref{app:m_sensitivity}"))
    se = load_csv(results_dir, "sensitivity_results.csv", required=False)
    if se is not None:
        se = se[~se["failed"].astype(bool)]
        se = se[se["config"].str.startswith("M=")]  # PAIR-CI rows only
        base = se[(se["M"] == 5) & (se["n_trees"] == 100) & (se["min_leaf"] == 5)]
        ks = sorted(base["K"].unique())
        vals = [base[base["K"] == k]["shd"].median() for k in ks]
        rows.append((r"$K$ (folds)", " / ".join(str(k) for k in ks), "median SHD, $p{=}10$ nonlinear MAR",
                     " / ".join(fmt_dec1(v) for v in vals), r"App.~\ref{app:sensitivity}"))
        cfg = se.groupby(["M", "K", "n_trees", "min_leaf"])["shd"].median()
        rows.append((r"Joint grid ($M, K$, trees, leaf)", f"{len(cfg)} configurations", "median SHD range",
                     f"{fmt_dec1(cfg.min())}--{fmt_dec1(cfg.max())}", r"App.~\ref{app:sensitivity}"))
    lc = load_csv(results_dir, "learner_comparison_results.csv", required=False)
    if lc is not None:
        lc = lc[~lc["failed"].astype(bool)]
        learners = [("rf", "RF"), ("extra_trees", "ExtraTrees"), ("lgbm", "LightGBM"),
                    ("lgbm_tuned", "LightGBM (tuned)"), ("xgboost", "XGBoost")]
        f = [_fpr(lc, signal=0.0, learner=k, mechanism="mnar") for k, _ in learners]
        p = [_fpr(lc, signal=0.3, learner=k, mechanism="mnar") for k, _ in learners]
        rows.append(("Learner", " / ".join(l for _, l in learners), "FPR, MNAR",
                     " / ".join(fmt_frac(v, 2) for v in f), r"App.~\ref{app:learner_comparison}"))
        rows.append(("", "", "power at 0.3, MNAR", " / ".join(fmt_frac(v, 2) for v in p), ""))
    al = load_csv(results_dir, "alpha_sensitivity_results.csv", required=False)
    if al is not None:
        al = al[~al["failed"].astype(bool)]
        alphas = sorted(al["alpha"].unique())
        for mech in ("mar", "mnar"):
            vals = [al[(al["alpha"] == a) & (al["mechanism"] == mech)]["f1"].median() for a in alphas]
            rows.append((r"$\alpha$", " / ".join(f"{a:g}" for a in alphas), f"median F1, $p{{=}}10$ {MECH_LABELS[mech]}",
                         " / ".join(fmt_frac(v, 2) for v in vals), r"App.~\ref{app:pr_frontier}"))
    lines = [r"\begin{tabular}{lllll}", r"    \toprule",
             r"    Factor & Levels & Metric & Values & Detail \\", r"    \midrule"]
    for r in rows:
        lines.append("    " + " & ".join(r) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Graph-recovery tables with MVPC and full metrics ────────────────

def _er_cell_table(results_dir: Path) -> pd.DataFrame:
    """Tidy per-(dgp, p, mechanism, method) medians for the ER grid."""
    rows = []
    for dgp in ("linear", "nonlinear"):
        for p in (10, 20, 30):
            df = load_er(results_dir, p, dgp)
            if df is None:
                continue
            for _, key in CONSTRAINT_METHODS + SCORE_METHODS:
                for mech in ("complete", "mar", "mnar", "mixed"):
                    sub = df[(df["method_key"] == key) & (df["mechanism"] == mech)]
                    rows.append(dict(dgp=dgp, p=p, mechanism=mech, method=key, **cell_stats(sub)))
    return pd.DataFrame(rows)


def _scaling_table(results_dir: Path, metric: str) -> str:
    """Table 2 layout: median across mechanism-level medians per (method, dgp, p)."""
    cells = _er_cell_table(results_dir)
    if cells.empty:
        return "% MISSING: p-sweep CSVs"
    agg = cells.groupby(["method", "dgp", "p"])[["shd", "skel", "f1", "edges", "prec", "rec"]].median()

    def val(key, dgp, p, m):
        try:
            return agg.loc[(key, dgp, p), m]
        except KeyError:
            return np.nan

    bold_mask = {}
    for dgp in ("linear", "nonlinear"):
        for p in (10, 20, 30):
            col = [val(k, dgp, p, "shd" if metric in ("shd", "both") else "f1") for _, k in CONSTRAINT_METHODS]
            if metric in ("shd", "both"):
                mask = bold_lowest(col, tol=0.5)
            else:
                mx = np.nanmax(col) if not all(pd.isna(col)) else np.nan
                mask = [not pd.isna(c) and (mx - c) < 0.005 for c in col]
            for (_, k), b in zip(CONSTRAINT_METHODS, mask):
                bold_mask[(k, dgp, p)] = b

    def render(key, dgp, p):
        if metric == "pre":
            pr, rc, e = val(key, dgp, p, "prec"), val(key, dgp, p, "rec"), val(key, dgp, p, "edges")
            if pd.isna(pr):
                return "--"
            body = f"{fmt_frac(pr, 2)}/{fmt_frac(rc, 2)}"
            body += "" if pd.isna(e) else f"\\,{{\\scriptsize({fmt_int(e)})}}"
            return gray_dagger(body) if key == "missdag" else body
        if metric == "both":
            v, sv, f = val(key, dgp, p, "shd"), val(key, dgp, p, "skel"), val(key, dgp, p, "f1")
            if pd.isna(v):
                return "--"
            body = bold(fmt_int(v), bold_mask.get((key, dgp, p), False))
            body += "" if pd.isna(sv) else f"\\,{{\\scriptsize({fmt_int(sv)})}}"
            body += " / " + ("--" if pd.isna(f) else fmt_frac(f, 2))
            return gray_dagger(body) if key == "missdag" else body
        if metric == "shd":
            v, sv = val(key, dgp, p, "shd"), val(key, dgp, p, "skel")
            if pd.isna(v):
                return "--"
            body = bold(fmt_int(v), bold_mask.get((key, dgp, p), False))
            body += "" if pd.isna(sv) else f"\\,{{\\scriptsize({fmt_int(sv)})}}"
        else:
            v, e = val(key, dgp, p, "f1"), val(key, dgp, p, "edges")
            if pd.isna(v):
                return "--"
            body = bold(fmt_frac(v, 2), bold_mask.get((key, dgp, p), False))
            body += "" if pd.isna(e) else f"\\,{{\\scriptsize({fmt_int(e)})}}"
        return gray_dagger(body) if key == "missdag" else body

    lines = [r"\begin{tabular}{lcccccc}", r"    \toprule",
             r"                  & \multicolumn{3}{c}{Linear} & \multicolumn{3}{c}{Nonlinear} \\",
             r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             r"    Method        & $p{=}10$ & $p{=}20$ & $p{=}30$ & $p{=}10$ & $p{=}20$ & $p{=}30$ \\",
             r"    \midrule"]
    for label, key in CONSTRAINT_METHODS:
        vals = [render(key, d, p) for d in ("linear", "nonlinear") for p in (10, 20, 30)]
        lines.append(f"    {label:<45} & " + " & ".join(vals) + r" \\")
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{7}{l}{\emph{Score-based (DAG output):}} \\")
    for label, key in SCORE_METHODS:
        vals = [render(key, d, p) for d in ("linear", "nonlinear") for p in (10, 20, 30)]
        lines.append(f"    {label:<45} & " + " & ".join(vals) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("scaling")
def build_scaling(results_dir: Path) -> str:
    """Total SHD (skeleton SHD) with MVPC (not in the paper; Table 2 uses
    scaling_combined)."""
    return _scaling_table(results_dir, "shd")


@register("scaling_f1")
def build_scaling_f1(results_dir: Path) -> str:
    """Skeleton F1 (median output edges) (not in the paper)."""
    return _scaling_table(results_dir, "f1")


@register("scaling_combined")
def build_scaling_combined(results_dir: Path) -> str:
    """Table 2, Panel A: total SHD (skeleton SHD) / skeleton F1 per cell."""
    return _scaling_table(results_dir, "both")


@register("scaling_pre")
def build_scaling_pre(results_dir: Path) -> str:
    """Table 2, Panel B: skeleton precision / recall (median output edges)
    per cell, aggregated as Panel A (median of mechanism-level medians)."""
    return _scaling_table(results_dir, "pre")


def _er_per_mech(results_dir: Path, dgp: str, p: int, full: bool) -> str:
    df = load_er(results_dir, p, dgp)
    if df is None:
        return f"% MISSING p={p} {dgp}"
    mechs = ("complete", "mar", "mnar", "mixed")
    if not full:
        lines = [r"\begin{tabular}{lcccc}", r"    \toprule",
                 r"    Method & Complete & MAR & MNAR & Mixed \\", r"    \midrule"]
        for label, key in CONSTRAINT_METHODS + SCORE_METHODS:
            cells = []
            for m in mechs:
                st = cell_stats(df[(df["method_key"] == key) & (df["mechanism"] == m)])
                if pd.isna(st["shd"]):
                    cells.append("--")
                    continue
                body = _fmt_metric("shd", st["shd"]) + f"\\,{{\\scriptsize({fmt_int(st['skel'])})}}"
                cells.append(gray_dagger(body) if key == "missdag" else body)
            lines.append(f"    {label:<49} & " + " & ".join(cells) + r" \\")
        lines += [r"    \bottomrule", r"\end{tabular}"]
        return "\n".join(lines)
    lines = [r"\begin{tabular}{llcccccc}", r"    \toprule",
             r"    Mechanism & Method & SHD & Skel.\ SHD & Precision & Recall & F1 & Edges \\",
             r"    \midrule"]
    for mi, m in enumerate(mechs):
        for i, (label, key) in enumerate(CONSTRAINT_METHODS + SCORE_METHODS):
            st = cell_stats(df[(df["method_key"] == key) & (df["mechanism"] == m)])
            if st["n"] == 0:
                continue
            cells = [_fmt_metric(k, st[k]) for k in ("shd", "skel", "prec", "rec", "f1", "edges")]
            if key == "missdag":
                cells[0] = gray_dagger(cells[0])
            lines.append(f"    {MECH_LABELS[m] if i == 0 else '':<9} & {SHORT_LABELS[key]:<18} & " + " & ".join(cells) + r" \\")
        if mi < 3:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("shd_linear")
def build_shd_linear(results_dir: Path) -> str:
    return _er_per_mech(results_dir, "linear", 10, full=False)


@register("shd_nonlinear")
def build_shd_nonlinear(results_dir: Path) -> str:
    return _er_per_mech(results_dir, "nonlinear", 10, full=False)


for _dgp in ("linear", "nonlinear"):
    for _p in (10, 20, 30):
        def _mk(dgp=_dgp, p=_p):
            def builder(results_dir: Path) -> str:
                """Appendix F: skeleton metrics by mechanism, one ER cell."""
                return _er_per_mech(results_dir, dgp, p, full=True)
            return builder
        REVISION_TABLES[f"graph_full_{_dgp}_p{_p}"] = _mk()


@register("realworld")
def build_realworld(results_dir: Path) -> str:
    """Table 3: the three network benchmarks at 20% MNAR with SHD,
    precision, recall, F1 and median output edges (MAR/mixed in the
    appendix, tab_network_metrics_*).  Best SHD and F1 among constraint-based
    methods in bold; MVPC's DNF replicates are excluded from its medians."""
    nets = ["alarm_linear", "alarm_nonlinear", "hailfinder"]
    data = {k: load_network(results_dir, k)[0] for k in nets}
    metrics = ("shd", "prec", "rec", "f1", "edges")
    st = {}
    for k in nets:
        d = data[k]
        for _, key in CONSTRAINT_METHODS:
            sub = d[(d["method_key"] == key) & (d["rate"] == 0.2) & (d["mechanism"] == "mnar")]
            st[(k, key)] = cell_stats(sub)
    bold_mask = {}
    for k in nets:
        shd = [st[(k, key)]["shd"] for _, key in CONSTRAINT_METHODS]
        f1 = [st[(k, key)]["f1"] for _, key in CONSTRAINT_METHODS]
        mx = np.nanmax(f1)
        for (_, key), b1, v in zip(CONSTRAINT_METHODS, bold_lowest(shd, tol=0.1), f1):
            bold_mask[(k, key, "shd")] = b1
            bold_mask[(k, key, "f1")] = (not pd.isna(v)) and (mx - v) < 0.005
    head = " & ".join(rf"\multicolumn{{5}}{{c}}{{{NETWORKS[k]['label']}}}" for k in nets)
    lines = [r"\begin{tabular}{l" + "ccccc" * 3 + "}", r"    \toprule",
             f"     & {head} \\\\",
             r"    \cmidrule(lr){2-6} \cmidrule(lr){7-11} \cmidrule(lr){12-16}",
             r"    Method & SHD & P & R & F1 & E & SHD & P & R & F1 & E & SHD & P & R & F1 & E \\",
             r"    \midrule"]
    for label, key in CONSTRAINT_METHODS:
        cells = []
        for k in nets:
            s = st[(k, key)]
            for m in metrics:
                v = _fmt_metric(m, s[m])
                if m in ("shd", "f1"):
                    v = bold(v, bold_mask[(k, key, m)])
                if key == "complete_case" and m == "f1" and not pd.isna(s["f1"]) and s["f1"] < 1e-6:
                    v = gray_dagger(v)
                cells.append(v)
        lines.append(f"    {label:<45} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("realworld_by_mech")
def build_realworld_by_mech(results_dir: Path) -> str:
    """Alternative Table 3 layout: median SHD [IQR] per mechanism at 20%
    missingness, F1 beneath, with MVPC and ALARM nonlinear (not in the
    paper)."""
    nets = ["alarm_linear", "alarm_nonlinear", "hailfinder"]
    data = {k: load_network(results_dir, k)[0] for k in nets}
    mechs = ("mar", "mnar", "mixed")
    st = {}
    for k in nets:
        for _, key in CONSTRAINT_METHODS:
            for m in mechs:
                d = data[k]
                st[(k, key, m)] = cell_stats(d[(d["method_key"] == key) & (d["rate"] == 0.2) & (d["mechanism"] == m)])
    bold_mask = {}
    for k in nets:
        for m in mechs:
            col = [st[(k, key, m)]["shd"] for _, key in CONSTRAINT_METHODS]
            for (_, key), b in zip(CONSTRAINT_METHODS, bold_lowest(col, tol=0.1)):
                bold_mask[(k, key, m)] = b
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{NETWORKS[k]['label']}}}" for k in nets)
    lines = [r"\begin{tabular}{l" + "ccc" * 3 + "}", r"    \toprule",
             f"     & {head} \\\\",
             r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
             r"    Method & MAR & MNAR & Mixed & MAR & MNAR & Mixed & MAR & MNAR & Mixed \\",
             r"    \midrule"]
    for label, key in CONSTRAINT_METHODS:
        shd_cells, f1_cells = [], []
        for k in nets:
            for m in mechs:
                s = st[(k, key, m)]
                if pd.isna(s["shd"]):
                    shd_cells.append("--"); f1_cells.append("")
                    continue
                body = bold(fmt_dec1(s["shd"]), bold_mask[(k, key, m)]) + f"\\,{{\\scriptsize[{fmt_int(s['shd_iqr'])}]}}"
                f1 = f"{{\\scriptsize F1 {fmt_frac(s['f1'], 2)}, E {fmt_int(s['edges'])}}}"
                if key == "complete_case" and s["f1"] < 1e-6:
                    body, f1 = gray_dagger(body), gray_dagger(f1)
                shd_cells.append(body); f1_cells.append(f1)
        lines.append(f"    {label:<45} & " + " & ".join(shd_cells) + r" \\")
        lines.append(f"    {'':<45} & " + " & ".join(f1_cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _network_metrics(results_dir: Path, key: str) -> str:
    df, failed = load_network(results_dir, key)
    lines = [r"\begin{tabular}{llccccccc}", r"    \toprule",
             r"    Mechanism & Method & SHD & Skel.\ SHD & Precision & Recall & F1 & Edges & Reps \\",
             r"    \midrule"]
    blocks = [("complete", 0.1)] + [(m, 0.2) for m in ("mar", "mnar", "mixed")]
    for bi, (mech, rate) in enumerate(blocks):
        for i, (label, mkey) in enumerate(CONSTRAINT_METHODS + SCORE_METHODS):
            sub = df[(df["method_key"] == mkey) & (df["mechanism"] == mech) & (df["rate"] == rate)]
            st = cell_stats(sub)
            if st["n"] == 0:
                continue
            cells = [_fmt_metric(k, st[k]) for k in ("shd", "skel", "prec", "rec", "f1", "edges")]
            if mkey == "missdag":
                cells[0] = gray_dagger(cells[0])
            nf = len(failed[(failed["method_key"] == mkey) & (failed["mechanism"] == mech) & (failed["rate"] == rate)])
            reps = f"{st['n']}" + (f" ({nf} DNF)" if nf else "")
            lab = (MECH_LABELS[mech] + (r" (0\%)" if mech == "complete" else r" (20\%)")) if i == 0 else ""
            lines.append(f"    {lab:<16} & {SHORT_LABELS[mkey]:<18} & " + " & ".join(cells) + f" & {reps} \\\\")
        if bi < 3:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


for _net in NETWORKS:
    def _mk_net(net=_net):
        def builder(results_dir: Path) -> str:
            """Apps H-I: full skeleton metrics for one network, complete
            data and 20% missingness."""
            return _network_metrics(results_dir, net)
        return builder
    REVISION_TABLES[f"network_metrics_{_net}"] = _mk_net()


def _network_rate_grid(results_dir: Path, key: str, metric: str) -> str:
    """Rate x mechanism rows, method columns, with MVPC (Apps H-I)."""
    df, _ = load_network(results_dir, key)
    methods = [k for _, k in CONSTRAINT_METHODS]
    if key != "hailfinder":
        methods += ["missdag"]
    n_true = NETWORKS[key]["n_true"]

    def cell(rate, mkey, mech):
        sub = df[(df["rate"] == rate) & (df["mechanism"] == mech) & (df["method_key"] == mkey)]
        if len(sub) == 0:
            return " "
        if metric == "shd":
            val = fmt_dec1(sub["shd"].median())
            skel = sub["skel"].median()
            val += "" if pd.isna(skel) else f"\\,{{\\scriptsize({fmt_int(skel)})}}"
        elif metric == "edges":
            val = _fmt_metric("edges", sub["edges"].median())
        else:
            val = fmt_frac(sub["f1"].median())
        if mkey == "missdag":
            val = gray_dagger(val)
        elif abs(sub["f1"].median()) < 1e-6:
            val = gray_dagger(val)
        return val

    lines = [r"\begin{tabular}{ll" + "c" * len(methods) + "}", r"    \toprule",
             "    Rate & Mechanism & " + " & ".join(SHORT_LABELS[m] for m in methods) + r" \\",
             r"    \midrule",
             "    0\\% & Complete & " + " & ".join(cell(0.1, m, "complete") for m in methods) + r" \\",
             r"    \midrule"]
    for ri, (rl, rate) in enumerate((("10\\%", 0.1), ("20\\%", 0.2), ("40\\%", 0.4))):
        for mi, mech in enumerate(("mar", "mnar", "mixed")):
            lines.append(f"    {rl if mi == 0 else '    '} & {MECH_LABELS[mech]:<5} & " + " & ".join(cell(rate, m, mech) for m in methods) + r" \\")
        if ri < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("alarm_full")
def build_alarm_full(results_dir: Path) -> str:
    return _network_rate_grid(results_dir, "alarm_linear", "shd")


@register("alarm_full_nonlinear")
def build_alarm_full_nonlinear(results_dir: Path) -> str:
    return _network_rate_grid(results_dir, "alarm_nonlinear", "shd")


@register("hailfinder_full")
def build_hailfinder_full(results_dir: Path) -> str:
    return _network_rate_grid(results_dir, "hailfinder", "shd")


@register("hailfinder_f1")
def build_hailfinder_f1(results_dir: Path) -> str:
    return _network_rate_grid(results_dir, "hailfinder", "f1")


# Median output edges on the rate x mechanism grid of the SHD tables.
@register("alarm_edges")
def build_alarm_edges(results_dir: Path) -> str:
    return _network_rate_grid(results_dir, "alarm_linear", "edges")


@register("alarm_nonlinear_edges")
def build_alarm_nonlinear_edges(results_dir: Path) -> str:
    return _network_rate_grid(results_dir, "alarm_nonlinear", "edges")


@register("hailfinder_edges")
def build_hailfinder_edges(results_dir: Path) -> str:
    return _network_rate_grid(results_dir, "hailfinder", "edges")


# ── Sparsity-matched comparison (Appendix T) ────────────────────────

def sparsity_match(results_dir: Path) -> pd.DataFrame:
    """For each benchmark cell, the alpha at which each tunable baseline's
    median output edge count is closest to PAIR-CI's at alpha = 0.05."""
    rows = []
    specs = [("ALARM (nonlinear), MNAR 20\\%", "alpha_sweep_baselines_alarm_nonlinear.csv", "alarm_nonlinear"),
             ("ALARM (linear), MNAR 20\\%", "alpha_sweep_baselines_alarm_linear.csv", "alarm_linear"),
             ("HAILFINDER, MNAR 20\\%", "alpha_sweep_baselines_hailfinder.csv", "hailfinder")]
    for label, sweep_file, net in specs:
        sw = load_csv(results_dir, sweep_file, required=False)
        if sw is None:
            continue
        sw = sw[(~sw["failed"].astype(bool)) & (sw["mechanism"] == "mnar")]
        d, _ = load_network(results_dir, net)
        ours = cell_stats(d[(d["method_key"] == "ours") & (d["rate"] == 0.2) & (d["mechanism"] == "mnar")])
        rows.append(dict(cell=label, method="ours", alpha=0.05, **{k: ours[k] for k in ("edges", "prec", "rec", "f1")}, matched=True))
        mv = cell_stats(d[(d["method_key"] == "mvpc") & (d["rate"] == 0.2) & (d["mechanism"] == "mnar")])
        rows.append(dict(cell=label, method="mvpc", alpha=0.05, **{k: mv[k] for k in ("edges", "prec", "rec", "f1")}, matched=False))
        for meth in ("testwise", "complete_case", "impute_fisherz", "micd_pooled"):
            s = sw[sw["method"] == meth].groupby("alpha").agg(edges=("n_edges_out", "median"), prec=("precision", "median"),
                                                              rec=("recall", "median"), f1=("f1", "median"))
            if s.empty:
                continue
            a_star = (s["edges"] - ours["edges"]).abs().idxmin()
            best = s["f1"].idxmax()
            rows.append(dict(cell=label, method=meth, alpha=a_star, **s.loc[a_star].to_dict(), matched=True))
            rows.append(dict(cell=label, method=meth, alpha=best, **s.loc[best].to_dict(), matched=False))
    return pd.DataFrame(rows)


@register("sparsity_matched")
def build_sparsity_matched(results_dir: Path) -> str:
    """Appendix T: each tunable baseline at the alpha matching PAIR-CI's
    edge count, and at its best-F1 alpha; MVPC at its native operating point."""
    S = sparsity_match(results_dir)
    if S.empty:
        return "% MISSING: alpha_sweep_baselines_*.csv"
    labels = {"ours": r"PAIR-CI ($\alpha{=}0.05$)", "mvpc": "MVPC (native)", "testwise": "Test-wise",
              "complete_case": "Complete case", "impute_fisherz": "FZ-vote", "micd_pooled": "FZ-Rubin"}
    lines = [r"\begin{tabular}{llccccc|c}", r"    \toprule",
             r"     & & \multicolumn{5}{c}{At PAIR-CI's sparsity} & Best swept \\",
             r"    \cmidrule(lr){3-7}",
             r"    Benchmark & Method & $\alpha$ & Edges & Precision & Recall & F1 & F1 ($\alpha$) \\", r"    \midrule"]
    for ci, cell in enumerate(S["cell"].unique()):
        sub = S[S["cell"] == cell]
        matched = sub[sub["matched"]].set_index("method")
        best = sub[~sub["matched"]].set_index("method")
        order = [m for m in ("ours", "mvpc", "testwise", "complete_case", "impute_fisherz", "micd_pooled")
                 if m in matched.index or m in best.index]
        for i, m in enumerate(order):
            r = matched.loc[m] if m in matched.index else best.loc[m]
            tail = "--" if m == "ours" else (f"{fmt_frac(best.loc[m, 'f1'], 2)} ({best.loc[m, 'alpha']:g})" if m in best.index and m != "mvpc" else "native")
            lines.append(f"    {cell if i == 0 else '':<32} & {labels[m]:<26} & {r['alpha']:g} & {_fmt_metric('edges', r['edges'])} & {fmt_frac(r['prec'], 2)} & {fmt_frac(r['rec'], 2)} & {fmt_frac(r['f1'], 2)} & {tail} \\\\")
        if ci < S["cell"].nunique() - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Layout assemblies of existing fragments ─────────────────────────

@register("shd_combined")
def build_shd_combined(results_dir: Path) -> str:
    """Appendix F: per-mechanism SHD at p = 10, linear and nonlinear panels."""
    return ("\\vspace{0.5em}\n\\textbf{Panel A: Linear Gaussian edges}\\\\[0.4em]\n"
            + REVISION_TABLES["shd_linear"](results_dir)
            + "\n\n\\vspace{1em}\n\\textbf{Panel B: Nonlinear edges}\\\\[0.4em]\n"
            + REVISION_TABLES["shd_nonlinear"](results_dir))


@register("degradation_combined")
def build_degradation_combined(results_dir: Path) -> str:
    """Appendix Q: imputation-strategy FPR (Panel A) and power (Panel B) in one table."""
    from build_tables import TABLES as BASE

    def rows(tex):
        out = []
        for line in tex.splitlines():
            if "&" not in line or "Strategy" in line:
                continue
            cells = [c.strip() for c in line.strip().rstrip("\\").split("&")]
            out.append(cells)
        return out

    lines = [r"\begin{tabular}{lccc}", r"    \toprule",
             r"    Strategy & Complete & MAR & MNAR \\", r"    \midrule",
             r"    \multicolumn{4}{l}{\textit{Panel A: False positive rate (signal $= 0$)}} \\"]
    for label, *vals in rows(BASE["degradation_fpr"](results_dir)):
        lines.append(f"    {label:<9} & " + " & ".join(vals) + r" \\")
    lines += [r"    \midrule", r"    \multicolumn{4}{l}{\textit{Panel B: Power (signal $= 0.6$)}} \\"]
    for label, *vals in rows(BASE["degradation_power"](results_dir)):
        dagger = any(r"$^\dagger$" in v for v in vals)
        vals = [v.replace(r"$^\dagger$", "") for v in vals]
        label = label + r"$^\dagger$" if dagger else f"{label:<9}"
        lines.append(f"    {label} & " + " & ".join(vals) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
