"""
Build the paper's LaTeX tables from the result CSVs.

Each builder writes tables/tab_{name}.tex, a bare tabular block that the
paper \\input{}s; captions and labels live in the paper. The builders here
produce the submitted version's tables; tables_revision.py adds the
remaining tables and replaces nine of these (calibration, scaling,
realworld, alarm_full, alarm_full_nonlinear, hailfinder_full,
hailfinder_f1, shd_linear, shd_nonlinear) with layouts that include MVPC.

Conventions: false positive rates and power are mean rejection rates over
replicates; graph metrics are per-cell medians, and cells pooled over
missingness mechanisms report the median of the per-mechanism medians.
Failed replicates are dropped.

Usage:
  python analysis/build_tables.py                   # all tables, to tables/
  python analysis/build_tables.py --table scaling   # one table
  python analysis/build_tables.py --results-dir results/ --output-dir tables/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Formatting helpers ───────────────────────────────────────────────

def fmt_shd(x: float) -> str:
    """Format SHD as an integer when possible, else 1 decimal."""
    if pd.isna(x):
        return "--"
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.1f}"


def round_half_up(x: float) -> int:
    """Round to nearest integer, breaking ties up (matches paper convention)."""
    if pd.isna(x):
        return None  # type: ignore
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def fmt_int(x: float) -> str:
    """Render as a rounded integer (round-half-up)."""
    if pd.isna(x):
        return "--"
    return str(round_half_up(x))


def fmt_dec1(x: float) -> str:
    """Render with exactly one decimal place."""
    if pd.isna(x):
        return "--"
    return f"{x:.1f}"


def fmt_frac(x: float, digits: int = 3) -> str:
    """Format a probability/rate to fixed decimal places."""
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def red(value: str, cond: bool) -> str:
    """Wrap in \\textcolor{red}{...} if the condition holds."""
    return f"\\textcolor{{red}}{{{value}}}" if cond else value


def bold(value: str, cond: bool) -> str:
    """Wrap in \\textbf{...} if the condition holds."""
    return f"\\textbf{{{value}}}" if cond else value


def gray_dagger(value: str) -> str:
    """Grey out a cell and add a dagger: the number is correct but is not a
    like-for-like comparison. Used for MissDAG (DAG rather than CPDAG
    output, which inflates total SHD), empty-graph collapses (median F1 = 0),
    and power cells whose false positive rate is inflated (mean or marginal
    imputation)."""
    return f"\\textcolor{{lightgray}}{{{value}}}$^\\dagger$"


def bold_lowest(values: list[float], tol: float = 1e-6) -> list[bool]:
    """Return a mask of which values are at the minimum (with tolerance)."""
    valid = [v for v in values if not pd.isna(v)]
    if not valid:
        return [False] * len(values)
    m = min(valid)
    return [not pd.isna(v) and (v - m) < tol for v in values]


# ── Data loading ─────────────────────────────────────────────────────

def load_csv(results_dir: Path, name: str, required: bool = True) -> pd.DataFrame | None:
    """Read results_dir/name, or return None (with a warning if required)."""
    f = results_dir / name
    if not f.exists():
        if required:
            print(f"  *** MISSING: {f}", file=sys.stderr)
        return None
    return pd.read_csv(f)


def load_p_sweep(results_dir: Path, p: int, dgp: str, include_failed: bool = False) -> pd.DataFrame | None:
    """Concatenate the 4 mechanism CSVs for a given p × dgp cell."""
    rows = []
    for mech in ("mar", "mnar", "mixed", "complete"):
        df = load_csv(results_dir, f"p{p}_{dgp}_{mech}.csv", required=False)
        if df is not None:
            rows.append(df)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    if not include_failed:
        df = df[~df["failed"]]
    return df


def load_newbase_p_sweep(results_dir: Path, p: int, dgp: str) -> pd.DataFrame | None:
    """Concatenate the 4 FZ-Rubin/MissDAG mechanism CSVs for a p × dgp cell."""
    rows = []
    for mech in ("mar", "mnar", "mixed", "complete"):
        df = load_csv(results_dir, f"newbase_p{p}_{dgp}_{mech}.csv", required=False)
        if df is not None:
            rows.append(df)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    return df[~df["failed"]]


def ours_row(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 'ours' rows, handling both ci_oracle_fast and ci_oracle_cached."""
    return df[df["method"].isin(["ci_oracle_fast", "ci_oracle_cached"])]


def method_row(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ours":
        return ours_row(df)
    return df[df["method"] == method]


def compute_skeleton_shd(sub: pd.DataFrame, n_true_edges: float | None = None) -> pd.Series:
    """Skeleton SHD = fp + fn (orientation ignored).

    Uses the logged tp/fp/fn where present. Rows that logged only precision
    and recall are back-derived from the true edge count (column
    n_true_edges, else the n_true_edges argument), row by row.
    """
    has_tpfpfn = {"tp", "fp", "fn"}.issubset(sub.columns)
    if has_tpfpfn and not sub["tp"].isna().all():
        result = sub["fp"] + sub["fn"]
        nan_mask = result.isna()
        if not nan_mask.any():
            return result
    else:
        result = pd.Series([float("nan")] * len(sub), index=sub.index)
        nan_mask = pd.Series([True] * len(sub), index=sub.index)

    # Back-derive for the NaN rows.
    if "precision" in sub.columns and "recall" in sub.columns:
        edges_col = sub.get("n_true_edges")
        if edges_col is None or edges_col.isna().all():
            if n_true_edges is None:
                return result
            edges = n_true_edges
        else:
            edges = edges_col.fillna(n_true_edges) if n_true_edges else edges_col
        tp = sub.loc[nan_mask, "recall"] * (edges.loc[nan_mask] if hasattr(edges, "loc") else edges)
        fp = tp * (1 / sub.loc[nan_mask, "precision"].clip(1e-10) - 1)
        fn = (edges.loc[nan_mask] if hasattr(edges, "loc") else edges) - tp
        result.loc[nan_mask] = fp + fn
    return result


# ── Table builders ───────────────────────────────────────────────────

TABLES: dict[str, callable] = {}


def register(name: str):
    """Decorator: register a table builder function."""
    def decorator(fn):
        TABLES[name] = fn
        return fn
    return decorator


@register("calibration_by_dgp")
def build_calibration_by_dgp(results_dir: Path) -> str:
    """Table 4: false positive rate at signal 0 by DGP and mechanism (MAR,
    MNAR)."""
    df = load_csv(results_dir, "ci_calibration_results.csv")
    if df is None:
        return "% MISSING: ci_calibration_results.csv"
    df = df[~df["failed"]]
    df = df[df["signal"] == 0]

    methods = [
        ("PAIR-CI",       "Ours",     "Integrated"),
        ("FZ-single",     "Fisher-Z", "Imputed"),
        ("FZ-Rubin",      "Fisher-Z", "Rubin's Rules"),
        ("GCM (Imputed)", "GCM",      "Imputed"),
        ("KCI (Imputed)", "KCI",      "Imputed"),
    ]
    dgps = [
        ("Linear Gaussian",   "linear_gaussian"),
        ("Post-nonlinear",    "post_nonlinear"),
        ("Latent confounder", "latent_confounder"),
    ]
    mechs = ("mar", "mnar")  # complete-data rates are in Table 1

    def cell(label, key_m, key_v, dgp_key, mech):
        sub = df[(df["method"] == key_m) & (df["variant"] == key_v) & (df["dgp"] == dgp_key) & (df["mechanism"] == mech)]
        if len(sub) == 0:
            return "--"
        v = sub["rejected"].mean()
        s = fmt_frac(v)
        if v > 0.05:
            s = red(s, True)
        return s

    lines = [
        r"\begin{tabular}{l*{6}{c}}",
        r"    \toprule",
        r"                  & \multicolumn{2}{c}{Linear Gaussian} & \multicolumn{2}{c}{Post-nonlinear} & \multicolumn{2}{c}{Latent confounder} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"    Method        & MAR & MNAR & MAR & MNAR & MAR & MNAR \\",
        r"    \midrule",
    ]
    for label, key_m, key_v in methods:
        cells = []
        for _, dgp_key in dgps:
            for mech in mechs:
                cells.append(cell(label, key_m, key_v, dgp_key, mech))
        lines.append(f"    {label:<14} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("calibration")
def build_calibration(results_dir: Path) -> str:
    """Submitted Table 1 layout: rejection rate at signal 0 by method and
    mechanism. Replaced by the tables_revision builder of the same name."""
    df = load_csv(results_dir, "ci_calibration_results.csv")
    if df is None:
        return "% MISSING: ci_calibration_results.csv"
    df = df[(df["signal"] == 0) & (~df["failed"])]

    methods = [
        ("PAIR-CI",        "Ours",      "Integrated",     True),
        ("FZ-single",                "Fisher-Z",  "Imputed",        False),
        ("FZ-Rubin",                 "Fisher-Z",  "Rubin's Rules",  False),
        ("GCM (Imputed)",            "GCM",       "Imputed",        False),
        ("KCI (Imputed)",            "KCI",       "Imputed",        False),
    ]

    lines = [
        r"\begin{tabular}{lccc}",
        r"    \toprule",
        r"    Method                   & Complete               & MAR                    & MNAR                   \\",
        r"    \midrule",
    ]
    for label, method, variant, is_ours in methods:
        sub = df[(df["method"] == method) & (df["variant"] == variant)]
        if len(sub) == 0:
            lines.append(f"    {label:<25} & -- & -- & -- \\\\")
            continue
        vals = {}
        for mech in ("complete", "mar", "mnar"):
            m = sub[sub["mechanism"] == mech]
            vals[mech] = m["reject"].mean() if "reject" in m.columns else m["rejected"].mean()
        # Red above 0.05; PAIR-CI's MNAR cell in bold.
        cells = []
        for mech in ("complete", "mar", "mnar"):
            v = vals[mech]
            s = fmt_frac(v)
            s = red(s, v > 0.05)
            if is_ours and mech == "mnar":
                s = bold(s, True)
            cells.append(s)
        lines.append(
            f"    {label:<25} & {cells[0]:<22} & {cells[1]:<22} & {cells[2]:<22} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("scaling")
def build_scaling(results_dir: Path) -> str:
    """Submitted Table 2 layout: total SHD (skeleton SHD) by method, DGP
    and p, as medians of per-mechanism medians. Skeleton SHD ignores
    orientation, so it compares CPDAG and DAG (MissDAG) outputs fairly.
    Replaced by the tables_revision builder of the same name."""
    constraint_methods = [
        ("PAIR-CI",                                                 "ours"),        ("Complete case",                                        "complete_case"),

        ("Test-wise",                                             "testwise"),
        ("FZ-vote",                                            "impute_fisherz"),
        (r"FZ-Rubin (\citeauthor{witte2022multiple})",           "micd_pooled"),
    ]
    score_methods = [
        (r"MissDAG$^\dagger$ (\citeauthor{gao2022missdag})",     "missdag"),
    ]
    all_methods = constraint_methods + score_methods

    def _compute_skeleton_shd(sub):
        # Use logged tp/fp/fn when present; back-deriving from precision
        # and recall is unstable when precision is near zero.
        if {"tp", "fp", "fn"}.issubset(sub.columns) and not sub["tp"].isna().all():
            return sub["fp"] + sub["fn"]
        tp = sub["recall"] * sub["n_true_edges"]
        fp = tp * (1 / sub["precision"].clip(1e-10) - 1)
        fn = sub["n_true_edges"] - tp
        return fp + fn

    medians: dict[tuple[str, str, int], float] = {}
    skel_medians: dict[tuple[str, str, int], float] = {}
    for _, key in all_methods:
        for dgp in ("linear", "nonlinear"):
            for p in (10, 20, 30):
                is_nb = key in ("micd_pooled", "missdag")
                df = load_newbase_p_sweep(results_dir, p, dgp) if is_nb \
                    else load_p_sweep(results_dir, p, dgp)
                if df is None:
                    medians[(key, dgp, p)] = float("nan")
                    skel_medians[(key, dgp, p)] = float("nan")
                    continue
                sub = ours_row(df) if key == "ours" else df[df["method"] == key]
                if len(sub) == 0:
                    medians[(key, dgp, p)] = float("nan")
                    skel_medians[(key, dgp, p)] = float("nan")
                    continue
                per_mech_shd = sub.groupby("mechanism")["shd"].median()
                medians[(key, dgp, p)] = float(per_mech_shd.median())
                skel = _compute_skeleton_shd(sub)
                per_mech_skel = skel.groupby(sub["mechanism"]).median()
                skel_medians[(key, dgp, p)] = float(per_mech_skel.median())

    # Bold the lowest total SHD among the constraint-based methods only;
    # MissDAG (score-based) is listed separately and never bolded.
    bold_mask: dict[tuple[str, str, int], bool] = {}
    for dgp in ("linear", "nonlinear"):
        for p in (10, 20, 30):
            col = [medians[(k, dgp, p)] for _, k in constraint_methods]
            mask = bold_lowest(col, tol=0.5)
            for (_, k), b in zip(constraint_methods, mask):
                bold_mask[(k, dgp, p)] = b
            for _, k in score_methods:
                bold_mask[(k, dgp, p)] = False

    def _render_val(k, dgp, p):
        v = medians[(k, dgp, p)]
        sv = skel_medians[(k, dgp, p)]
        if pd.isna(v):
            return "--"
        med_s = fmt_int(v)
        skel_s = "" if pd.isna(sv) else f"\\,{{\\scriptsize({fmt_int(sv)})}}"
        body = bold(med_s, bold_mask[(k, dgp, p)]) + skel_s
        if k == "missdag":
            body = gray_dagger(body)
        return body

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"    \toprule",
        r"                  & \multicolumn{3}{c}{Linear} & \multicolumn{3}{c}{Nonlinear}                                             \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"    Method        & $p{=}10$                   & $p{=}20$                      & $p{=}30$          & $p{=}10$          & $p{=}20$          & $p{=}30$          \\",
        r"    \midrule",
    ]
    for label, key in constraint_methods:
        vals = [_render_val(key, d, p) for d in ("linear", "nonlinear") for p in (10, 20, 30)]
        lines.append(
            f"    {label:<45}     & {vals[0]:<25} & {vals[1]:<25} & {vals[2]:<17} & {vals[3]:<17} & {vals[4]:<17} & {vals[5]:<17} \\\\"
        )
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{7}{l}{\emph{Score-based (DAG output):}} \\")
    for label, key in score_methods:
        vals = [_render_val(key, d, p) for d in ("linear", "nonlinear") for p in (10, 20, 30)]
        lines.append(
            f"    {label:<45}     & {vals[0]:<25} & {vals[1]:<25} & {vals[2]:<17} & {vals[3]:<17} & {vals[4]:<17} & {vals[5]:<17} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("alarm_full_nonlinear")
def build_alarm_full_nonlinear(results_dir: Path) -> str:
    """ALARM nonlinear, all rates (replaced by tables_revision)."""
    return build_alarm_full(results_dir, dgp="nonlinear")


@register("alarm_full")
def build_alarm_full(results_dir: Path, dgp: str = "linear") -> str:
    """ALARM, all rates 0.1/0.2/0.4 (replaced by tables_revision)."""
    fname = "alarm_results.csv" if dgp == "linear" else "alarm_nonlinear_results.csv"
    df = load_csv(results_dir, fname)
    if df is None:
        return f"% MISSING: {fname}"
    df = df[~df["failed"]]

    # Merge FZ-Rubin (micd_pooled) and MissDAG from the newbase CSV
    nb = load_csv(results_dir, f"newbase_alarm_{dgp}.csv", required=False)
    if nb is not None:
        nb = nb[(~nb["failed"]) & nb["method"].isin(["micd_pooled", "missdag"])]
        df = pd.concat([df, nb], ignore_index=True)

    methods = [
        ("P-CI",                "ci_oracle_fast"),
        ("CC",                  "complete_case"),
        ("TW",                  "testwise"),
        ("FZ-v",                "impute_fisherz"),
        ("FZ-R",                "micd_pooled"),
        (r"MissDAG$^\dagger$",  "missdag"),
    ]

    def _cell(rate: float, method: str, mech: str) -> str:
        sub = df[(df["rate"] == rate) & (df["mechanism"] == mech) & (df["method"] == method)]
        if len(sub) == 0:
            return " "
        med = fmt_dec1(sub["shd"].median())
        skel = compute_skeleton_shd(sub, n_true_edges=46).median()  # ALARM has 46 true edges
        skel_s = "" if pd.isna(skel) else f"\\,{{\\scriptsize({fmt_int(skel)})}}"
        val = med + skel_s
        if method == "missdag":
            val = gray_dagger(val)
        return val

    col_spec = "ll" + "c" * len(methods)
    header_methods = " & ".join(label for label, _ in methods)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Rate & Mechanism & {header_methods} \\\\",
        r"    \midrule",
    ]
    # 0% block: single row for the no-missingness baseline (Complete mechanism)
    cells = [_cell(0.1, key, "complete") for _, key in methods]
    lines.append("    0\\% & Complete & " + " & ".join(cells) + r" \\")
    lines.append(r"    \midrule")
    # Rate-stratified blocks: one row per (rate, mechanism)
    for ri, (rate_label, rate) in enumerate((("10\\%", 0.1), ("20\\%", 0.2), ("40\\%", 0.4))):
        for mi, (mech_label, mech_key) in enumerate(
                (("MAR", "mar"), ("MNAR", "mnar"), ("Mixed", "mixed"))):
            rate_col = rate_label if mi == 0 else "    "
            cells = [_cell(rate, key, mech_key) for _, key in methods]
            lines.append(
                f"    {rate_col} & {mech_label:<5} & " + " & ".join(cells) + r" \\"
            )
        if ri < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("power")
def build_power(results_dir: Path) -> str:
    """Power for sample-size planning (Appendix L).

    PAIR-CI rejection rate at signal > 0, averaged over MAR and MNAR at
    dim_x = 5, by DGP, signal and n in {500, 1000, 2000}.
    """
    df = load_csv(results_dir, "ci_calibration_results.csv")
    if df is None:
        return "% MISSING: ci_calibration_results.csv"
    df = df[~df["failed"]]
    col = "rejected" if "rejected" in df.columns else "reject"
    df = df[(df["method"] == "Ours") & (df["variant"] == "Integrated")
            & (df["dim_x"] == 5) & (df["signal"] > 0)
            & (df["mechanism"].isin(["mar", "mnar"]))]

    dgps = [("Linear Gaussian", "linear_gaussian"),
            ("Post-nonlinear",   "post_nonlinear"),
            ("Latent confounder", "latent_confounder")]
    signals = [0.3, 0.6, 1.0]
    ns = [500, 1000, 2000]

    lines = [
        r"\begin{tabular}{llccc}",
        r"    \toprule",
        r"    DGP & Signal & $n{=}500$ & $n{=}1000$ & $n{=}2000$ \\",
        r"    \midrule",
    ]
    for i, (dgp_label, dgp_key) in enumerate(dgps):
        for j, sig in enumerate(signals):
            row_label = dgp_label if j == 0 else "   "
            sub = df[(df["dgp"] == dgp_key) & (df["signal"] == sig)]
            cells = []
            for n in ns:
                s = sub[sub["n"] == n]
                v = s[col].mean() if len(s) else float("nan")
                cells.append(fmt_frac(v, digits=2))
            lines.append(
                f"    {row_label:<18} & {sig:.1f} & {cells[0]:<6} & {cells[1]:<6} & {cells[2]:<6} \\\\"
            )
        if i < len(dgps) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("realworld")
def build_realworld(results_dir: Path) -> str:
    """Submitted Table 3 layout: ALARM and HAILFINDER at 20% missingness,
    median SHD [IQR]. Replaced by the tables_revision builder of the same
    name."""
    alarm = load_csv(results_dir, "alarm_results.csv")
    hailfinder = load_csv(results_dir, "hailfinder_results.csv")
    nb_alarm = load_csv(results_dir, "newbase_alarm_linear.csv", required=False)
    nb_hf = load_csv(results_dir, "newbase_hailfinder_nonlinear.csv", required=False)
    if any(x is None for x in (alarm, hailfinder)):
        return "% MISSING: one of alarm/hailfinder results.csv"
    alarm = alarm[(~alarm["failed"]) & (alarm["rate"] == 0.2)]
    hailfinder = hailfinder[(~hailfinder["failed"]) & (hailfinder["rate"] == 0.2)]
    # Merge FZ-Rubin (micd_pooled) from newbase CSVs if available
    if nb_alarm is not None:
        nb_alarm = nb_alarm[(~nb_alarm["failed"]) & (nb_alarm["rate"] == 0.2)
                            & (nb_alarm["method"] == "micd_pooled")]
        alarm = pd.concat([alarm, nb_alarm], ignore_index=True)
    if nb_hf is not None:
        nb_hf = nb_hf[(~nb_hf["failed"]) & (nb_hf["rate"] == 0.2)
                      & (nb_hf["method"] == "micd_pooled")]
        hailfinder = pd.concat([hailfinder, nb_hf], ignore_index=True)

    methods = [
        ("PAIR-CI (fast)",                              "ci_oracle_fast"),        ("Complete case",                               "complete_case"),

        ("Test-wise",                                   "testwise"),
        ("FZ-vote",                             "impute_fisherz"),
        (r"FZ-Rubin (\citeauthor{witte2022multiple})",  "micd_pooled"),
    ]
    mechs = ("mar", "mnar", "mixed")
    networks = [("alarm", alarm), ("hf", hailfinder)]

    # Collect medians and IQRs
    medians: dict[tuple[str, str, str], float] = {}
    iqrs: dict[tuple[str, str, str], float] = {}
    for _, key in methods:
        for net_name, net_df in networks:
            for mech in mechs:
                r = net_df[(net_df["method"] == key) & (net_df["mechanism"] == mech)]
                if len(r):
                    medians[(key, net_name, mech)] = float(r["shd"].median())
                    q1, q3 = r["shd"].quantile([0.25, 0.75])
                    iqrs[(key, net_name, mech)] = float(q3 - q1)
                else:
                    medians[(key, net_name, mech)] = float("nan")
                    iqrs[(key, net_name, mech)] = float("nan")

    # Bold lowest per column
    bold_mask: dict[tuple[str, str, str], bool] = {}
    for net_name, _ in networks:
        for mech in mechs:
            col = [medians[(k, net_name, mech)] for _, k in methods]
            mask = bold_lowest(col, tol=0.1)
            for (_, k), b in zip(methods, mask):
                bold_mask[(k, net_name, mech)] = b

    def _cell(k, net_name, mech):
        v = medians[(k, net_name, mech)]
        iqr = iqrs[(k, net_name, mech)]
        if pd.isna(v):
            return "--"
        med_s = fmt_dec1(v)
        iqr_s = "" if pd.isna(iqr) else f"\\,{{\\scriptsize[{fmt_int(iqr)}]}}"
        body = bold(med_s, bold_mask[(k, net_name, mech)]) + iqr_s
        # Complete case on HAILFINDER MNAR returns the empty skeleton in
        # every replicate (IQR 0), so its SHD hides F1 = 0.
        if k == "complete_case" and net_name == "hf" and mech == "mnar" and iqr == 0:
            body = gray_dagger(body)
        return body

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"    \toprule",
        r"                  & \multicolumn{3}{c}{ALARM ($p{=}37$)} & \multicolumn{3}{c}{HAILFINDER ($p{=}56$)} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"    Method        & MAR   & MNAR  & Mixed  & MAR    & MNAR   & Mixed \\",
        r"    \midrule",
    ]
    for label, key in methods:
        cells = [_cell(key, n, m) for (n, _) in networks for m in mechs]
        lines.append(
            f"    {label:<15} & {cells[0]:<6} & {cells[1]:<6} & {cells[2]:<7} & {cells[3]:<7} & {cells[4]:<7} & {cells[5]:<6} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("hailfinder_full")
def build_hailfinder_full(results_dir: Path) -> str:
    """HAILFINDER SHD, all rates (replaced by tables_revision)."""
    return _build_hailfinder_grid(results_dir, metric="shd")


@register("hailfinder_f1")
def build_hailfinder_f1(results_dir: Path) -> str:
    """HAILFINDER F1, all rates (replaced by tables_revision)."""
    return _build_hailfinder_grid(results_dir, metric="f1")


def _build_hailfinder_grid(results_dir: Path, metric: str) -> str:
    df = load_csv(results_dir, "hailfinder_results.csv")
    if df is None:
        return "% MISSING: hailfinder_results.csv"
    df = df[~df["failed"]]

    # Merge FZ-Rubin (micd_pooled) from the newbase CSV. MissDAG is omitted
    # because it was not run on HAILFINDER (runtime cost).
    nb = load_csv(results_dir, "newbase_hailfinder_nonlinear.csv", required=False)
    if nb is not None:
        nb = nb[(~nb["failed"]) & (nb["method"] == "micd_pooled")]
        df = pd.concat([df, nb], ignore_index=True)

    methods = [
        ("P-CI",   "ci_oracle_fast"),
        ("CC",     "complete_case"),
        ("TW",     "testwise"),
        ("FZ-v",   "impute_fisherz"),
        ("FZ-R",   "micd_pooled"),
    ]

    def _cell(rate: float, method: str, mech: str) -> str:
        sub = df[(df["rate"] == rate) & (df["mechanism"] == mech) & (df["method"] == method)]
        if len(sub) == 0:
            return " "
        v = sub[metric].median()
        s = fmt_dec1(v) if metric == "shd" else fmt_frac(v)
        # Median F1 of 0 means an empty skeleton, whose SHD can look
        # competitive; flag the cell in both the SHD and F1 tables.
        median_f1 = sub["f1"].median()
        if not pd.isna(median_f1) and abs(median_f1) < 1e-6:
            s = gray_dagger(s)
        return s

    col_spec = "ll" + "c" * len(methods)
    header_methods = " & ".join(label for label, _ in methods)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Rate & Mechanism & {header_methods} \\\\",
        r"    \midrule",
    ]
    # 0% block: single row for the no-missingness baseline
    cells = [_cell(0.1, key, "complete") for _, key in methods]
    lines.append("    0\\% & Complete & " + " & ".join(cells) + r" \\")
    lines.append(r"    \midrule")
    # Rate-stratified blocks: one row per (rate, mechanism)
    for ri, (rate_label, rate) in enumerate((("10\\%", 0.1), ("20\\%", 0.2), ("40\\%", 0.4))):
        for mi, (mech_label, mech_key) in enumerate(
                (("MAR", "mar"), ("MNAR", "mnar"), ("Mixed", "mixed"))):
            rate_col = rate_label if mi == 0 else "    "
            cells = [_cell(rate, key, mech_key) for _, key in methods]
            lines.append(
                f"    {rate_col} & {mech_label:<5} & " + " & ".join(cells) + r" \\"
            )
        if ri < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("sachs_linear")
def build_sachs_linear(results_dir: Path) -> str:
    """Sachs, real data, rates 0.1/0.2/0.4 (Appendix J)."""
    df = load_csv(results_dir, "sachs_results.csv")
    if df is None:
        return "% MISSING: sachs_results.csv"
    df = df[~df["failed"]]

    # Merge FZ-Rubin (micd_pooled) and MissDAG from the newbase CSV
    nb = load_csv(results_dir, "newbase_sachs_linear.csv", required=False)
    if nb is not None:
        nb = nb[(~nb["failed"]) & nb["method"].isin(["micd_pooled", "missdag"])]
        df = pd.concat([df, nb], ignore_index=True)

    # MVPC, run on the same datasets by experiments/mvpc_sachs.py
    mv = load_csv(results_dir, "mvpc_sachs_real.csv", required=False)
    if mv is not None:
        mv = mv[(~mv["failed"].astype(bool)) & (mv["method"] == "mvpc")]
        df = pd.concat([df, mv], ignore_index=True)

    methods = [
        ("P-CI",                "ci_oracle_fast"),
        ("CC",                  "complete_case"),
        ("TW",                  "testwise"),
        ("FZ-v",                "impute_fisherz"),
        ("FZ-R",                "micd_pooled"),
        (r"MVPC$^\ddagger$",    "mvpc"),
        (r"MissDAG$^\dagger$",  "missdag"),
    ]

    def _cell(rate: float, method: str, mech: str) -> str:
        sub = df[(df["rate"] == rate) & (df["mechanism"] == mech) & (df["method"] == method)]
        if len(sub) == 0:
            return " "
        med = fmt_dec1(sub["shd"].median())
        skel = compute_skeleton_shd(sub, n_true_edges=17).median()  # Sachs has 17 true edges
        skel_s = "" if pd.isna(skel) else f"\\,{{\\scriptsize({fmt_int(skel)})}}"
        val = med + skel_s
        if method == "missdag":
            val = gray_dagger(val)
        return val

    col_spec = "ll" + "c" * len(methods)
    header_methods = " & ".join(label for label, _ in methods)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Rate & Mechanism & {header_methods} \\\\",
        r"    \midrule",
    ]
    cells = [_cell(0.1, key, "complete") for _, key in methods]
    lines.append("    0\\% & Complete & " + " & ".join(cells) + r" \\")
    lines.append(r"    \midrule")
    for ri, (rate_label, rate) in enumerate((("10\\%", 0.1), ("20\\%", 0.2), ("40\\%", 0.4))):
        for mi, (mech_label, mech_key) in enumerate(
                (("MAR", "mar"), ("MNAR", "mnar"), ("Mixed", "mixed"))):
            rate_col = rate_label if mi == 0 else "    "
            cells = [_cell(rate, key, mech_key) for _, key in methods]
            lines.append(
                f"    {rate_col} & {mech_label:<5} & " + " & ".join(cells) + r" \\"
            )
        if ri < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("sachs_nonlinear")
def build_sachs_nonlinear(results_dir: Path) -> str:
    """Sachs topology, simulated nonlinear data, rates 0.1/0.2/0.4 (App. J)."""
    df = load_csv(results_dir, "sachs_nonlinear_results.csv")
    if df is None:
        return "% MISSING: sachs_nonlinear_results.csv"
    df = df[~df["failed"]]

    # Merge FZ-Rubin (micd_pooled) and MissDAG from the newbase CSV
    nb = load_csv(results_dir, "newbase_sachs_nonlinear.csv", required=False)
    if nb is not None:
        nb = nb[(~nb["failed"]) & nb["method"].isin(["micd_pooled", "missdag"])]
        df = pd.concat([df, nb], ignore_index=True)

    # MVPC, run on the same datasets by experiments/mvpc_sachs.py
    mv = load_csv(results_dir, "mvpc_sachs_nonlinear.csv", required=False)
    if mv is not None:
        mv = mv[(~mv["failed"].astype(bool)) & (mv["method"] == "mvpc")]
        df = pd.concat([df, mv], ignore_index=True)

    methods = [
        ("P-CI",                "ci_oracle_fast"),
        ("CC",                  "complete_case"),
        ("TW",                  "testwise"),
        ("FZ-v",                "impute_fisherz"),
        ("FZ-R",                "micd_pooled"),
        (r"MVPC$^\ddagger$",    "mvpc"),
        (r"MissDAG$^\dagger$",  "missdag"),
    ]

    def _cell(rate: float, method: str, mech: str) -> str:
        sub = df[(df["rate"] == rate) & (df["mechanism"] == mech) & (df["method"] == method)]
        if len(sub) == 0:
            return " "
        med = fmt_dec1(sub["shd"].median())
        skel = compute_skeleton_shd(sub, n_true_edges=17).median()  # Sachs has 17 true edges
        skel_s = "" if pd.isna(skel) else f"\\,{{\\scriptsize({fmt_int(skel)})}}"
        val = med + skel_s
        if method == "missdag":
            val = gray_dagger(val)
        return val

    col_spec = "ll" + "c" * len(methods)
    header_methods = " & ".join(label for label, _ in methods)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Rate & Mechanism & {header_methods} \\\\",
        r"    \midrule",
    ]
    cells = [_cell(0.1, key, "complete") for _, key in methods]
    lines.append("    0\\% & Complete & " + " & ".join(cells) + r" \\")
    lines.append(r"    \midrule")
    for ri, (rate_label, rate) in enumerate((("10\\%", 0.1), ("20\\%", 0.2), ("40\\%", 0.4))):
        for mi, (mech_label, mech_key) in enumerate(
                (("MAR", "mar"), ("MNAR", "mnar"), ("Mixed", "mixed"))):
            rate_col = rate_label if mi == 0 else "    "
            cells = [_cell(rate, key, mech_key) for _, key in methods]
            lines.append(
                f"    {rate_col} & {mech_label:<5} & " + " & ".join(cells) + r" \\"
            )
        if ri < 2:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("shd_linear")
def build_shd_linear(results_dir: Path) -> str:
    """p = 10 linear, SHD by mechanism (replaced by tables_revision)."""
    return _build_p10_per_mech(results_dir, "linear")


@register("shd_nonlinear")
def build_shd_nonlinear(results_dir: Path) -> str:
    """p = 10 nonlinear, SHD by mechanism (replaced by tables_revision)."""
    return _build_p10_per_mech(results_dir, "nonlinear")


def _build_p10_per_mech(results_dir: Path, dgp: str) -> str:
    df = load_p_sweep(results_dir, 10, dgp)
    if df is None:
        return "% MISSING p=10 {dgp}"
    # Merge FZ-Rubin (micd_pooled) and MissDAG from the newbase CSVs.
    nb = load_newbase_p_sweep(results_dir, 10, dgp)
    if nb is not None:
        df = pd.concat([df, nb], ignore_index=True)
    methods = [
        ("PAIR-CI",                                                 "ours"),        ("Complete case",                                        "complete_case"),

        ("Test-wise",                                             "testwise"),
        ("FZ-vote",                                            "impute_fisherz"),
        (r"FZ-Rubin (\citeauthor{witte2022multiple})",           "micd_pooled"),
        (r"MissDAG$^\dagger$ (\citeauthor{gao2022missdag})",     "missdag"),
    ]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Method                                            & Complete & MAR & MNAR & Mixed \\",
        r"    \midrule",
    ]
    for label, key in methods:
        sub = ours_row(df) if key == "ours" else df[df["method"] == key]
        per_mech_shd = sub.groupby("mechanism")["shd"].median()
        skel = compute_skeleton_shd(sub)
        per_mech_skel = skel.groupby(sub["mechanism"]).median()
        cells = []
        for m in ("complete", "mar", "mnar", "mixed"):
            v = per_mech_shd.get(m, float('nan'))
            sv = per_mech_skel.get(m, float('nan'))
            med_s = fmt_int(v)
            skel_s = "" if pd.isna(sv) else f"\\,{{\\scriptsize({fmt_int(sv)})}}"
            body = med_s + skel_s
            if key == "missdag":
                body = gray_dagger(body)
            cells.append(body)
        lines.append(
            f"    {label:<49} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _robustness_table(results_dir: Path, mech: str) -> str:
    """Median SHD by missingness rate and n, p = 10 nonlinear, one mechanism."""
    methods = [
        ("PAIR-CI",          "ours"),        ("Complete case", "complete_case"),

        ("Test-wise",      "testwise"),
        ("FZ-vote",     "impute_fisherz"),
        (r"MVPC$^\ddagger$", "mvpc"),
    ]
    ns = (500, 1000, 2000)
    rates = (0.1, 0.3, 0.5)
    # MVPC was run on the same datasets by experiments/mvpc_robustness.py.
    mv = load_csv(results_dir, "mvpc_robustness.csv", required=False)
    if mv is not None:
        mv = mv[(~mv["failed"].astype(bool)) & (mv["method"] == "mvpc") & (mv["mechanism"] == mech)]

    # Compute per (method, n, rate)
    values: dict[tuple[str, int, float], float] = {}
    for _, key in methods:
        for n in ns:
            for rate in rates:
                if key == "mvpc":
                    sub = mv[(mv["n_obs"] == n) & (mv["miss_rate"] == rate)] if mv is not None else []
                    values[(key, n, rate)] = float(sub["shd"].median()) if len(sub) else float("nan")
                    continue
                df = load_csv(results_dir, f"robustness_{mech}_{rate}_{n}.csv", required=False)
                if df is None:
                    values[(key, n, rate)] = float("nan")
                    continue
                df = df[~df["failed"]]
                sub = ours_row(df) if key == "ours" else df[df["method"] == key]
                values[(key, n, rate)] = float(sub["shd"].median()) if len(sub) else float("nan")

    lines = [
        r"\begin{tabular}{lccccccccc}",
        r"    \toprule",
        r"                  & \multicolumn{3}{c}{$n{=}500$} & \multicolumn{3}{c}{$n{=}1000$} & \multicolumn{3}{c}{$n{=}2000$} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"    Method        & 10\% & 30\% & 50\%            & 10\%  & 30\%  & 50\%           & 10\%  & 30\%  & 50\%  \\",
        r"    \midrule",
    ]
    for label, key in methods:
        # Robustness table uses 1-decimal precision (10.0, 9.0, etc.)
        cells = [fmt_dec1(values[(key, n, r)]) for n in ns for r in rates]
        lines.append(
            f"    {label:<13} & {cells[0]:>4} & {cells[1]:>4} & {cells[2]:>4} & {cells[3]:>4} & {cells[4]:>4} & {cells[5]:>4} & {cells[6]:>4} & {cells[7]:>4} & {cells[8]:>4} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


@register("robustness_mar")
def build_robustness_mar(results_dir: Path) -> str:
    """Robustness to rate and n, p = 10 nonlinear, MAR (Appendix F)."""
    return _robustness_table(results_dir, "mar")


@register("robustness_mnar")
def build_robustness_mnar(results_dir: Path) -> str:
    """Robustness to rate and n, p = 10 nonlinear, MNAR (Appendix F)."""
    return _robustness_table(results_dir, "mnar")


@register("degradation_fpr")
def build_degradation_fpr(results_dir: Path) -> str:
    """Imputation degradation, FPR at signal 0 (Appendix Q)."""
    return _build_degradation(results_dir, signal=0.0)


@register("degradation_power")
def build_degradation_power(results_dir: Path) -> str:
    """Imputation degradation, power at signal 0.6 (Appendix Q)."""
    return _build_degradation(results_dir, signal=0.6)


def _build_degradation(results_dir: Path, signal: float) -> str:
    df = load_csv(results_dir, "imputation_degradation.csv")
    if df is None:
        return "% MISSING: imputation_degradation.csv"
    df = df[df["signal"] == signal]
    is_fpr = signal == 0.0

    strategies = [("MICE", "mice"), ("Mean", "mean"), ("Marginal", "sample")]

    lines = [
        r"\begin{tabular}{lccc}",
        r"    \toprule",
        r"    Strategy  & Complete & MAR                    & MNAR                   \\",
        r"    \midrule",
    ]
    for label, key in strategies:
        sub = df[df["strategy"] == key]
        vals = {}
        for mech in ("complete", "mar", "mnar"):
            m = sub[sub["mechanism"] == mech]
            vals[mech] = m["reject"].mean() if "reject" in m.columns else m["rejected"].mean()
        # Highlight red if > 0.05 (FPR only)
        cells = []
        for mech in ("complete", "mar", "mnar"):
            v = vals[mech]
            s = fmt_frac(v)
            if is_fpr:
                s = red(s, v > 0.05)
            # Mean and marginal imputation inflate the FPR under MAR/MNAR
            # (tab_degradation_fpr), so these cells cannot be read as power.
            if not is_fpr and key in ("mean", "sample") and mech in ("mar", "mnar"):
                s = gray_dagger(s)
            cells.append(s)
        lines.append(
            f"    {label:<10}& {cells[0]:<8}& {cells[1]:<22}& {cells[2]:<22}\\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build every paper table directly from result CSVs."
    )
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing the result CSVs")
    parser.add_argument("--output-dir", type=str, default="tables",
                        help="Where to write the .tex fragments")
    parser.add_argument("--table", type=str, default=None,
                        help=f"Build only one table (choices: {', '.join(sorted(TABLES))})")
    parser.add_argument("--print", action="store_true",
                        help="Also print tables to stdout")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # tables_revision builders replace those of the same name here and add
    # the remaining tables.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tables_revision import REVISION_TABLES
    TABLES.update(REVISION_TABLES)

    if args.table and args.table not in TABLES:
        print(f"Unknown table: {args.table}. Choices: {sorted(TABLES)}", file=sys.stderr)
        sys.exit(1)

    names = [args.table] if args.table else list(TABLES)

    print(f"Building {len(names)} table(s) from {results_dir}/ -> {output_dir}/")
    for name in names:
        builder = TABLES[name]
        try:
            tex = builder(results_dir)
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        out = output_dir / f"tab_{name}.tex"
        out.write_text(tex + "\n")
        print(f"  ✓ {name} -> {out}")
        if args.print:
            print()
            print(tex)
            print()


if __name__ == "__main__":
    main()
