"""
Print a report of headline numbers from the result CSVs, with optional
LaTeX row fragments, and flag missing result files.

A diagnostic aid: the section titles follow the submitted version's table
numbering, and the camera-ready tables are built by build_tables.py.

Usage:
  python analysis/extract_paper_numbers.py                # full report
  python analysis/extract_paper_numbers.py --results-dir results/
  python analysis/extract_paper_numbers.py --latex         # emit LaTeX table fragments
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")


def load(name, required=True):
    path = RESULTS_DIR / name
    if not path.exists():
        if required:
            print(f"  *** MISSING: {name}")
        return None
    df = pd.read_csv(path)
    return df


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def subsection(title):
    print(f"\n--- {title} ---\n")


# =====================================================================
# Standalone calibration (Table 1)
# =====================================================================

def table_calibration(latex=False):
    section("TABLE 1: CALIBRATION (FPR under H0)")
    df = load("ci_calibration_results.csv")
    if df is None:
        return

    null = df[df["signal"] == 0]
    methods = [
        ("Ours (Integrated)", "Ours", "Integrated"),
        ("Fisher-Z (Complete Case)", "Fisher-Z", "Complete Case"),
        ("Fisher-Z (Imputed)", "Fisher-Z", "Imputed"),
        ("Fisher-Z (Rubin's Rules)", "Fisher-Z", "Rubin's Rules"),
        ("GCM (Complete Case)", "GCM", "Complete Case"),
        ("GCM (Imputed)", "GCM", "Imputed"),
        ("KCI (Complete Case)", "KCI", "Complete Case"),
        ("KCI (Imputed)", "KCI", "Imputed"),
    ]

    for label, method, variant in methods:
        sub = null[(null["method"] == method) & (null["variant"] == variant)]
        if len(sub) == 0:
            print(f"  {label}: NO DATA")
            continue
        by_mech = sub.groupby("mechanism")["rejected"].mean()
        c = by_mech.get("complete", float("nan"))
        m = by_mech.get("mar", float("nan"))
        n = by_mech.get("mnar", float("nan"))
        flag = lambda v: f"\\textcolor{{red}}{{{v:.3f}}}" if v > 0.05 else f"{v:.3f}"
        if latex:
            print(f"        {label:30s} & {flag(c)} & {flag(m)} & {flag(n)} \\\\")
        else:
            print(f"  {label:30s}  complete={c:.3f}  MAR={m:.3f}  MNAR={n:.3f}")

    subsection("Prose numbers")
    ours = null[(null["method"] == "Ours")]
    print(f"  Ours overall FPR: {ours['rejected'].mean():.4f}")
    for mech in ["complete", "mar", "mnar"]:
        sub = ours[ours["mechanism"] == mech]
        print(f"  Ours {mech}: {sub['rejected'].mean():.4f}")


# =====================================================================
# p = 10 graph recovery
# =====================================================================

def table_p10(latex=False):
    section("TABLES 2-3: p=10 GRAPH RECOVERY")
    for dgp, label in [("linear", "Linear Gaussian"), ("nonlinear", "Nonlinear")]:
        df = load(f"p10_{dgp}_results.csv")
        if df is None:
            df_parts = []
            for mech in ["complete", "mar", "mnar", "mixed"]:
                part = load(f"p10_{dgp}_{mech}.csv", required=False)
                if part is not None:
                    df_parts.append(part)
            if df_parts:
                df = pd.concat(df_parts, ignore_index=True)
            else:
                print(f"  *** MISSING: p10_{dgp} results")
                continue

        df = df[df["failed"] == False]
        subsection(f"p=10 {label}: median SHD")
        METHOD_MAP = {
            "ci_oracle_cached": "Ours",
            "testwise": "Testwise",
            "impute_fisherz": "Impute-FZ",
            "complete_case": "Complete case",
        }
        for method, name in METHOD_MAP.items():
            sub = df[df["method"] == method]
            if len(sub) == 0:
                continue
            by_mech = sub.groupby("mechanism")["shd"].median()
            c = by_mech.get("complete", float("nan"))
            m = by_mech.get("mar", float("nan"))
            n = by_mech.get("mnar", float("nan"))
            x = by_mech.get("mixed", float("nan"))
            if latex:
                print(f"        {name:15s} & {c:.0f} & {m:.0f} & {n:.0f} & {x:.0f} \\\\")
            else:
                print(f"  {name:15s}  complete={c:.1f}  MAR={m:.1f}  MNAR={n:.1f}  Mixed={x:.1f}")

        subsection(f"p=10 {label}: IQR for Ours")
        ours = df[df["method"] == "ci_oracle_cached"]
        for mech in ["complete", "mar", "mnar", "mixed"]:
            sub = ours[ours["mechanism"] == mech]["shd"]
            print(f"  {mech}: median={sub.median():.1f}, Q25={sub.quantile(0.25):.1f}, Q75={sub.quantile(0.75):.1f}")


# =====================================================================
# Scaling comparison (Table 2)
# =====================================================================

def table_scaling(latex=False):
    section("TABLE 4: SCALING COMPARISON (median SHD across mechanisms)")

    all_data = {}
    for pval in [10, 20, 30]:
        fname = f"p{pval}_full_results.csv"
        df = load(fname, required=False)
        if df is None:
            # Try assembling from parts
            parts = []
            for dgp in ["linear", "nonlinear"]:
                for mech in ["complete", "mar", "mnar", "mixed"]:
                    part = load(f"p{pval}_{dgp}_{mech}.csv", required=False)
                    if part is not None:
                        parts.append(part)
            if parts:
                df = pd.concat(parts, ignore_index=True)
        if df is not None:
            all_data[pval] = df[df["failed"] == False]

    METHOD_MAP = {
        "ci_oracle_cached": "Ours",
        "ci_oracle_fast": "Ours",
        "testwise": "Testwise",
        "impute_fisherz": "Impute-FZ",
        "complete_case": "Complete case",
    }

    for dgp in ["linear", "nonlinear"]:
        subsection(f"{dgp}: median SHD by p")
        for method_key in ["ci_oracle_cached", "ci_oracle_fast", "testwise",
                           "impute_fisherz", "complete_case"]:
            name = METHOD_MAP[method_key]
            vals = []
            for pval in [10, 20, 30]:
                if pval not in all_data:
                    vals.append("?")
                    continue
                df = all_data[pval]
                sub = df[(df["method"] == method_key) &
                         (df["dgp"] == dgp if "dgp" in df.columns else True)]
                if len(sub) == 0:
                    # Both method names denote PAIR-CI.
                    alt = "ci_oracle_fast" if method_key == "ci_oracle_cached" else "ci_oracle_cached"
                    sub = df[(df["method"] == alt) &
                             (df["dgp"] == dgp if "dgp" in df.columns else True)]
                if len(sub) > 0:
                    vals.append(f"{sub['shd'].median():.0f}")
                else:
                    vals.append("?")
            if all(v == "?" for v in vals):
                continue
            print(f"  {name:15s}  p=10: {vals[0]:>4s}  p=20: {vals[1]:>4s}  p=30: {vals[2]:>4s}")

    subsection("p=30 nonlinear: SHD by mechanism (mechanism stability)")
    if 30 in all_data:
        df = all_data[30]
        for method in ["ci_oracle_fast", "testwise", "impute_fisherz", "complete_case"]:
            sub = df[(df["method"] == method) &
                     (df["dgp"] == "nonlinear" if "dgp" in df.columns else True)]
            if len(sub) == 0:
                continue
            by_mech = sub.groupby("mechanism")["shd"].median()
            rng = by_mech.max() - by_mech.min()
            print(f"  {METHOD_MAP.get(method, method):15s}  {dict(by_mech.round(1))}  range={rng:.1f}")


# =====================================================================
# Robustness to missingness rate and sample size (Appendix F)
# =====================================================================

def table_robustness(latex=False):
    section("TABLE 7: ROBUSTNESS (p=10 nonlinear)")
    df = load("robustness_results.csv")
    if df is None:
        return

    METHOD_MAP = {
        "ci_oracle_cached": "Ours",
        "testwise": "Testwise",
        "impute_fisherz": "Impute-FZ",
        "complete_case": "Complete case",
    }

    for mech in ["mnar", "mar"]:
        subsection(f"Robustness {mech.upper()}: median SHD")
        sub = df[df["mechanism"] == mech]
        for method, name in METHOD_MAP.items():
            msub = sub[sub["method"] == method]
            vals = []
            for n_obs in [500, 1000, 2000]:
                for mr in [0.1, 0.3, 0.5]:
                    cell = msub[(msub["n_obs"] == n_obs) &
                                (msub["miss_rate"] == mr)]["shd"].median()
                    vals.append(f"{cell:.1f}" if not np.isnan(cell) else "?")
            if latex:
                print(f"        {name:13s} & " + " & ".join(vals) + " \\\\")
            else:
                print(f"  {name:15s}  " + "  ".join(f"{v:>5s}" for v in vals))

    subsection("Impute-FZ degradation with n")
    fz_mnar = df[(df["method"] == "impute_fisherz") & (df["mechanism"] == "mnar")]
    for n_obs in [500, 1000, 2000]:
        med = fz_mnar[fz_mnar["n_obs"] == n_obs]["shd"].median()
        print(f"  n={n_obs}: median SHD = {med:.1f}")


# =====================================================================
# ALARM (Appendix H)
# =====================================================================

def table_alarm(latex=False):
    section("TABLE 8: ALARM (p=37)")

    for dgp_label, fname in [("LINEAR", "alarm_results.csv"),
                              ("NONLINEAR", "alarm_nonlinear_results.csv")]:
        df = load(fname, required=(dgp_label == "LINEAR"))
        if df is None:
            continue

        subsection(f"ALARM {dgp_label}: median SHD at 20% missingness")
        # Complete baseline from 10% rate
        comp = df[(df["rate"] == 0.1) & (df["mechanism"] == "complete")]
        r20 = df[df["rate"] == 0.2]

        for method in ["ci_oracle_fast", "testwise", "impute_fisherz", "complete_case"]:
            c = comp[comp["method"] == method]["shd"].median()
            m = r20[(r20["method"] == method) & (r20["mechanism"] == "mar")]["shd"].median()
            n = r20[(r20["method"] == method) & (r20["mechanism"] == "mnar")]["shd"].median()
            x = r20[(r20["method"] == method) & (r20["mechanism"] == "mixed")]["shd"].median()
            print(f"  {method:20s}  complete={c:.1f}  MAR={m:.1f}  MNAR={n:.1f}  Mixed={x:.1f}")

        subsection(f"ALARM {dgp_label}: full table (all rates)")
        for rate in [0.1, 0.2, 0.4]:
            sub = df[df["rate"] == rate]
            print(f"  Rate={rate}:")
            for method in ["ci_oracle_fast", "testwise", "impute_fisherz", "complete_case"]:
                by_mech = sub[sub["method"] == method].groupby("mechanism")["shd"].median()
                print(f"    {method:20s}  {dict(by_mech.round(1))}")


# =====================================================================
# HAILFINDER (Appendix I)
# =====================================================================

def table_hailfinder(latex=False):
    section("HAILFINDER (p=56)")
    df = load("hailfinder_results.csv")
    if df is None:
        return

    subsection("HAILFINDER: median SHD at 20% missingness")
    comp = df[(df["rate"] == 0.1) & (df["mechanism"] == "complete")]
    r20 = df[df["rate"] == 0.2]

    for method in ["ci_oracle_fast", "testwise", "impute_fisherz", "complete_case"]:
        c = comp[comp["method"] == method]["shd"].median()
        m = r20[(r20["method"] == method) & (r20["mechanism"] == "mar")]["shd"].median()
        n = r20[(r20["method"] == method) & (r20["mechanism"] == "mnar")]["shd"].median()
        x = r20[(r20["method"] == method) & (r20["mechanism"] == "mixed")]["shd"].median()
        print(f"  {method:20s}  complete={c:.1f}  MAR={m:.1f}  MNAR={n:.1f}  Mixed={x:.1f}")

    subsection("HAILFINDER: median runtime (s)")
    print(df.groupby("method")["runtime"].median().round(1).to_string())


# =====================================================================
# Variance estimator comparison (Appendix K)
# =====================================================================

def table_variance_comparison():
    section("APPENDIX: VARIANCE COMPARISON (NB vs Bayle)")
    vc = load("variance_comparison_calibration.csv")
    vg = load("variance_comparison_graph.csv")

    if vc is not None:
        for signal, label in [(0, "FPR (signal=0)"), (0.3, "Power (signal=0.3)"),
                               (0.6, "Power (signal=0.6)"), (1.0, "Power (signal=1.0)")]:
            subsection(label)
            sub = vc[vc["signal"] == signal]
            print(sub.groupby(["variance_estimator", "mechanism"])["reject"]
                  .mean().unstack("mechanism").round(3).to_string())

    if vg is not None:
        subsection("Graph recovery: median SHD")
        print(vg.groupby(["variance_estimator", "mechanism"])["shd"]
              .median().unstack("mechanism").to_string())
        subsection("Graph recovery: median F1")
        print(vg.groupby(["variance_estimator", "mechanism"])["f1"]
              .median().unstack("mechanism").round(3).to_string())


# =====================================================================
# Early stopping (Appendix P)
# =====================================================================

def table_early_stopping():
    section("APPENDIX: EARLY STOPPING")
    df = load("early_stopping_calibration.csv")
    if df is None:
        return

    fpr_es = df["reject_early_stop"].mean()
    fpr_no = df["reject_no_early_stop"].mean()
    agree = (df["reject_early_stop"] == df["reject_no_early_stop"]).mean()
    print(f"  FPR with early stopping:    {fpr_es:.4f}")
    print(f"  FPR without early stopping: {fpr_no:.4f}")
    print(f"  Decision agreement:         {agree:.4f}")

    subsection("By DGP and mechanism")
    print("Early stop:")
    print(df.groupby(["dgp", "mechanism"])["reject_early_stop"]
          .mean().unstack("mechanism").round(4).to_string())
    print("\nNo early stop:")
    print(df.groupby(["dgp", "mechanism"])["reject_no_early_stop"]
          .mean().unstack("mechanism").round(4).to_string())


# =====================================================================
# Imputation degradation (Appendix Q)
# =====================================================================

def table_imputation_degradation():
    section("APPENDIX: IMPUTATION DEGRADATION")
    df = load("imputation_degradation.csv")
    if df is None:
        return

    null = df[df["signal"] == 0]
    subsection("FPR under null (target: 5%)")
    print(null.groupby(["strategy", "mechanism"])["reject"]
          .mean().unstack("mechanism").round(3).to_string())

    subsection("Power at signal=0.6")
    power = df[df["signal"] == 0.6]
    print(power.groupby(["strategy", "mechanism"])["reject"]
          .mean().unstack("mechanism").round(3).to_string())


# =====================================================================
# Cached vs per-query imputation agreement (Section 3.2)
# =====================================================================

def table_caching():
    section("SECTION 5.2: CACHING AGREEMENT")
    df = load("caching_validation_large_p.csv")
    if df is None:
        return

    print("Agreement by p:")
    print(df.groupby("p")["agreement"].mean().round(4).to_string())


# =====================================================================
# Sachs (Appendix J)
# =====================================================================

def table_sachs():
    section("SACHS (p=11)")
    df = load("sachs_results.csv")
    if df is None:
        return

    r20 = df[df["rate"] == 0.2]
    subsection("Sachs: median SHD at 20%")
    print(r20.groupby(["method", "mechanism"])["shd"]
          .median().unstack("mechanism").round(1).to_string())
    subsection("Sachs: median F1 at 20%")
    print(r20.groupby(["method", "mechanism"])["f1"]
          .median().unstack("mechanism").round(3).to_string())


# =====================================================================
# K = 10 smoke test (a development check, not reported in the paper)
# =====================================================================

def table_k10_smoke():
    section("K10 SMOKE TEST (Bayle + K=10 + n_trees=200)")
    df = load("k10_smoke_test.csv")
    if df is None:
        return

    print("FPR by DGP and mechanism:")
    print(df.groupby(["dgp", "mechanism"])["reject"]
          .mean().unstack("mechanism").round(3).to_string())
    print(f"\nOverall by mechanism:")
    print(df.groupby("mechanism")["reject"].mean().round(3).to_string())
    worst = df.groupby(["dgp", "mechanism"])["reject"].mean().max()
    print(f"\nWorst-case FPR: {worst:.3f}")
    if worst < 0.05:
        print(">>> GO: FPR below 5%")
    else:
        print(">>> NO-GO: FPR exceeds 5%")


# =====================================================================
# Numbers quoted in the prose of the submitted version
# =====================================================================

def prose_numbers():
    section("PROSE NUMBER CHECKLIST")

    # Abstract
    print("ABSTRACT:")
    df = load("ci_calibration_results.csv")
    if df is not None:
        null = df[df["signal"] == 0]
        ours_mnar = null[(null["method"] == "Ours") & (null["mechanism"] == "mnar")]["rejected"].mean()
        print(f"  'FPR of X% under MNAR': {ours_mnar:.3f} ({ours_mnar*100:.1f}%)")

        fz_imp_mnar = null[(null["method"] == "Fisher-Z") & (null["variant"] == "Imputed") & (null["mechanism"] == "mnar")]["rejected"].mean()
        gcm_imp_mnar = null[(null["method"] == "GCM") & (null["variant"] == "Imputed") & (null["mechanism"] == "mnar")]["rejected"].mean()
        print(f"  'Fisher-Z and GCM reject 59-78%': Fisher-Z={fz_imp_mnar:.3f}, GCM={gcm_imp_mnar:.3f}")

    # Scaling gaps
    print("\nSCALING GAPS (nonlinear):")
    for pval in [10, 20, 30]:
        fname = f"p{pval}_full_results.csv"
        df = load(fname, required=False)
        if df is None:
            parts = []
            for dgp in ["linear", "nonlinear"]:
                for mech in ["complete", "mar", "mnar", "mixed"]:
                    p = load(f"p{pval}_{dgp}_{mech}.csv", required=False)
                    if p is not None:
                        parts.append(p)
            if parts:
                df = pd.concat(parts, ignore_index=True)
        if df is None:
            continue
        df = df[df["failed"] == False]
        nl = df[df["dgp"] == "nonlinear"] if "dgp" in df.columns else df
        ours_key = "ci_oracle_fast" if "ci_oracle_fast" in nl["method"].values else "ci_oracle_cached"
        ours = nl[nl["method"] == ours_key]["shd"].median()
        tw = nl[nl["method"] == "testwise"]["shd"].median()
        fz = nl[nl["method"] == "impute_fisherz"]["shd"].median()
        print(f"  p={pval}: Ours={ours:.0f}, Testwise={tw:.0f} (gap={tw-ours:.0f}), "
              f"Impute-FZ={fz:.0f} (gap={fz-ours:.0f})")

    # Caching
    print("\nCACHING AGREEMENT:")
    cv = load("caching_validation_large_p.csv")
    if cv is not None:
        for p in [5, 10, 20]:
            ag = cv[cv["p"] == p]["agreement"].mean()
            print(f"  p={p}: {ag:.1%}")


# =====================================================================
# Main
# =====================================================================

def main():
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description="Extract paper numbers from results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--latex", action="store_true",
                        help="Emit LaTeX table fragments")
    args = parser.parse_args()
    RESULTS_DIR = Path(args.results_dir)

    print("=" * 70)
    print("  PAPER NUMBERS EXTRACTION")
    print(f"  Results directory: {RESULTS_DIR.resolve()}")
    print("=" * 70)

    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    print(f"\n  Found {len(csvs)} CSV files in {RESULTS_DIR}/")
    for f in csvs:
        rows = sum(1 for _ in open(f)) - 1
        print(f"    {f.name}: {rows} rows")

    table_calibration(args.latex)
    table_p10(args.latex)
    table_scaling(args.latex)
    table_robustness(args.latex)
    table_alarm(args.latex)
    table_hailfinder(args.latex)
    table_sachs()
    table_caching()
    table_variance_comparison()
    table_early_stopping()
    table_imputation_degradation()
    table_k10_smoke()
    prose_numbers()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
