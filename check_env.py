#!/usr/bin/env python
"""Pre-flight check for the causal_ci conda environment used by pair-ci_replication.

Run this once on the HPC after `conda env create -f environment.yml` and
before launching `reproduce_all.sh`. Exits 0 if every dependency the
pipeline actually uses imports cleanly; exits 1 with a diagnostic print
if anything is missing.

Usage:
    conda activate causal_ci
    python check_env.py
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

FAIL = False


def _mark_fail(msg: str) -> None:
    global FAIL
    FAIL = True
    print(f"  FAIL: {msg}")


def check_python() -> None:
    major, minor = sys.version_info[:2]
    print(f"Python: {sys.version.split()[0]}")
    if (major, minor) != (3, 11):
        _mark_fail(
            f"expected Python 3.11 (pinned in environment.yml), got {major}.{minor}"
        )


def check_top_level_packages() -> None:
    print("\nTop-level packages:")
    pkgs = [
        "numpy", "pandas", "scipy", "sklearn", "matplotlib",
        "causallearn", "citest", "lightgbm", "xgboost",
    ]
    for name in pkgs:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {name:14s} {ver}")
        except Exception as exc:  # pragma: no cover
            _mark_fail(f"{name}: {type(exc).__name__}: {exc}")


def check_matplotlib_headless() -> None:
    print("\nMatplotlib headless render:")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        tmp = Path(tempfile.gettempdir()) / "mpl_env_check.pdf"
        fig.savefig(tmp)
        plt.close(fig)
        tmp.unlink()
        print("  Agg backend writes PDF: OK")
    except Exception as exc:
        _mark_fail(f"matplotlib Agg render: {type(exc).__name__}: {exc}")


def check_deep_imports() -> None:
    print("\nDeep imports used by the pipeline:")
    tests = [
        # sklearn.experimental.enable_iterative_imputer is a module whose only
        # job is to register IterativeImputer as a side-effect on import; it
        # has no attribute of the same name, so check importability only.
        ("sklearn.experimental.enable_iterative_imputer", None),
        ("sklearn.impute", "IterativeImputer"),
        ("sklearn.linear_model", "BayesianRidge"),
        ("sklearn.linear_model", "LogisticRegression"),
        ("sklearn.ensemble", "RandomForestClassifier"),
        ("sklearn.ensemble", "RandomForestRegressor"),
        ("sklearn.ensemble", "ExtraTreesClassifier"),
        ("sklearn.ensemble", "ExtraTreesRegressor"),
        ("sklearn.ensemble", "HistGradientBoostingClassifier"),
        ("sklearn.ensemble", "HistGradientBoostingRegressor"),
        ("sklearn.model_selection", "KFold"),
        ("sklearn.model_selection", "cross_val_predict"),
        ("sklearn.neighbors", "NearestNeighbors"),
        ("scipy.stats", "pearsonr"),
        ("scipy.stats", "mannwhitneyu"),
        ("scipy.linalg", None),
        ("scipy.optimize", None),
        ("causallearn.search.ConstraintBased.PC", "pc"),
        ("causallearn.utils.cit", "fisherz"),
        ("causallearn.utils.cit", "CIT"),
        ("citest", "CIMissTest"),
        ("citest.data", "Dataset"),
        ("lightgbm", "LGBMClassifier"),
        ("lightgbm", "LGBMRegressor"),
        ("xgboost", "XGBClassifier"),
        ("xgboost", "XGBRegressor"),
    ]
    for mod_name, attr in tests:
        try:
            mod = importlib.import_module(mod_name)
            if attr is not None and not hasattr(mod, attr):
                _mark_fail(f"{mod_name}.{attr} missing (module loaded but attr absent)")
                continue
            label = f"{mod_name}.{attr}" if attr else mod_name
            print(f"  {label}: OK")
        except Exception as exc:
            label = f"{mod_name}.{attr}" if attr else mod_name
            _mark_fail(f"{label}: {type(exc).__name__}: {exc}")


def check_local_modules() -> None:
    print("\nLocal pipeline modules:")
    # Add src/ and experiments/ to sys.path so we can verify the repo layout
    here = Path(__file__).resolve().parent
    for sub in ("src", "experiments"):
        sys.path.insert(0, str(here / sub))
    modules = [
        "pc_integration", "ci_oracle", "ci_oracle_fast", "ci_oracle_linear",
        "baselines_new", "dgps", "conditional_permutation",
        "graph_recovery_p10", "graph_recovery_p20", "graph_recovery_p30",
        "alarm", "hailfinder", "sachs", "sachs_nonlinear",
        "calibration", "new_baselines", "learner_comparison",
        "adversarial_kappa", "m_sensitivity", "sensitivity", "robustness",
        "alpha_sensitivity",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"  {name}: OK")
        except Exception as exc:
            _mark_fail(f"{name}: {type(exc).__name__}: {exc}")


def main() -> int:
    print("=" * 60)
    print("pair-ci_replication environment check")
    print("=" * 60)
    check_python()
    check_top_level_packages()
    check_matplotlib_headless()
    check_deep_imports()
    check_local_modules()
    print()
    if FAIL:
        print("RESULT: one or more checks FAILED — fix before running the pipeline.")
        return 1
    print("RESULT: every dependency the pipeline uses imports cleanly.")
    print("Safe to launch `./reproduce_all.sh`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
