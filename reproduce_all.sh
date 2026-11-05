#!/bin/bash
# =====================================================================
# reproduce_all.sh: submit every experiment in the paper to SLURM
# =====================================================================
# Experiment jobs are submitted at once, with no dependencies between them;
# only the combine jobs and the final table/figure build wait for their
# inputs.
#
# All in-house constraint-based methods share one simplified PC search
# (src/pc_integration.py: _pc_skeleton + _orient_edges), so that differences
# between them come from the CI test rather than the search. MVPC runs
# causal-learn's PC-stable search plus its correction step; the search check
# (experiments/search_check_baselines.py) reruns the baselines under PC-stable.
#
# Usage:
#   ./reproduce_all.sh                        # full run (153 jobs)
#   ./reproduce_all.sh --dry-run              # print the sbatch calls only
#   ./reproduce_all.sh --partition <name>     # named partition (default: the
#                                             # cluster's default partition)
#   ./reproduce_all.sh --supplementary        # add the experiments not reported
#                                             # in the paper (see README)
#   ./reproduce_all.sh --quick                # smoke test: 1 graph, 2 datasets,
#                                             # reduced reps (see README)
#   ./reproduce_all.sh --quick --force        # allow --quick over a populated results/
#
# Jobs take the conda env's python from PATH, located once at submit time
# (slurm/_common.sh explains why); set PYBIN=/path/to/python to override.
#
# With full parallelism on 16-core jobs the run takes about five days, set by
# the longest single jobs (alarm_nonlinear, the p30 nonlinear cells,
# sensitivity and alpha_sensitivity, 4-5 days each). README, "Compute and
# runtime", gives per-experiment runtimes.
# =====================================================================

set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASEDIR"

# --- Configuration ---
PARTITION=""          # empty = cluster default; override with --partition <name>
CONDA_ENV="causal_ci"
CPUS=16
MEM_STD="4G"
MEM_LARGE="8G"
TIME_SHORT="12:00:00"
TIME_MEDIUM="2-00:00:00"
TIME_LONG="5-00:00:00"
TIME_XLONG="7-00:00:00"
TIME_XXLONG="10-00:00:00"

DRY_RUN=false
QUICK=false
SUPPLEMENTARY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true; shift ;;
        --quick)          QUICK=true; shift ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --supplementary)  SUPPLEMENTARY=true; shift ;;
        --force)          FORCE=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Guard: --quick writes reduced-replicate output to the same results/*.csv
# paths as the full run, so on the released package it would overwrite the
# paper's result files. Smoke-test in a copy without results/ instead, e.g.
#   rsync -a --exclude results/ --exclude logs/ ./ ../pairci_smoke/
#   cd ../pairci_smoke && mkdir results && ./reproduce_all.sh --quick
if $QUICK && ! $DRY_RUN && ! $FORCE && [ -f results/ci_calibration_results.csv ]; then
    echo "ERROR: --quick would overwrite the released results/*.csv with smoke-test" >&2
    echo "output. Run it in a copy of the package with an empty results/ (see the" >&2
    echo "comment above this check), or pass --force to overwrite deliberately." >&2
    exit 1
fi

# --- Smoke test overrides ---
if $QUICK; then
    N_GRAPHS=1
    N_DATASETS=2
    N_REPS=5
    CALIB_REPS=3
    SACHS_REPS=3
    ALARM_REPS=3
    HAILFINDER_REPS=3
    EARLY_STOP_REPS=10
    IMP_DEGRAD_REPS=20
    M_SENS_REPS=3
    LEARNER_REPS=3
    ADV_REPS=5
    # Camera-ready experiments use each script's own reduced mode where that
    # mode keeps the --output path; gcm_ols and pooling_validity --smoke would
    # write elsewhere, so they get small explicit grids instead.
    AUDIT_QUICK="--quick"
    GCMOLS_QUICK="--n_reps 5 --sample_sizes 300 --dim_xs 2"
    IMPQ_QUICK="--quick"
    VDEC_QUICK="--quick"
    TESTVAR_QUICK="--quick"
    LROB_QUICK="--quick"
    PLACEBO_QUICK="--smoke"
    POOL_QUICK="--n_reps 2 --sample_sizes 300 --dim_xs 2 --dgps linear_gaussian"
    HEAVY_QUICK="--smoke"
    TIME_SHORT="0:30:00"
    TIME_MEDIUM="1:00:00"
    TIME_LONG="1:00:00"
    TIME_XLONG="1:00:00"
    TIME_XXLONG="1:00:00"
else
    N_GRAPHS=10
    N_DATASETS=20
    N_REPS=50
    CALIB_REPS=100
    SACHS_REPS=20
    ALARM_REPS=20
    HAILFINDER_REPS=20
    EARLY_STOP_REPS=100
    IMP_DEGRAD_REPS=200
    M_SENS_REPS=50
    LEARNER_REPS=50
    ADV_REPS=100
    AUDIT_QUICK=""; GCMOLS_QUICK=""; IMPQ_QUICK=""; VDEC_QUICK=""
    TESTVAR_QUICK=""; LROB_QUICK=""; PLACEBO_QUICK=""; POOL_QUICK=""; HEAVY_QUICK=""
fi

$DRY_RUN || mkdir -p logs results results/search_check figures tables

# --- Conda environment: create it if missing, then locate its python once ---
# Jobs run under /bin/sh (sbatch --wrap), where `conda activate` fails, so the
# env's bin/ is put on PATH instead (see slurm/_common.sh).
PYBIN="${PYBIN:-}"
if ! $DRY_RUN; then
    echo "Checking conda environment '$CONDA_ENV'..."
    if conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
        echo "  Environment '$CONDA_ENV' already exists."
    else
        echo "  Creating environment '$CONDA_ENV' from environment.yml..."
        conda env create -f "$BASEDIR/environment.yml" -n "$CONDA_ENV"
        echo "  Done."
    fi
fi
if [ -z "$PYBIN" ]; then
    PYBIN="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || true)"
fi
if [ -z "$PYBIN" ]; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda /opt/miniconda3; do
        if [ -x "$c/envs/$CONDA_ENV/bin/python" ]; then PYBIN="$c/envs/$CONDA_ENV/bin/python"; break; fi
    done
fi
if [ -z "$PYBIN" ]; then
    if $DRY_RUN; then
        PYBIN="/path/to/envs/$CONDA_ENV/bin/python"
    else
        echo "ERROR: cannot locate the $CONDA_ENV env's python (set PYBIN=/path/to/python)." >&2
        exit 1
    fi
fi
echo "  Jobs will use: $PYBIN"
echo ""

# --- Common preamble for all jobs ---
PREAMBLE="set -eu
cd $BASEDIR
export PATH=$(dirname "$PYBIN"):\$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=\${SLURM_CPUS_PER_TASK:-$CPUS}
export TQDM_DISABLE=1
export PYTHONPATH=$BASEDIR/src:\${PYTHONPATH:-}
"

# --- Helper: submit a job and capture its ID ---
# submit_job NAME TIME MEM CMD [DEPS] [CPUS]   (CPUS defaults to $CPUS)
submit_job() {
    local name="$1"
    local time="$2"
    local mem="$3"
    local cmd="$4"
    local deps="${5:-}"
    local cpus="${6:-$CPUS}"

    local dep_flag=""
    if [[ -n "$deps" ]]; then
        dep_flag="--dependency=afterok:${deps}"
    fi
    local part_flag=""
    if [[ -n "$PARTITION" ]]; then
        part_flag="--partition=$PARTITION"
    fi

    if $DRY_RUN; then
        # Printed to stderr so that the $(...) capturing the job ID does not swallow it.
        echo "[DRY-RUN] sbatch --job-name=$name $part_flag --time=$time --mem=$mem --cpus-per-task=$cpus $dep_flag" >&2
        echo "          CMD: $cmd" >&2
        echo "0"
        return
    fi

    local job_id
    job_id=$(sbatch --parsable \
        --job-name="$name" \
        $part_flag \
        --time="$time" \
        --mem="$mem" \
        --cpus-per-task="$cpus" \
        --output="logs/slurm_${name}_%j.out" \
        --error="logs/slurm_${name}_%j.err" \
        $dep_flag \
        --wrap="$PREAMBLE
$cmd")

    echo "  Submitted $name -> $job_id" >&2
    echo "$job_id"
}

echo "============================================================"
echo "  MASTER REPLICATION PIPELINE"
echo "============================================================"
echo "  Base directory: $BASEDIR"
echo "  Partition:      ${PARTITION:-<cluster default>}"
echo "  Quick mode:     $QUICK"
echo "  Supplementary:  $SUPPLEMENTARY"
echo "  Dry run:        $DRY_RUN"
echo "============================================================"
echo ""
echo "All experiment jobs submitted immediately (no artificial"
echo "sequencing). SLURM handles scheduling. Only combine/verify"
echo "jobs have dependencies on their input experiments."
echo ""

# =====================================================================
# EXPERIMENT JOBS — all submitted immediately, no inter-dependencies
# =====================================================================

# --- Caching Benchmark ---
echo "--- Caching Benchmark ---"
JOB_CACHE=$(submit_job "cache_bench" "$TIME_LONG" "$MEM_STD" \
    "python experiments/caching_benchmark.py --output results/caching_benchmark_results.csv 2>/dev/null && \
     python experiments/caching_validation.py --output results/caching_validation_large_p.csv 2>/dev/null")
echo ""

# --- CI Calibration (3 jobs: fast methods || KCI || merge) ---
# A rerun matches the released files only up to Monte Carlo error: the
# corrected MAR x |X|=2 generator in src/dgps.py consumes the random stream
# differently (README, "Changes since the submitted version"). The released
# mar x dim2 rows came from `bash slurm/submit_diagnostics.sh mar2`
# (seeds 4242/4243).
echo "--- CI Calibration (3 jobs: fast + kci + merge) ---"
JOB_CALIB_FAST=$(submit_job "calib_fast" "$TIME_LONG" "$MEM_STD" \
    "python experiments/calibration.py --n_reps $CALIB_REPS --methods ours fisher_z gcm fisher_z_rubin --output results/ci_calibration_fast.csv 2>/dev/null")
JOB_CALIB_KCI=$(submit_job "calib_kci" "$TIME_XLONG" "$MEM_STD" \
    "python experiments/calibration.py --n_reps $((CALIB_REPS / 2 > 0 ? CALIB_REPS / 2 : 1)) --methods kci --sample_sizes 1000 --signals 0 0.6 --output results/ci_calibration_kci.csv 2>/dev/null")
JOB_CALIB=$(submit_job "calib_merge" "$TIME_SHORT" "$MEM_STD" \
    "python -c \"import pandas as pd; fast=pd.read_csv('results/ci_calibration_fast.csv'); kci=pd.read_csv('results/ci_calibration_kci.csv'); pd.concat([fast,kci],ignore_index=True).to_csv('results/ci_calibration_results.csv',index=False); print('Merged calibration results')\"" \
    "$JOB_CALIB_FAST:$JOB_CALIB_KCI")
echo ""

# --- p=10 Graph Recovery (linear + nonlinear, 8 jobs) ---
echo "--- p=10 Graph Recovery (8 jobs) ---"
P10_JOBS=""
for DGP in linear nonlinear; do
    for MECH in mar mnar mixed complete; do
        JID=$(submit_job "p10_${DGP:0:3}_${MECH}" "$TIME_LONG" "$MEM_STD" \
            "python experiments/graph_recovery_p10.py --mechanism ${MECH} --dgp ${DGP} --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/p10_${DGP}_${MECH}.csv 2>/dev/null")
        P10_JOBS="${P10_JOBS:+$P10_JOBS:}$JID"
    done
done
echo ""

# --- p=20 Graph Recovery (8 jobs) ---
echo "--- p=20 Graph Recovery (8 jobs) ---"
P20_JOBS=""
for DGP in linear nonlinear; do
    for MECH in mar mnar mixed complete; do
        JID=$(submit_job "p20_${DGP:0:3}_${MECH}" "$TIME_XXLONG" "$MEM_STD" \
            "python experiments/graph_recovery_p20.py --mechanism ${MECH} --dgp ${DGP} --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/p20_${DGP}_${MECH}.csv 2>/dev/null")
        P20_JOBS="${P20_JOBS:+$P20_JOBS:}$JID"
    done
done
echo ""

# --- p=30 Graph Recovery (8 jobs) ---
echo "--- p=30 Graph Recovery (8 jobs) ---"
P30_JOBS=""
for DGP in linear nonlinear; do
    for MECH in mar mnar mixed complete; do
        JID=$(submit_job "p30_${DGP:0:3}_${MECH}" "$TIME_XXLONG" "$MEM_LARGE" \
            "python experiments/graph_recovery_p30.py --mechanism ${MECH} --dgp ${DGP} --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/p30_${DGP}_${MECH}.csv 2>/dev/null")
        P30_JOBS="${P30_JOBS:+$P30_JOBS:}$JID"
    done
done
echo ""

# --- Robustness: varying miss_rate and n_obs at p=10 (18 jobs) ---
echo "--- Robustness (18 jobs: 2 mechanisms x 3 miss_rates x 3 sample sizes) ---"
ROBUST_JOBS=""
for MECH in mar mnar; do
    for MR in 0.1 0.3 0.5; do
        for NOBS in 500 1000 2000; do
            JID=$(submit_job "robust_${MECH}_${MR}_${NOBS}" "$TIME_LONG" "$MEM_STD" \
                "python experiments/robustness.py --mechanism ${MECH} --miss_rate ${MR} --n_obs ${NOBS} --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/robustness_${MECH}_${MR}_${NOBS}.csv 2>/dev/null")
            ROBUST_JOBS="${ROBUST_JOBS:+$ROBUST_JOBS:}$JID"
        done
    done
done
echo ""

# --- Sachs Protein Network (real flow cytometry data) ---
# Reads data/sachs_data.csv, which is not shipped: run
# `bash data/fetch_sachs_data.sh` before submitting.
echo "--- Sachs Experiment (real data) ---"
JOB_SACHS=$(submit_job "sachs" "$TIME_LONG" "$MEM_STD" \
    "python experiments/sachs.py --n_reps $SACHS_REPS --data data/sachs_data.csv --output results/sachs_results.csv 2>/dev/null")
echo ""

# --- Sachs DAG with Synthetic Nonlinear Edges (10 per-cell jobs + merge) ---
echo "--- Sachs Synthetic Nonlinear Experiment (10 per-cell jobs) ---"
SACHS_NL_JOBS=""
for CELL in "0.1 complete" "0.1 mar" "0.1 mnar" "0.1 mixed" \
            "0.2 mar" "0.2 mnar" "0.2 mixed" \
            "0.4 mar" "0.4 mnar" "0.4 mixed"; do
    read -r RATE MECH <<< "$CELL"
    JID=$(submit_job "sachs_nl_${RATE}_${MECH}" "$TIME_MEDIUM" "$MEM_STD" \
        "python experiments/sachs_nonlinear.py --n_obs 1000 --n_reps $SACHS_REPS --dgp nonlinear --rate ${RATE} --mechanism ${MECH} --output results/sachs_nonlinear_${RATE}_${MECH}.csv 2>/dev/null")
    SACHS_NL_JOBS="${SACHS_NL_JOBS:+$SACHS_NL_JOBS:}$JID"
done
JOB_SACHS_NL=$(submit_job "sachs_nl_merge" "$TIME_SHORT" "$MEM_STD" \
    "python analysis/combine_sachs_nonlinear.py" \
    "$SACHS_NL_JOBS")
echo ""

# --- Sensitivity Analysis ---
echo "--- Sensitivity Analysis ---"
JOB_SENS=$(submit_job "sensitivity" "$TIME_XXLONG" "$MEM_STD" \
    "python experiments/sensitivity.py --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/sensitivity_results.csv 2>/dev/null")
echo ""

# --- Alpha Sensitivity ---
echo "--- Alpha Sensitivity ---"
JOB_ALPHA=$(submit_job "alpha_sens" "$TIME_XLONG" "$MEM_STD" \
    "python experiments/alpha_sensitivity.py --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/alpha_sensitivity_results.csv 2>/dev/null")
echo ""

# --- ALARM Network (linear + nonlinear) ---
echo "--- ALARM Experiment (linear) ---"
JOB_ALARM=$(submit_job "alarm" "$TIME_XXLONG" "$MEM_STD" \
    "python experiments/alarm.py --n_obs 1000 --n_reps $ALARM_REPS --dgp linear --output results/alarm_results.csv 2>/dev/null")
echo "--- ALARM Experiment (nonlinear) ---"
JOB_ALARM_NL=$(submit_job "alarm_nl" "$TIME_XXLONG" "$MEM_STD" \
    "python experiments/alarm.py --n_obs 1000 --n_reps $ALARM_REPS --dgp nonlinear --output results/alarm_nonlinear_results.csv 2>/dev/null")
echo ""

# --- HAILFINDER Network (nonlinear, 10 per-cell jobs + merge) ---
# One job per rate x mechanism cell, merged by combine_hailfinder.py. The
# per-cell jobs match a single-job run's datasets only in the 10% complete and
# 10% MAR cells (README, reproducibility caveats). The two slowest released
# cells logged 49-51 h, just above TIME_MEDIUM (2 days); raise it if your
# nodes are not faster.
echo "--- HAILFINDER Experiment (nonlinear, 10 per-cell jobs) ---"
HAIL_JOBS=""
for CELL in "0.1 complete" "0.1 mar" "0.1 mnar" "0.1 mixed" \
            "0.2 mar" "0.2 mnar" "0.2 mixed" \
            "0.4 mar" "0.4 mnar" "0.4 mixed"; do
    read -r RATE MECH <<< "$CELL"
    JID=$(submit_job "hail_${RATE}_${MECH}" "$TIME_MEDIUM" "$MEM_LARGE" \
        "python experiments/hailfinder.py --n_obs 1000 --n_reps $HAILFINDER_REPS --dgp nonlinear --rate ${RATE} --mechanism ${MECH} --output results/hailfinder_${RATE}_${MECH}.csv 2>/dev/null")
    HAIL_JOBS="${HAIL_JOBS:+$HAIL_JOBS:}$JID"
done
JOB_HAIL=$(submit_job "hail_merge" "$TIME_SHORT" "$MEM_STD" \
    "python analysis/combine_hailfinder.py" \
    "$HAIL_JOBS")
echo ""

# --- Early Stopping Calibration ---
echo "--- Early Stopping Calibration ---"
JOB_EARLY=$(submit_job "early_stop" "$TIME_SHORT" "$MEM_STD" \
    "python experiments/early_stopping_calibration.py --n_reps $EARLY_STOP_REPS --output results/early_stopping_calibration.csv 2>/dev/null")
echo ""

# --- Variance Comparison (NB vs Bayle) ---
echo "--- Variance Comparison ---"
JOB_VARCOMP=$(submit_job "var_comp" "$TIME_LONG" "$MEM_STD" \
    "python experiments/variance_comparison.py --mode both --n_reps $N_REPS --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output_calib results/variance_comparison_calibration.csv --output_graph results/variance_comparison_graph.csv 2>/dev/null")
echo ""

# --- Imputation Degradation ---
echo "--- Imputation Degradation ---"
JOB_IMPDEGRAD=$(submit_job "imp_degrad" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/imputation_degradation.py --n_reps $IMP_DEGRAD_REPS --n_obs 500 --dim_x 5 --seed 42 --output results/imputation_degradation.csv 2>/dev/null")
echo ""

# --- M-Sensitivity (power as function of number of imputations) ---
echo "--- M-Sensitivity Experiment ---"
JOB_MSENS=$(submit_job "m_sensitivity" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/m_sensitivity.py --n_reps $M_SENS_REPS --Ms 3 5 10 20 --K 5 --dgps linear_gaussian post_nonlinear --sample_sizes 500 1000 --dim_x 5 --signals 0 0.3 0.6 1.0 --mechanisms complete mar mnar --output results/m_sensitivity_results.csv 2>/dev/null")
echo ""

# --- Learner Comparison (RF, ExtraTrees, LightGBM, XGBoost) ---
# LightGBM and XGBoost are pip-installed into the env if they are missing.
# The released run logged 52 h, above TIME_MEDIUM (2 days); raise it if your
# nodes are not faster.
echo "--- Learner Comparison Experiment ---"
JOB_LEARNER=$(submit_job "learner_cmp" "$TIME_MEDIUM" "$MEM_STD" \
    "python -c 'import lightgbm' 2>/dev/null || python -m pip install lightgbm --quiet 2>/dev/null
     python -c 'import xgboost' 2>/dev/null || python -m pip install xgboost --quiet 2>/dev/null
     python experiments/learner_comparison.py --n_reps $LEARNER_REPS --learners rf extra_trees lgbm lgbm_tuned xgboost --M 5 --K 5 --dgps linear_gaussian post_nonlinear --sample_sizes 500 1000 --dim_x 5 --signals 0 0.3 0.6 1.0 --mechanisms complete mar mnar --output results/learner_comparison_results.csv 2>/dev/null")
echo ""

# --- Adversarial Kappa (stress test of paired-design cancellation) ---
echo "--- Adversarial Kappa Experiment ---"
JOB_ADV=$(submit_job "adv_kappa" "$TIME_MEDIUM" "$MEM_STD" \
    "python -u experiments/adversarial_kappa.py --n_reps $ADV_REPS --dgps hub chain dense_block weak_hub branch_separator hub_nonlinear --sample_sizes 500 --rates 0.3 0.5 --cond_strategies full missing --imputers linear rf --output results/adversarial_kappa_results.csv 2>/dev/null")
echo ""

# --- New Baselines: micd-pooled and missDAG (Erdos-Renyi sweep, 24 jobs) ---
echo "--- New Baselines: ER sweep (24 jobs) ---"
NB_ER_JOBS=""
for P in 10 20 30; do
    for DGP in linear nonlinear; do
        for MECH in mar mnar mixed complete; do
            JID=$(submit_job "nb_p${P}_${DGP:0:3}_${MECH}" "$TIME_LONG" "$MEM_STD" \
                "python experiments/new_baselines.py --benchmark erdos --p ${P} --mechanism ${MECH} --dgp ${DGP} --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/newbase_p${P}_${DGP}_${MECH}.csv 2>/dev/null")
            NB_ER_JOBS="${NB_ER_JOBS:+$NB_ER_JOBS:}$JID"
        done
    done
done
echo ""

# --- New Baselines: ALARM (linear + nonlinear, 2 jobs) ---
echo "--- New Baselines: ALARM (2 jobs) ---"
JOB_NB_ALARM=$(submit_job "nb_alarm_lin" "$TIME_MEDIUM" "$MEM_LARGE" \
    "python experiments/new_baselines.py --benchmark alarm --dgp linear --n_reps $ALARM_REPS --output results/newbase_alarm_linear.csv 2>/dev/null")
JOB_NB_ALARM_NL=$(submit_job "nb_alarm_nl" "$TIME_MEDIUM" "$MEM_LARGE" \
    "python experiments/new_baselines.py --benchmark alarm --dgp nonlinear --n_reps $ALARM_REPS --output results/newbase_alarm_nonlinear.csv 2>/dev/null")
echo ""

# --- New Baselines: HAILFINDER (nonlinear, 1 job) ---
# Replays the single-job HAILFINDER stream: datasets matched to the per-cell
# PAIR-CI runs in distribution only.
echo "--- New Baselines: HAILFINDER (1 job) ---"
JOB_NB_HAIL=$(submit_job "nb_hailfinder" "$TIME_LONG" "$MEM_LARGE" \
    "python experiments/new_baselines.py --benchmark hailfinder --dgp nonlinear --methods micd_pooled --n_reps $HAILFINDER_REPS --rep_timeout 3600 --output results/newbase_hailfinder_nonlinear.csv 2>/dev/null")
echo ""

# --- New Baselines: Sachs (linear + nonlinear, 2 jobs) ---
# The nonlinear job replays the single-job stream: datasets matched to the
# per-cell PAIR-CI runs in distribution only.
echo "--- New Baselines: Sachs (2 jobs) ---"
JOB_NB_SACHS=$(submit_job "nb_sachs_lin" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/new_baselines.py --benchmark sachs --dgp real --n_reps $SACHS_REPS --data_path data/sachs_data.csv --output results/newbase_sachs_linear.csv 2>/dev/null")
JOB_NB_SACHS_NL=$(submit_job "nb_sachs_nl" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/new_baselines.py --benchmark sachs --dgp nonlinear --n_reps $SACHS_REPS --output results/newbase_sachs_nonlinear.csv 2>/dev/null")
echo ""

# =====================================================================
# EXPERIMENTS ADDED FOR THE CAMERA-READY — no inter-dependencies
# =====================================================================
# Same commands as slurm/submit_rebuttal.sh (h1-h7), submit_anticipation.sh
# (a2) and submit_search_check.sh, which produced the released CSVs, with
# the same CPU counts (6th argument of submit_job); walltimes follow the
# TIME_* classes above.

# --- MVPC (causal-learn PC-stable + MV_Crtn_Fisher_Z correction; 27 jobs) ---
# Datasets replay the p-sweep / network RNG streams (bit-identical inputs;
# HAILFINDER matched in distribution). The correction step draws from NumPy's
# unseeded global generator, so outputs reproduce in distribution only.
echo "--- MVPC: ER sweep (24 jobs) + ALARM (2) + HAILFINDER (1) + Sachs (2) + p = 10 sweep (1) ---"
MVPC_JOBS=""
for P in 10 20 30; do
    for DGP in linear nonlinear; do
        for MECH in mar mnar mixed complete; do
            JID=$(submit_job "mvpc_p${P}_${DGP:0:3}_${MECH}" "$TIME_MEDIUM" "$MEM_STD" \
                "python experiments/mvpc_baseline.py --benchmark erdos --p ${P} --dgp ${DGP} --mechanism ${MECH} --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/mvpc_p${P}_${DGP}_${MECH}.csv 2>/dev/null" "" 2)
            MVPC_JOBS="${MVPC_JOBS:+$MVPC_JOBS:}$JID"
        done
    done
done
for DGP in linear nonlinear; do
    JID=$(submit_job "mvpc_alarm_${DGP:0:3}" "$TIME_MEDIUM" "$MEM_LARGE" \
        "python experiments/mvpc_baseline.py --benchmark alarm --dgp ${DGP} --n_reps $ALARM_REPS --output results/mvpc_alarm_${DGP}.csv 2>/dev/null" "" 2)
    MVPC_JOBS="${MVPC_JOBS}:$JID"
done
JID=$(submit_job "mvpc_hailfinder" "$TIME_MEDIUM" "$MEM_LARGE" \
    "python experiments/mvpc_baseline.py --benchmark hailfinder --dgp nonlinear --n_reps $HAILFINDER_REPS --rep_timeout 3600 --output results/mvpc_hailfinder_nonlinear.csv 2>/dev/null" "" 2)
MVPC_JOBS="${MVPC_JOBS}:$JID"
# Sachs (real data and nonlinear) and the p = 10 rate x n sweep: both replay
# the original streams and rerun test-wise deletion to confirm identical data.
for SDGP in real nonlinear; do
    JID=$(submit_job "mvpc_sachs_${SDGP}" "$TIME_SHORT" "$MEM_STD" \
        "python experiments/mvpc_sachs.py --dgp ${SDGP} --output results/mvpc_sachs_${SDGP}.csv 2>/dev/null" "" 2)
    MVPC_JOBS="${MVPC_JOBS}:$JID"
done
JID=$(submit_job "mvpc_robustness" "$TIME_SHORT" "$MEM_STD" \
    "python experiments/mvpc_robustness.py --n_jobs 6 --output results/mvpc_robustness.csv 2>/dev/null" "" 6)
MVPC_JOBS="${MVPC_JOBS}:$JID"
echo ""

# --- Baseline alpha sweeps (PR frontier, sparsity-matched table; 4 jobs) ---
# ER replays alpha_sensitivity.py's datasets; networks replay alarm.py /
# single-job hailfinder.py streams at rate 0.2 (HAILFINDER matched to the
# released PAIR-CI data in distribution only).
echo "--- Baseline alpha sweeps (4 jobs) ---"
ASWEEP_JOBS=$(submit_job "asweep_p10" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/alpha_sweep_baselines.py --benchmark erdos --n_graphs $N_GRAPHS --n_datasets $N_DATASETS --output results/alpha_sweep_baselines_p10.csv 2>/dev/null" "" 8)
for DGP in linear nonlinear; do
    JID=$(submit_job "asweep_alarm_${DGP:0:3}" "$TIME_MEDIUM" "$MEM_LARGE" \
        "python experiments/alpha_sweep_baselines.py --benchmark alarm --dgp ${DGP} --n_reps $ALARM_REPS --rates 0.2 --mechs mar mnar mixed --output results/alpha_sweep_baselines_alarm_${DGP}.csv 2>/dev/null" "" 8)
    ASWEEP_JOBS="${ASWEEP_JOBS}:$JID"
done
JID=$(submit_job "asweep_hailf" "$TIME_MEDIUM" "$MEM_LARGE" \
    "python experiments/alpha_sweep_baselines.py --benchmark hailfinder --dgp nonlinear --n_reps $HAILFINDER_REPS --rates 0.2 --mechs mar mnar mixed --alphas 0.01 0.05 0.10 0.20 --output results/alpha_sweep_baselines_hailfinder.csv 2>/dev/null" "" 8)
ASWEEP_JOBS="${ASWEEP_JOBS}:$JID"
echo ""

# --- GCM/KCI complete-data audit + GCM with OLS (2 jobs) ---
echo "--- GCM/KCI audit + GCM-OLS (2 jobs) ---"
JOB_AUDIT=$(submit_job "kci_gcm_audit" "$TIME_MEDIUM" "$MEM_LARGE" \
    "python experiments/kci_gcm_audit.py --n_jobs 16 $AUDIT_QUICK --output results/kci_gcm_audit.csv 2>/dev/null" "" 16)
JOB_GCMOLS=$(submit_job "gcm_ols" "$TIME_SHORT" "$MEM_STD" \
    "python experiments/gcm_ols.py --n_jobs 8 $GCMOLS_QUICK --output results/gcm_ols_audit.csv 2>/dev/null" "" 8)
echo ""

# --- Imputer quality (calibration grid + p=10 ER; 2 jobs) ---
echo "--- Imputer quality (2 jobs) ---"
JOB_IMPQ_CAL=$(submit_job "imputer_qual_cal" "$TIME_SHORT" "$MEM_STD" \
    "python experiments/imputer_quality.py --benchmark calibration --n_jobs 16 $IMPQ_QUICK --output results/imputer_quality_calibration.csv 2>/dev/null" "" 16)
JOB_IMPQ_ER=$(submit_job "imputer_qual_er" "$TIME_SHORT" "$MEM_STD" \
    "python experiments/imputer_quality.py --benchmark erdos --n_jobs 16 $IMPQ_QUICK --output results/imputer_quality_erdos.csv 2>/dev/null" "" 16)
echo ""

# --- Variance decomposition, test-variable missingness, learner robustness ---
# variance_decomposition: with the corrected generator its 4 MAR x |X|=2 cells
# are genuine MAR (labelled `mar`); the released file keeps the pre-fix runs of
# those cells relabelled `mnar_1col` (README, "Changes since the submitted version").
echo "--- Variance decomposition / test-variable missingness / learner robustness (3 jobs) ---"
JOB_VDEC=$(submit_job "var_decomp" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/variance_decomposition.py $VDEC_QUICK --output results/variance_decomposition.csv 2>/dev/null" "" 16)
JOB_TESTVAR=$(submit_job "testvar_miss" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/test_variable_missingness.py $TESTVAR_QUICK --output results/test_variable_missingness.csv 2>/dev/null" "" 16)
JOB_LROB=$(submit_job "learner_robust" "$TIME_MEDIUM" "$MEM_STD" \
    "python experiments/learner_robustness.py $LROB_QUICK --output results/learner_robustness.csv 2>/dev/null" "" 16)
echo ""

# --- Search check: baselines under the PC-stable search (2 jobs) ---
# Fixed design (Table 3's 20%-MNAR cells, 20 replicates) with no reduced
# mode, so skipped under --quick; the summary step in the build job then
# finds no search-check CSVs (unless --force kept the released ones) and
# reports a failure without failing the job.
SEARCH_JOBS=""
if ! $QUICK; then
    echo "--- Search check: baselines, simplified vs PC-stable search (2 jobs) ---"
    SEARCH_JOBS=$(submit_job "sc_base_alarm_nl" "$TIME_SHORT" "$MEM_STD" \
        "python -u experiments/search_check_baselines.py --network alarm_nonlinear 2>/dev/null" "" 2)
    JID=$(submit_job "sc_base_hailf" "$TIME_MEDIUM" "$MEM_LARGE" \
        "python -u experiments/search_check_baselines.py --network hailfinder 2>/dev/null" "" 2)
    SEARCH_JOBS="${SEARCH_JOBS}:$JID"
    echo ""
else
    echo "--- Search check: skipped under --quick (no reduced mode) ---"
    echo ""
fi

# =====================================================================
# SUPPLEMENTARY EXPERIMENTS (opt-in: --supplementary) — not reported in the
# paper; diagnostics kept for provenance.
# =====================================================================
SUPP_JOBS=""
if $SUPPLEMENTARY; then
    echo "--- SUPPLEMENTARY (not reported in the paper) ---"
    # Placebo-quality diagnostics at |X|=10 (3 jobs; builder tab_placebo_diag, not \input)
    for DGP in linear_gaussian post_nonlinear latent_confounder; do
        JID=$(submit_job "placebo_${DGP}" "$TIME_MEDIUM" "$MEM_STD" \
            "python experiments/placebo_diagnostics.py --dgp ${DGP} $PLACEBO_QUICK --output results/placebo_diag_${DGP}.csv 2>/dev/null" "" 16)
        SUPP_JOBS="${SUPP_JOBS:+$SUPP_JOBS:}$JID"
    done
    # Dependence-valid pooling for imputed GCM/KCI (1 job)
    JID=$(submit_job "pooling_validity" "$TIME_MEDIUM" "$MEM_LARGE" \
        "python experiments/pooling_validity.py --n_jobs 16 $POOL_QUICK --output results/pooling_validity.csv 2>/dev/null" "" 16)
    SUPP_JOBS="${SUPP_JOBS:+$SUPP_JOBS:}$JID"
    # p=10 with all non-root variables incomplete (4 jobs)
    for DGP in linear nonlinear; do
        for MECH in mar mnar; do
            JID=$(submit_job "heavy_${DGP:0:3}_${MECH}" "$TIME_MEDIUM" "$MEM_STD" \
                "python experiments/heavy_missingness_p10.py --dgp ${DGP} --mechanism ${MECH} $HEAVY_QUICK --output results/heavy_miss_p10_${DGP}_${MECH}.csv 2>/dev/null" "" 4)
            SUPP_JOBS="${SUPP_JOBS:+$SUPP_JOBS:}$JID"
        done
    done
    # PAIR-CI under the PC-stable search (40 jobs, up to 12 h each; never run;
    # no reduced mode, so skipped under --quick)
    if ! $QUICK; then
        for NET in alarm_nonlinear hailfinder; do
            for REP in $(seq 0 19); do
                JID=$(submit_job "sc_pairci_${NET:0:5}_${REP}" "$TIME_SHORT" "$MEM_LARGE" \
                    "python -u experiments/search_check_pairci.py --network ${NET} --rep ${REP} --out results/search_check/" "" 8)
                SUPP_JOBS="${SUPP_JOBS:+$SUPP_JOBS:}$JID"
            done
        done
    fi
    echo ""
fi

# =====================================================================
# COMBINE JOBS — depend on their input experiments
# =====================================================================
echo "--- Combine jobs (with dependencies) ---"

JOB_COMBINE_P10=$(submit_job "combine_p10" "$TIME_SHORT" "$MEM_STD" \
    "python analysis/combine_p10.py" \
    "$P10_JOBS")

JOB_COMBINE_P20=$(submit_job "combine_p20" "$TIME_SHORT" "$MEM_STD" \
    "python analysis/combine_p20.py" \
    "$P20_JOBS")

# combine_p30 needs p10+p20+p30 for the scaling comparison
SCALING_DEPS="${P10_JOBS}:${P20_JOBS}:${P30_JOBS}"
JOB_COMBINE_P30=$(submit_job "combine_p30" "$TIME_SHORT" "$MEM_STD" \
    "python analysis/combine_p30.py" \
    "$SCALING_DEPS")

JOB_COMBINE_ROBUST=$(submit_job "combine_robust" "$TIME_SHORT" "$MEM_STD" \
    "python analysis/combine_robustness.py" \
    "$ROBUST_JOBS")

echo ""

# =====================================================================
# FINAL BUILD STEP — depends on every combine and experiment job
# =====================================================================
# Generates the paper tables and figures from the collected results, in
# dependency order: table fragments; paired bootstrap (writes
# results/paired_bootstrap_contrasts.csv + tables/tab_bootstrap_contrasts.tex);
# PR frontier (figures/pr_frontier.pdf + results/pr_frontier_points.csv);
# kappa trace (results/kappa_trace.csv) then Figure kappa_fpr_trace +
# results/kappa_fit.csv; power and PR-scatter figures; the worked example.
# Then two checks: search_check_summary.py (the appendix search-check
# numbers) and check_rebuttal_numbers.py (201 numbers quoted in the authors'
# responses during review). Both pass on the released results/; after a
# from-scratch regeneration they can fail through Monte Carlo noise alone
# (the calibration grid, MVPC and FZ-vote are not bit-reproducible; README),
# so their exit status is reported but does not fail the job.
echo "--- Final build step (tables + figures + checks) ---"
ALL_DEPS="${JOB_COMBINE_P10}:${JOB_COMBINE_P20}:${JOB_COMBINE_P30}:${JOB_COMBINE_ROBUST}"
ALL_DEPS="${ALL_DEPS}:${JOB_CALIB}:${JOB_CACHE}:${JOB_SACHS}:${JOB_SACHS_NL}"
ALL_DEPS="${ALL_DEPS}:${JOB_SENS}:${JOB_ALPHA}:${JOB_ALARM}:${JOB_ALARM_NL}"
ALL_DEPS="${ALL_DEPS}:${JOB_HAIL}:${JOB_EARLY}:${JOB_VARCOMP}:${JOB_IMPDEGRAD}"
ALL_DEPS="${ALL_DEPS}:${JOB_MSENS}:${JOB_LEARNER}:${JOB_ADV}"
ALL_DEPS="${ALL_DEPS}:${NB_ER_JOBS}:${JOB_NB_ALARM}:${JOB_NB_ALARM_NL}"
ALL_DEPS="${ALL_DEPS}:${JOB_NB_HAIL}:${JOB_NB_SACHS}:${JOB_NB_SACHS_NL}"
ALL_DEPS="${ALL_DEPS}:${MVPC_JOBS}:${ASWEEP_JOBS}:${JOB_AUDIT}:${JOB_GCMOLS}"
ALL_DEPS="${ALL_DEPS}:${JOB_IMPQ_CAL}:${JOB_IMPQ_ER}:${JOB_VDEC}:${JOB_TESTVAR}:${JOB_LROB}"
[[ -n "$SEARCH_JOBS" ]] && ALL_DEPS="${ALL_DEPS}:${SEARCH_JOBS}"
[[ -n "$SUPP_JOBS" ]] && ALL_DEPS="${ALL_DEPS}:${SUPP_JOBS}"

JOB_BUILD=$(submit_job "build_tables" "$TIME_SHORT" "$MEM_STD" \
    "python analysis/build_tables.py --output-dir tables/ && \
     python analysis/paired_bootstrap.py --results-dir results --output-dir tables && \
     python analysis/pr_frontier.py --results-dir results --output figures/pr_frontier.pdf && \
     python analysis/compute_kappa_trace.py && \
     python analysis/plot_kappa_trace.py && \
     python analysis/generate_figures.py --results-dir results/ && \
     python analysis/plot_pr_scatter.py --results-dir results/ --output figures/pr_scatter.pdf && \
     python analysis/worked_example_closed_form.py && \
     python analysis/worked_example_closed_form.py --draw_noise || exit 1
     python analysis/search_check_summary.py --results-dir results || echo 'search_check_summary: some appendix numbers not reproduced (see README)'
     python analysis/check_rebuttal_numbers.py --results-dir results/ || echo 'check_rebuttal_numbers: some posted numbers not reproduced (see README)'" \
    "$ALL_DEPS")
echo ""

echo ""
echo "============================================================"
echo "  All jobs submitted. Monitor with: squeue -u \$USER"
echo "  Logs in: $BASEDIR/logs/"
echo "  Results in: $BASEDIR/results/"
echo "============================================================"
