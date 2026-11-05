#!/bin/bash
# Experiments added for the camera-ready, one target each:
#   h1  MVPC on the Erdos-Renyi sweep (24 jobs), ALARM (2) and HAILFINDER (1);
#       h1_hailfinder submits the HAILFINDER job alone
#   h2  alpha sweeps of the tunable baselines: p=10, ALARM x2, HAILFINDER
#   h3  complete-data audit of GCM and KCI configurations
#   h4  imputer quality on the calibration grid and on p=10 graphs
#   h5  variance decomposition
#   h6  missingness on the test variables
#   h7  learner robustness
# h3 and h4 run their tests in 16 parallel processes; h5 and h6 run them in
# sequence, with parallel forests.
#
# Usage:
#   bash slurm/submit_rebuttal.sh            # h1-h6 (h7 only when named)
#   bash slurm/submit_rebuttal.sh h2 h5      # chosen targets
#
# This script does not source slurm/_common.sh. It has its own copy of that
# setup and honours PARTITION only: DRY_RUN, PYBIN and CONDA_ENV are ignored,
# so it always submits, with the causal_ci env. `./reproduce_all.sh --dry-run`
# lists the same commands.
#
# To check the MVPC install first (about a minute):
#     conda run -n causal_ci python experiments/mvpc_baseline.py --smoke

set -euo pipefail
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASEDIR"
mkdir -p logs results

# Locate the causal_ci env's python once, at submit time, and put it on each
# job's PATH. --wrap payloads run under /bin/sh, where `conda activate` fails
# (silently, since stderr is discarded) and the job would fall back to the
# base python.
PYBIN="$(conda run -n causal_ci which python 2>/dev/null || true)"
if [ -z "$PYBIN" ]; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /opt/miniconda3; do
        if [ -x "$c/envs/causal_ci/bin/python" ]; then
            PYBIN="$c/envs/causal_ci/bin/python"; break
        fi
    done
fi
if [ -z "$PYBIN" ]; then
    echo "ERROR: cannot locate the causal_ci env's python. Create the env" >&2
    echo "(conda env create -f environment.yml) or submit from a shell where" >&2
    echo "conda is on PATH." >&2
    exit 1
fi
echo "Jobs will use: $PYBIN"

ENV_SETUP="cd $BASEDIR
export PATH=$(dirname "$PYBIN"):\$PATH
export TQDM_DISABLE=1 PYTHONPATH=$BASEDIR/src:\${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=\${SLURM_CPUS_PER_TASK:-2}"
# BLAS pools are pinned to 1 in every job: unpinned numpy starts one thread
# per core on the node, whatever the job's allocation, and oversubscribes it.
# sklearn/joblib parallelism (forests with n_jobs=-1, the h3/h4 process pools)
# is capped at the allocation by LOKY_MAX_CPU_COUNT, which joblib.cpu_count()
# respects.

submit() { # name time mem cpus _legacy_pin_arg command...
    local NAME=$1 TIME=$2 MEM=$3 CPUS=$4; shift 5
    # The 5th argument (pin/nopin) is ignored: every job gets the pinned
    # ENV_SETUP. No --partition unless PARTITION is set, so jobs go to the
    # cluster's default partition.
    local PART_FLAG=""
    [ -n "${PARTITION:-}" ] && PART_FLAG="--partition=$PARTITION"
    # Retry transient slurmctld errors ("Unexpected message received" and
    # the like) instead of aborting the batch, and space submissions 1 s
    # apart so as not to flood the controller.
    local ok=""
    for attempt in 1 2 3; do
        if sbatch --parsable --job-name="$NAME" $PART_FLAG \
            --time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS" \
            --output="logs/slurm_${NAME}_%j.out" \
            --error="logs/slurm_${NAME}_%j.err" \
            --wrap="$ENV_SETUP
$* 2>/dev/null"; then
            ok=1; break
        fi
        echo "  sbatch failed for $NAME (attempt $attempt/3); retrying in 10s..." >&2
        sleep 10
    done
    if [ -n "$ok" ]; then
        echo "Submitted $NAME"
    else
        echo "FAILED to submit $NAME after 3 attempts" >&2
        FAILED_JOBS="${FAILED_JOBS:-} $NAME"
    fi
    sleep 1
}

do_h1() {
    # MVPC: Erdos-Renyi grid, ALARM, HAILFINDER (nonlinear).
    for P in 10 20 30; do
        TIME="12:00:00"; MEM="4G"
        [ "$P" = "30" ] && TIME="24:00:00"
        for DGP in linear nonlinear; do
            for MECH in mar mnar mixed complete; do
                submit "mvpc_p${P}_${DGP}_${MECH}" "$TIME" "$MEM" 2 nopin \
                    python experiments/mvpc_baseline.py \
                        --benchmark erdos --p $P --dgp $DGP --mechanism $MECH \
                        --n_graphs 10 --n_datasets 20 \
                        --output results/mvpc_p${P}_${DGP}_${MECH}.csv
            done
        done
    done
    for DGP in linear nonlinear; do
        submit "mvpc_alarm_${DGP}" "24:00:00" "8G" 2 nopin \
            python experiments/mvpc_baseline.py \
                --benchmark alarm --dgp $DGP --n_reps 20 \
                --output results/mvpc_alarm_${DGP}.csv
    done
    do_h1_hailfinder
}

do_h1_hailfinder() {
    submit "mvpc_hailfinder" "48:00:00" "8G" 2 nopin \
        python experiments/mvpc_baseline.py \
            --benchmark hailfinder --dgp nonlinear --n_reps 20 \
            --rep_timeout 3600 \
            --output results/mvpc_hailfinder_nonlinear.csv
}

do_h2() {
    # Baseline alpha sweeps. The p=10 sweep replays the datasets of PAIR-CI's
    # own sweep (alpha_sensitivity.py); the networks replay the alarm.py and
    # single-job hailfinder.py streams at rate 0.2 (on HAILFINDER, matched to
    # the released PAIR-CI data in distribution only).
    submit "asweep_p10" "24:00:00" "4G" 8 nopin \
        python experiments/alpha_sweep_baselines.py --benchmark erdos \
            --n_graphs 10 --n_datasets 20 \
            --output results/alpha_sweep_baselines_p10.csv
    for DGP in linear nonlinear; do
        submit "asweep_alarm_${DGP}" "24:00:00" "8G" 8 nopin \
            python experiments/alpha_sweep_baselines.py --benchmark alarm \
                --dgp $DGP --n_reps 20 --rates 0.2 --mechs mar mnar mixed \
                --output results/alpha_sweep_baselines_alarm_${DGP}.csv
    done
    # HAILFINDER: 4 alphas only, to keep micd's repeated 56-column imputation
    # inside the walltime; results are saved incrementally.
    submit "asweep_hailf" "48:00:00" "8G" 8 nopin \
        python experiments/alpha_sweep_baselines.py --benchmark hailfinder \
            --dgp nonlinear --n_reps 20 --rates 0.2 --mechs mar mnar mixed \
            --alphas 0.01 0.05 0.10 0.20 \
            --output results/alpha_sweep_baselines_hailfinder.csv
}

do_h3() {
    submit "kci_gcm_audit" "24:00:00" "8G" 16 pin \
        python experiments/kci_gcm_audit.py --n_jobs 16 \
            --output results/kci_gcm_audit.csv
}

do_h4() {
    submit "imputer_qual_cal" "12:00:00" "4G" 16 pin \
        python experiments/imputer_quality.py --benchmark calibration \
            --n_jobs 16 --output results/imputer_quality_calibration.csv
    submit "imputer_qual_er" "12:00:00" "4G" 16 pin \
        python experiments/imputer_quality.py --benchmark erdos \
            --n_jobs 16 --output results/imputer_quality_erdos.csv
}

do_h5() {
    submit "var_decomp" "48:00:00" "4G" 16 nopin \
        python experiments/variance_decomposition.py \
            --output results/variance_decomposition.csv
}

do_h6() {
    submit "testvar_miss" "24:00:00" "4G" 16 nopin \
        python experiments/test_variable_missingness.py \
            --output results/test_variable_missingness.csv
}

do_h7() {
    # Calibration under weak or misspecified learners.
    submit "learner_robust" "24:00:00" "4G" 16 nopin \
        python experiments/learner_robustness.py \
            --output results/learner_robustness.csv
}

if [ $# -eq 0 ]; then
    do_h1; do_h2; do_h3; do_h4; do_h5; do_h6
else
    for target in "$@"; do "do_${target}"; done
fi
if [ -n "${FAILED_JOBS:-}" ]; then
    echo "WARNING -- these jobs did NOT submit:${FAILED_JOBS}" >&2
    echo "Re-run with just those targets, e.g.: bash slurm/submit_rebuttal.sh h1_hailfinder" >&2
else
    echo "All requested rebuttal jobs submitted."
fi
