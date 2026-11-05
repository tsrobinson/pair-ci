#!/bin/bash
# Reruns after the MAR x |X|=2 generator fix in src/dgps.py, plus a placebo
# diagnostic. One target each:
#   mar2     the calibration grid's MAR x |X|=2 cells on the corrected
#            generator: main methods on the full signal and n grid (100
#            reps), and KCI at n=1000, signals {0, 0.6}, 50 reps, as in
#            submit_calibration.sh. Writes results/mar_dim2_rerun.csv and
#            results/mar_dim2_rerun_kci.csv, whose rows replace the mar x dim2
#            rows of ci_calibration_*.csv in the released files.
#   imputer  imputer quality on the corrected generator. The script has no
#            cell filter, so the whole calibration grid is rerun; only the MAR
#            x |X|=2 cells change. Writes imputer_quality_calibration_v2.csv;
#            the released imputer_quality_calibration.csv is this file, renamed.
#   placebo  placebo-quality diagnostics at |X|=10, one job per DGP: why some
#            complete-data cells at |X|=10 exceed nominal. Supplementary, not
#            in the paper.
#
# Usage:
#   bash slurm/submit_diagnostics.sh                 # all three
#   bash slurm/submit_diagnostics.sh mar2 placebo    # chosen targets
#
# Same setup as slurm/submit_rebuttal.sh (python located at submit time, BLAS
# pinned, 1 s pacing with retries). It honours PARTITION only; DRY_RUN, PYBIN
# and CONDA_ENV are ignored, so it always submits, with the causal_ci env.

set -euo pipefail
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASEDIR"
mkdir -p logs results

PYBIN="$(conda run -n causal_ci which python 2>/dev/null || true)"
if [ -z "$PYBIN" ]; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /opt/miniconda3; do
        if [ -x "$c/envs/causal_ci/bin/python" ]; then
            PYBIN="$c/envs/causal_ci/bin/python"; break
        fi
    done
fi
if [ -z "$PYBIN" ]; then
    echo "ERROR: cannot locate the causal_ci env's python." >&2
    exit 1
fi
echo "Jobs will use: $PYBIN"

ENV_SETUP="cd $BASEDIR
export PATH=$(dirname "$PYBIN"):\$PATH
export TQDM_DISABLE=1 PYTHONPATH=$BASEDIR/src:\${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=\${SLURM_CPUS_PER_TASK:-2}"

submit() { # name time mem cpus command...
    local NAME=$1 TIME=$2 MEM=$3 CPUS=$4; shift 4
    local PART_FLAG=""
    [ -n "${PARTITION:-}" ] && PART_FLAG="--partition=$PARTITION"
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

do_mar2() {
    # Full signal and n grid, 100 reps: the slice of ci_calibration_fast.csv
    # that these rows replace. New seed: the corrected generator consumes the
    # random stream differently, so the original stream cannot be replayed;
    # these are independent draws.
    submit "mar2_rerun" "24:00:00" "4G" 16 \
        python experiments/calibration.py \
            --mechanisms mar --dim_xs 2 --n_reps 100 \
            --methods ours fisher_z gcm fisher_z_rubin \
            --seed 4242 --output results/mar_dim2_rerun.csv
    # KCI slice, with the settings of submit_calibration.sh step 2.
    submit "mar2_kci" "12:00:00" "4G" 8 \
        python experiments/calibration.py \
            --mechanisms mar --dim_xs 2 --sample_sizes 1000 \
            --signals 0 0.6 --n_reps 50 --methods kci \
            --seed 4243 --output results/mar_dim2_rerun_kci.csv
}

do_imputer() {
    submit "imputer_qual_v2" "12:00:00" "4G" 16 \
        python experiments/imputer_quality.py --benchmark calibration \
            --n_jobs 16 --output results/imputer_quality_calibration_v2.csv
}

do_placebo() {
    for DGP in linear_gaussian post_nonlinear latent_confounder; do
        submit "placebo_${DGP}" "36:00:00" "4G" 16 \
            python experiments/placebo_diagnostics.py --dgp "$DGP" \
                --output "results/placebo_diag_${DGP}.csv"
    done
}

if [ $# -eq 0 ]; then
    do_mar2; do_imputer; do_placebo
else
    for target in "$@"; do "do_${target}"; done
fi
if [ -n "${FAILED_JOBS:-}" ]; then
    echo "WARNING -- these jobs did NOT submit:${FAILED_JOBS}" >&2
else
    echo "All requested diagnostics jobs submitted."
fi
