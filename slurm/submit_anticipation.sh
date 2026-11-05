#!/bin/bash
# Three further experiments, one target each:
#   a1  pooling_validity   dependence-valid pooling rules for imputed GCM/KCI,
#                          plus RCIT if the installed causal-learn provides it
#                          (0.1.3.7 does not); supplementary, not in the paper
#   a2  gcm_ols            GCM with a correctly specified linear learner
#                          (App. R.1)
#   a3  heavy_miss         p=10 with every non-root variable incomplete,
#                          4 jobs: {linear,nonlinear} x {mar,mnar};
#                          supplementary, not in the paper
#
# Usage:
#   bash slurm/submit_anticipation.sh          # all three
#   bash slurm/submit_anticipation.sh a2       # chosen targets
#
# Quick checks first (about 2 min in total):
#   conda run -n causal_ci python experiments/pooling_validity.py --smoke --n_jobs 2
#   conda run -n causal_ci python experiments/gcm_ols.py --smoke
#   conda run -n causal_ci python experiments/heavy_missingness_p10.py \
#       --smoke --mechanism mnar --dgp linear --output results/heavy_smoke.csv
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
    [ -n "$ok" ] && echo "Submitted $NAME" || {
        echo "FAILED to submit $NAME after 3 attempts" >&2
        FAILED_JOBS="${FAILED_JOBS:-} $NAME"
    }
    sleep 1
}

do_a1() {
    # KCI dominates the runtime; the default n in {500, 1000, 2000} keeps the
    # job to about a night on 16 cores. To add n = 5000, pass
    # --sample_sizes 500 1000 2000 5000.
    submit "pooling_validity" "24:00:00" "8G" 16 \
        python experiments/pooling_validity.py --n_jobs 16 \
            --output results/pooling_validity.csv
}

do_a2() {
    submit "gcm_ols" "04:00:00" "4G" 8 \
        python experiments/gcm_ols.py --n_jobs 8 \
            --output results/gcm_ols_audit.csv
}

do_a3() {
    for DGP in linear nonlinear; do
        for MECH in mar mnar; do
            submit "heavy_${DGP}_${MECH}" "24:00:00" "4G" 4 \
                python experiments/heavy_missingness_p10.py \
                    --dgp $DGP --mechanism $MECH \
                    --output results/heavy_miss_p10_${DGP}_${MECH}.csv
        done
    done
}

if [ $# -eq 0 ]; then
    do_a1; do_a2; do_a3
else
    for target in "$@"; do "do_${target}"; done
fi
if [ -n "${FAILED_JOBS:-}" ]; then
    echo "WARNING -- these jobs did NOT submit:${FAILED_JOBS}" >&2
else
    echo "All anticipation jobs submitted."
fi
