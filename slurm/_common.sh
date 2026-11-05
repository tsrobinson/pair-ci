# slurm/_common.sh: submit-time setup shared by the slurm/submit_*.sh wrappers.
# It is sourced by them; do not run or sbatch it directly.
#
# - The env's python is located once, here, and put on each job's PATH.
#   sbatch --wrap payloads run under /bin/sh, where `conda activate` fails
#   (silently, because payloads discard stderr) and the job would fall back
#   to the base python.
# - No --partition unless PARTITION is set, so jobs go to the cluster's
#   default partition instead of a hard-coded name that may not exist.
# - BLAS thread pools are pinned to 1: unpinned numpy starts one thread per
#   core on the node, whatever the job's allocation. sklearn/joblib
#   parallelism is capped at the allocation through LOKY_MAX_CPU_COUNT.
# - sbatch calls are retried (3 attempts) and spaced 1 s apart, so that a
#   loop of submissions does not overload slurmctld.
# - The package root is found from this file's location, so the wrappers
#   work from any directory. Run them with `bash`, not `sbatch`: sbatch runs
#   a spooled copy of the script, from which neither the package nor this
#   file can be found.
#
# Environment overrides:
#   PARTITION=<name>   submit to this partition
#   CONDA_ENV=<name>   env to take python from (default: causal_ci)
#   PYBIN=/path/python use this interpreter and skip the lookup
#   DRY_RUN=1          print the sbatch commands instead of submitting

set -euo pipefail
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASEDIR"
CONDA_ENV="${CONDA_ENV:-causal_ci}"
DRY_RUN="${DRY_RUN:-0}"
[ "$DRY_RUN" = "1" ] || mkdir -p logs results
PYBIN="${PYBIN:-}"
if [ -z "$PYBIN" ]; then
    PYBIN="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || true)"
fi
if [ -z "$PYBIN" ]; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda /opt/miniconda3; do
        if [ -x "$c/envs/$CONDA_ENV/bin/python" ]; then
            PYBIN="$c/envs/$CONDA_ENV/bin/python"; break
        fi
    done
fi
if [ -z "$PYBIN" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        PYBIN="/path/to/envs/$CONDA_ENV/bin/python"
        echo "WARNING (dry run): cannot locate the $CONDA_ENV env's python; using a placeholder." >&2
    else
        echo "ERROR: cannot locate the $CONDA_ENV env's python. Create the env" >&2
        echo "(conda env create -f environment.yml), submit from a shell where" >&2
        echo "conda is on PATH, or set PYBIN=/path/to/python." >&2
        exit 1
    fi
fi
echo "Jobs will use: $PYBIN"

ENV_SETUP="cd $BASEDIR
export PATH=$(dirname "$PYBIN"):\$PATH
export TQDM_DISABLE=1 PYTHONPATH=$BASEDIR/src:\${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=\${SLURM_CPUS_PER_TASK:-2}"

FAILED_JOBS=""

# submit NAME TIME MEM CPUS LOGSTEM PAYLOAD
#   MEM may be "" (cluster default). LOGSTEM names logs/slurm_<LOGSTEM>_%j.{out,err}.
#   PAYLOAD is POSIX sh, run after ENV_SETUP; it carries its own 2>/dev/null.
submit() {
    local NAME=$1 TIME=$2 MEM=$3 CPUS=$4 LOGSTEM=$5 PAYLOAD=$6
    local PART_FLAG="" MEM_FLAG=""
    [ -n "${PARTITION:-}" ] && PART_FLAG="--partition=$PARTITION"
    [ -n "$MEM" ] && MEM_FLAG="--mem=$MEM"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY-RUN] sbatch --job-name=$NAME $PART_FLAG --time=$TIME $MEM_FLAG --cpus-per-task=$CPUS --output=logs/slurm_${LOGSTEM}_%j.out"
        echo "$PAYLOAD" | sed 's/^/          | /'
        return 0
    fi
    local ok=""
    for attempt in 1 2 3; do
        if sbatch --parsable --job-name="$NAME" $PART_FLAG \
            --time="$TIME" $MEM_FLAG --cpus-per-task="$CPUS" \
            --output="logs/slurm_${LOGSTEM}_%j.out" \
            --error="logs/slurm_${LOGSTEM}_%j.err" \
            --wrap="$ENV_SETUP
$PAYLOAD"; then
            ok=1; break
        fi
        echo "  sbatch failed for $NAME (attempt $attempt/3); retrying in 10s..." >&2
        sleep 10
    done
    if [ -n "$ok" ]; then
        echo "Submitted $NAME"
    else
        echo "FAILED to submit $NAME after 3 attempts" >&2
        FAILED_JOBS="$FAILED_JOBS $NAME"
    fi
    sleep 1
}

finish() {
    if [ -n "$FAILED_JOBS" ]; then
        echo "WARNING -- these jobs did NOT submit:$FAILED_JOBS" >&2
        return 1
    fi
    return 0
}
