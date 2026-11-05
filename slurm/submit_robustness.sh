#!/bin/bash
# Robustness sweep at p=10 (App. F): 18 parallel jobs, 2 mechanisms x
# 3 missingness rates x 3 sample sizes; the same jobs as the robust_* loop
# in reproduce_all.sh.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_robustness.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

for MECH in mar mnar; do
    for MR in 0.1 0.3 0.5; do
        for NOBS in 500 1000 2000; do
            submit "robust_${MECH}_${MR}_${NOBS}" "5-00:00:00" "4G" 16 "robust_${MECH}_${MR}_${NOBS}" \
                "python experiments/robustness.py --mechanism ${MECH} --miss_rate ${MR} --n_obs ${NOBS} --n_graphs 10 --n_datasets 20 --output results/robustness_${MECH}_${MR}_${NOBS}.csv 2>/dev/null"
        done
    done
done
finish
