#!/bin/bash
# FZ-Rubin (micd-pooled) and MissDAG baselines on the Erdos-Renyi sweeps
# (p in {10,20,30} x {linear,nonlinear} x {mar,mnar,mixed,complete}: 24 jobs)
# and on ALARM (2 jobs). Walltime is 2 days at p<=20 and 4 days at p=30;
# NOTEARS, inside MissDAG's M-step, dominates the runtime.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_new_baselines.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

submit_erdos() {
    local P=$1 TIME=$2
    for DGP in linear nonlinear; do
        for MECH in mar mnar mixed complete; do
            submit "nb_p${P}_${DGP}_${MECH}" "$TIME" "4G" 8 "newbase_p${P}_${DGP}_${MECH}" \
                "python experiments/new_baselines.py --benchmark erdos --p ${P} --mechanism ${MECH} --dgp ${DGP} --n_graphs 10 --n_datasets 20 --output results/newbase_p${P}_${DGP}_${MECH}.csv 2>/dev/null"
        done
    done
}

submit_alarm() {
    for DGP in linear nonlinear; do
        submit "nb_alarm_${DGP}" "2-00:00:00" "8G" 8 "newbase_alarm_${DGP}" \
            "python experiments/new_baselines.py --benchmark alarm --dgp ${DGP} --n_reps 20 --output results/newbase_alarm_${DGP}.csv 2>/dev/null"
    done
}

# Submit all sweeps. Comment out lines to run a subset.
submit_erdos 10 2-00:00:00
submit_erdos 20 2-00:00:00
submit_erdos 30 4-00:00:00
submit_alarm
finish
