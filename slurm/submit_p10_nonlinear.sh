#!/bin/bash
# p=10 graph recovery, nonlinear edges: 4 parallel jobs, one per mechanism
# (Table 2, App. F).
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_p10_nonlinear.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

for MECH in mar mnar mixed complete; do
    submit "p10_nl_${MECH}" "5-00:00:00" "4G" 16 "p10_nonlinear_${MECH}" \
        "python experiments/graph_recovery_p10.py --mechanism ${MECH} --dgp nonlinear --n_graphs 10 --n_datasets 20 --output results/p10_nonlinear_${MECH}.csv 2>/dev/null"
done
finish
