#!/bin/bash
# p=30 graph recovery: 8 parallel jobs, 2 edge types x 4 mechanisms
# (Table 2, App. F).
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_p30.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

for DGP in linear nonlinear; do
    for MECH in mar mnar mixed complete; do
        submit "p30_${DGP}_${MECH}" "5-00:00:00" "8G" 16 "p30_${DGP}_${MECH}" \
            "python experiments/graph_recovery_p30.py --mechanism ${MECH} --dgp ${DGP} --n_graphs 10 --n_datasets 20 --output results/p30_${DGP}_${MECH}.csv 2>/dev/null"
    done
done
finish
