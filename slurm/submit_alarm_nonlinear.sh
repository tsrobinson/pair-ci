#!/bin/bash
# ALARM benchmark, nonlinear edges (Table 3, App. H). One job.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_alarm_nonlinear.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== ALARM benchmark (37 nodes, 46 edges, NONLINEAR) ==="
python experiments/alarm.py \
    --n_obs 1000 --n_reps 20 --dgp nonlinear \
    --output results/alarm_nonlinear_results.csv 2>/dev/null
echo "Done."
EOF
)
submit "alarm_nl" "10-00:00:00" "4G" 16 "alarm_nonlinear" "$PAYLOAD"
finish
