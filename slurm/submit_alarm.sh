#!/bin/bash
# ALARM benchmark, linear edges (Table 3, App. H). One job.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_alarm.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== ALARM benchmark (37 nodes, 46 edges) ==="
python experiments/alarm.py \
    --n_obs 1000 --n_reps 20 --dgp linear \
    --output results/alarm_results.csv 2>/dev/null
echo "Done."
EOF
)
submit "alarm" "10-00:00:00" "4G" 16 "alarm" "$PAYLOAD"
finish
