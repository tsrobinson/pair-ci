#!/bin/bash
# Early-stopping calibration check (App. P). One job.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_early_stopping.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== Early stopping calibration check ==="
python experiments/early_stopping_calibration.py \
    --n_reps 100 \
    --output results/early_stopping_calibration.csv 2>/dev/null
echo "Done."
EOF
)
submit "early_stop" "12:00:00" "4G" 16 "early_stop" "$PAYLOAD"
finish
