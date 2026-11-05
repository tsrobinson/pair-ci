#!/bin/bash
# PAIR-CI alpha sweep at p=10 (App. T.1, T.2). One job.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_alpha_sensitivity.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
set -e
python experiments/alpha_sensitivity.py \
    --n_graphs 10 --n_datasets 20 \
    --output results/alpha_sensitivity_results.csv 2>/dev/null
echo "Done."
EOF
)
submit "alpha_sens" "7-00:00:00" "4G" 16 "alpha_sens" "$PAYLOAD"
finish
