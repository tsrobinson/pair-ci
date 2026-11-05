#!/bin/bash
# Sachs protein network, real flow-cytometry data (App. J). One job.
# Needs data/sachs_data.csv: run `bash data/fetch_sachs_data.sh` first.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_sachs.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
set -e
python experiments/sachs.py \
    --n_reps 20 \
    --data data/sachs_data.csv \
    --output results/sachs_results.csv 2>/dev/null
echo "Done."
EOF
)
submit "sachs" "5-00:00:00" "4G" 16 "sachs" "$PAYLOAD"
finish
