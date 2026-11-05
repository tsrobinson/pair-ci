#!/bin/bash
# Naive versus Bayle within-imputation variance estimator (App. K). One job.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_variance_comparison.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== Variance estimator comparison (NB vs Bayle) ==="
python experiments/variance_comparison.py \
    --mode both --n_reps 50 --n_graphs 10 --n_datasets 20 \
    --output_calib results/variance_comparison_calibration.csv \
    --output_graph results/variance_comparison_graph.csv 2>/dev/null
echo "Done."
EOF
)
submit "var_comp" "3-00:00:00" "4G" 16 "var_comp" "$PAYLOAD"
finish
