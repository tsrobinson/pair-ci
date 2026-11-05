#!/bin/bash
# Sensitivity to the number of imputations M (App. M). One job.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_m_sensitivity.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "Starting M-sensitivity experiment at $(date)"
echo "Node: $(hostname)"
echo ""
python experiments/m_sensitivity.py \
    --n_reps 50 \
    --Ms 3 5 10 20 \
    --K 5 \
    --dgps linear_gaussian post_nonlinear \
    --sample_sizes 500 1000 \
    --dim_x 5 \
    --signals 0 0.3 0.6 1.0 \
    --mechanisms complete mar mnar \
    --output results/m_sensitivity_results.csv \
    2>/dev/null
echo ""
echo "Finished at $(date)"
EOF
)
submit "m_sensitivity" "2-00:00:00" "4G" 16 "m_sensitivity" "$PAYLOAD"
finish
