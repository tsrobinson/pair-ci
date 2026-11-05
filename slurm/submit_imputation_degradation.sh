#!/bin/bash
# Imputation-degradation experiment (App. Q). One job.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_imputation_degradation.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== Imputation degradation experiment ==="
python experiments/imputation_degradation.py \
    --n_reps 200 \
    --n_obs 500 \
    --dim_x 5 \
    --seed 42 \
    --output results/imputation_degradation.csv 2>/dev/null
echo "Done."
EOF
)
submit "imp_degrad" "1-00:00:00" "4G" 16 "imputation_degradation" "$PAYLOAD"
finish
