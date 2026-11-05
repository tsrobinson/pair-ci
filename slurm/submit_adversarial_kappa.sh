#!/bin/bash
# Adversarial kappa experiment (App. N). One job.
# Asks for 200 replicates; the released CSV has 100, as run by reproduce_all.sh.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_adversarial_kappa.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "Starting adversarial kappa experiment at $(date)"
echo "Node: $(hostname)"
echo ""
# Focused adversarial grid (cached imputation throughout, comparing imputer types):
#   6 DGPs x n=500 x 2 rates x 2 cond strategies x 2 imputer types
#   = 48 configs x 200 reps = 9,600 tests.
python -u experiments/adversarial_kappa.py \
    --n_reps 200 \
    --dgps hub chain dense_block weak_hub branch_separator hub_nonlinear \
    --sample_sizes 500 \
    --rates 0.3 0.5 \
    --cond_strategies full missing \
    --imputers linear rf \
    --output results/adversarial_kappa_results.csv
echo ""
echo "Finished at $(date)"
EOF
)
submit "adversarial_kappa" "2-00:00:00" "4G" 16 "adversarial_kappa" "$PAYLOAD"
finish
