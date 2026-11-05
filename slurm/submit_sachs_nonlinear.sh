#!/bin/bash
# Sachs DAG with synthetic nonlinear edges (App. J) as one job. The released
# results come from reproduce_all.sh's 10 per-cell jobs plus
# analysis/combine_sachs_nonlinear.py, which match this job's datasets only in
# the 10% complete and 10% MAR cells; elsewhere they are identically
# distributed but different.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_sachs_nonlinear.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== Sachs synthetic benchmark (11 nodes, 17 edges, NONLINEAR) ==="
python experiments/sachs_nonlinear.py \
    --n_obs 1000 --n_reps 20 --dgp nonlinear \
    --output results/sachs_nonlinear_results.csv 2>/dev/null
echo "Done."
EOF
)
submit "sachs_nl" "10-00:00:00" "4G" 16 "sachs_nonlinear" "$PAYLOAD"
finish
