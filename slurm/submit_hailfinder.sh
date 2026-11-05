#!/bin/bash
# HAILFINDER benchmark (Table 3, App. I) as one job. The released results come
# from reproduce_all.sh's 10 per-cell jobs plus analysis/combine_hailfinder.py,
# which match this job's datasets only in the 10% complete and 10% MAR cells;
# elsewhere they are identically distributed but different.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_hailfinder.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== HAILFINDER benchmark (56 nodes, 66 edges, NONLINEAR) ==="
python experiments/hailfinder.py \
    --n_obs 1000 --n_reps 20 --dgp nonlinear \
    --output results/hailfinder_results.csv 2>/dev/null
echo "Done."
EOF
)
submit "hailfinder" "14-00:00:00" "8G" 16 "hailfinder" "$PAYLOAD"
finish
