#!/bin/bash
# FZ-Rubin (micd-pooled) baseline on HAILFINDER (Table 3, App. I). One job.
# Replays the single-job HAILFINDER stream, so its datasets match the released
# per-cell PAIR-CI runs in distribution only (README, reproducibility caveats).
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_newbase_hailfinder.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "Starting HAILFINDER new baselines (micd_pooled only; missDAG excluded"
echo "because its Gaussian linear assumptions are violated by the nonlinear DGP) at $(date)"
python -u experiments/new_baselines.py \
    --benchmark hailfinder --dgp nonlinear \
    --methods micd_pooled \
    --n_reps 20 \
    --rep_timeout 3600 \
    --output results/newbase_hailfinder_nonlinear.csv \
    2>/dev/null
echo "Finished at $(date)"
EOF
)
submit "nb_hailfinder" "7-00:00:00" "8G" 16 "nb_hailfinder" "$PAYLOAD"
finish
