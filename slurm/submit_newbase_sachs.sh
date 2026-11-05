#!/bin/bash
# FZ-Rubin (micd-pooled) and MissDAG baselines on Sachs, real data and
# synthetic nonlinear edges (App. J). One job, two steps. The nonlinear step
# replays the single-job stream, so its datasets match the released per-cell
# PAIR-CI runs in distribution only (README, reproducibility caveats).
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_newbase_sachs.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "Starting Sachs new baselines at $(date)"
# Real (linear) Sachs -- bootstrap from flow cytometry data
python -u experiments/new_baselines.py \
    --benchmark sachs --dgp real \
    --n_reps 20 --data_path data/sachs_data.csv \
    --output results/newbase_sachs_linear.csv \
    2>/dev/null
# Synthetic nonlinear Sachs
python -u experiments/new_baselines.py \
    --benchmark sachs --dgp nonlinear \
    --n_reps 20 \
    --output results/newbase_sachs_nonlinear.csv \
    2>/dev/null
echo "Finished at $(date)"
EOF
)
submit "nb_sachs" "2-00:00:00" "4G" 16 "nb_sachs" "$PAYLOAD"
finish
