#!/bin/bash
# Learner comparison: RF, ExtraTrees, LightGBM, XGBoost (App. O). One job.
# The released run logged 52 h of test time, more than the 1-day walltime
# requested here; raise --time if your nodes are not faster.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_learner_comparison.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
# Install LightGBM + XGBoost into the env if not available (best-effort; the
# script silently skips any learner whose package import fails).
python -c "import lightgbm" 2>/dev/null || python -m pip install lightgbm --quiet 2>/dev/null || echo "LightGBM not installed; will skip lgbm/lgbm_tuned"
python -c "import xgboost" 2>/dev/null || python -m pip install xgboost --quiet 2>/dev/null || echo "XGBoost not installed; will skip xgboost"
echo "Starting learner comparison experiment at $(date)"
echo "Node: $(hostname)"
echo ""
# Learner list matches reproduce_all.sh so the standalone rerun produces the
# same columns the paper appendix reports.
python experiments/learner_comparison.py \
    --n_reps 50 \
    --learners rf extra_trees lgbm lgbm_tuned xgboost \
    --M 5 \
    --K 5 \
    --dgps linear_gaussian post_nonlinear \
    --sample_sizes 500 1000 \
    --dim_x 5 \
    --signals 0 0.3 0.6 1.0 \
    --mechanisms complete mar mnar \
    --output results/learner_comparison_results.csv \
    2>/dev/null
echo ""
echo "Finished at $(date)"
EOF
)
submit "learner_comparison" "1-00:00:00" "4G" 16 "learner_comparison" "$PAYLOAD"
finish
