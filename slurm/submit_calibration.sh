#!/bin/bash
# Standalone calibration grid (Table 1): fast methods, then KCI, then merge.
# One job, three steps. A rerun matches the released files only up to Monte
# Carlo error: the corrected MAR x |X|=2 generator in src/dgps.py consumes the
# random stream differently (README, 'Changes since the submitted version').
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_calibration.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
set -e
echo "=== Step 1: Fast methods (ours, fisher_z, gcm, fisher_z_rubin) ==="
python experiments/calibration.py \
    --n_reps 100 \
    --methods ours fisher_z gcm fisher_z_rubin \
    --output results/ci_calibration_fast.csv 2>/dev/null
echo "=== Step 2: KCI (n=1000 only, 50 reps) ==="
python experiments/calibration.py \
    --n_reps 50 \
    --methods kci \
    --sample_sizes 1000 \
    --signals 0 0.6 \
    --output results/ci_calibration_kci.csv 2>/dev/null
echo "=== Step 3: Merge ==="
python -c "
import pandas as pd
fast = pd.read_csv('results/ci_calibration_fast.csv')
kci = pd.read_csv('results/ci_calibration_kci.csv')
merged = pd.concat([fast, kci], ignore_index=True)
merged.to_csv('results/ci_calibration_results.csv', index=False)
print(f'Merged: {len(merged)} rows -> results/ci_calibration_results.csv')
"
echo "Done."
EOF
)
submit "calib_unified" "7-00:00:00" "4G" 16 "calibration" "$PAYLOAD"
finish
