#!/bin/bash
# Caching benchmark (p=5) and validation (p=5, 10, 20) (Section 3.2, App. C).
# One job, two steps.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_caching.sh
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"

PAYLOAD=$(cat <<'EOF'
echo "=== Caching benchmark (p=5) ==="
python experiments/caching_benchmark.py \
    --output results/caching_benchmark_results.csv 2>/dev/null
echo "=== Caching validation (p=5, 10, 20) ==="
python experiments/caching_validation.py \
    --output results/caching_validation_large_p.csv 2>/dev/null
echo "Done."
EOF
)
submit "caching" "5-00:00:00" "4G" 16 "caching" "$PAYLOAD"
finish
