#!/bin/bash
# Search check (App. E, 'PC algorithm with cached imputations'). Targets:
#   baselines  test-wise deletion, FZ-vote and complete case under both the
#              simplified and the PC-stable search, on Table 3's 20%-MNAR
#              cells; 2 jobs -> results/search_check/rerun_{alarm_nl,hailf}.csv
#              (reported in the paper)
#   pairci     PAIR-CI under the PC-stable search, one job per replicate,
#              2 networks x 20 = 40 jobs
#              -> results/search_check/pairci_stdsearch_*_repNN.csv
#              (optional; never run, not reported)
# With no target, only baselines is submitted.
#
# Usage: [PARTITION=<name>] [DRY_RUN=1] bash slurm/submit_search_check.sh [baselines] [pairci]
# Run with bash, not sbatch; slurm/_common.sh describes the options.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/_common.sh" ] || { echo "Run as: bash slurm/$(basename "$0") from the package directory tree (not via sbatch)." >&2; exit 1; }
. "$HERE/_common.sh"


[ "$DRY_RUN" = "1" ] || mkdir -p results/search_check

do_baselines() {
    submit "sc_base_alarm_nl" "12:00:00" "4G" 2 "sc_base_alarm_nl" \
        "python -u experiments/search_check_baselines.py --network alarm_nonlinear 2>/dev/null"
    submit "sc_base_hailf" "24:00:00" "8G" 2 "sc_base_hailf" \
        "python -u experiments/search_check_baselines.py --network hailfinder 2>/dev/null"
}

do_pairci() {
    for NET in alarm_nonlinear hailfinder; do
        for REP in $(seq 0 19); do
            submit "sc_${NET}_${REP}" "12:00:00" "" 8 "sc_${NET}_${REP}" \
                "python -u experiments/search_check_pairci.py --network $NET --rep $REP --out results/search_check/"
        done
    done
}

if [ $# -eq 0 ]; then
    do_baselines
else
    for target in "$@"; do "do_${target}"; done
fi
finish
