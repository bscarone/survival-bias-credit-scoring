#!/usr/bin/env bash
# =============================================================================
# Run the controlled-exploration sweep for all six paper configurations.
#
# Reproduces the results behind Section 5 (Tables 2-4, Figures 4-6) and
# Appendix F: three datasets x two models x three sampling strategies x
# eleven exploration rates x five runs.
#
# Usage:
#   bash run_all_exploration.sh          # all six paper configurations
#   bash run_all_exploration.sh --quick  # quick test (Default/RF, 1 run)
#
# Note on down-sampling: PPDai and LendingClub are down-sampled to a 77%
# non-default rate (--imbalance-ratio 0.77), producing result files with an
# _imb77 suffix. Default is used as distributed (77.9% non-default) and must
# NOT be down-sampled. This is why the two groups run separately below.
#
# Expected CSV rows per dataset/model (data rows + 1 header):
#   full:   11 rates x 3 rankings x 5 runs x 11 iterations = 1815 + 1 = 1816
#   --quick: 6 rates x 1 ranking  x 1 run  x 11 iterations =   66 + 1 =   67
# Verify with:
#   wc -l experiments/exploration/results/exploration_<dataset>_<model>.csv
# =============================================================================

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-experiments/exploration/results}"

MODELS="${MODELS:-rf gbdt}"
RANKINGS="${RANKINGS:-least_risky random most_risky}"
GRANULARITY="${GRANULARITY:-standard}"
N_RUNS="${N_RUNS:-5}"
T="${T:-10}"
TAU="${TAU:-0.5}"

# Datasets used as distributed, and datasets requiring down-sampling.
DATASETS_RAW="${DATASETS_RAW:-default}"
DATASETS_IMB77="${DATASETS_IMB77:-ppdai lendingclub}"

if [[ "${1:-}" == "--quick" ]]; then
    echo "=== QUICK MODE (Default/RF, 1 run, coarse grid) ==="
    DATASETS_RAW="default"
    DATASETS_IMB77=""
    MODELS="rf"
    RANKINGS="least_risky"
    GRANULARITY="coarse"
    N_RUNS=1
    RESULTS_DIR="experiments/exploration/results_quick"
fi

echo "============================================================"
echo "EXPLORATION EXPERIMENTS"
echo "  Datasets (as distributed): ${DATASETS_RAW:-none}"
echo "  Datasets (imb 0.77):       ${DATASETS_IMB77:-none}"
echo "  Models:      ${MODELS}"
echo "  Rankings:    ${RANKINGS}"
echo "  Granularity: ${GRANULARITY}"
echo "  Runs:        ${N_RUNS}"
echo "  Results:     ${RESULTS_DIR}"
echo "============================================================"

mkdir -p "${RESULTS_DIR}"

if [[ -n "${DATASETS_RAW}" ]]; then
    echo ""
    echo ">>> Datasets used as distributed: ${DATASETS_RAW}"
    python run_exploration_experiments.py \
        --datasets ${DATASETS_RAW} \
        --models ${MODELS} \
        --rankings ${RANKINGS} \
        --granularity "${GRANULARITY}" \
        --n-runs "${N_RUNS}" \
        --T "${T}" \
        --tau "${TAU}" \
        --results-dir "${RESULTS_DIR}"
fi

if [[ -n "${DATASETS_IMB77}" ]]; then
    echo ""
    echo ">>> Down-sampled datasets (--imbalance-ratio 0.77): ${DATASETS_IMB77}"
    python run_exploration_experiments.py \
        --datasets ${DATASETS_IMB77} \
        --models ${MODELS} \
        --rankings ${RANKINGS} \
        --granularity "${GRANULARITY}" \
        --n-runs "${N_RUNS}" \
        --T "${T}" \
        --tau "${TAU}" \
        --imbalance-ratio 0.77 \
        --results-dir "${RESULTS_DIR}"
fi

echo ""
echo "============================================================"
echo "DONE"
echo ""
echo "Results: ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}"/*.csv 2>/dev/null || echo "  (no CSVs found)"
echo "============================================================"