#!/usr/bin/env bash
# ==============================================================================
# Hardware Validation of Pruned ResNet-18 on NVIDIA Jetson AGX Orin (Batch Size 32)
# ==============================================================================
# This script:
#   1. (Optional) Materializes the exact pruned architecture from a JouleNAS
#      pruning JSON log using materialize_pruned_model.py.
#   2. Executes the physical measurement campaign on the AGX Orin board at BS=32
#      for both Dense and Pruned ResNet-18 models.
#   3. Compares physical INA226 measurements against JouleNAS predictions.
#
# Usage:
#   ./validate_agx_orin_bs32.sh --pruning-log /path/to/model_pruning.json [options]
#   ./validate_agx_orin_bs32.sh /path/to/model_pruning.json [options]
#   ./validate_agx_orin_bs32.sh --dry-run
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BOARD="${BOARD:-agx_orin_bs32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKEND="${BACKEND:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PRUNING_LOG="${PRUNING_LOG:-}"
CAMPAIGN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./validate_agx_orin_bs32.sh [options] [path/to/pruning_log.json]

Options:
  --pruning-log, -p PATH   Path to JouleNAS pruning JSON log. If provided,
                           automatically materializes the architecture into
                           layers/resnet.py before launching measurements.
  --board BOARD            Board label (default: agx_orin_bs32).
  --batch-size BS          Batch size for inference (default: 32).
  --backend BACKEND        Execution backend: cuda, cpu (default: cuda).
  --dry-run                Dry-run mode: plan campaign without starting measurements.
  --help, -h               Show this help message.

Any additional options are forwarded directly to run_measurement_campaign.sh.
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pruning-log|-p)
            PRUNING_LOG="$2"
            shift 2
            ;;
        --pruning-log=*)
            PRUNING_LOG="${1#*=}"
            shift 1
            ;;
        --board)
            BOARD="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *.json)
            if [[ -z "${PRUNING_LOG}" && -f "$1" ]]; then
                PRUNING_LOG="$1"
            else
                CAMPAIGN_ARGS+=("$1")
            fi
            shift 1
            ;;
        *)
            CAMPAIGN_ARGS+=("$1")
            shift 1
            ;;
    esac
done

# Step 0: Materialize pruned architecture if a pruning log is provided
if [[ -n "${PRUNING_LOG}" ]]; then
    if [[ ! -f "${PRUNING_LOG}" ]]; then
        echo "Error: Pruning log file not found: ${PRUNING_LOG}" >&2
        exit 1
    fi
    echo "======================================================================"
    echo " [Step 0] Materializing Pruned Architecture from JSON Log"
    echo " Log: ${PRUNING_LOG}"
    echo "======================================================================"
    "${PYTHON_BIN}" materialize_pruned_model.py \
        --pruning-log "${PRUNING_LOG}" \
        --update-resnet
    echo ""
fi

echo "======================================================================"
echo " [Step 1] Running Physical Measurement Campaign on AGX Orin (BS=${BATCH_SIZE})"
echo " Board: ${BOARD} | Suite: pruned_validation_orin_bs32 | Backend: ${BACKEND}"
echo "======================================================================"

# Step 1: Execute physical measurement campaign for dense and pruned models at BS=32
BACKEND="${BACKEND}" NETWORK_BATCH_SIZE="${BATCH_SIZE}" ./run_measurement_campaign.sh \
    --board "${BOARD}" \
    --suite pruned_validation_orin_bs32 \
    "${CAMPAIGN_ARGS[@]}"

echo ""
echo "======================================================================"
echo " [Step 2] Comparing Physical Measurements vs JouleNAS Predictions"
echo "======================================================================"

VALIDATE_ARGS=(--board "${BOARD}" --batch-size "${BATCH_SIZE}")
if [[ -n "${PRUNING_LOG}" ]]; then
    VALIDATE_ARGS+=(--pruning-log "${PRUNING_LOG}")
fi

"${PYTHON_BIN}" processing_report/validate_pruned_measurements.py "${VALIDATE_ARGS[@]}"
