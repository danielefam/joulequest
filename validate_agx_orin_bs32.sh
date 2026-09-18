#!/usr/bin/env bash
# ==============================================================================
# Hardware Validation of Pruned ResNet-18 on NVIDIA Jetson AGX Orin (Batch Size 32)
# ==============================================================================
# This script measures both the dense baseline and the pruned ResNet-18 model on
# physical Jetson AGX Orin hardware at Batch Size 32, then compares physical INA226
# measurements against the JouleNAS differentiable lookup predictions.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BOARD="${BOARD:-agx_orin_bs32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKEND="${BACKEND:-cuda}"

echo "======================================================================"
echo " Starting Physical Measurement Campaign on AGX Orin at Batch Size 32"
echo " Board: ${BOARD} | Batch Size: ${BATCH_SIZE} | Backend: ${BACKEND}"
echo "======================================================================"

# Step 1: Execute physical measurement campaign for dense and pruned models at BS=32
BACKEND="${BACKEND}" NETWORK_BATCH_SIZE="${BATCH_SIZE}" ./run_measurement_campaign.sh \
    --board "${BOARD}" \
    --suite pruned_validation_orin_bs32 \
    "$@"

echo ""
echo "======================================================================"
echo " Physical measurement finished. Comparing with JouleNAS predictions..."
echo "======================================================================"

# Step 2: Validate and generate comparison report
python3 processing_report/validate_pruned_measurements.py --board "${BOARD}" --batch-size "${BATCH_SIZE}"

