#!/usr/bin/env bash

set -Eeuo pipefail

# Run a complete measurement campaign on exactly one board. This script only
# schedules experiments; automated_measurement.py still owns one experiment.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# User-configurable campaign parameters. Every value can also be overridden by
# exporting an environment variable with the same name before running.
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
CONNECTION_CONFIG="${CONNECTION_CONFIG:-${SCRIPT_DIR}/measurement_hosts.local.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/measurements/runs}"

BACKEND="${BACKEND:-cuda}"
MODEL_ROOT="${MODEL_ROOT:-Models/CUDA}"
MODEL_SUFFIX="${MODEL_SUFFIX:-.pt}"
LINEAR_MODEL_DIRECTORY="${LINEAR_MODEL_DIRECTORY:-${MODEL_ROOT}/Linear}"
CONV_MODEL_DIRECTORY="${CONV_MODEL_DIRECTORY:-${MODEL_ROOT}/Conv}"
LENET_MODEL_DIRECTORY="${LENET_MODEL_DIRECTORY:-${MODEL_ROOT}/Lenet}"
LENET_MODEL_PATH="${LENET_MODEL_PATH:-${LENET_MODEL_DIRECTORY}/Lenet${MODEL_SUFFIX}}"
POOL_MODEL_DIRECTORY="${POOL_MODEL_DIRECTORY:-${MODEL_ROOT}/Pool}"
RELU_MODEL_DIRECTORY="${RELU_MODEL_DIRECTORY:-${MODEL_ROOT}/ReLU}"
FLATTEN_MODEL_DIRECTORY="${FLATTEN_MODEL_DIRECTORY:-${MODEL_ROOT}/Flatten}"
RESNET18_MODEL_DIRECTORY="${RESNET18_MODEL_DIRECTORY:-${MODEL_ROOT}/ResNet18}"
RESNET18_MODEL_PATH="${RESNET18_MODEL_PATH:-${RESNET18_MODEL_DIRECTORY}/ResNet18_${RESNET18_IMAGE_SIZE:-224}_${RESNET18_NUM_CLASSES:-1000}${MODEL_SUFFIX}}"
RESNET50_MODEL_DIRECTORY="${RESNET50_MODEL_DIRECTORY:-${MODEL_ROOT}/ResNet50}"
RESNET50_MODEL_PATH="${RESNET50_MODEL_PATH:-${RESNET50_MODEL_DIRECTORY}/ResNet50_${RESNET50_IMAGE_SIZE:-224}_${RESNET50_NUM_CLASSES:-1000}${MODEL_SUFFIX}}"

NUMBER_OF_CYCLES="${NUMBER_OF_CYCLES:-100}"
# BATCH_SIZE remains a compatibility override when the two suite-specific
# values are not supplied. The default campaign uses batch 1 for every suite.
BATCH_SIZE="${BATCH_SIZE:-}"
LINEAR_CONV_BATCH_SIZE="${LINEAR_CONV_BATCH_SIZE:-${BATCH_SIZE:-1}}"
NETWORK_BATCH_SIZE="${NETWORK_BATCH_SIZE:-${BATCH_SIZE:-1}}"
SLEEP_TIME="${SLEEP_TIME:-3}"
TARGET_BURST_SECONDS="${TARGET_BURST_SECONDS:-0}"
SAMPLING_RATE_HZ="${SAMPLING_RATE_HZ:-100}"
MIN_ACTIVE_SAMPLES="${MIN_ACTIVE_SAMPLES:-100}"
WARMUP_INFERENCES="${WARMUP_INFERENCES:-20}"
WARMUP_SECONDS="${WARMUP_SECONDS:-0}"
WARMUP_COOLDOWN_SECONDS="${WARMUP_COOLDOWN_SECONDS:-}"
CALIBRATION_INITIAL_INFERENCES="${CALIBRATION_INITIAL_INFERENCES:-20}"
CALIBRATION_TARGET_SECONDS="${CALIBRATION_TARGET_SECONDS:-1}"
CALIBRATION_REPETITIONS="${CALIBRATION_REPETITIONS:-8}"
CALIBRATION_SIZING_MAX_ATTEMPTS="${CALIBRATION_SIZING_MAX_ATTEMPTS:-3}"
CALIBRATION_DURATION_TOLERANCE="${CALIBRATION_DURATION_TOLERANCE:-0.20}"
MAX_RELATIVE_MAD="${MAX_RELATIVE_MAD:-0.15}"
BURST_DURATION_MARGIN="${BURST_DURATION_MARGIN:-1.2}"
VALIDATION_REPETITIONS="${VALIDATION_REPETITIONS:-3}"
VALIDATION_MAX_ROUNDS="${VALIDATION_MAX_ROUNDS:-3}"
VALIDATION_SAFETY_MARGIN="${VALIDATION_SAFETY_MARGIN:-1.1}"
VALIDATION_COOLDOWN_SECONDS="${VALIDATION_COOLDOWN_SECONDS:-}"
CLOCK_SYNC_EXCHANGES="${CLOCK_SYNC_EXCHANGES:-10}"
MAX_CLOCK_UNCERTAINTY_FRACTION="${MAX_CLOCK_UNCERTAINTY_FRACTION:-0.50}"
MAX_CALIBRATION_INFERENCES="${MAX_CALIBRATION_INFERENCES:-1000000}"
LEADING_IDLE_SECONDS="${LEADING_IDLE_SECONDS:-5}"
TRAILING_IDLE_SECONDS="${TRAILING_IDLE_SECONDS:-30}"
SAFETY_MARGIN_SECONDS="${SAFETY_MARGIN_SECONDS:-2}"
SHUNT_OHMS="${SHUNT_OHMS:-0.012}"
MAX_EXPECTED_CURRENT_A="${MAX_EXPECTED_CURRENT_A:-5.0}"
INA226_PORT="${INA226_PORT:-}"
EXPERIMENT_COOLDOWN_SECONDS="${EXPERIMENT_COOLDOWN_SECONDS:-20}"

LINEAR_SIZES="${LINEAR_SIZES:-64 128 256 512 1024 2048 4096 8192}"

# The protocol table displays input channels as rows and image sizes as
# columns. To avoid implausibly expensive high-channel/high-resolution cases,
# the campaign caps each input-channel range at a realistic maximum resolution.
# CONV_IMAGE_SIZES remains a global filter over those allowed resolutions.
# Filenames deliberately use the runner convention:
# Conv_<input_channels>_<image_size>_<kernel_size>_<padding>.pt
CONV_INPUT_CHANNELS="${CONV_INPUT_CHANNELS:-1 2 4 8 16 32 64 128 256 512}"
CONV_OUTPUT_CHANNELS="${CONV_OUTPUT_CHANNELS:-1 8 16 32 64 128 256 512}"
CONV_IMAGE_SIZES="${CONV_IMAGE_SIZES:-32 64 128 256 512 1024}"
CONV_KERNEL_PADDING="${CONV_KERNEL_PADDING:-3:0 3:1 5:0 5:1}"

# Ordered standalone operations in TorchRunner's LeNet definition. The first
# dimension of ReLU and Flatten models is the batch dimension and is replaced
# by NETWORK_BATCH_SIZE by the runner.
LENET_COMPONENT_MODELS=(
    "${CONV_MODEL_DIRECTORY}/Conv_1_32_5_0_8${MODEL_SUFFIX}"
    "${POOL_MODEL_DIRECTORY}/MaxPool_8_28_2${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_8_14_14${MODEL_SUFFIX}"
    "${CONV_MODEL_DIRECTORY}/Conv_8_14_5_0_32${MODEL_SUFFIX}"
    "${POOL_MODEL_DIRECTORY}/MaxPool_32_10_2${MODEL_SUFFIX}"
    "${POOL_MODEL_DIRECTORY}/AdaPool_32_5_4${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_32_4_4${MODEL_SUFFIX}"
    "${FLATTEN_MODEL_DIRECTORY}/Flatten_1_32_4_4${MODEL_SUFFIX}"
    "${LINEAR_MODEL_DIRECTORY}/Linear_512_128${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_128${MODEL_SUFFIX}"
    "${LINEAR_MODEL_DIRECTORY}/Linear_128_64${MODEL_SUFFIX}"
)

# ResNet-18 uses the standard [2, 2, 2, 2] basic-block layout. This list contains
# each distinct shape-specific operation once; repeat counts are part of the graph.
RESNET18_IMAGE_SIZE="${RESNET18_IMAGE_SIZE:-224}"
RESNET18_NUM_CLASSES="${RESNET18_NUM_CLASSES:-1000}"
RESNET50_IMAGE_SIZE="${RESNET50_IMAGE_SIZE:-224}"
RESNET50_NUM_CLASSES="${RESNET50_NUM_CLASSES:-1000}"
RESNET_STEM_SIZE=$(((RESNET18_IMAGE_SIZE + 1) / 2))
RESNET_STAGE1_SIZE=$(((RESNET_STEM_SIZE + 1) / 2))
RESNET_STAGE2_SIZE=$(((RESNET_STAGE1_SIZE + 1) / 2))
RESNET_STAGE3_SIZE=$(((RESNET_STAGE2_SIZE + 1) / 2))
RESNET_STAGE4_SIZE=$(((RESNET_STAGE3_SIZE + 1) / 2))
RESNET18_COMPONENT_MODELS=(
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_3_64_${RESNET18_IMAGE_SIZE}_7_2_3${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetBatchNorm_64_${RESNET_STEM_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_64_${RESNET_STEM_SIZE}_${RESNET_STEM_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetMaxPool_64_${RESNET_STEM_SIZE}_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_64_${RESNET_STAGE1_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetBatchNorm_64_${RESNET_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_64_${RESNET_STAGE1_SIZE}_${RESNET_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetResidualAdd_64_${RESNET_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_${RESNET_STAGE1_SIZE}_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_${RESNET_STAGE1_SIZE}_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetBatchNorm_128_${RESNET_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_128_${RESNET_STAGE2_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_128_${RESNET_STAGE2_SIZE}_${RESNET_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetResidualAdd_128_${RESNET_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_${RESNET_STAGE2_SIZE}_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_${RESNET_STAGE2_SIZE}_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetBatchNorm_256_${RESNET_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_256_${RESNET_STAGE3_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_256_${RESNET_STAGE3_SIZE}_${RESNET_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetResidualAdd_256_${RESNET_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_${RESNET_STAGE3_SIZE}_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_${RESNET_STAGE3_SIZE}_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetBatchNorm_512_${RESNET_STAGE4_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_512_512_${RESNET_STAGE4_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_512_${RESNET_STAGE4_SIZE}_${RESNET_STAGE4_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetResidualAdd_512_${RESNET_STAGE4_SIZE}${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetAvgPool_512_${RESNET_STAGE4_SIZE}_1${MODEL_SUFFIX}"
    "${FLATTEN_MODEL_DIRECTORY}/Flatten_1_512_1_1${MODEL_SUFFIX}"
    "${LINEAR_MODEL_DIRECTORY}/Linear_512_${RESNET18_NUM_CLASSES}${MODEL_SUFFIX}"
)

# ResNet-50 uses the standard [3, 4, 6, 3] bottleneck layout. As with ResNet-18,
# each distinct shape-specific operation is measured once.
RESNET50_STEM_SIZE=$(((RESNET50_IMAGE_SIZE + 1) / 2))
RESNET50_STAGE1_SIZE=$(((RESNET50_STEM_SIZE + 1) / 2))
RESNET50_STAGE2_SIZE=$(((RESNET50_STAGE1_SIZE + 1) / 2))
RESNET50_STAGE3_SIZE=$(((RESNET50_STAGE2_SIZE + 1) / 2))
RESNET50_STAGE4_SIZE=$(((RESNET50_STAGE3_SIZE + 1) / 2))
RESNET50_COMPONENT_MODELS=(
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_3_64_${RESNET50_IMAGE_SIZE}_7_2_3${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_64_${RESNET50_STEM_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_64_${RESNET50_STEM_SIZE}_${RESNET50_STEM_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetMaxPool_64_${RESNET50_STEM_SIZE}_3_2_1${MODEL_SUFFIX}"

    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_64_64_${RESNET50_STAGE1_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_256_64_${RESNET50_STAGE1_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_64_64_${RESNET50_STAGE1_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_64_256_${RESNET50_STAGE1_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_64_${RESNET50_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_64_${RESNET50_STAGE1_SIZE}_${RESNET50_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_256_${RESNET50_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetResidualAdd_256_${RESNET50_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_256_${RESNET50_STAGE1_SIZE}_${RESNET50_STAGE1_SIZE}${MODEL_SUFFIX}"

    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_256_128_${RESNET50_STAGE1_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_128_${RESNET50_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_128_${RESNET50_STAGE1_SIZE}_${RESNET50_STAGE1_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_128_128_${RESNET50_STAGE1_SIZE}_3_2_1${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_256_512_${RESNET50_STAGE1_SIZE}_1_2_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_128_${RESNET50_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_128_${RESNET50_STAGE2_SIZE}_${RESNET50_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_128_512_${RESNET50_STAGE2_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_512_128_${RESNET50_STAGE2_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_128_128_${RESNET50_STAGE2_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_512_${RESNET50_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetResidualAdd_512_${RESNET50_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_512_${RESNET50_STAGE2_SIZE}_${RESNET50_STAGE2_SIZE}${MODEL_SUFFIX}"

    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_512_256_${RESNET50_STAGE2_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_256_${RESNET50_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_256_${RESNET50_STAGE2_SIZE}_${RESNET50_STAGE2_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_256_256_${RESNET50_STAGE2_SIZE}_3_2_1${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_512_1024_${RESNET50_STAGE2_SIZE}_1_2_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_256_${RESNET50_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_256_${RESNET50_STAGE3_SIZE}_${RESNET50_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_256_1024_${RESNET50_STAGE3_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_1024_256_${RESNET50_STAGE3_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_256_256_${RESNET50_STAGE3_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_1024_${RESNET50_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetResidualAdd_1024_${RESNET50_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_1024_${RESNET50_STAGE3_SIZE}_${RESNET50_STAGE3_SIZE}${MODEL_SUFFIX}"

    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_1024_512_${RESNET50_STAGE3_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_512_${RESNET50_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_512_${RESNET50_STAGE3_SIZE}_${RESNET50_STAGE3_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_512_512_${RESNET50_STAGE3_SIZE}_3_2_1${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_1024_2048_${RESNET50_STAGE3_SIZE}_1_2_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_512_${RESNET50_STAGE4_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_512_${RESNET50_STAGE4_SIZE}_${RESNET50_STAGE4_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_512_2048_${RESNET50_STAGE4_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_2048_512_${RESNET50_STAGE4_SIZE}_1_1_0${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetConv_512_512_${RESNET50_STAGE4_SIZE}_3_1_1${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetBatchNorm_2048_${RESNET50_STAGE4_SIZE}${MODEL_SUFFIX}"
    "${RESNET50_MODEL_DIRECTORY}/ResNetResidualAdd_2048_${RESNET50_STAGE4_SIZE}${MODEL_SUFFIX}"
    "${RELU_MODEL_DIRECTORY}/ReLU_1_2048_${RESNET50_STAGE4_SIZE}_${RESNET50_STAGE4_SIZE}${MODEL_SUFFIX}"

    "${RESNET50_MODEL_DIRECTORY}/ResNetAvgPool_2048_${RESNET50_STAGE4_SIZE}_1${MODEL_SUFFIX}"
    "${FLATTEN_MODEL_DIRECTORY}/Flatten_1_2048_1_1${MODEL_SUFFIX}"
    "${LINEAR_MODEL_DIRECTORY}/Linear_2048_${RESNET50_NUM_CLASSES}${MODEL_SUFFIX}"
)

# Set REPEAT_COMPLETED=1 to rerun models that already have a COMPLETE manifest
# in this board's output directory. Set CONTINUE_ON_ERROR=1 to keep scheduling
# after an experiment fails.
REPEAT_COMPLETED="${REPEAT_COMPLETED:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

usage() {
    cat <<'EOF'
Usage:
  ./run_measurement_campaign.sh --board BOARD_LABEL [options]

Options:
  --board LABEL       Safe label used only for the local result directory.
    --suite SUITES      Comma-separated suite list: linear, conv, lenet,
                                            resnet18, resnet50, or all (default: all). all runs
                                            Linear, Conv, and LeNet only.
  --dry-run           Print commands without running measurements.
  --continue-on-error Continue after an experiment exits nonzero.
  --repeat-completed  Rerun experiments with an existing COMPLETE manifest.
    --experiment-cooldown-seconds SECONDS  Inter-experiment idle (default: 20).
  -h, --help          Show this message.

The connection target and remote paths come from measurement_hosts.local.json.
The script runs one board per invocation. Change the physical board only after
the script finishes, then invoke it again with a new BOARD_LABEL/configuration.

Examples:
  ./run_measurement_campaign.sh --board jetson_nano --suite all
    ./run_measurement_campaign.sh --board jetson_nano --suite all,resnet18,resnet50
  ./run_measurement_campaign.sh --board pi5 --suite linear --dry-run

Override parameters without editing the script:
    BACKEND=cpu LINEAR_CONV_BATCH_SIZE=16 NETWORK_BATCH_SIZE=8 \
    MAX_EXPECTED_CURRENT_A=3.0 \
    ./run_measurement_campaign.sh --board pi5 --suite all
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

is_positive_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] &&
        awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

is_nonnegative_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

suite_is_selected() {
    [[ -n "${SELECTED_SUITES[$1]+selected}" ]]
}

conv_max_image_size() {
    local input_channels="$1"

    if ((input_channels <= 8)); then
        printf '1024\n'
    elif ((input_channels <= 32)); then
        printf '512\n'
    elif ((input_channels <= 64)); then
        printf '256\n'
    elif ((input_channels <= 128)); then
        printf '256\n'
    elif ((input_channels <= 256)); then
        printf '128\n'
    else
        printf '64\n'
    fi
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

manifest_exists_for_model() {
    local model_path="$1"
    local batch_size="$2"
    "$PYTHON_BIN" - "$OUTPUT_DIRECTORY" "$model_path" "$batch_size" <<'PY'
import json
import re
import sys
from pathlib import Path

output_directory = Path(sys.argv[1])
model_path = sys.argv[2]
batch_size = int(sys.argv[3])
model_path_object = Path(model_path)
completed_model_paths = {model_path}
legacy_conv_match = re.fullmatch(
    r"(Conv_\d+_\d+_\d+_\d+)_1(\.[^.]+)", model_path_object.name
)
if legacy_conv_match:
    completed_model_paths.add(
        str(
            model_path_object.with_name(
                f"{legacy_conv_match.group(1)}{legacy_conv_match.group(2)}"
            )
        )
    )
for manifest_path in output_directory.glob("*.json"):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if (
        manifest.get("status") == "COMPLETE"
        and manifest.get("model_path") in completed_model_paths
        and manifest.get("input_batch_size", 1) == batch_size
    ):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

BOARD_LABEL=""
SUITE="all"
DRY_RUN=0

while (($#)); do
    case "$1" in
        --board)
            (($# >= 2)) || die "--board requires a value"
            BOARD_LABEL="$2"
            shift 2
            ;;
        --suite)
            (($# >= 2)) || die "--suite requires a value"
            SUITE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        --repeat-completed)
            REPEAT_COMPLETED=1
            shift
            ;;
        --experiment-cooldown-seconds)
            (($# >= 2)) || die "--experiment-cooldown-seconds requires a value"
            EXPERIMENT_COOLDOWN_SECONDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$BOARD_LABEL" ]] || die "--board is required"
[[ "$BOARD_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "--board may contain only letters, numbers, dot, underscore, and dash"
IFS=',' read -r -a REQUESTED_SUITES <<<"$SUITE"
declare -A SELECTED_SUITES=()
for suite_name in "${REQUESTED_SUITES[@]}"; do
    case "$suite_name" in
        all)
            SELECTED_SUITES[linear]=1
            SELECTED_SUITES[conv]=1
            SELECTED_SUITES[lenet]=1
            ;;
        linear|conv|lenet|resnet18|resnet50)
            SELECTED_SUITES["$suite_name"]=1
            ;;
        *)
            die "--suite must be a comma-separated list of linear, conv, lenet, resnet18, resnet50, or all"
            ;;
    esac
done
[[ "$BACKEND" == "cpu" || "$BACKEND" == "cuda" || "$BACKEND" == "tpu" ]] ||
    die "BACKEND must be cpu, cuda, or tpu"
[[ "$NUMBER_OF_CYCLES" =~ ^[1-9][0-9]*$ ]] ||
    die "NUMBER_OF_CYCLES must be a positive integer"
[[ -z "$BATCH_SIZE" || "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
    die "BATCH_SIZE must be a positive integer when set"
[[ "$LINEAR_CONV_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
    die "LINEAR_CONV_BATCH_SIZE must be a positive integer"
[[ "$NETWORK_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
    die "NETWORK_BATCH_SIZE must be a positive integer"
[[ "$RESNET18_IMAGE_SIZE" =~ ^[3-9][0-9]*$ || "$RESNET18_IMAGE_SIZE" =~ ^[1-9][0-9]{2,}$ ]] ||
    die "RESNET18_IMAGE_SIZE must be an integer of at least 32"
[[ "$RESNET18_NUM_CLASSES" =~ ^[1-9][0-9]*$ ]] ||
    die "RESNET18_NUM_CLASSES must be a positive integer"
[[ "$RESNET50_IMAGE_SIZE" =~ ^[3-9][0-9]*$ || "$RESNET50_IMAGE_SIZE" =~ ^[1-9][0-9]{2,}$ ]] ||
    die "RESNET50_IMAGE_SIZE must be an integer of at least 32"
[[ "$RESNET50_NUM_CLASSES" =~ ^[1-9][0-9]*$ ]] ||
    die "RESNET50_NUM_CLASSES must be a positive integer"
[[ "$MIN_ACTIVE_SAMPLES" =~ ^[1-9][0-9]*$ ]] ||
    die "MIN_ACTIVE_SAMPLES must be a positive integer"
[[ "$WARMUP_INFERENCES" =~ ^[0-9]+$ ]] ||
    die "WARMUP_INFERENCES must be a nonnegative integer"
[[ "$CALIBRATION_INITIAL_INFERENCES" =~ ^[1-9][0-9]*$ ]] ||
    die "CALIBRATION_INITIAL_INFERENCES must be a positive integer"
[[ "$CALIBRATION_REPETITIONS" =~ ^[0-9]+$ ]] &&
    ((CALIBRATION_REPETITIONS >= 2)) ||
    die "CALIBRATION_REPETITIONS must be an integer of at least 2"
[[ "$CALIBRATION_SIZING_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] ||
    die "CALIBRATION_SIZING_MAX_ATTEMPTS must be a positive integer"
[[ "$VALIDATION_REPETITIONS" =~ ^[1-9][0-9]*$ ]] ||
    die "VALIDATION_REPETITIONS must be a positive integer"
[[ "$VALIDATION_MAX_ROUNDS" =~ ^[1-9][0-9]*$ ]] ||
    die "VALIDATION_MAX_ROUNDS must be a positive integer"
[[ "$MAX_CALIBRATION_INFERENCES" =~ ^[1-9][0-9]*$ ]] ||
    die "MAX_CALIBRATION_INFERENCES must be a positive integer"
is_nonnegative_number "$SLEEP_TIME" || die "SLEEP_TIME must be nonnegative"
is_nonnegative_number "$TARGET_BURST_SECONDS" ||
    die "TARGET_BURST_SECONDS must be nonnegative"
is_nonnegative_number "$WARMUP_SECONDS" ||
    die "WARMUP_SECONDS must be nonnegative"
if [[ -n "$WARMUP_COOLDOWN_SECONDS" ]]; then
    is_nonnegative_number "$WARMUP_COOLDOWN_SECONDS" ||
        die "WARMUP_COOLDOWN_SECONDS must be nonnegative"
fi
is_positive_number "$CALIBRATION_TARGET_SECONDS" ||
    die "CALIBRATION_TARGET_SECONDS must be positive"
is_nonnegative_number "$CALIBRATION_DURATION_TOLERANCE" &&
    awk -v value="$CALIBRATION_DURATION_TOLERANCE" 'BEGIN { exit !(value < 1) }' ||
    die "CALIBRATION_DURATION_TOLERANCE must be in the range [0, 1)"
is_nonnegative_number "$MAX_RELATIVE_MAD" ||
    die "MAX_RELATIVE_MAD must be nonnegative"
is_positive_number "$BURST_DURATION_MARGIN" &&
    awk -v value="$BURST_DURATION_MARGIN" 'BEGIN { exit !(value >= 1) }' ||
    die "BURST_DURATION_MARGIN must be at least 1"
is_positive_number "$VALIDATION_SAFETY_MARGIN" &&
    awk -v value="$VALIDATION_SAFETY_MARGIN" 'BEGIN { exit !(value >= 1) }' ||
    die "VALIDATION_SAFETY_MARGIN must be at least 1"
if [[ -n "$VALIDATION_COOLDOWN_SECONDS" ]]; then
    is_nonnegative_number "$VALIDATION_COOLDOWN_SECONDS" ||
        die "VALIDATION_COOLDOWN_SECONDS must be nonnegative"
fi
[[ "$CLOCK_SYNC_EXCHANGES" =~ ^[1-9][0-9]*$ ]] ||
    die "CLOCK_SYNC_EXCHANGES must be a positive integer"
is_positive_number "$MAX_CLOCK_UNCERTAINTY_FRACTION" ||
    die "MAX_CLOCK_UNCERTAINTY_FRACTION must be positive"
is_nonnegative_number "$LEADING_IDLE_SECONDS" ||
    die "LEADING_IDLE_SECONDS must be nonnegative"
is_nonnegative_number "$TRAILING_IDLE_SECONDS" ||
    die "TRAILING_IDLE_SECONDS must be nonnegative"
is_nonnegative_number "$SAFETY_MARGIN_SECONDS" ||
    die "SAFETY_MARGIN_SECONDS must be nonnegative"
is_positive_number "$SAMPLING_RATE_HZ" ||
    die "SAMPLING_RATE_HZ must be positive"
is_positive_number "$SHUNT_OHMS" || die "SHUNT_OHMS must be positive"
is_positive_number "$MAX_EXPECTED_CURRENT_A" ||
    die "MAX_EXPECTED_CURRENT_A must be positive"
is_nonnegative_number "$EXPERIMENT_COOLDOWN_SECONDS" ||
    die "EXPERIMENT_COOLDOWN_SECONDS must be nonnegative"

[[ -f "${SCRIPT_DIR}/automated_measurement.py" ]] ||
    die "automated_measurement.py not found beside this script"
[[ -f "$CONNECTION_CONFIG" ]] ||
    die "connection config not found: $CONNECTION_CONFIG"
command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
    die "Python executable not found: $PYTHON_BIN"

read -r -a LINEAR_SIZE_LIST <<<"$LINEAR_SIZES"
read -r -a CONV_INPUT_CHANNEL_LIST <<<"$CONV_INPUT_CHANNELS"
read -r -a CONV_OUTPUT_CHANNEL_LIST <<<"$CONV_OUTPUT_CHANNELS"
read -r -a CONV_IMAGE_SIZE_LIST <<<"$CONV_IMAGE_SIZES"
read -r -a CONV_KERNEL_PADDING_LIST <<<"$CONV_KERNEL_PADDING"

conv_image_pair_total=0
for input_channels in "${CONV_INPUT_CHANNEL_LIST[@]}"; do
    max_image_size="$(conv_max_image_size "$input_channels")"
    for image_size in "${CONV_IMAGE_SIZE_LIST[@]}"; do
        if ((image_size <= max_image_size)); then
            conv_image_pair_total=$((conv_image_pair_total + 1))
        fi
    done
done

for value in "${LINEAR_SIZE_LIST[@]}" "${CONV_INPUT_CHANNEL_LIST[@]}" \
    "${CONV_OUTPUT_CHANNEL_LIST[@]}" "${CONV_IMAGE_SIZE_LIST[@]}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] ||
        die "matrix values must be positive integers: $value"
done
for pair in "${CONV_KERNEL_PADDING_LIST[@]}"; do
    [[ "$pair" =~ ^[1-9][0-9]*:[0-9]+$ ]] ||
        die "kernel/padding entries must use K:P format: $pair"
done

OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${BOARD_LABEL}"
LOG_DIRECTORY="${OUTPUT_DIRECTORY}/logs"
SUMMARY_PATH="${OUTPUT_DIRECTORY}/campaign_summary.tsv"

COMMON_ARGS=(
    "$PYTHON_BIN"
    "${SCRIPT_DIR}/automated_measurement.py"
    --connection-config "$CONNECTION_CONFIG"
    --backend "$BACKEND"
    --output-directory "$OUTPUT_DIRECTORY"
    --shunt-ohms "$SHUNT_OHMS"
    --max-expected-current-a "$MAX_EXPECTED_CURRENT_A"
    --number_of_cycles "$NUMBER_OF_CYCLES"
    --sleep_time "$SLEEP_TIME"
    --target_burst_seconds "$TARGET_BURST_SECONDS"
    --sampling_rate_hz "$SAMPLING_RATE_HZ"
    --min_active_samples "$MIN_ACTIVE_SAMPLES"
    --warmup_inferences "$WARMUP_INFERENCES"
    --warmup_seconds "$WARMUP_SECONDS"
    --calibration_initial_inferences "$CALIBRATION_INITIAL_INFERENCES"
    --calibration_target_seconds "$CALIBRATION_TARGET_SECONDS"
    --calibration_repetitions "$CALIBRATION_REPETITIONS"
    --calibration-sizing-max-attempts "$CALIBRATION_SIZING_MAX_ATTEMPTS"
    --calibration-duration-tolerance "$CALIBRATION_DURATION_TOLERANCE"
    --max_relative_mad "$MAX_RELATIVE_MAD"
    --burst-duration-margin "$BURST_DURATION_MARGIN"
    --validation-repetitions "$VALIDATION_REPETITIONS"
    --validation-max-rounds "$VALIDATION_MAX_ROUNDS"
    --validation-safety-margin "$VALIDATION_SAFETY_MARGIN"
    --clock-sync-exchanges "$CLOCK_SYNC_EXCHANGES"
    --max-clock-uncertainty-fraction "$MAX_CLOCK_UNCERTAINTY_FRACTION"
    --max_calibration_inferences "$MAX_CALIBRATION_INFERENCES"
    --leading_idle_seconds "$LEADING_IDLE_SECONDS"
    --trailing_idle_seconds "$TRAILING_IDLE_SECONDS"
    --safety_margin_seconds "$SAFETY_MARGIN_SECONDS"
)
if [[ -n "$WARMUP_COOLDOWN_SECONDS" ]]; then
    COMMON_ARGS+=(--warmup_cooldown_seconds "$WARMUP_COOLDOWN_SECONDS")
fi
if [[ -n "$VALIDATION_COOLDOWN_SECONDS" ]]; then
    COMMON_ARGS+=(--validation-cooldown-seconds "$VALIDATION_COOLDOWN_SECONDS")
fi
if [[ -n "$INA226_PORT" ]]; then
    COMMON_ARGS+=(--port "$INA226_PORT")
fi

linear_total=0
conv_total=0
lenet_total=0
resnet18_total=0
resnet50_total=0
if suite_is_selected linear; then
    linear_total=$((${#LINEAR_SIZE_LIST[@]} * ${#LINEAR_SIZE_LIST[@]}))
fi
if suite_is_selected conv; then
    conv_total=$((conv_image_pair_total * ${#CONV_OUTPUT_CHANNEL_LIST[@]} * ${#CONV_KERNEL_PADDING_LIST[@]}))
fi
if suite_is_selected lenet; then
    lenet_total=$((1 + ${#LENET_COMPONENT_MODELS[@]}))
fi
if suite_is_selected resnet18; then
    resnet18_total=$((1 + ${#RESNET18_COMPONENT_MODELS[@]}))
fi
if suite_is_selected resnet50; then
    resnet50_total=$((1 + ${#RESNET50_COMPONENT_MODELS[@]}))
fi
total=$((linear_total + conv_total + lenet_total + resnet18_total + resnet50_total))

printf 'Board label: %s\n' "$BOARD_LABEL"
printf 'Suite: %s (%d Linear, %d Conv, %d LeNet, %d ResNet-18, %d ResNet-50, %d total)\n' \
    "$SUITE" "$linear_total" "$conv_total" "$lenet_total" "$resnet18_total" "$resnet50_total" "$total"
printf 'Batch sizes: Linear/Conv=%s; LeNet/ResNet-18=%s; ResNet-50=1\n' \
    "$LINEAR_CONV_BATCH_SIZE" "$NETWORK_BATCH_SIZE"
printf 'Results: %s\n' "$OUTPUT_DIRECTORY"
((DRY_RUN == 0)) || printf 'Mode: dry-run (no measurements will start)\n'

if ((DRY_RUN == 0)); then
    mkdir -p "$LOG_DIRECTORY"
    if [[ ! -e "$SUMMARY_PATH" ]]; then
        printf 'timestamp_utc\tmodel\tstatus\texit_code\tlog\n' >"$SUMMARY_PATH"
    fi
fi

attempted=0
completed=0
skipped=0
failed=0
experiment_started=0
interrupted=0

handle_interrupt() {
    interrupted=1
}

trap handle_interrupt INT

run_experiment() {
    local model_path="$1"
    local batch_size="$2"
    local model_name="${model_path##*/}"
    local model_stem="${model_name%.*}"
    local log_path="${LOG_DIRECTORY}/${model_stem}.log"
    local timestamp
    local exit_code
    local -a pipeline_status
    local -a command=("${COMMON_ARGS[@]}" --batch-size "$batch_size" --model "$model_path")

    attempted=$((attempted + 1))
    printf '[%d/%d] %s\n' "$attempted" "$total" "$model_path"

    if ((REPEAT_COMPLETED == 0)) && manifest_exists_for_model "$model_path" "$batch_size"; then
        skipped=$((skipped + 1))
        printf '  skipped: COMPLETE manifest already exists\n'
        return 0
    fi

    if ((DRY_RUN)); then
        print_command "${command[@]}"
        return 0
    fi

    if ((experiment_started)) && is_positive_number "$EXPERIMENT_COOLDOWN_SECONDS"; then
        printf '  cooling board for %s seconds before next experiment\n' \
            "$EXPERIMENT_COOLDOWN_SECONDS"
        set +e
        sleep "$EXPERIMENT_COOLDOWN_SECONDS"
        exit_code=$?
        set -e
        if ((interrupted || exit_code == 130)); then
            printf 'Campaign interrupted during board cooldown.\n' >&2
            return 130
        fi
        if ((exit_code != 0)); then
            printf '  cooldown failed with exit code %d\n' "$exit_code" >&2
            return "$exit_code"
        fi
    fi
    experiment_started=1

    set +e
    "${command[@]}" 2>&1 | tee "$log_path"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    exit_code="${pipeline_status[0]}"
    if ((interrupted || exit_code == 130 || pipeline_status[1] == 130)); then
        exit_code=130
    elif ((exit_code == 0 && pipeline_status[1] != 0)); then
        exit_code="${pipeline_status[1]}"
    fi
    timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

    if ((exit_code == 0)); then
        completed=$((completed + 1))
        printf '%s\t%s\tCOMPLETE\t0\t%s\n' \
            "$timestamp" "$model_path" "$log_path" >>"$SUMMARY_PATH"
        return 0
    fi

    failed=$((failed + 1))
    printf '%s\t%s\tFAILED\t%d\t%s\n' \
        "$timestamp" "$model_path" "$exit_code" "$log_path" >>"$SUMMARY_PATH"
    printf '  failed with exit code %d; log: %s\n' "$exit_code" "$log_path" >&2
    if ((exit_code == 130)); then
        printf 'Campaign interrupted by user; completed experiments are preserved.\n' >&2
        return 130
    fi
    if ((CONTINUE_ON_ERROR)); then
        return 0
    fi
    return "$exit_code"
}

if suite_is_selected linear; then
    for input_size in "${LINEAR_SIZE_LIST[@]}"; do
        for output_size in "${LINEAR_SIZE_LIST[@]}"; do
            run_experiment \
                "${LINEAR_MODEL_DIRECTORY}/Linear_${input_size}_${output_size}${MODEL_SUFFIX}" \
                "$LINEAR_CONV_BATCH_SIZE"
        done
    done
fi

if suite_is_selected conv; then
    for kernel_padding in "${CONV_KERNEL_PADDING_LIST[@]}"; do
        kernel_size="${kernel_padding%%:*}"
        padding="${kernel_padding##*:}"
        for input_channels in "${CONV_INPUT_CHANNEL_LIST[@]}"; do
            max_image_size="$(conv_max_image_size "$input_channels")"
            for output_channels in "${CONV_OUTPUT_CHANNEL_LIST[@]}"; do
                for image_size in "${CONV_IMAGE_SIZE_LIST[@]}"; do
                    ((image_size <= max_image_size)) || continue
                    run_experiment \
                        "${CONV_MODEL_DIRECTORY}/Conv_${input_channels}_${image_size}_${kernel_size}_${padding}_${output_channels}${MODEL_SUFFIX}" \
                        "$LINEAR_CONV_BATCH_SIZE"
                done
            done
        done
    done
fi

if suite_is_selected lenet; then
    run_experiment "$LENET_MODEL_PATH" "$NETWORK_BATCH_SIZE"
    for model_path in "${LENET_COMPONENT_MODELS[@]}"; do
        run_experiment "$model_path" "$NETWORK_BATCH_SIZE"
    done
fi

if suite_is_selected resnet18; then
    run_experiment "$RESNET18_MODEL_PATH" "$NETWORK_BATCH_SIZE"
    for model_path in "${RESNET18_COMPONENT_MODELS[@]}"; do
        run_experiment "$model_path" "$NETWORK_BATCH_SIZE"
    done
fi

if suite_is_selected resnet50; then
    run_experiment "$RESNET50_MODEL_PATH" 1
    for model_path in "${RESNET50_COMPONENT_MODELS[@]}"; do
        run_experiment "$model_path" 1
    done
fi

printf '\nCampaign finished for board %s.\n' "$BOARD_LABEL"
printf 'Scheduled: %d; completed: %d; skipped: %d; failed: %d.\n' \
    "$total" "$completed" "$skipped" "$failed"
if ((DRY_RUN)); then
    printf 'Dry-run only: no experiment was executed.\n'
else
    printf 'You can now power down this setup, change the board, update the local\n'
    printf 'connection/current settings, and start a new campaign with a new label.\n'
fi

if ((failed > 0)); then
    exit 1
fi
