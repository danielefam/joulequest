#!/usr/bin/env bash

set -Eeuo pipefail

# Run a focused measurement campaign on NVIDIA Jetson AGX Orin (or compatible target)
# specifically covering the layers and dimensions required for CIFAR-10 and Imagenette
# ResNet-18 pruning experiments.
#
# Default batch size is 32 to saturate the GPU and amortize CUDA kernel launch overhead.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# User-configurable campaign parameters.
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
CONNECTION_CONFIG="${CONNECTION_CONFIG:-${SCRIPT_DIR}/measurement_hosts.local.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/measurements/runs}"

BACKEND="${BACKEND:-cuda}"
MODEL_ROOT="${MODEL_ROOT:-Models/CUDA}"
MODEL_SUFFIX="${MODEL_SUFFIX:-.pt}"
LINEAR_MODEL_DIRECTORY="${LINEAR_MODEL_DIRECTORY:-${MODEL_ROOT}/Linear}"
CONV_MODEL_DIRECTORY="${CONV_MODEL_DIRECTORY:-${MODEL_ROOT}/Conv}"
RESNET18_MODEL_DIRECTORY="${RESNET18_MODEL_DIRECTORY:-${MODEL_ROOT}/ResNet18}"

# Default batch size is 1 (can be overridden via --batch-size N or BATCH_SIZE=N).
BATCH_SIZE="${BATCH_SIZE:-1}"
NUMBER_OF_CYCLES="${NUMBER_OF_CYCLES:-100}"
SLEEP_TIME="${SLEEP_TIME:-3}"
TARGET_BURST_SECONDS="${TARGET_BURST_SECONDS:-0}"
SAMPLING_RATE_HZ="${SAMPLING_RATE_HZ:-100}"
MIN_ACTIVE_SAMPLES="${MIN_ACTIVE_SAMPLES:-100}"
WARMUP_INFERENCES="${WARMUP_INFERENCES:-3}"
WARMUP_SECONDS="${WARMUP_SECONDS:-0}"
WARMUP_COOLDOWN_SECONDS="${WARMUP_COOLDOWN_SECONDS:-}"
CALIBRATION_INITIAL_INFERENCES="${CALIBRATION_INITIAL_INFERENCES:-3}"
CALIBRATION_TARGET_SECONDS="${CALIBRATION_TARGET_SECONDS:-1}"
CALIBRATION_REPETITIONS="${CALIBRATION_REPETITIONS:-8}"
CALIBRATION_SIZING_MAX_ATTEMPTS="${CALIBRATION_SIZING_MAX_ATTEMPTS:-5}"
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
EXPERIMENT_COOLDOWN_SECONDS="${EXPERIMENT_COOLDOWN_SECONDS:-10}"
CPU_THREADS="${CPU_THREADS:-}"

# ---------------------------------------------------------------------------
# Targeted Layer Matrices for CIFAR-10 & Imagenette ResNet-18 Pruning
# ---------------------------------------------------------------------------
# BasicBlock 3x3 shape-preserving convolutions (stride 1, padding 1)
CONV_3X3_INPUT_CHANNELS="${CONV_3X3_INPUT_CHANNELS:-1 2 4 8 16 32 64 128 256 512}"
CONV_3X3_OUTPUT_CHANNELS="${CONV_3X3_OUTPUT_CHANNELS:-1 8 16 32 64 128 256 512}"
CONV_3X3_IMAGE_SIZES="${CONV_3X3_IMAGE_SIZES:-2 32 64 128}"

# Pointwise 1x1 convolutions (stride 1, padding 0) for channel projection
CONV_1X1_CHANNELS="${CONV_1X1_CHANNELS:-1 8 64 512}"
CONV_1X1_IMAGE_SIZES="${CONV_1X1_IMAGE_SIZES:-2 8 32 64}"

# Linear classifier layers (up to 512 inputs, including 10 classes)
LINEAR_INPUT_SIZES="${LINEAR_INPUT_SIZES:-1 8 32 64 128 256 512}"
LINEAR_OUTPUT_SIZES="${LINEAR_OUTPUT_SIZES:-1 8 10 16 32 64 128 256 512}"

# ResNet-specific stride-2 transition layers
RESNET_STRIDE2_CONV_MODELS=(
    # CIFAR-10 transitions
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_16_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_8_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_4_3_2_1${MODEL_SUFFIX}"
    # Imagenette 128 transitions
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_32_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_16_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_8_3_2_1${MODEL_SUFFIX}"
    # Imagenette 224 transitions
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_56_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_28_3_2_1${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_14_3_2_1${MODEL_SUFFIX}"
)

RESNET_STRIDE2_DOWNSAMPLE_MODELS=(
    # CIFAR-10 downsample shortcuts (1x1, stride 2)
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_16_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_8_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_4_1_2_0${MODEL_SUFFIX}"
    # Imagenette 128 downsample shortcuts
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_32_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_16_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_8_1_2_0${MODEL_SUFFIX}"
    # Imagenette 224 downsample shortcuts
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_64_128_56_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_128_256_28_1_2_0${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_256_512_14_1_2_0${MODEL_SUFFIX}"
)

# Network stem convolutions
STEM_MODELS=(
    "${CONV_MODEL_DIRECTORY}/Conv_3_32_3_1_64${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_3_64_128_7_2_3${MODEL_SUFFIX}"
    "${RESNET18_MODEL_DIRECTORY}/ResNetConv_3_64_224_7_2_3${MODEL_SUFFIX}"
)

REPEAT_COMPLETED="${REPEAT_COMPLETED:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

usage() {
    cat <<'EOF'
Usage:
  ./run_cifar_imagenette_campaign.sh [options]

Options:
  --board LABEL                  Safe label for result directory (default: agx_orin).
  --batch-size N                 Inference batch size (default: 1).
  --backend BACKEND              Target backend: cuda, cpu, or tpu (default: cuda).
  --dry-run                      Print scheduled commands without executing.
  --continue-on-error            Continue after an experiment exits nonzero.
  --repeat-completed             Rerun experiments with an existing COMPLETE manifest.
  --experiment-cooldown-seconds S Inter-experiment cooldown in seconds (default: 10).
  -h, --help                     Show this message.

Examples:
  # Standard run with default batch size 1 on AGX Orin:
  ./run_cifar_imagenette_campaign.sh --board agx_orin

  # Run with batch size 32 to isolate CUDA kernel launch overhead:
  ./run_cifar_imagenette_campaign.sh --board agx_orin_bs32 --batch-size 32

  # Dry-run inspection:
  ./run_cifar_imagenette_campaign.sh --dry-run
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

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

declare -A COMPLETED_MANIFESTS=()
MANIFEST_CACHE_LOADED=0

load_completed_manifests() {
    COMPLETED_MANIFESTS=()
    MANIFEST_CACHE_LOADED=1
    [[ -d "$OUTPUT_DIRECTORY" ]] || return 0
    local completed_model completed_batch
    while IFS=$'\t' read -r completed_model completed_batch; do
        [[ -n "$completed_model" ]] || continue
        COMPLETED_MANIFESTS["${completed_model}|${completed_batch}"]=1
    done < <(
        "$PYTHON_BIN" - "$OUTPUT_DIRECTORY" <<'PY'
import json
import re
import sys
from pathlib import Path

output_directory = Path(sys.argv[1])
if not output_directory.is_dir():
    sys.exit(0)

for manifest_path in output_directory.glob("*.json"):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if manifest.get("status") != "COMPLETE":
        continue
    raw_path = manifest.get("model_path")
    if not raw_path:
        continue
    batch_size = int(manifest.get("input_batch_size", 1))
    print(f"{raw_path}\t{batch_size}")
    model_name = Path(raw_path).name
    legacy_conv_match = re.fullmatch(
        r"(Conv_\d+_\d+_\d+_\d+)_1(\.[^.]+)", model_name
    )
    if legacy_conv_match:
        norm_name = f"{legacy_conv_match.group(1)}{legacy_conv_match.group(2)}"
        print(f"{Path(raw_path).with_name(norm_name)}\t{batch_size}")
    legacy_conv_without = re.fullmatch(
        r"(Conv_\d+_\d+_\d+_\d+)(\.[^.]+)", model_name
    )
    if legacy_conv_without:
        alt_name = f"{legacy_conv_without.group(1)}_1{legacy_conv_without.group(2)}"
        print(f"{Path(raw_path).with_name(alt_name)}\t{batch_size}")
PY
    )
}

manifest_exists_for_model() {
    local model_path="$1"
    local batch_size="$2"
    if ((MANIFEST_CACHE_LOADED == 0)); then
        load_completed_manifests
    fi
    [[ -n "${COMPLETED_MANIFESTS["${model_path}|${batch_size}"]+exists}" ]]
}

BOARD_LABEL="${BOARD_LABEL:-agx_orin}"
DRY_RUN=0

while (($#)); do
    case "$1" in
        --board)
            (($# >= 2)) || die "--board requires a value"
            BOARD_LABEL="$2"
            shift 2
            ;;
        --batch-size)
            (($# >= 2)) || die "--batch-size requires a value"
            BATCH_SIZE="$2"
            shift 2
            ;;
        --backend)
            (($# >= 2)) || die "--backend requires a value"
            BACKEND="$2"
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
        --experiment-cooldown-seconds|--cooldown)
            (($# >= 2)) || die "--cooldown requires a value"
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
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
    die "--batch-size must be a positive integer"
[[ "$BACKEND" == "cpu" || "$BACKEND" == "cuda" || "$BACKEND" == "tpu" ]] ||
    die "BACKEND must be cpu, cuda, or tpu"

# Auto-adapt model path root if backend is CPU
if [[ "$BACKEND" == "cpu" && "$MODEL_ROOT" == "Models/CUDA" ]]; then
    MODEL_ROOT="Models/CPU"
    LINEAR_MODEL_DIRECTORY="${MODEL_ROOT}/Linear"
    CONV_MODEL_DIRECTORY="${MODEL_ROOT}/Conv"
    RESNET18_MODEL_DIRECTORY="${MODEL_ROOT}/ResNet18"
fi

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

if [[ -n "$INA226_PORT" ]]; then
    COMMON_ARGS+=(--port "$INA226_PORT")
fi
if [[ -n "$WARMUP_COOLDOWN_SECONDS" ]]; then
    COMMON_ARGS+=(--warmup_cooldown_seconds "$WARMUP_COOLDOWN_SECONDS")
fi
if [[ -n "$VALIDATION_COOLDOWN_SECONDS" ]]; then
    COMMON_ARGS+=(--validation-cooldown-seconds "$VALIDATION_COOLDOWN_SECONDS")
fi
if [[ -n "$CPU_THREADS" ]]; then
    [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || die "CPU_THREADS must be a positive integer: $CPU_THREADS"
    COMMON_ARGS+=(--cpu-threads "$CPU_THREADS")
fi

read -r -a CONV_3X3_IN_LIST <<<"$CONV_3X3_INPUT_CHANNELS"
read -r -a CONV_3X3_OUT_LIST <<<"$CONV_3X3_OUTPUT_CHANNELS"
read -r -a CONV_3X3_SZ_LIST <<<"$CONV_3X3_IMAGE_SIZES"

read -r -a CONV_1X1_CH_LIST <<<"$CONV_1X1_CHANNELS"
read -r -a CONV_1X1_SZ_LIST <<<"$CONV_1X1_IMAGE_SIZES"

read -r -a LINEAR_IN_LIST <<<"$LINEAR_INPUT_SIZES"
read -r -a LINEAR_OUT_LIST <<<"$LINEAR_OUTPUT_SIZES"

conv_3x3_total=$((${#CONV_3X3_IN_LIST[@]} * ${#CONV_3X3_OUT_LIST[@]} * ${#CONV_3X3_SZ_LIST[@]}))
conv_1x1_total=$((${#CONV_1X1_CH_LIST[@]} * ${#CONV_1X1_CH_LIST[@]} * ${#CONV_1X1_SZ_LIST[@]}))
stride2_total=$((${#RESNET_STRIDE2_CONV_MODELS[@]} + ${#RESNET_STRIDE2_DOWNSAMPLE_MODELS[@]}))
stem_total=${#STEM_MODELS[@]}
linear_total=$((${#LINEAR_IN_LIST[@]} * ${#LINEAR_OUT_LIST[@]}))

total=$((conv_3x3_total + conv_1x1_total + stride2_total + stem_total + linear_total))

printf '==============================================================================\n'
printf ' JOULEQUEST FOCUSED CAMPAIGN: CIFAR-10 & IMAGENETTE RESNET-18 (BATCH %d)\n' "$BATCH_SIZE"
printf '==============================================================================\n'
printf 'Board label : %s\n' "$BOARD_LABEL"
printf 'Backend     : %s\n' "$BACKEND"
printf 'Batch size  : %d\n' "$BATCH_SIZE"
printf 'Total models: %d (Conv3x3: %d, Pointwise1x1: %d, Stride2: %d, Stem: %d, Linear: %d)\n' \
    "$total" "$conv_3x3_total" "$conv_1x1_total" "$stride2_total" "$stem_total" "$linear_total"
printf 'Results dir : %s\n' "$OUTPUT_DIRECTORY"
((DRY_RUN == 0)) || printf 'Mode        : DRY-RUN (no physical measurements will start)\n'
printf '==============================================================================\n\n'

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
consecutive_failures=0
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

    if ((DRY_RUN)); then
        print_command "${command[@]}"
        return 0
    fi

    if ((REPEAT_COMPLETED == 0)) && manifest_exists_for_model "$model_path" "$batch_size"; then
        skipped=$((skipped + 1))
        printf '  skipped: COMPLETE manifest already exists\n'
        return 0
    fi

    if ((interrupted)); then
        printf 'Campaign interrupted by user; no next model will start.\n' >&2
        return 130
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
        consecutive_failures=0
        COMPLETED_MANIFESTS["${model_path}|${batch_size}"]=1
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
    consecutive_failures=$((consecutive_failures + 1))
    if ((CONTINUE_ON_ERROR)); then
        return 0
    fi
    if ((consecutive_failures < 2)); then
        printf '  continuing after failure; the campaign stops after two consecutive failures\n' >&2
        return 0
    fi
    printf '  stopping after %d consecutive failures\n' "$consecutive_failures" >&2
    return "$exit_code"
}

# 1. 3x3 Shape-preserving Convolutions
for image_size in "${CONV_3X3_SZ_LIST[@]}"; do
    for in_ch in "${CONV_3X3_IN_LIST[@]}"; do
        for out_ch in "${CONV_3X3_OUT_LIST[@]}"; do
            run_experiment \
                "${CONV_MODEL_DIRECTORY}/Conv_${in_ch}_${image_size}_3_1_${out_ch}${MODEL_SUFFIX}" \
                "$BATCH_SIZE"
        done
    done
done

# 2. 1x1 Pointwise Convolutions
for image_size in "${CONV_1X1_SZ_LIST[@]}"; do
    for in_ch in "${CONV_1X1_CH_LIST[@]}"; do
        for out_ch in "${CONV_1X1_CH_LIST[@]}"; do
            run_experiment \
                "${CONV_MODEL_DIRECTORY}/Conv_${in_ch}_${image_size}_1_0_${out_ch}${MODEL_SUFFIX}" \
                "$BATCH_SIZE"
        done
    done
done

# 3. ResNet Stride-2 Convolutions & Downsamples
for model_path in "${RESNET_STRIDE2_CONV_MODELS[@]}"; do
    run_experiment "$model_path" "$BATCH_SIZE"
done

for model_path in "${RESNET_STRIDE2_DOWNSAMPLE_MODELS[@]}"; do
    run_experiment "$model_path" "$BATCH_SIZE"
done

# 4. Stem Convolutions
for model_path in "${STEM_MODELS[@]}"; do
    run_experiment "$model_path" "$BATCH_SIZE"
done

# 5. Linear Classifier Layers
for in_features in "${LINEAR_IN_LIST[@]}"; do
    for out_features in "${LINEAR_OUT_LIST[@]}"; do
        run_experiment \
            "${LINEAR_MODEL_DIRECTORY}/Linear_${in_features}_${out_features}${MODEL_SUFFIX}" \
            "$BATCH_SIZE"
    done
done

printf '\nCampaign finished for board %s.\n' "$BOARD_LABEL"
printf 'Scheduled: %d; completed: %d; skipped: %d; failed: %d.\n' \
    "$total" "$completed" "$skipped" "$failed"

if ((failed > 0)); then
    exit 1
fi

