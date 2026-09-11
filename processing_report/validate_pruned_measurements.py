#!/usr/bin/env python3
"""Validate and compare JouleNAS pruning predictions against JouleQuest physical measurements."""

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_DATA_PATHS = {
    "agx_orin": {
        "pruning_log": "jouleNAS/experiments/pruning_logs/resnet18_icpr_weight_0p005_lookup_agx_orin_seed_0_energy_mode_discrete_978062_pruning.json",
        "metrics_csv": "jouleNAS/experiments/csv_files/cifar10/agxorin/resnet18_icpr_weight_0p005_lookup_agxorin_energy_lookup_out_of_range_extrapolate_seed_0_energy_mode_discrete_978062_metrics.csv",
        "summary_csv": "joulequest/measurements/lookup_summaries/summaries/agx_orin.csv",
        "dense_model_names": ["ResNet18_32_10.pt"],
        "pruned_model_names": ["PrunedOrinResNet18_32_10.pt", "PrunedResNet18_32_10.pt"],
    },
    "pi5": {
        "pruning_log": "jouleNAS/experiments/pruning_logs/resnet18_icpr_weight_0p005_lookup_pi5_energy_lookup_out_of_range_extrapolate_seed_0_energy_mode_discrete_987718_pruning.json",
        "metrics_csv": "jouleNAS/experiments/csv_files/cifar10/pi5/resnet18_icpr_weight_0p005_lookup_pi5_energy_lookup_out_of_range_extrapolate_seed_0_energy_mode_discrete_987718_metrics.csv",
        "summary_csv": "joulequest/measurements/lookup_summaries/summaries/pi5.csv",
        "dense_model_names": ["ResNet18_32_10.pt"],
        "pruned_model_names": ["PrunedPi5ResNet18_32_10.pt"],
    },
    "pi5_4threads": {
        "pruning_log": "jouleNAS/experiments/pruning_logs/resnet18_icpr_weight_0p005_lookup_pi5_energy_lookup_out_of_range_extrapolate_seed_0_energy_mode_discrete_987718_pruning.json",
        "metrics_csv": "jouleNAS/experiments/csv_files/cifar10/pi5/resnet18_icpr_weight_0p005_lookup_pi5_energy_lookup_out_of_range_extrapolate_seed_0_energy_mode_discrete_987718_metrics.csv",
        "summary_csv": "joulequest/measurements/lookup_summaries/summaries/pi5_4threads.csv",
        "dense_model_names": ["ResNet18_32_10.pt"],
        "pruned_model_names": ["PrunedPi5ResNet18_32_10.pt"],
    },
}


def load_joulenas_metrics(metrics_csv_path):
    """Extract baseline and final epoch metrics from JouleNAS metrics CSV."""
    path = Path(metrics_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        raise ValueError(f"Empty metrics CSV: {path}")

    # Baseline (epoch 0)
    baseline_row = None
    for row in reader:
        if row.get("epoch") == "0" and row.get("phase") in {"initial_masks", "mask_training"}:
            baseline_row = row
            break
    if baseline_row is None:
        baseline_row = reader[0]

    # Final epoch row
    final_row = reader[-1]

    pred_dense_energy = float(baseline_row["energy_mJ"])
    # Use post_epoch_energy_mJ if available and non-empty, otherwise energy_mJ
    final_energy_str = final_row.get("post_epoch_energy_mJ") or final_row["energy_mJ"]
    pred_pruned_energy = float(final_energy_str)

    dense_acc = float(baseline_row.get("test_accuracy") or reader[0].get("test_accuracy") or 0.0)
    pruned_acc = float(final_row.get("test_accuracy") or 0.0)
    active_fraction = float(final_row.get("active_fraction") or 1.0)
    hard_sparsity = float(final_row.get("hard_sparsity") or 0.0)

    pred_energy_fraction = pred_pruned_energy / pred_dense_energy if pred_dense_energy > 0 else 0.0
    pred_savings_percent = (1.0 - pred_energy_fraction) * 100.0

    return {
        "pred_dense_energy_mJ": pred_dense_energy,
        "pred_pruned_energy_mJ": pred_pruned_energy,
        "pred_energy_fraction": pred_energy_fraction,
        "pred_savings_percent": pred_savings_percent,
        "dense_accuracy": dense_acc,
        "pruned_accuracy": pruned_acc,
        "accuracy_delta": pruned_acc - dense_acc,
        "active_fraction": active_fraction,
        "hard_sparsity": hard_sparsity,
    }


def load_physical_measurements(summary_csv_path, dense_names, pruned_names, batch_size=1):
    """Extract physical measurements from JouleQuest summary CSV."""
    path = Path(summary_csv_path)
    if not path.is_file():
        return None

    with open(path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    dense_row = None
    pruned_row = None

    for row in reversed(reader):
        if row.get("status") != "COMPLETE":
            continue
        model_path = row.get("model_path", "")
        campaign_id = row.get("campaign_id", "")
        if f"_bs{batch_size}_" not in campaign_id:
            continue

        model_name = Path(model_path).name
        if dense_row is None and any(model_name == d for d in dense_names):
            dense_row = row
        if pruned_row is None and any(model_name == p for p in pruned_names):
            pruned_row = row

        if dense_row and pruned_row:
            break

    if not dense_row or not pruned_row:
        return {
            "dense_found": dense_row is not None,
            "pruned_found": pruned_row is not None,
            "dense_measurement": dense_row,
            "pruned_measurement": pruned_row,
        }

    meas_dense_energy = float(dense_row["energy_mean_mJ"])
    meas_pruned_energy = float(pruned_row["energy_mean_mJ"])
    meas_dense_power = float(dense_row.get("power_mean_W", 0.0))
    meas_pruned_power = float(pruned_row.get("power_mean_W", 0.0))

    meas_fraction = meas_pruned_energy / meas_dense_energy if meas_dense_energy > 0 else 0.0
    meas_savings = (1.0 - meas_fraction) * 100.0

    return {
        "dense_found": True,
        "pruned_found": True,
        "meas_dense_energy_mJ": meas_dense_energy,
        "meas_pruned_energy_mJ": meas_pruned_energy,
        "meas_dense_power_W": meas_dense_power,
        "meas_pruned_power_W": meas_pruned_power,
        "meas_energy_fraction": meas_fraction,
        "meas_savings_percent": meas_savings,
        "dense_campaign_id": dense_row["campaign_id"],
        "pruned_campaign_id": pruned_row["campaign_id"],
    }


def compare_predictions_and_measurements(predictions, measurements):
    """Compute comparison metrics between predictions and physical measurements."""
    if not measurements or not measurements.get("dense_found") or not measurements.get("pruned_found"):
        return {"has_measurements": False}

    pred_ratio = predictions["pred_energy_fraction"]
    meas_ratio = measurements["meas_energy_fraction"]

    ratio_discrepancy = abs(meas_ratio - pred_ratio)
    savings_delta_pp = measurements["meas_savings_percent"] - predictions["pred_savings_percent"]

    dense_gap = measurements["meas_dense_energy_mJ"] - predictions["pred_dense_energy_mJ"]
    dense_gap_pct = (dense_gap / predictions["pred_dense_energy_mJ"]) * 100.0

    pruned_gap = measurements["meas_pruned_energy_mJ"] - predictions["pred_pruned_energy_mJ"]
    pruned_gap_pct = (pruned_gap / predictions["pred_pruned_energy_mJ"]) * 100.0

    return {
        "has_measurements": True,
        "predicted_ratio": pred_ratio,
        "measured_ratio": meas_ratio,
        "ratio_discrepancy": ratio_discrepancy,
        "savings_delta_percentage_points": savings_delta_pp,
        "dense_gap_mJ": dense_gap,
        "dense_gap_pct": dense_gap_pct,
        "pruned_gap_mJ": pruned_gap,
        "pruned_gap_pct": pruned_gap_pct,
    }


def format_report(board, predictions, measurements, comparison):
    """Render a clean summary report."""
    lines = []
    lines.append("=" * 78)
    lines.append(f" JOULEECOSYSTEM VALIDATION REPORT: {board.upper()}")
    lines.append("=" * 78)

    lines.append("\n[1] JouleNAS Pruning Prediction (Lookup Table Simulation):")
    lines.append(f"  - Dense Baseline Energy : {predictions['pred_dense_energy_mJ']:>8.3f} mJ")
    lines.append(f"  - Pruned Model Energy   : {predictions['pred_pruned_energy_mJ']:>8.3f} mJ")
    lines.append(f"  - Predicted Ratio       : {predictions['pred_energy_fraction']:>8.2%}")
    lines.append(f"  - Predicted Savings     : {predictions['pred_savings_percent']:>8.2f}%")
    lines.append(f"  - Active Channels       : {predictions['active_fraction']:>8.2%}")
    lines.append(f"  - Test Accuracy (Dense) : {predictions['dense_accuracy']:>8.2%}")
    lines.append(f"  - Test Accuracy (Pruned): {predictions['pruned_accuracy']:>8.2%}")
    lines.append(f"  - Accuracy Difference   : {predictions['accuracy_delta']:>+8.2%}")

    if not comparison.get("has_measurements"):
        lines.append("\n[2] JouleQuest Physical Hardware Measurements:")
        lines.append("  * Status: Physical measurements have not been executed yet on this board.")
        lines.append("  * To collect physical measurements, run on the host:")
        if board == "pi5":
            lines.append("      BACKEND=cpu MODEL_ROOT=Models/CPU ./run_measurement_campaign.sh --board pi5 --suite pruned_validation_pi5")
        else:
            lines.append(f"      ./run_measurement_campaign.sh --board {board} --suite pruned_validation_orin")
    else:
        lines.append("\n[2] JouleQuest Physical Hardware Measurements (INA226):")
        lines.append(f"  - Dense Baseline Energy : {measurements['meas_dense_energy_mJ']:>8.3f} mJ (Power: {measurements['meas_dense_power_W']:.2f} W)")
        lines.append(f"  - Pruned Model Energy   : {measurements['meas_pruned_energy_mJ']:>8.3f} mJ (Power: {measurements['meas_pruned_power_W']:.2f} W)")
        lines.append(f"  - Measured Ratio        : {measurements['meas_energy_fraction']:>8.2%}")
        lines.append(f"  - Measured Savings      : {measurements['meas_savings_percent']:>8.2f}%")

        lines.append("\n[3] Physical Validation & Composition Discrepancy:")
        lines.append(f"  - Energy Ratio Diff     : {comparison['ratio_discrepancy']:>8.2%}")
        lines.append(f"  - Savings Delta (pp)    : {comparison['savings_delta_percentage_points']:>+8.2f} percentage points")
        lines.append(f"  - Dense Gap (Meas - Pred): {comparison['dense_gap_mJ']:>+8.3f} mJ ({comparison['dense_gap_pct']:>+6.1f}%)")
        lines.append(f"  - Pruned Gap (Meas-Pred): {comparison['pruned_gap_mJ']:>+8.3f} mJ ({comparison['pruned_gap_pct']:>+6.1f}%)")

    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and compare JouleNAS pruning predictions against JouleQuest physical measurements."
    )
    parser.add_argument(
        "--board",
        choices=["agx_orin", "pi5", "pi5_4threads"],
        default="pi5",
        help="Target hardware board (default: pi5)",
    )
    parser.add_argument(
        "--pruning-log",
        help="Path to JouleNAS pruning JSON log",
    )
    parser.add_argument(
        "--metrics-csv",
        help="Path to JouleNAS metrics CSV",
    )
    parser.add_argument(
        "--summary-csv",
        help="Path to JouleQuest summary CSV",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Only display predictions from JouleNAS without reading summary CSV",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to extract from summary CSV (default: 1)",
    )

    args = parser.parse_args()

    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2] if this_file.parent.name == "processing_report" else this_file.parents[1]
    board_defaults = DEFAULT_DATA_PATHS[args.board]

    metrics_csv = Path(args.metrics_csv or (repo_root / board_defaults["metrics_csv"]))
    summary_csv = Path(args.summary_csv or (repo_root / board_defaults["summary_csv"]))

    try:
        predictions = load_joulenas_metrics(metrics_csv)
    except Exception as e:
        print(f"Error loading JouleNAS metrics: {e}", file=sys.stderr)
        return 1

    measurements = None
    if not args.predictions_only:
        measurements = load_physical_measurements(
            summary_csv,
            board_defaults["dense_model_names"],
            board_defaults["pruned_model_names"],
            batch_size=args.batch_size,
        )

    comparison = compare_predictions_and_measurements(predictions, measurements)

    if args.json:
        output = {
            "board": args.board,
            "predictions": predictions,
            "measurements": measurements,
            "comparison": comparison,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(args.board, predictions, measurements, comparison))

    return 0


if __name__ == "__main__":
    sys.exit(main())
