"""Process INA226 CSV measurements and visualize the resulting traces."""
# The following script was generated entirely by GPT-5.6 
# and is intended solely for presentation purposes.
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import data_processing as dp

try:
    from .artifact_paths import load_measurement_metadata
except ImportError:
    from artifact_paths import load_measurement_metadata


POWER_COLUMN = "EVM1 POWER Results (W)"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process all INA226 CSV files in a folder and create plots.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "measurements" / "runs" / "jetson_nano_base",
        help="Directory containing measurement CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "measurements" / "Plot" / "jetson_nano_base",
        help="Directory receiving the summary CSV.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Manifest directory; defaults to --data-dir for colocated artifacts.",
    )
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument(
        "--frequency",
        type=float,
        default=None,
        help="Sampling-rate override; defaults to each manifest's achieved rate.",
    )
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=41)
    return parser


def process_file(csv_path, metadata, args, combined_axis):
    df = dp.load_data(csv_path)
    if POWER_COLUMN not in df.columns:
        raise ValueError(f"Missing column '{POWER_COLUMN}'")

    inferences_per_cycle = metadata["inferences_per_cycle"]
    frequency = args.frequency or metadata["sampling_rate_hz"]

    df["median_filtered"] = dp.median_filter_data(
        df[POWER_COLUMN], args.kernel_size
    )
    df["lowpass_filtered"] = dp.lowpass_filter(
        df["median_filtered"], args.cutoff, frequency
    )
    df["smoothed"] = dp.average_data(
        df["lowpass_filtered"], args.window_size
    )

    valid_smoothed = df["smoothed"].dropna()
    threshold = dp.get_threshold(valid_smoothed.to_numpy())
    results = dp.compute_means_variances(
        df,
        threshold,
        sampling_interval=1 / frequency,
        inferences_per_cycle=inferences_per_cycle,
    )

    time_seconds = df["Sample"] / frequency
    combined_axis.plot(
        time_seconds,
        df["smoothed"],
        label=csv_path.stem,
        linewidth=0.5,
    )

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(time_seconds, df[POWER_COLUMN], alpha=0.3, linewidth=0.2, label="Raw power")
    axis.plot(time_seconds, df["smoothed"], linewidth=0.3, label="Smoothed power")
    axis.axhline(threshold, color="tab:red", linestyle="--", linewidth=0.5, label="Otsu threshold")
    axis.set_title(csv_path.stem)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Power (W)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{csv_path.stem}.pdf", format="pdf")
    plt.show()
    plt.close(figure)

    return {
        "campaign_id": metadata["campaign_id"],
        "model_path": metadata["model_path"],
        "status": metadata["status"],
        "quality_status": metadata["quality_status"],
        "file": str(csv_path.relative_to(args.data_dir)),
        "samples": len(df),
        "sampling_rate_hz": frequency,
        "inferences_per_cycle": inferences_per_cycle,
        "expected_cycles": metadata["expected_cycles"],
        "detected_regions": results["region_count"],
        "threshold_W": threshold,
        "power_mean_W": results["power_avg_W"],
        "power_variance_W2": results["power_var_W2"],
        "energy_mean_J": results["energy_avg_J"],
        "energy_variance_J2": results["energy_var_J2"],
    }


def main():
    args = build_parser().parse_args()
    manifest_dir = args.manifest_dir or args.data_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(args.data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

    summary_rows = []
    skipped_files = 0
    combined_figure, combined_axis = plt.subplots(figsize=(15, 7))
    for csv_path in csv_files:
        metadata = load_measurement_metadata(manifest_dir, csv_path)
        if metadata is None:
            skipped_files += 1
            print(
                f"Skipping {csv_path}: no matching COMPLETE manifest",
                file=sys.stderr,
            )
            continue
        try:
            summary_rows.append(
                process_file(csv_path, metadata, args, combined_axis)
            )
        except (ValueError, IndexError) as error:
            raise ValueError(f"Could not process {csv_path}: {error}") from error

    if not summary_rows:
        raise RuntimeError("No CSV file has a matching COMPLETE manifest")

    combined_axis.set_title("Smoothed power comparison")
    combined_axis.set_xlabel("Time (s)")
    combined_axis.set_ylabel("Power (W)")
    combined_axis.grid(True, alpha=0.3)
    combined_axis.legend()
    combined_figure.tight_layout()
    combined_figure.savefig(
        args.output_dir / "smoothed_power_comparison.pdf", format="pdf"
    )
    plt.show()
    plt.close(combined_figure)

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "summary.csv", index=False
    )
    print(f"Processed {len(summary_rows)} files")
    print(f"Skipped {skipped_files} incomplete or unpaired files")
    print(f"Summary saved to: {args.output_dir}")


if __name__ == "__main__":
    main()