"""Process INA226 CSV measurements and visualize the resulting traces."""
# The following script was generated entirely by GPT-5.6 
# and is intended solely for presentation purposes.
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import data_processing as dp


POWER_COLUMN = "EVM1 POWER Results (W)"


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process all INA226 CSV files in a folder and create plots.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data/v3"),
        help="Directory containing measurement CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Plot/v3"),
        help="Directory receiving plots and the summary CSV.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("measurements_jetson"),
        help="Directory containing the JSON manifest for each CSV measurement.",
    )
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--frequency", type=float, default=100.0)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the combined plot after processing.",
    )
    return parser


def load_inferences_per_cycle(manifest_dir, csv_path):
    manifest_path = dp.get_manifest_file_path(manifest_dir, csv_path)
    if manifest_path is None:
        raise FileNotFoundError(
            f"No manifest found for {csv_path.name} in {manifest_dir}"
        )

    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    try:
        return manifest["plan"]["inferences_per_cycle"]
    except KeyError as error:
        raise KeyError(
            f"Manifest {manifest_path} does not contain plan.inferences_per_cycle"
        ) from error


def process_file(csv_path, args, combined_axis):
    df = dp.load_data(csv_path)
    if POWER_COLUMN not in df.columns:
        raise ValueError(f"Missing column '{POWER_COLUMN}'")

    inferences_per_cycle = load_inferences_per_cycle(
        args.manifest_dir, csv_path
    )

    df["median_filtered"] = dp.median_filter_data(
        df[POWER_COLUMN], args.kernel_size
    )
    df["lowpass_filtered"] = dp.lowpass_filter(
        df["median_filtered"], args.cutoff, args.frequency
    )
    df["smoothed"] = dp.average_data(
        df["lowpass_filtered"], args.window_size
    )

    valid_smoothed = df["smoothed"].dropna()
    threshold = dp.get_threshold(valid_smoothed.to_numpy())
    results = dp.compute_means_variances(
        df,
        threshold,
        sampling_interval=1 / args.frequency,
        inferences_per_cycle=inferences_per_cycle,
    )

    time_seconds = df["Sample"] / args.frequency
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
    plt.close(figure)

    return {
        "file": csv_path.name,
        "samples": len(df),
        "inferences_per_cycle": inferences_per_cycle,
        "threshold_W": threshold,
        "power_mean_W": results["power_avg_W"],
        "power_variance_W2": results["power_var_W2"],
        "energy_mean_J": results["energy_avg_J"],
        "energy_variance_J2": results["energy_var_J2"],
    }


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(args.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

    summary_rows = []
    combined_figure, combined_axis = plt.subplots(figsize=(15, 7))
    for csv_path in csv_files:
        try:
            summary_rows.append(process_file(csv_path, args, combined_axis))
        except (ValueError, IndexError) as error:
            raise ValueError(f"Could not process {csv_path}: {error}") from error

    combined_axis.set_title("Smoothed power comparison")
    combined_axis.set_xlabel("Time (s)")
    combined_axis.set_ylabel("Power (W)")
    combined_axis.grid(True, alpha=0.3)
    combined_axis.legend()
    combined_figure.tight_layout()
    combined_figure.savefig(
        args.output_dir / "all_measurements.pdf", format="pdf"
    )
    if args.show:
        plt.show()
    else:
        plt.close(combined_figure)

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "summary.csv", index=False
    )
    print(f"Processed {len(summary_rows)} files")
    print(f"Plots and summary saved to: {args.output_dir}")


if __name__ == "__main__":
    main()