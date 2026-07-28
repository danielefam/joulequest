"""Process INA226 CSV measurements and visualize the resulting traces."""
# The following script was generated entirely by GPT-5.6 
# and is intended solely for presentation purposes.
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import data_processing_new as dp
except ImportError:
    import data_processing_new as dp

try:
    from .artifact_paths import get_manifest_file_path, load_measurement_metadata
except ImportError:
    from artifact_paths import get_manifest_file_path, load_measurement_metadata


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
    manifest_path = get_manifest_file_path(
        args.manifest_dir or args.data_dir,
        csv_path,
    )
    result = dp.process_measurement(
        csv_path,
        manifest_path=manifest_path,
        config=dp.ProcessingConfig(sampling_rate_hz=args.frequency),
    )
    samples = result.samples
    regions = result.regions
    frequency = result.summary["sampling_rate_hz"]

    time_seconds = samples["time_s"]
    combined_axis.plot(
        time_seconds,
        samples["power_smoothed_W"],
        label=csv_path.stem,
        linewidth=0.2,
    )

    dp.plot_measurement(result, args.output_dir / f"{csv_path.stem}.pdf")

    power_offsets = regions["power_offset_W"]
    energies = regions["energy_per_inference_J"]

    return {
        "campaign_id": metadata["campaign_id"],
        "model_path": metadata["model_path"],
        "status": metadata["status"],
        "quality_status": metadata["quality_status"],
        "file": str(csv_path.relative_to(args.data_dir)),
        "samples": len(samples),
        "sampling_rate_hz": frequency,
        "inferences_per_cycle": metadata["inferences_per_cycle"],
        "expected_cycles": metadata["expected_cycles"],
        "detected_regions": result.summary["active_region_count"],
        "threshold_W": result.summary["threshold_W"],
        "power_mean_W": power_offsets.mean(),
        "power_variance_W2": power_offsets.var(ddof=1),
        "energy_mean_J": energies.mean(),
        "energy_variance_J2": energies.var(ddof=1),
        "outlier_count": result.summary["outlier_count"],
        "active_classification_source": result.summary[
            "active_classification_source"
        ],
        "preparation_classification_source": result.summary[
            "preparation_classification_source"
        ],
    }


def main():
    args = build_parser().parse_args()
    manifest_dir = args.manifest_dir or args.data_dir
    args.manifest_dir = manifest_dir
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