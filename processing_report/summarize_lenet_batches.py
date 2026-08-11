"""Print LeNet and standalone-operation energy summaries by platform and batch."""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


OPERATIONS = (
    ("Conv1", re.compile(r"^conv_1_32_5_0_8$", re.IGNORECASE)),
    ("MaxPool1", re.compile(r"^maxpool_8_28_2$", re.IGNORECASE)),
    ("ReLU1", re.compile(r"^relu_1_8_14_14$", re.IGNORECASE)),
    ("Conv2", re.compile(r"^conv_8_14_5_0_32$", re.IGNORECASE)),
    ("MaxPool2", re.compile(r"^maxpool_32_10_2$", re.IGNORECASE)),
    ("AdaptiveMaxPool", re.compile(r"^adapool_32_5_4$", re.IGNORECASE)),
    ("ReLU2", re.compile(r"^relu_1_32_4_4$", re.IGNORECASE)),
    ("Flatten", re.compile(r"^flatten_1_32_4_4$", re.IGNORECASE)),
    ("Linear1", re.compile(r"^linear_512_128$", re.IGNORECASE)),
    ("ReLU3", re.compile(r"^relu_1_128$", re.IGNORECASE)),
    ("Linear2", re.compile(r"^linear_128_64$", re.IGNORECASE)),
)
FULL_MODEL = re.compile(r"^lenet$", re.IGNORECASE)
BATCH_MARKER = re.compile(r"(?:^|_)bs(\d+)(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Measurement:
    operation: str
    energy_mj: float
    quality: str
    campaign_id: str


def _batch(row):
    match = BATCH_MARKER.search(row.get("campaign_id", ""))
    return int(match.group(1)) if match else 1


def _operation(row):
    model_name = Path(row.get("model_path", "")).stem
    if FULL_MODEL.fullmatch(model_name):
        return "LeNet"
    for operation, pattern in OPERATIONS:
        if pattern.fullmatch(model_name):
            return operation
    return None


def load_measurements(path, batches):
    selected = {}
    with path.open(newline="", encoding="utf-8") as summary_file:
        for row in csv.DictReader(summary_file):
            operation = _operation(row)
            batch = _batch(row)
            if operation is None or batch not in batches:
                continue
            try:
                energy = float(row["energy_mean_mJ"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (batch, operation)
            candidate = Measurement(
                operation,
                energy,
                row.get("quality_status", "UNKNOWN").upper(),
                row.get("campaign_id", ""),
            )
            # Campaign IDs start with UTC timestamps, so this keeps the latest run.
            if key not in selected or candidate.campaign_id > selected[key].campaign_id:
                selected[key] = candidate
    return selected


def _format_energy(value):
    return "missing" if value is None else f"{value:.6f}"


def print_platform(label, path, batches):
    print(f"\n=== {label}: {path} ===")
    measurements = load_measurements(path, batches)
    for batch in batches:
        values = {
            operation: measurements.get((batch, operation))
            for operation, _ in OPERATIONS
        }
        complete = measurements.get((batch, "LeNet"))
        print(f"\n{label} batch {batch} (energy_mean_mJ: per input sample)")
        print("operation          energy_mJ   quality")
        print("------------------  ----------  -------")
        for operation, _ in OPERATIONS:
            measurement = values[operation]
            if measurement is None:
                print(f"{operation:<18}  {'missing':>10}  {'-':>7}")
            else:
                print(
                    f"{operation:<18}  {measurement.energy_mj:10.6f}  "
                    f"{measurement.quality:>7}"
                )
        print(
            f"{'LeNet complete':<18}  "
            f"{_format_energy(complete.energy_mj if complete else None):>10}  "
            f"{complete.quality if complete else '-':>7}"
        )

        available = [measurement for measurement in values.values() if measurement]
        ok_values = [measurement for measurement in available if measurement.quality == "OK"]
        all_sum = sum(measurement.energy_mj for measurement in available)
        ok_sum = sum(measurement.energy_mj for measurement in ok_values)
        missing = [operation for operation, measurement in values.items() if measurement is None]
        review = [measurement.operation for measurement in available if measurement.quality != "OK"]
        print(f"Standalone sum (available): {all_sum:.6f} mJ/sample ({len(available)}/{len(OPERATIONS)} operations)")
        print(f"Standalone sum (OK only):   {ok_sum:.6f} mJ/sample ({len(ok_values)}/{len(OPERATIONS)} operations)")
        if missing:
            print("Missing: " + ", ".join(missing))
        if review:
            print("REVIEW rows: " + ", ".join(review))

        if complete and not missing:
            gap = complete.energy_mj - all_sum
            ratio = complete.energy_mj / all_sum if all_sum else float("nan")
            coverage = 100 * all_sum / complete.energy_mj if complete.energy_mj else float("nan")
            qualifier = "" if not review and complete.quality == "OK" else " (provisional)"
            print(f"Gap (LeNet - standalone){qualifier}: {gap:+.6f} mJ/sample")
            print(f"Ratio LeNet/standalone{qualifier}:    {ratio:.4f}")
            print(f"Standalone coverage{qualifier}:      {coverage:.2f}%")
        else:
            print("Comparison: unavailable as a definitive total (missing or REVIEW data).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano-summary", type=Path, default=Path("measurements/Plot/nano_base/summary.csv"))
    parser.add_argument("--pi5-summary", type=Path, default=Path("measurements/Plot/pi5/summary.csv"))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 128])
    args = parser.parse_args()

    print_platform("Nano", args.nano_summary, args.batches)
    print_platform("Pi5", args.pi5_summary, [1])


if __name__ == "__main__":
    main()