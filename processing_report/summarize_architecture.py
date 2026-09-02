"""Print LeNet and ResNet18 standalone-operation energy summaries."""

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

# Each entry is a distinct batch-8 benchmark and its call count in ResNet18.
RESNET18_OPERATIONS = (
    ("Stem Conv 3->64", r"resnetconv_3_64_224_7_2_3", 1),
    ("Stem BatchNorm 64x112", r"resnetbatchnorm_64_112", 1),
    ("Stem ReLU 64x112", r"relu_1_64_112_112", 1),
    ("Stem MaxPool", r"resnetmaxpool_64_112_3_2_1", 1),
    ("Stage1 Conv 64->64", r"resnetconv_64_64_56_3_1_1", 4),
    ("Stage1 BatchNorm", r"resnetbatchnorm_64_56", 4),
    ("Stage1 ReLU", r"relu_1_64_56_56", 4),
    ("Stage1 ResidualAdd", r"resnetresidualadd_64_56", 2),
    ("Stage2 Conv 64->128", r"resnetconv_64_128_56_3_2_1", 1),
    ("Stage2 Shortcut Conv", r"resnetconv_64_128_56_1_2_0", 1),
    ("Stage2 BatchNorm", r"resnetbatchnorm_128_28", 5),
    ("Stage2 Conv 128->128", r"resnetconv_128_128_28_3_1_1", 3),
    ("Stage2 ReLU", r"relu_1_128_28_28", 4),
    ("Stage2 ResidualAdd", r"resnetresidualadd_128_28", 2),
    ("Stage3 Conv 128->256", r"resnetconv_128_256_28_3_2_1", 1),
    ("Stage3 Shortcut Conv", r"resnetconv_128_256_28_1_2_0", 1),
    ("Stage3 BatchNorm", r"resnetbatchnorm_256_14", 5),
    ("Stage3 Conv 256->256", r"resnetconv_256_256_14_3_1_1", 3),
    ("Stage3 ReLU", r"relu_1_256_14_14", 4),
    ("Stage3 ResidualAdd", r"resnetresidualadd_256_14", 2),
    ("Stage4 Conv 256->512", r"resnetconv_256_512_14_3_2_1", 1),
    ("Stage4 Shortcut Conv", r"resnetconv_256_512_14_1_2_0", 1),
    ("Stage4 BatchNorm", r"resnetbatchnorm_512_7", 5),
    ("Stage4 Conv 512->512", r"resnetconv_512_512_7_3_1_1", 3),
    ("Stage4 ReLU", r"relu_1_512_7_7", 4),
    ("Stage4 ResidualAdd", r"resnetresidualadd_512_7", 2),
    ("AdaptiveAvgPool", r"resnetavgpool_512_7_1", 1),
    ("Flatten", r"flatten_1_512_1_1", 1),
    ("Linear 512->classes", r"linear_512_(?:10|1000)", 1),
)
RESNET18_FULL_MODEL = re.compile(r"^resnet18_224_(?:10|1000)$", re.IGNORECASE)
RESNET18_NON_COMPARABLE = frozenset(
    {"Stem ReLU 64x112", "Stage1 ReLU", "Stage2 ReLU", "Stage3 ReLU", "Stage4 ReLU", "Flatten"}
)


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


def load_resnet18_measurements(path, batch=8):
    selected = {}
    complete = None
    with path.open(newline="", encoding="utf-8") as summary_file:
        for row in csv.DictReader(summary_file):
            if _batch(row) != batch:
                continue
            model_name = Path(row.get("model_path", "")).stem
            if RESNET18_FULL_MODEL.fullmatch(model_name):
                operation = "ResNet18 complete"
            else:
                operation = next(
                    (
                        label
                        for label, pattern, _ in RESNET18_OPERATIONS
                        if re.fullmatch(pattern, model_name, re.IGNORECASE)
                    ),
                    None,
                )
                if operation is None:
                    continue
            try:
                energy = float(row["energy_mean_mJ"])
            except (KeyError, TypeError, ValueError):
                continue
            candidate = Measurement(
                operation,
                energy,
                row.get("quality_status", "UNKNOWN").upper(),
                row.get("campaign_id", ""),
            )
            if operation == "ResNet18 complete":
                if complete is None or candidate.campaign_id > complete.campaign_id:
                    complete = candidate
            elif operation not in selected or candidate.campaign_id > selected[operation].campaign_id:
                selected[operation] = candidate
    return selected, complete


def print_resnet18(path, batch=8, label="Nano"):
    measurements, complete = load_resnet18_measurements(path, batch)
    print(f"\n=== {label} ResNet18 224x224 batch {batch}: {path} ===")
    print("module                    count  each_mJ   weighted_mJ  quality")
    print("------------------------  -----  --------  -----------  -------")
    weighted_sum = 0.0
    ok_weighted_sum = 0.0
    comparable_sum = 0.0
    comparable_ok_sum = 0.0
    missing = []
    review = []
    for label, _, count in RESNET18_OPERATIONS:
        measurement = measurements.get(label)
        if measurement is None:
            missing.append(label)
            print(f"{label:<24}  {count:5d}  {'missing':>8}  {'missing':>11}  {'-':>7}")
            continue
        weighted_energy = count * measurement.energy_mj
        weighted_sum += weighted_energy
        if measurement.quality == "OK":
            ok_weighted_sum += weighted_energy
        else:
            review.append(label)
        if label not in RESNET18_NON_COMPARABLE:
            comparable_sum += weighted_energy
            if measurement.quality == "OK":
                comparable_ok_sum += weighted_energy
        print(
            f"{label:<24}  {count:5d}  {measurement.energy_mj:8.6f}  "
            f"{weighted_energy:11.6f}  {measurement.quality:>7}"
        )

    total_calls = sum(count for _, _, count in RESNET18_OPERATIONS)
    comparable_calls = sum(
        count
        for label, _, count in RESNET18_OPERATIONS
        if label not in RESNET18_NON_COMPARABLE
    )
    print(f"Weighted standalone sum (all):       {weighted_sum:.6f} mJ/sample ({total_calls} module calls)")
    print(f"Weighted OK-only sum (all):           {ok_weighted_sum:.6f} mJ/sample")
    print(
        f"Weighted standalone sum (comparable): {comparable_sum:.6f} mJ/sample "
        f"({comparable_calls} module calls)"
    )
    print(f"Weighted OK-only sum (comparable):    {comparable_ok_sum:.6f} mJ/sample")
    print("Excluded from comparable sum: ReLU and Flatten")
    print(
        "ResNet18 complete:       "
        f"{_format_energy(complete.energy_mj if complete else None)} mJ/sample "
        f"({complete.quality if complete else '-'})"
    )
    if missing:
        print("Missing: " + ", ".join(missing))
    if review:
        print("REVIEW rows: " + ", ".join(review))

    if complete and not missing:
        gap = complete.energy_mj - comparable_sum
        ratio = complete.energy_mj / comparable_sum if comparable_sum else float("nan")
        coverage = 100 * comparable_sum / complete.energy_mj if complete.energy_mj else float("nan")
        qualifier = "" if not review and complete.quality == "OK" else " (provisional)"
        print(f"Gap (ResNet18 - comparable sum){qualifier}: {gap:+.6f} mJ/sample")
        print(f"Ratio ResNet18/comparable sum{qualifier}: {ratio:.4f}")
        print(f"Comparable standalone coverage{qualifier}: {coverage:.2f}%")
    else:
        print("Comparison unavailable because complete or component data is missing.")


def _default_summary_path(name, legacy_subpath):
    summaries_path = Path("measurements/summaries") / f"{name}.csv"
    if summaries_path.is_file():
        return summaries_path
    legacy_path = Path(f"measurements/Plot/{legacy_subpath}/summary.csv")
    if legacy_path.is_file():
        return legacy_path
    return summaries_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nano-summary",
        type=Path,
        default=_default_summary_path("nano_base", "nano_base"),
    )
    parser.add_argument(
        "--pi5-summary",
        type=Path,
        default=_default_summary_path("pi5", "pi5"),
    )
    parser.add_argument(
        "--agx-orin-summary",
        type=Path,
        default=_default_summary_path("agx_orin", "agx_orin"),
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 128])
    args = parser.parse_args()

    print_platform("Nano", args.nano_summary, args.batches)
    print_platform("Pi5", args.pi5_summary, [1])
    print_resnet18(args.nano_summary, label="Nano")
    print_resnet18(args.agx_orin_summary, batch=1, label="AGX Orin")


if __name__ == "__main__":
    main()