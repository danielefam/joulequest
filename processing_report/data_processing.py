"""Robustly clean INA226 traces and identify measurement phases.

Manifest burst timestamps are preferred over signal-only classification because
they describe when inference actually ran.  Preparation is inferred from an
elevated pre-burst signal; one-time model loading occurs before acquisition and
cannot be observed in the CSV.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu


POWER_COLUMN = "EVM1 POWER Results (W)"
ELAPSED_COLUMN = "Elapsed Time (s)"
TIMESTAMP_COLUMN = "Timestamp UTC"


@dataclass(frozen=True)
class ProcessingConfig:
    """Tunable durations are expressed in seconds, not sample counts."""

    sampling_rate_hz: float | None = None
    hampel_window_seconds: float = 0.21
    hampel_sigma: float = 4.5
    smoothing_window_seconds: float = 0.09
    minimum_active_seconds: float = 0.10
    minimum_idle_seconds: float = 0.05
    maximum_preparation_seconds: float = 2.0
    max_clock_uncertainty_fraction: float = 0.50

    def __post_init__(self):
        positive_values = {
            "hampel_window_seconds": self.hampel_window_seconds,
            "hampel_sigma": self.hampel_sigma,
            "smoothing_window_seconds": self.smoothing_window_seconds,
            "minimum_active_seconds": self.minimum_active_seconds,
            "minimum_idle_seconds": self.minimum_idle_seconds,
            "maximum_preparation_seconds": self.maximum_preparation_seconds,
            "max_clock_uncertainty_fraction": (
                self.max_clock_uncertainty_fraction
            ),
        }
        if self.sampling_rate_hz is not None:
            positive_values["sampling_rate_hz"] = self.sampling_rate_hz
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Configuration values must be positive: "
                + ", ".join(invalid)
            )


@dataclass
class ProcessingResult:
    samples: pd.DataFrame
    regions: pd.DataFrame
    summary: dict


def _odd_window(duration_seconds, sampling_rate_hz, minimum=3):
    samples = max(minimum, int(round(duration_seconds * sampling_rate_hz)))
    return samples if samples % 2 else samples + 1


def _load_manifest(csv_path, manifest_path=None):
    path = (
        Path(manifest_path)
        if manifest_path
        else Path(csv_path).with_suffix(".json")
    )
    if not path.is_file():
        return None, None
    with path.open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file), path


def _sampling_rate(data, manifest, override):
    if override is not None:
        return float(override), "command line"
    if manifest is not None:
        acquisition = manifest.get("acquisition", {})
        rate = acquisition.get("achieved_sampling_rate_hz")
        if rate is None:
            rate = acquisition.get("sampling_rate_hz")
        if rate is not None and float(rate) > 0:
            return float(rate), "manifest"
    if ELAPSED_COLUMN in data:
        elapsed = pd.to_numeric(data[ELAPSED_COLUMN], errors="coerce")
        intervals = elapsed.diff().dropna()
        intervals = intervals[intervals > 0]
        if not intervals.empty:
            return float(1.0 / intervals.median()), "CSV elapsed time"
    raise ValueError(
        "Sampling rate is unavailable; provide --sampling-rate-hz or a manifest"
    )


def _time_axis(data, sampling_rate_hz):
    if ELAPSED_COLUMN in data:
        elapsed = pd.to_numeric(data[ELAPSED_COLUMN], errors="coerce")
        if elapsed.notna().all() and elapsed.is_monotonic_increasing:
            return elapsed
    if "Sample" not in data:
        raise ValueError(f"CSV must contain '{ELAPSED_COLUMN}' or 'Sample'")
    samples = pd.to_numeric(data["Sample"], errors="coerce")
    if samples.isna().any():
        raise ValueError("Sample contains non-numeric values")
    return (samples - samples.iloc[0]) / sampling_rate_hz


def _parse_utc_timestamps(values):
    try:
        return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return values.map(
            lambda value: pd.to_datetime(value, utc=True, errors="coerce")
        )


def _hampel_clean(power, window, sigma):
    local_median = power.rolling(window, center=True, min_periods=1).median()
    absolute_deviation = (power - local_median).abs()
    local_mad = absolute_deviation.rolling(
        window, center=True, min_periods=1
    ).median()
    robust_scale = 1.4826 * local_mad
    global_scale = 1.4826 * float(np.median(np.abs(power - power.median())))
    fallback_scale = global_scale if global_scale > 0 else np.finfo(float).eps
    robust_scale = robust_scale.mask(robust_scale <= 0, fallback_scale)
    previous_value = power.shift(1)
    next_value = power.shift(-1)
    neighbors_agree = (
        (previous_value - next_value).abs() <= 2.5 * robust_scale
    ).fillna(False)
    is_outlier = (absolute_deviation > sigma * robust_scale) & neighbors_agree
    cleaned = power.mask(is_outlier).interpolate(limit_direction="both")
    return cleaned, is_outlier


def _run_bounds(mask):
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts, ends))


def _remove_short_runs(mask, value, minimum_samples):
    result = np.asarray(mask, dtype=bool).copy()
    target = result if value else ~result
    for start, end in _run_bounds(target):
        if end - start < minimum_samples:
            result[start:end] = not value
    return result


def _signal_active_mask(smoothed, threshold, sampling_rate_hz, config):
    lower_values = smoothed[smoothed <= threshold]
    upper_values = smoothed[smoothed > threshold]
    if lower_values.empty or upper_values.empty:
        raise ValueError(
            "Power trace does not contain separable idle and active levels"
        )

    idle_level = float(lower_values.median())
    active_level = float(upper_values.median())
    level_gap = active_level - idle_level
    lower_threshold = idle_level + 0.40 * level_gap
    upper_threshold = idle_level + 0.60 * level_gap

    active = np.zeros(len(smoothed), dtype=bool)
    state = False
    for index, value in enumerate(smoothed.to_numpy()):
        if not state and value >= upper_threshold:
            state = True
        elif state and value <= lower_threshold:
            state = False
        active[index] = state

    return active


def _aligned_active_mask(time_axis, manifest, sampling_rate_hz, config):
    threshold_seconds = (
        config.max_clock_uncertainty_fraction / sampling_rate_hz
    )
    diagnostics = {
        "status": "UNAVAILABLE",
        "uncertainty_seconds": None,
        "uncertainty_threshold_seconds": threshold_seconds,
        "fallback_reason": None,
    }
    if manifest is None:
        diagnostics["fallback_reason"] = "MANIFEST_UNAVAILABLE"
        return None, diagnostics
    if manifest.get("schema_version") != 2:
        diagnostics["fallback_reason"] = "LEGACY_MANIFEST_NO_ALIGNMENT"
        return None, diagnostics

    alignment = manifest.get("acquisition", {}).get("clock_alignment")
    if not isinstance(alignment, dict):
        diagnostics["fallback_reason"] = "CLOCK_ALIGNMENT_UNAVAILABLE"
        return None, diagnostics
    diagnostics["status"] = alignment.get("status", "UNAVAILABLE")
    diagnostics["fallback_reason"] = alignment.get("fallback_reason")
    if diagnostics["status"] != "COMPLETE":
        diagnostics["fallback_reason"] = (
            diagnostics["fallback_reason"]
            or "CLOCK_ALIGNMENT_NOT_COMPLETE"
        )
        return None, diagnostics
    try:
        uncertainty_seconds = float(alignment["uncertainty_seconds"])
    except (KeyError, TypeError, ValueError):
        diagnostics["fallback_reason"] = "CLOCK_UNCERTAINTY_UNAVAILABLE"
        return None, diagnostics
    diagnostics["uncertainty_seconds"] = uncertainty_seconds
    if not np.isfinite(uncertainty_seconds) or uncertainty_seconds < 0:
        diagnostics["fallback_reason"] = "CLOCK_UNCERTAINTY_INVALID"
        return None, diagnostics
    if uncertainty_seconds > threshold_seconds:
        diagnostics["fallback_reason"] = "CLOCK_UNCERTAINTY_EXCEEDED"
        return None, diagnostics

    cycles = manifest.get("measurement", {}).get("cycles", [])
    if not cycles:
        diagnostics["fallback_reason"] = "ALIGNED_BURST_BOUNDS_UNAVAILABLE"
        return None, diagnostics
    times = np.asarray(time_axis, dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        diagnostics["fallback_reason"] = "CSV_ELAPSED_TIME_INVALID"
        return None, diagnostics
    active = np.zeros(len(times), dtype=bool)
    for cycle in cycles:
        try:
            start = float(cycle["start_event"]["aligned_elapsed_seconds"])
            end = float(cycle["end_event"]["aligned_elapsed_seconds"])
        except (KeyError, TypeError, ValueError):
            diagnostics["fallback_reason"] = "ALIGNED_BURST_BOUNDS_UNAVAILABLE"
            return None, diagnostics
        if not (
            np.isfinite(start)
            and np.isfinite(end)
            and times[0] <= start < end <= times[-1]
        ):
            diagnostics["fallback_reason"] = "ALIGNED_BURST_BOUNDS_INVALID"
            return None, diagnostics
        active |= (times >= start) & (times <= end)
    if not active.any() or len(_run_bounds(active)) != len(cycles):
        diagnostics["fallback_reason"] = "ALIGNED_BURST_SAMPLES_INVALID"
        return None, diagnostics
    diagnostics["fallback_reason"] = None
    return active, diagnostics


def _aligned_preparation_mask(time_axis, manifest, active, config):
    cycles = manifest.get("measurement", {}).get("cycles", [])
    plan = manifest.get("plan", {})
    if not cycles or "sleep_time_seconds" not in plan:
        return None

    times = np.asarray(time_axis, dtype=float)
    maximum_duration = pd.to_timedelta(
        config.maximum_preparation_seconds, unit="s"
    )
    preparation = np.zeros(len(times), dtype=bool)
    previous_end = None
    for cycle_index, cycle in enumerate(cycles):
        try:
            burst_start = float(
                cycle["start_event"]["aligned_elapsed_seconds"]
            )
            if cycle_index == 0:
                preparation_start = float(
                    plan.get("leading_idle_seconds", 0.0)
                )
            else:
                preparation_start = previous_end + float(
                    plan["sleep_time_seconds"]
                )
            previous_end = float(
                cycle["end_event"]["aligned_elapsed_seconds"]
            )
        except (KeyError, TypeError, ValueError):
            return None

        preparation_start = max(
            preparation_start,
            burst_start - maximum_duration.total_seconds(),
        )
        if preparation_start < burst_start:
            preparation |= (times >= preparation_start) & (times < burst_start)
    preparation &= ~np.asarray(active, dtype=bool)
    return preparation if preparation.any() else None


def _infer_preparation_mask(
    smoothed, active, threshold, sampling_rate_hz, config
):
    idle_values = smoothed[~active]
    if idle_values.empty:
        return np.zeros(len(smoothed), dtype=bool)
    idle_level = float(idle_values.median())
    idle_mad = float(np.median(np.abs(idle_values - idle_level)))
    noise_scale = max(1.4826 * idle_mad, np.finfo(float).eps)
    preparation_threshold = idle_level + max(
        4.0 * noise_scale, 0.15 * max(threshold - idle_level, 0.0)
    )
    elevated = (smoothed >= preparation_threshold).to_numpy() & ~active
    maximum_samples = max(
        1, round(config.maximum_preparation_seconds * sampling_rate_hz)
    )
    preparation = np.zeros(len(smoothed), dtype=bool)

    previous_active_end = 0
    for active_start, active_end in _run_bounds(active):
        lower_bound = max(previous_active_end, active_start - maximum_samples)
        cursor = active_start - 1
        while cursor >= lower_bound and elevated[cursor]:
            cursor -= 1
        if cursor < active_start - 1:
            preparation[cursor + 1 : active_start] = True
        previous_active_end = active_end
    return preparation


def _idle_baseline(samples, manifest):
    times = samples["time_s"]
    fallback_reason = None
    safety_margin_seconds = None
    if manifest is not None:
        safety_margin_seconds = manifest.get("plan", {}).get(
            "safety_margin_seconds"
        )
    try:
        safety_margin_seconds = float(safety_margin_seconds)
    except (TypeError, ValueError):
        safety_margin_seconds = None

    if safety_margin_seconds is not None and safety_margin_seconds > 0:
        window_start = float(times.iloc[-1] - safety_margin_seconds)
        candidates = samples.loc[
            (times >= window_start) & (samples["phase"] == "idle"),
            "power_clean_W",
        ]
        if len(candidates) < 2:
            fallback_reason = "SAFETY_MARGIN_IDLE_SAMPLES_INSUFFICIENT"
    else:
        fallback_reason = "SAFETY_MARGIN_UNAVAILABLE"

    if fallback_reason is not None:
        source = "classified idle fallback"
        candidates = samples.loc[
            samples["phase"] == "idle", "power_clean_W"
        ]
    else:
        source = "final measured safety margin"

    if candidates.empty:
        return np.nan, {
            "source": "unavailable",
            "fallback_reason": fallback_reason or "IDLE_SAMPLES_UNAVAILABLE",
            "sample_count": 0,
            "start_time_s": None,
            "end_time_s": None,
            "power_median_W": None,
            "power_mad_W": None,
        }

    baseline = float(candidates.median())
    candidate_times = times.loc[candidates.index]
    mad = float(np.median(np.abs(candidates.to_numpy() - baseline)))
    return baseline, {
        "source": source,
        "fallback_reason": fallback_reason,
        "sample_count": int(len(candidates)),
        "start_time_s": float(candidate_times.iloc[0]),
        "end_time_s": float(candidate_times.iloc[-1]),
        "power_median_W": baseline,
        "power_mad_W": mad,
    }


def _single_sample_duration(times, index, sampling_rate_hz):
    if len(times) == 1:
        return 1.0 / sampling_rate_hz
    if index == 0:
        return float(times.iloc[1] - times.iloc[0])
    if index == len(times) - 1:
        return float(times.iloc[-1] - times.iloc[-2])
    return float((times.iloc[index + 1] - times.iloc[index - 1]) / 2.0)


def _build_regions(samples, sampling_rate_hz, manifest):
    regions = []
    inferences_per_cycle = None
    input_batch_size = 1
    if manifest is not None:
        inferences_per_cycle = manifest.get("plan", {}).get(
            "inferences_per_cycle"
        )
        input_batch_size = manifest.get("input_batch_size", 1)
    if isinstance(input_batch_size, bool) or not isinstance(
        input_batch_size, (int, float)
    ) or input_batch_size <= 0:
        raise ValueError("Manifest input_batch_size must be a positive number")
    input_batch_size = float(input_batch_size)

    idle_baseline, baseline_metadata = _idle_baseline(samples, manifest)
    for cycle, (start, end) in enumerate(
        _run_bounds(samples["is_active"]), start=1
    ):
        active_power = samples["power_clean_W"].iloc[start:end]
        active_times = samples["time_s"].iloc[start:end]
        if len(active_times) > 1:
            deltas = np.diff(active_times.to_numpy(dtype=float))
            if np.any(deltas <= 0):
                raise ValueError("Elapsed Time (s) must be strictly increasing")
            duration = float(active_times.iloc[-1] - active_times.iloc[0])
            interval_power = (
                active_power.to_numpy(dtype=float)[:-1]
                + active_power.to_numpy(dtype=float)[1:]
            ) / 2.0
            active_energy = float(np.sum(interval_power * deltas))
            active_power_mean = active_energy / duration
        else:
            duration = _single_sample_duration(
                samples["time_s"], start, sampling_rate_hz
            )
            active_power_mean = float(active_power.iloc[0])
            active_energy = active_power_mean * duration
        offset_power = float(active_power_mean - idle_baseline)
        energy = float(active_energy - idle_baseline * duration)
        regions.append(
            {
                "cycle": cycle,
                "start_time_s": float(samples["time_s"].iloc[start]),
                "end_time_s": float(samples["time_s"].iloc[end - 1]),
                "duration_s": duration,
                "active_power_mean_W": active_power_mean,
                "idle_power_median_W": idle_baseline,
                "idle_baseline_source": baseline_metadata["source"],
                "power_offset_W": offset_power,
                "energy_per_cycle_J": energy,
                "energy_per_inference_J": (
                    energy / (inferences_per_cycle * input_batch_size)
                    if inferences_per_cycle
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(regions), baseline_metadata


def process_measurement(csv_path, manifest_path=None, config=None):
    """Clean one CSV and return auditable sample, region, and summary tables."""

    config = config or ProcessingConfig()
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path)
    if POWER_COLUMN not in data:
        raise ValueError(f"Missing column '{POWER_COLUMN}'")
    power = pd.to_numeric(data[POWER_COLUMN], errors="coerce")
    if power.isna().any() or not np.isfinite(power).all():
        raise ValueError(
            f"Column '{POWER_COLUMN}' must contain only finite numbers"
        )

    manifest, resolved_manifest_path = _load_manifest(csv_path, manifest_path)
    sampling_rate_hz, rate_source = _sampling_rate(
        data, manifest, config.sampling_rate_hz
    )
    hampel_window = _odd_window(config.hampel_window_seconds, sampling_rate_hz)
    smoothing_window = _odd_window(
        config.smoothing_window_seconds, sampling_rate_hz
    )
    cleaned, is_outlier = _hampel_clean(
        power, hampel_window, config.hampel_sigma
    )
    smoothed = cleaned.rolling(
        smoothing_window, center=True, min_periods=1
    ).mean()
    if np.allclose(smoothed, smoothed.iloc[0]):
        raise ValueError("Power trace is constant; phases cannot be identified")
    threshold = float(threshold_otsu(smoothed.to_numpy()))
    time_axis = _time_axis(data, sampling_rate_hz)

    manifest_active, alignment_diagnostics = _aligned_active_mask(
        time_axis,
        manifest,
        sampling_rate_hz,
        config,
    )
    if manifest_active is None:
        active = _signal_active_mask(smoothed, threshold, sampling_rate_hz, config)
        active_source = "signal hysteresis (Otsu fallback)"
    else:
        active = manifest_active
        active_source = "synchronized manifest elapsed time"
    preparation = (
        _aligned_preparation_mask(time_axis, manifest, active, config)
        if manifest_active is not None
        else None
    )
    if preparation is None:
        preparation = _infer_preparation_mask(
            smoothed, active, threshold, sampling_rate_hz, config
        )
        preparation_source = "signal before each active region"
    else:
        preparation_source = "synchronized manifest lifecycle timing"

    phase = np.full(len(data), "idle", dtype=object)
    phase[preparation] = "input_and_parameter_preparation"
    phase[active] = "active_inference"
    samples = data.copy()
    samples["time_s"] = time_axis
    samples["power_raw_W"] = power
    samples["power_clean_W"] = cleaned
    samples["power_smoothed_W"] = smoothed
    samples["is_outlier"] = is_outlier
    samples["is_active"] = active
    samples["phase"] = phase
    regions, idle_baseline_metadata = _build_regions(
        samples,
        sampling_rate_hz,
        manifest,
    )

    workload_policy = manifest.get("workload_policy", {}) if manifest else {}
    summary = {
        "csv_path": str(csv_path),
        "manifest_path": (
            str(resolved_manifest_path) if resolved_manifest_path else None
        ),
        "sampling_rate_hz": sampling_rate_hz,
        "sampling_rate_source": rate_source,
        "threshold_W": threshold,
        "outlier_count": int(is_outlier.sum()),
        "outlier_fraction": float(is_outlier.mean()),
        "active_region_count": len(regions),
        "active_classification_source": active_source,
        "clock_alignment_status": alignment_diagnostics["status"],
        "clock_alignment_uncertainty_seconds": alignment_diagnostics[
            "uncertainty_seconds"
        ],
        "clock_alignment_threshold_seconds": alignment_diagnostics[
            "uncertainty_threshold_seconds"
        ],
        "clock_alignment_fallback_reason": alignment_diagnostics[
            "fallback_reason"
        ],
        "idle_baseline": idle_baseline_metadata,
        "preparation_classification_source": preparation_source,
        "preparation_phase": (
            "inferred interval after the configured idle sleep and before "
            "BURST_START; it can include a brief acquisition health check"
        ),
        "input_policy": workload_policy.get("input", "unknown"),
        "parameter_policy": workload_policy.get("parameters", "unknown"),
        "one_time_model_preparation_observed": False,
        "one_time_model_preparation_note": (
            "Model construction/loading, warm-up, and calibration occur before "
            "INA226 acquisition starts and are not present in this CSV."
        ),
        "configuration": asdict(config),
    }
    return ProcessingResult(samples=samples, regions=regions, summary=summary)


def plot_measurement(result, output_path, title=None):
    """Save a clean trace with phase shading and visible rejected outliers."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    samples = result.samples
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.plot(
        samples["time_s"],
        samples["power_raw_W"],
        color="0.72",
        linewidth=0.15,
        alpha=0.55,
        label="Raw power",
    )
    axis.plot(
        samples["time_s"],
        samples["power_smoothed_W"],
        color="#143642",
        linewidth=0.35,
        label="Cleaned power",
    )
    outliers = samples[samples["is_outlier"]]
    if not outliers.empty:
        axis.scatter(
            outliers["time_s"],
            outliers["power_raw_W"],
            color="#c44900",
            marker="x",
            s=13,
            linewidths=0.4,
            label="Rejected outlier",
            zorder=4,
        )

    phase_colors = {
        "active_inference": ("#e9c46a", 0.22),
        "input_and_parameter_preparation": ("#2a9d8f", 0.24),
    }
    for phase_name, (color, alpha) in phase_colors.items():
        for start, end in _run_bounds(samples["phase"] == phase_name):
            left = samples["time_s"].iloc[start]
            right = samples["time_s"].iloc[min(end, len(samples) - 1)]
            axis.axvspan(left, right, color=color, alpha=alpha, linewidth=0)

    axis.axhline(
        result.summary["threshold_W"],
        color="#9b2226",
        linestyle="--",
        linewidth=0.4,
        label="Signal threshold",
    )
    axis.set(
        title=title or Path(result.summary["csv_path"]).stem,
        xlabel="Time (s)",
        ylabel="Power (W)",
    )
    axis.set_xlim(samples["time_s"].iloc[0], samples["time_s"].iloc[-1])
    axis.grid(True, color="0.88", linewidth=0.3)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor="#e9c46a", alpha=0.35, label="Active inference"),
            Patch(
                facecolor="#2a9d8f",
                alpha=0.35,
                label="Inferred input/parameter preparation",
            ),
        ]
    )
    labels.extend(["Active inference", "Inferred input/parameter preparation"])
    axis.legend(handles, labels, loc="upper right", frameon=True, ncols=2)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_result(result, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.samples.to_csv(output_dir / f"{stem}_cleaned.csv", index=False)
    result.regions.to_csv(output_dir / f"{stem}_regions.csv", index=False)
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_measurement(result, output_dir / f"{stem}_cleaned.pdf")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Clean and annotate one INA226 measurement CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("cleaned_output"))
    parser.add_argument("--sampling-rate-hz", type=float)
    parser.add_argument(
        "--max-clock-uncertainty-fraction",
        type=float,
        default=0.50,
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = ProcessingConfig(
        sampling_rate_hz=args.sampling_rate_hz,
        max_clock_uncertainty_fraction=(
            args.max_clock_uncertainty_fraction
        ),
    )
    result = process_measurement(args.csv_path, args.manifest, config)
    save_result(result, args.output_dir, args.csv_path.stem)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
