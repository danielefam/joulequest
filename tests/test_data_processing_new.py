import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from processing_report.data_processing import (
    POWER_COLUMN,
    ProcessingConfig,
    process_measurement,
)


class CleanDataProcessingTests(unittest.TestCase):
    def test_isolated_spike_is_replaced_without_removing_step_edges(self):
        sampling_rate_hz = 100.0
        sample_count = 400
        time_seconds = np.arange(sample_count) / sampling_rate_hz
        power = 0.30 + 0.005 * np.sin(np.arange(sample_count))
        power[100:200] += 1.20
        power[50] = 3.0

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "synthetic.csv"
            pd.DataFrame(
                {
                    "Sample": np.arange(sample_count),
                    "Elapsed Time (s)": time_seconds,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)

            result = process_measurement(
                csv_path,
                config=ProcessingConfig(sampling_rate_hz=sampling_rate_hz),
            )

        self.assertTrue(result.samples.loc[50, "is_outlier"])
        self.assertAlmostEqual(
            result.samples.loc[50, "power_clean_W"], 0.30, places=2
        )
        self.assertFalse(result.samples.loc[100, "is_outlier"])
        self.assertFalse(result.samples.loc[199, "is_outlier"])
        self.assertEqual(result.summary["active_region_count"], 1)
        self.assertEqual(len(result.samples), sample_count)

    def test_legacy_manifest_uses_otsu_and_signal_preparation(self):
        sampling_rate_hz = 100.0
        sample_count = 500
        start = pd.Timestamp("2026-07-27T10:00:00Z")
        timestamps = start + pd.to_timedelta(
            np.arange(sample_count) / sampling_rate_hz, unit="s"
        )
        power = np.full(sample_count, 0.3)
        power[100:120] = 0.6
        power[120:221] = 1.5
        power[320:330] = 0.6
        power[330:431] = 1.5

        def event_time(offset_seconds):
            return (start + pd.to_timedelta(offset_seconds, unit="s")).isoformat()

        manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "workload_policy": {
                "input": "fresh_per_burst",
                "parameters": "fresh_per_burst",
            },
            "plan": {
                "inferences_per_cycle": 100,
                "number_of_cycles": 2,
                "leading_idle_seconds": 1.0,
                "sleep_time_seconds": 1.0,
            },
            "acquisition": {
                "started_at_utc": event_time(0),
                "sampling_rate_hz": sampling_rate_hz,
            },
            "measurement": {
                "cycles": [
                    {
                        "start_event": {"wall_time_utc": event_time(1.2)},
                        "end_event": {"wall_time_utc": event_time(2.2)},
                    },
                    {
                        "start_event": {"wall_time_utc": event_time(3.3)},
                        "end_event": {"wall_time_utc": event_time(4.3)},
                    },
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            csv_path = directory / "synthetic.csv"
            manifest_path = directory / "synthetic.json"
            pd.DataFrame(
                {
                    "Sample": np.arange(sample_count),
                    "Timestamp UTC": timestamps,
                    "Elapsed Time (s)": np.arange(sample_count) / sampling_rate_hz,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = process_measurement(csv_path)

        phase_counts = result.samples["phase"].value_counts()
        self.assertEqual(result.summary["active_region_count"], 2)
        self.assertEqual(
            result.summary["active_classification_source"],
            "signal hysteresis (Otsu fallback)",
        )
        self.assertEqual(
            result.summary["preparation_classification_source"],
            "signal before each active region",
        )
        self.assertEqual(
            result.summary["clock_alignment_fallback_reason"],
            "LEGACY_MANIFEST_NO_ALIGNMENT",
        )
        self.assertGreater(phase_counts["input_and_parameter_preparation"], 0)
        self.assertFalse(result.summary["one_time_model_preparation_observed"])

    def test_aligned_elapsed_bounds_allow_half_sample_clock_uncertainty(self):
        sampling_rate_hz = 100.0
        sample_count = 500
        elapsed = np.arange(sample_count) / sampling_rate_hz
        power = np.full(sample_count, 0.3)
        power[120:221] = 1.5
        power[330:431] = 1.5
        manifest = {
            "schema_version": 2,
            "workload_policy": {
                "input": "fresh_per_burst",
                "parameters": "fresh_per_burst",
            },
            "plan": {
                "inferences_per_cycle": 100,
                "number_of_cycles": 2,
                "leading_idle_seconds": 1.0,
                "sleep_time_seconds": 1.0,
            },
            "acquisition": {
                "sampling_rate_hz": sampling_rate_hz,
                "clock_alignment": {
                    "status": "COMPLETE",
                    "classification_eligible": False,
                    "uncertainty_seconds": 0.005,
                    "fallback_reason": "CLOCK_UNCERTAINTY_EXCEEDED",
                },
            },
            "measurement": {
                "cycles": [
                    {
                        "start_event": {"aligned_elapsed_seconds": 1.2},
                        "end_event": {"aligned_elapsed_seconds": 2.2},
                    },
                    {
                        "start_event": {"aligned_elapsed_seconds": 3.3},
                        "end_event": {"aligned_elapsed_seconds": 4.3},
                    },
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            csv_path = directory / "synthetic.csv"
            manifest_path = directory / "synthetic.json"
            pd.DataFrame(
                {
                    "Sample": np.arange(sample_count),
                    "Elapsed Time (s)": elapsed,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            aligned = process_measurement(csv_path)
            manifest["acquisition"]["clock_alignment"][
                "uncertainty_seconds"
            ] = 0.00501
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fallback = process_measurement(csv_path)

        self.assertEqual(
            aligned.summary["active_classification_source"],
            "synchronized manifest elapsed time",
        )
        self.assertEqual(aligned.summary["active_region_count"], 2)
        self.assertIsNone(aligned.summary["clock_alignment_fallback_reason"])
        self.assertEqual(
            fallback.summary["active_classification_source"],
            "signal hysteresis (Otsu fallback)",
        )
        self.assertEqual(
            fallback.summary["clock_alignment_fallback_reason"],
            "CLOCK_UNCERTAINTY_EXCEEDED",
        )

    def test_final_safety_margin_idle_and_actual_time_deltas_drive_energy(self):
        elapsed = np.array([0.0, 0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 1.0])
        power = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        manifest = {
            "schema_version": 2,
            "workload_policy": {},
            "plan": {
                "inferences_per_cycle": 10,
                "number_of_cycles": 1,
                "leading_idle_seconds": 0.0,
                "sleep_time_seconds": 0.0,
                "safety_margin_seconds": 0.2,
            },
            "acquisition": {
                "sampling_rate_hz": 10.0,
                "clock_alignment": {
                    "status": "COMPLETE",
                    "classification_eligible": True,
                    "uncertainty_seconds": 0.001,
                    "fallback_reason": None,
                },
            },
            "measurement": {
                "cycles": [
                    {
                        "start_event": {"aligned_elapsed_seconds": 0.2},
                        "end_event": {"aligned_elapsed_seconds": 0.7},
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            csv_path = directory / "synthetic.csv"
            manifest_path = directory / "synthetic.json"
            pd.DataFrame(
                {
                    "Sample": np.arange(len(elapsed)),
                    "Elapsed Time (s)": elapsed,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = process_measurement(
                csv_path,
                config=ProcessingConfig(
                    hampel_window_seconds=0.01,
                    smoothing_window_seconds=0.01,
                ),
            )

        region = result.regions.iloc[0]
        baseline = result.summary["idle_baseline"]
        self.assertEqual(baseline["source"], "final measured safety margin")
        self.assertEqual(baseline["sample_count"], 3)
        self.assertAlmostEqual(baseline["power_median_W"], 1.0)
        self.assertAlmostEqual(region["duration_s"], 0.5)
        self.assertAlmostEqual(region["power_offset_W"], 1.0)
        self.assertAlmostEqual(region["energy_per_cycle_J"], 0.5)
        self.assertAlmostEqual(region["energy_per_inference_J"], 0.05)

    def test_missing_safety_margin_uses_idle_fallback_without_discarding(self):
        sampling_rate_hz = 20.0
        elapsed = np.arange(40) / sampling_rate_hz
        power = np.full(40, 0.5)
        power[10:21] = 1.5

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            csv_path = directory / "synthetic.csv"
            manifest_path = directory / "synthetic.json"
            pd.DataFrame(
                {
                    "Sample": np.arange(len(elapsed)),
                    "Elapsed Time (s)": elapsed,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "plan": {"inferences_per_cycle": 10},
                        "acquisition": {
                            "sampling_rate_hz": sampling_rate_hz
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = process_measurement(csv_path)

        self.assertEqual(len(result.regions), 1)
        self.assertTrue(np.isfinite(result.regions.iloc[0]["energy_per_cycle_J"]))
        self.assertEqual(
            result.summary["idle_baseline"]["source"],
            "classified idle fallback",
        )
        self.assertEqual(
            result.summary["idle_baseline"]["fallback_reason"],
            "SAFETY_MARGIN_UNAVAILABLE",
        )


if __name__ == "__main__":
    unittest.main()
