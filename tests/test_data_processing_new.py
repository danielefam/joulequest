import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from processing_report.data_processing_new import (
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

    def test_manifest_timestamps_identify_active_and_preparation_phases(self):
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
            "manifest burst timestamps",
        )
        self.assertEqual(
            result.summary["preparation_classification_source"],
            "manifest lifecycle timing",
        )
        self.assertGreater(phase_counts["input_and_parameter_preparation"], 0)
        self.assertFalse(result.summary["one_time_model_preparation_observed"])


if __name__ == "__main__":
    unittest.main()
