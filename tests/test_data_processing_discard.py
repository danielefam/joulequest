import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from processing_report.data_processing_discard import (
    POWER_COLUMN,
    ProcessingConfig,
    process_measurement,
)


class TailTrimProcessingTests(unittest.TestCase):
    def test_boundary_outliers_shrink_region_and_inference_count(self):
        elapsed = np.arange(13) / 10.0
        power = np.array(
            [1.0, 10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0, 1.0]
        )
        manifest = {
            "schema_version": 2,
            "plan": {
                "inferences_per_cycle": 100,
                "sleep_time_seconds": 0.0,
                "safety_margin_seconds": 0.2,
            },
            "acquisition": {
                "sampling_rate_hz": 10.0,
                "clock_alignment": {
                    "status": "COMPLETE",
                    "uncertainty_seconds": 0.001,
                },
            },
            "measurement": {
                "cycles": [
                    {
                        "start_event": {"aligned_elapsed_seconds": 0.1},
                        "end_event": {"aligned_elapsed_seconds": 1.0},
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "synthetic.csv"
            pd.DataFrame(
                {
                    "Sample": np.arange(len(elapsed)),
                    "Elapsed Time (s)": elapsed,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)
            csv_path.with_suffix(".json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = process_measurement(
                csv_path,
                config=ProcessingConfig(tail_trim_fraction=0.1),
            )

        region = result.regions.iloc[0]
        self.assertEqual(result.summary["outlier_count"], 2)
        self.assertEqual(region["discarded_leading_sample_count"], 1)
        self.assertEqual(region["discarded_trailing_sample_count"], 1)
        self.assertEqual(region["discarded_interior_sample_count"], 0)
        self.assertAlmostEqual(region["original_start_time_s"], 0.1)
        self.assertAlmostEqual(region["original_end_time_s"], 1.0)
        self.assertAlmostEqual(region["start_time_s"], 0.2)
        self.assertAlmostEqual(region["end_time_s"], 0.9)
        self.assertAlmostEqual(region["duration_s"], 0.8)
        self.assertAlmostEqual(region["effective_inference_count"], 80.0)
        self.assertAlmostEqual(region["discarded_inference_count"], 20.0)
        self.assertAlmostEqual(
            region["energy_per_inference_J"], 3.6 / 80.0
        )

    def test_interior_outliers_are_skipped_without_bridging_the_gap(self):
        elapsed = np.arange(8) / 10.0
        power = np.array([1.0, 2.0, 10.0, 3.0, 0.0, 4.0, 1.0, 1.0])
        manifest = {
            "schema_version": 2,
            "plan": {
                "inferences_per_cycle": 100,
                "sleep_time_seconds": 0.0,
                "safety_margin_seconds": 0.1,
            },
            "acquisition": {
                "sampling_rate_hz": 10.0,
                "clock_alignment": {
                    "status": "COMPLETE",
                    "uncertainty_seconds": 0.001,
                },
            },
            "measurement": {
                "cycles": [
                    {
                        "start_event": {"aligned_elapsed_seconds": 0.1},
                        "end_event": {"aligned_elapsed_seconds": 0.5},
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "synthetic.csv"
            pd.DataFrame(
                {
                    "Sample": np.arange(len(elapsed)),
                    "Elapsed Time (s)": elapsed,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)
            csv_path.with_suffix(".json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = process_measurement(
                csv_path,
                config=ProcessingConfig(tail_trim_fraction=0.2),
            )

        region = result.regions.iloc[0]
        self.assertEqual(region["discarded_interior_sample_count"], 2)
        self.assertEqual(region["discarded_leading_sample_count"], 0)
        self.assertEqual(region["discarded_trailing_sample_count"], 0)
        self.assertAlmostEqual(region["duration_s"], 0.2)
        self.assertAlmostEqual(region["energy_per_cycle_J"], 0.4)
        self.assertAlmostEqual(region["effective_inference_count"], 60.0)
        self.assertAlmostEqual(
            region["energy_per_inference_J"], 0.4 / 60.0
        )

    def test_initial_time_trim_is_controlled_by_flag(self):
        elapsed = np.arange(8) / 10.0
        power = np.array([1.0, 2.0, 10.0, 3.0, 0.0, 4.0, 1.0, 1.0])
        manifest = {
            "schema_version": 2,
            "plan": {
                "inferences_per_cycle": 100,
                "sleep_time_seconds": 0.0,
                "safety_margin_seconds": 0.1,
            },
            "acquisition": {
                "sampling_rate_hz": 10.0,
                "clock_alignment": {
                    "status": "COMPLETE",
                    "uncertainty_seconds": 0.001,
                },
            },
            "measurement": {
                "cycles": [
                    {
                        "start_event": {"aligned_elapsed_seconds": 0.1},
                        "end_event": {"aligned_elapsed_seconds": 0.5},
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "synthetic.csv"
            pd.DataFrame(
                {
                    "Sample": np.arange(len(elapsed)),
                    "Elapsed Time (s)": elapsed,
                    POWER_COLUMN: power,
                }
            ).to_csv(csv_path, index=False)
            csv_path.with_suffix(".json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            without_initial_trim = process_measurement(
                csv_path,
                config=ProcessingConfig(
                    tail_trim_fraction=0.2,
                    discard_initial_samples=False,
                    initial_trim_fraction=0.2,
                ),
            )
            with_initial_trim = process_measurement(
                csv_path,
                config=ProcessingConfig(
                    tail_trim_fraction=0.2,
                    discard_initial_samples=True,
                    initial_trim_fraction=0.2,
                ),
            )

        without_region = without_initial_trim.regions.iloc[0]
        with_region = with_initial_trim.regions.iloc[0]
        self.assertEqual(without_region["discarded_initial_sample_count"], 0)
        self.assertAlmostEqual(without_region["start_time_s"], 0.1)
        self.assertEqual(with_region["discarded_initial_sample_count"], 1)
        self.assertAlmostEqual(with_region["start_time_s"], 0.3)
        self.assertAlmostEqual(with_region["effective_inference_count"], 40.0)


if __name__ == "__main__":
    unittest.main()