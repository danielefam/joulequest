import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from processing_report import process_and_visualize
from processing_report.data_processing_discard import (
    POWER_COLUMN,
    ProcessingConfig,
    build_parser,
    process_measurement,
)


class TailTrimProcessingTests(unittest.TestCase):
    @staticmethod
    def _aligned_manifest(inferences_per_cycle=100):
        return {
            "schema_version": 2,
            "plan": {
                "inferences_per_cycle": inferences_per_cycle,
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
                        "start_event": {"aligned_elapsed_seconds": 0.2},
                        "end_event": {"aligned_elapsed_seconds": 1.1},
                    }
                ]
            },
        }

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

    def test_trimming_is_skipped_below_minimum_inference_count(self):
        elapsed = np.arange(13) / 10.0
        power = np.array(
            [1.0, 10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0, 1.0]
        )

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
                json.dumps(self._aligned_manifest(inferences_per_cycle=5)),
                encoding="utf-8",
            )

            result = process_measurement(
                csv_path,
                config=ProcessingConfig(
                    tail_trim_fraction=0.1,
                    discard_initial_samples=True,
                    initial_trim_fraction=0.2,
                ),
            )

        self.assertFalse(result.summary["trimming_enabled"])
        self.assertEqual(
            result.summary["trimming_skip_reason"],
            "TRIMMING_SKIPPED_TOO_FEW_INFERENCES",
        )
        self.assertEqual(result.summary["outlier_count"], 0)
        self.assertEqual(result.regions.iloc[0]["retained_sample_count"], 10)

    def test_optional_filters_run_after_discarding_extreme_samples(self):
        elapsed = np.arange(15) / 10.0
        power = np.array(
            [1.0, 1.0, 2.0, 100.0, 2.0, 2.0, 20.0, 2.0, 2.0,
             2.0, 0.0, 2.0, 1.0, 1.0, 1.0]
        )

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
                json.dumps(self._aligned_manifest()),
                encoding="utf-8",
            )

            unfiltered = process_measurement(
                csv_path,
                config=ProcessingConfig(tail_trim_fraction=0.1),
            )
            filtered = process_measurement(
                csv_path,
                config=ProcessingConfig(
                    tail_trim_fraction=0.1,
                    filter_after_discard=True,
                    hampel_window_seconds=0.3,
                    rolling_window_seconds=0.3,
                ),
            )

        self.assertEqual(unfiltered.samples.loc[6, "power_clean_W"], 20.0)
        self.assertFalse(unfiltered.summary["filter_after_discard"])
        self.assertEqual(filtered.summary["outlier_count"], 2)
        self.assertEqual(filtered.summary["hampel_outlier_count"], 1)
        self.assertTrue(filtered.summary["filter_after_discard"])
        self.assertTrue(filtered.samples.loc[3, "is_power_outlier"])
        self.assertTrue(filtered.samples.loc[10, "is_power_outlier"])
        self.assertTrue(filtered.samples.loc[6, "is_hampel_outlier"])
        self.assertTrue(np.isnan(filtered.samples.loc[3, "power_clean_W"]))
        self.assertAlmostEqual(filtered.samples.loc[6, "power_clean_W"], 2.0)
        self.assertAlmostEqual(filtered.samples.loc[3, "power_smoothed_W"], 2.0)
        self.assertAlmostEqual(filtered.samples.loc[6, "power_smoothed_W"], 2.0)
        self.assertLess(
            filtered.regions.iloc[0]["energy_per_cycle_J"],
            unfiltered.regions.iloc[0]["energy_per_cycle_J"],
        )

    def test_filter_after_discard_cli_flag_is_optional(self):
        parser = build_parser()
        disabled = parser.parse_args(["capture.csv"])
        enabled = parser.parse_args(
            ["capture.csv", "--filter-after-discard"]
        )

        self.assertFalse(disabled.filter_after_discard)
        self.assertTrue(enabled.filter_after_discard)

    def test_batch_processor_forwards_hybrid_filter_options(self):
        args = process_and_visualize.build_parser().parse_args(
            [
                "--frequency", "100",
                "--tail-trim-percentage", "2.5",
                "--discard-initial-samples",
                "--initial-trim-percentage", "4",
                "--filter-after-discard",
                "--hampel-window-seconds", "0.31",
                "--hampel-sigma", "3.5",
                "--rolling-window-seconds", "0.11",
                "--max-clock-uncertainty-fraction", "0.25",
            ]
        )

        config = process_and_visualize._processing_config(args)

        self.assertEqual(config.sampling_rate_hz, 100.0)
        self.assertEqual(config.tail_trim_fraction, 0.025)
        self.assertTrue(config.discard_initial_samples)
        self.assertEqual(config.initial_trim_fraction, 0.04)
        self.assertEqual(config.min_inferences_for_trimming, 10)
        self.assertTrue(config.filter_after_discard)
        self.assertEqual(config.hampel_window_seconds, 0.31)
        self.assertEqual(config.hampel_sigma, 3.5)
        self.assertEqual(config.rolling_window_seconds, 0.11)
        self.assertEqual(config.max_clock_uncertainty_fraction, 0.25)


if __name__ == "__main__":
    unittest.main()