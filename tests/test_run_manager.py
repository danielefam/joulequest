import json
import tempfile
import unittest
from pathlib import Path

from base_runner import BurstResult
from run_manager import (
    RunManager,
    calculate_capture_plan,
    calculate_inference_count,
    calculate_required_burst_seconds,
)


class FakeRunner:
    """Deterministic runner used to verify workload accounting."""

    instances = []

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.prepared = False
        self.closed = False
        self.calls = []
        self.burst_preparations = 0
        self.parameter_states = []
        self.input_states = []
        self._calibration_call = 0
        FakeRunner.instances.append(self)

    def prepare(self):
        self.prepared = True

    def prepare_burst(self):
        self.burst_preparations += 1
        self.parameter_states.append(object())
        self.input_states.append(object())

    def run_burst(self, inference_count):
        self.prepare_burst()
        self.calls.append(inference_count)
        # Ten milliseconds per inference gives exact, deterministic planning.
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * 0.01,
        )

    def run_prepared_burst(self, inference_count):
        self.calls.append(inference_count)
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * 0.01,
        )

    def close(self):
        self.closed = True


class UnstableFakeRunner(FakeRunner):
    def run_burst(self, inference_count):
        self.prepare_burst()
        self.calls.append(inference_count)
        self._calibration_call += 1
        multiplier = 1.0 if self._calibration_call % 2 else 2.0
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * 0.01 * multiplier,
        )

    def run_prepared_burst(self, inference_count):
        self.calls.append(inference_count)
        self._calibration_call += 1
        multiplier = 1.0 if self._calibration_call % 2 else 2.0
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * 0.01 * multiplier,
        )


class PlanningTests(unittest.TestCase):
    def test_required_duration_respects_time_and_sample_constraints(self):
        self.assertEqual(calculate_required_burst_seconds(2.0, 50, 10.0), 5.0)
        self.assertEqual(calculate_required_burst_seconds(10.0, 50, 10.0), 10.0)

    def test_inference_count_rounds_up(self):
        self.assertEqual(
            calculate_inference_count(
                stable_latency_seconds=0.003,
                target_burst_seconds=1.0,
                min_active_samples=20,
                sampling_rate_hz=10.0,
            ),
            667,
        )

    def test_capture_plan_counts_only_inter_cycle_sleeps(self):
        plan = calculate_capture_plan(
            number_of_cycles=3,
            estimated_burst_seconds=2.0,
            sleep_time_seconds=4.0,
            leading_idle_seconds=5.0,
            trailing_idle_seconds=6.0,
            safety_margin_seconds=1.0,
            sampling_rate_hz=10.0,
        )
        self.assertEqual(plan.capture_seconds, 26.0)
        self.assertEqual(plan.capture_samples, 260)


class RunManagerTests(unittest.TestCase):
    def setUp(self):
        FakeRunner.instances.clear()

    def _manager(self, manifest_directory, runner_cls=FakeRunner, **overrides):
        settings = {
            "runner_cls": runner_cls,
            "model_path": "Linear_64_64.pt",
            "number_of_cycles": 2,
            "sleep_time": 0.0,
            "backend": "cpu",
            "inferences_per_cycle": None,
            "target_burst_seconds": 1.0,
            "sampling_rate_hz": 10.0,
            "min_active_samples": 10,
            "warmup_inferences": 5,
            "warmup_seconds": 0.05,
            "calibration_initial_inferences": 2,
            "calibration_target_seconds": 0.1,
            "calibration_repetitions": 4,
            "max_relative_mad": 0.05,
            "leading_idle_seconds": 0.0,
            "trailing_idle_seconds": 0.0,
            "safety_margin_seconds": 0.0,
            "wait_for_acquisition": False,
            "manifest_directory": manifest_directory,
            "sleep_fn": lambda _seconds: None,
        }
        settings.update(overrides)
        return RunManager(**settings)

    def test_warmup_calibration_and_measurement_are_accounted_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = self._manager(temp_dir).execute()
            runner = FakeRunner.instances[-1]

            self.assertTrue(runner.prepared)
            self.assertTrue(runner.closed)
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["warmup"]["executed_inferences"], 5)
            self.assertEqual(manifest["calibration"]["discarded_batches"], 1)
            self.assertEqual(manifest["plan"]["inferences_per_cycle"], 100)
            self.assertEqual(
                manifest["workload_policy"]["parameters"],
                "fresh_per_burst",
            )
            self.assertEqual(
                manifest["workload_policy"]["input"],
                "fresh_per_burst",
            )
            self.assertEqual(
                manifest["measurement"]["total_executed_inferences"], 200
            )
            self.assertEqual(len(manifest["measurement"]["cycles"]), 2)

            # Calls: one warm-up, one sizing pilot, four calibration batches,
            # and exactly two measured cycles.
            self.assertEqual(len(runner.calls), 8)
            self.assertEqual(runner.calls[-2:], [100, 100])
            self.assertEqual(runner.burst_preparations, len(runner.calls))
            self.assertEqual(len(set(map(id, runner.parameter_states))), 8)
            self.assertEqual(len(set(map(id, runner.input_states))), 8)

            manifest_path = Path(manifest["manifest_path"])
            self.assertTrue(manifest_path.exists())
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["campaign_id"], manifest["campaign_id"])

    def test_manual_override_that_is_too_short_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self._manager(temp_dir, inferences_per_cycle=5)
            with self.assertRaisesRegex(ValueError, "minimum active samples"):
                manager.execute()

    def test_unstable_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self._manager(
                temp_dir,
                runner_cls=UnstableFakeRunner,
                max_relative_mad=0.01,
            )
            with self.assertRaisesRegex(RuntimeError, "did not stabilize"):
                manager.execute()


if __name__ == "__main__":
    unittest.main()
