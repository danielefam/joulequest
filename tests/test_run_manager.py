import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_manager as run_manager_module
from base_runner import BurstResult
from run_manager import (
    RunManager,
    StdioAcquisitionController,
    calculate_capture_plan,
    calculate_clock_sync_sample,
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


class PilotRegimeShiftRunner(FakeRunner):
    def run_burst(self, inference_count):
        self.prepare_burst()
        self.calls.append(inference_count)
        latency_seconds = 0.01 if inference_count == 2 else 0.002
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * latency_seconds,
        )

    def run_prepared_burst(self, inference_count):
        self.calls.append(inference_count)
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * 0.002,
        )


class FinalCountRegimeShiftRunner(FakeRunner):
    def run_burst(self, inference_count):
        self.prepare_burst()
        self.calls.append(inference_count)
        latency_seconds = 0.002 if inference_count >= 100 else 0.01
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * latency_seconds,
        )

    def run_prepared_burst(self, inference_count):
        self.calls.append(inference_count)
        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=inference_count * 0.002,
        )


class FailingMeasuredRunner(FakeRunner):
    def __init__(self, model_path, device):
        super().__init__(model_path, device)
        self.measured_calls = 0

    def run_prepared_burst(self, inference_count):
        self.measured_calls += 1
        if self.measured_calls == 2:
            raise RuntimeError("measured workload failed")
        return super().run_prepared_burst(inference_count)


class FakeAcquisitionController:
    def __init__(self, timeline, start_error=None):
        self.timeline = timeline
        self.start_error = start_error
        self.active = False
        self.stop_calls = 0

    def describe(self, campaign_id, plan):
        self.timeline.append(("describe", campaign_id))
        return {
            "csv_path": f"/tmp/{campaign_id}.csv",
            "sampling_rate_hz": plan["sampling_rate_hz"],
        }

    def start(self):
        self.timeline.append(("start", None))
        if self.start_error is not None:
            raise self.start_error
        self.active = True
        return {"status": "RUNNING", "sample_count": 1}

    def synchronize_clock(self, round_name, exchange_count):
        self.timeline.append(("sync", round_name))
        return {
            "status": "UNAVAILABLE",
            "round": round_name,
            "requested_exchange_count": exchange_count,
        }

    def check_health(self):
        self.timeline.append(("health", None))
        return None

    def stop(self):
        self.timeline.append(("stop", None))
        self.stop_calls += 1
        self.active = False
        return {"status": "COMPLETE", "sample_count": 20}


class PlanningTests(unittest.TestCase):
    def test_clock_sync_sample_calculates_offset_rtt_and_uncertainty(self):
        sample = calculate_clock_sync_sample(1.0, 11.002, 11.003, 1.005)

        self.assertAlmostEqual(
            sample["controller_minus_runner_seconds"], 10.0
        )
        self.assertAlmostEqual(sample["round_trip_seconds"], 0.004)
        self.assertAlmostEqual(sample["uncertainty_seconds"], 0.002)
        self.assertAlmostEqual(
            sample["runner_midpoint_monotonic_seconds"], 1.0025
        )

    def test_required_duration_respects_time_and_sample_constraints(self):
        self.assertEqual(calculate_required_burst_seconds(2.0, 50, 10.0), 6.0)
        self.assertEqual(calculate_required_burst_seconds(10.0, 50, 10.0), 12.0)

    def test_inference_count_rounds_up(self):
        self.assertEqual(
            calculate_inference_count(
                stable_latency_seconds=0.003,
                target_burst_seconds=1.0,
                min_active_samples=20,
                sampling_rate_hz=10.0,
            ),
            800,
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


class StdioAcquisitionControllerTests(unittest.TestCase):
    def test_clock_sync_selects_the_minimum_round_trip_exchange(self):
        input_stream = io.StringIO(
            json.dumps(
                {
                    "command": "CLOCK_SYNC_RESPONSE",
                    "campaign_id": "campaign-123",
                    "request_id": "pre:1",
                    "result": {
                        "controller_received_monotonic_seconds": 11.004,
                        "controller_sent_monotonic_seconds": 11.005,
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "command": "CLOCK_SYNC_RESPONSE",
                    "campaign_id": "campaign-123",
                    "request_id": "pre:2",
                    "result": {
                        "controller_received_monotonic_seconds": 12.002,
                        "controller_sent_monotonic_seconds": 12.003,
                    },
                }
            )
            + "\n"
        )
        monotonic_values = iter([1.0, 1.009, 2.0, 2.005])
        output_stream = io.StringIO()
        controller = StdioAcquisitionController(
            input_stream,
            output_stream,
            monotonic_fn=lambda: next(monotonic_values),
        )
        controller.describe("campaign-123", {})

        result = controller.synchronize_clock("pre", 2)

        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["valid_exchange_count"], 2)
        self.assertEqual(result["selected_sample"]["request_id"], "pre:2")
        self.assertAlmostEqual(
            result["selected_sample"]["controller_minus_runner_seconds"],
            10.0,
        )
        requests = [
            json.loads(line) for line in output_stream.getvalue().splitlines()
        ]
        self.assertEqual(
            [request["request_id"] for request in requests],
            ["pre:1", "pre:2"],
        )

    def test_clock_sync_stops_early_only_with_precise_samples(self):
        def synchronize(
            exchange_count,
            runner_elapsed,
            controller_processing,
            controller_offsets=None,
            fail_first=False,
        ):
            responses = []
            monotonic_values = []
            for exchange_index in range(1, exchange_count + 1):
                runner_sent = float(exchange_index)
                controller_offset = (
                    10.0
                    if controller_offsets is None
                    else controller_offsets[exchange_index - 1]
                )
                controller_received = runner_sent + controller_offset + 0.0005
                response = json.dumps(
                    {
                        "command": "CLOCK_SYNC_RESPONSE",
                        "campaign_id": "campaign-123",
                        "request_id": f"pre:{exchange_index}",
                        "result": {
                            "controller_received_monotonic_seconds": (
                                controller_received
                            ),
                            "controller_sent_monotonic_seconds": (
                                controller_received + controller_processing
                            ),
                        },
                    }
                )
                responses.append(
                    "invalid JSON"
                    if fail_first and exchange_index == 1
                    else response
                )
                monotonic_values.extend(
                    [runner_sent, runner_sent + runner_elapsed]
                )

            output_stream = io.StringIO()
            values = iter(monotonic_values)
            controller = StdioAcquisitionController(
                io.StringIO("\n".join(responses) + "\n"),
                output_stream,
                monotonic_fn=lambda: next(values),
                max_clock_uncertainty_fraction=0.10,
            )
            controller.describe(
                "campaign-123",
                {"sampling_rate_hz": 10.0},
            )
            result = controller.synchronize_clock("pre", exchange_count)
            requests = output_stream.getvalue().splitlines()
            return result, requests

        precise, precise_requests = synchronize(
            exchange_count=10,
            runner_elapsed=0.002,
            controller_processing=0.0005,
        )
        self.assertTrue(precise["stopped_early"])
        self.assertEqual(precise["attempted_exchange_count"], 3)
        self.assertEqual(precise["valid_exchange_count"], 3)
        self.assertEqual(len(precise_requests), 3)

        noisy, noisy_requests = synchronize(
            exchange_count=4,
            runner_elapsed=0.020,
            controller_processing=0.001,
        )
        self.assertFalse(noisy["stopped_early"])
        self.assertEqual(noisy["attempted_exchange_count"], 4)
        self.assertEqual(noisy["valid_exchange_count"], 4)
        self.assertEqual(len(noisy_requests), 4)

        inconsistent, inconsistent_requests = synchronize(
            exchange_count=4,
            runner_elapsed=0.002,
            controller_processing=0.0005,
            controller_offsets=[10.0, 10.02, 10.0, 10.0],
        )
        self.assertFalse(inconsistent["stopped_early"])
        self.assertEqual(inconsistent["attempted_exchange_count"], 4)
        self.assertEqual(len(inconsistent_requests), 4)

        failed, failed_requests = synchronize(
            exchange_count=5,
            runner_elapsed=0.002,
            controller_processing=0.0005,
            fail_first=True,
        )
        self.assertFalse(failed["stopped_early"])
        self.assertEqual(failed["attempted_exchange_count"], 5)
        self.assertEqual(failed["valid_exchange_count"], 4)
        self.assertEqual(len(failed["errors"]), 1)
        self.assertEqual(len(failed_requests), 5)

    def test_json_handshake_uses_campaign_id_and_preserves_results(self):
        input_stream = io.StringIO(
            json.dumps(
                {
                    "command": "ACQUISITION_STARTED",
                    "campaign_id": "campaign-123",
                    "result": {"status": "RUNNING", "sample_count": 1},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "command": "ACQUISITION_STOPPED",
                    "campaign_id": "campaign-123",
                    "result": {"status": "COMPLETE", "sample_count": 20},
                }
            )
            + "\n"
        )
        output_stream = io.StringIO()
        controller = StdioAcquisitionController(input_stream, output_stream)

        description = controller.describe(
            "campaign-123",
            {"sampling_rate_hz": 10.0},
        )
        started = controller.start()
        stopped = controller.stop()

        self.assertEqual(description["control_protocol"], "stdio_json_v2")
        self.assertEqual(description["acquisition_role"], "controller_host")
        self.assertNotIn("acquisition_host", description)
        self.assertEqual(started["status"], "RUNNING")
        self.assertEqual(stopped["sample_count"], 20)
        events = [
            json.loads(line)["event"]
            for line in output_stream.getvalue().splitlines()
        ]
        self.assertEqual(
            events,
            ["ACQUISITION_START_REQUEST", "ACQUISITION_STOP_REQUEST"],
        )

    def test_cli_emits_complete_manifest_for_ssh_client(self):
        manifest = {
            "campaign_id": "campaign-123",
            "status": "COMPLETE",
        }
        manager = Mock()
        manager.execute.return_value = manifest
        manager.last_manifest = manifest
        args = SimpleNamespace(
            backend="cpu",
            model="Linear_64_64.pt",
            number_of_cycles=1,
            sleep_time=0.0,
            inferences_per_cycle=None,
            target_burst_seconds=1.0,
            sampling_rate_hz=10.0,
            min_active_samples=10,
            warmup_inferences=1,
            warmup_seconds=0.0,
            warmup_cooldown_seconds=0.0,
            calibration_initial_inferences=1,
            calibration_target_seconds=0.1,
            calibration_repetitions=2,
            calibration_sizing_max_attempts=3,
            calibration_duration_tolerance=0.20,
            max_relative_mad=0.15,
            burst_duration_margin=1.2,
            validation_repetitions=3,
            validation_max_rounds=3,
            validation_safety_margin=1.1,
            validation_cooldown_seconds=0.0,
            clock_sync_exchanges=10,
            max_clock_uncertainty_fraction=0.10,
            max_calibration_inferences=1000,
            leading_idle_seconds=0.0,
            trailing_idle_seconds=0.0,
            safety_margin_seconds=0.0,
            wait_for_acquisition=False,
            stdio_acquisition=True,
            manifest_directory="measurements_jetson",
        )
        parser = Mock()
        parser.parse_args.return_value = args
        runner_module = SimpleNamespace(TorchRunner=object)
        output = io.StringIO()

        with (
            patch.object(
                run_manager_module,
                "build_argument_parser",
                return_value=parser,
            ),
            patch.object(
                run_manager_module,
                "RunManager",
                return_value=manager,
            ) as manager_class,
            patch.dict("sys.modules", {"runner": runner_module}),
            redirect_stdout(output),
        ):
            run_manager_module.main()

        acquisition_controller = manager_class.call_args.kwargs[
            "acquisition_controller"
        ]
        self.assertEqual(
            acquisition_controller.max_clock_uncertainty_fraction,
            0.10,
        )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["event"], "RUN_MANIFEST")
        self.assertEqual(payload["manifest"], manifest)


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
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["input_batch_size"], 1)
            self.assertEqual(manifest["warmup"]["executed_inferences"], 5)
            self.assertEqual(manifest["calibration"]["discarded_batches"], 1)
            self.assertEqual(manifest["plan"]["inferences_per_cycle"], 120)
            self.assertEqual(manifest["plan"]["required_burst_seconds"], 1.2)
            self.assertEqual(manifest["plan"]["burst_duration_margin"], 1.2)
            self.assertTrue(manifest["burst_validation"]["passed"])
            self.assertFalse(manifest["burst_validation"]["adjusted"])
            self.assertEqual(
                manifest["workload_policy"]["parameters"],
                "fresh_per_burst",
            )
            self.assertEqual(
                manifest["workload_policy"]["input"],
                "fresh_per_burst",
            )
            self.assertEqual(
                manifest["measurement"]["total_executed_inferences"], 240
            )
            self.assertEqual(len(manifest["measurement"]["cycles"]), 2)

            # Calls: warm-up, sizing pilot, four calibration batches, three
            # excluded validation bursts, and exactly two measured cycles.
            self.assertEqual(len(runner.calls), 11)
            self.assertEqual(runner.calls[-2:], [120, 120])
            self.assertEqual(runner.burst_preparations, len(runner.calls))
            self.assertEqual(len(set(map(id, runner.parameter_states))), 11)
            self.assertEqual(len(set(map(id, runner.input_states))), 11)

            manifest_path = Path(manifest["manifest_path"])
            self.assertTrue(manifest_path.exists())
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["campaign_id"], manifest["campaign_id"])

    def test_calibration_resizes_batch_after_pilot_regime_shift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = self._manager(
                temp_dir,
                runner_cls=PilotRegimeShiftRunner,
            ).execute()

            attempts = manifest["calibration"]["sizing_attempts"]
            self.assertEqual([attempt["inferences"] for attempt in attempts], [10, 50])
            self.assertTrue(manifest["calibration"]["sizing_converged"])
            self.assertEqual(manifest["calibration"]["batch_inferences"], 50)
            self.assertEqual(manifest["plan"]["inferences_per_cycle"], 600)

    def test_validation_corrects_for_faster_final_count_regime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = self._manager(
                temp_dir,
                runner_cls=FinalCountRegimeShiftRunner,
            ).execute()

            validation = manifest["burst_validation"]
            self.assertTrue(validation["adjusted"])
            self.assertEqual(len(validation["rounds"]), 2)
            self.assertEqual(
                validation["rounds"][0]["inferences_per_burst"],
                120,
            )
            self.assertGreaterEqual(
                validation["final_minimum_burst_seconds"],
                manifest["plan"]["required_burst_seconds"],
            )
            self.assertGreaterEqual(
                manifest["plan"]["inferences_per_cycle"],
                660,
            )
            self.assertTrue(
                all(
                    "UNDER_RESOLVED" not in cycle["quality_flags"]
                    for cycle in manifest["measurement"]["cycles"]
                )
            )

    def test_manual_override_that_is_too_short_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self._manager(temp_dir, inferences_per_cycle=5)
            with self.assertRaisesRegex(ValueError, "minimum active samples"):
                manager.execute()

    def test_unstable_calibration_is_retained_for_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self._manager(
                temp_dir,
                runner_cls=UnstableFakeRunner,
                max_relative_mad=0.01,
            )
            manifest = manager.execute()

            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["quality_status"], "REVIEW")
            self.assertIn("CALIBRATION_UNSTABLE", manifest["quality_flags"])
            self.assertFalse(manifest["calibration"]["is_stable"])
            self.assertEqual(len(manifest["measurement"]["cycles"]), 2)

    def test_burst_boundary_events_are_recorded_without_stdout_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            manager = self._manager(temp_dir, number_of_cycles=1)

            with redirect_stdout(output):
                manager.execute()

            events = [
                json.loads(line)["event"]
                for line in output.getvalue().splitlines()
            ]
            self.assertEqual(
                events,
                [
                    "WARMUP_START",
                    "WARMUP_END",
                    "CALIBRATION_START",
                    "CALIBRATION_END",
                    "READY",
                    "COMPLETE",
                ],
            )

    def test_automated_acquisition_wraps_only_the_capture_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = []
            controller = FakeAcquisitionController(timeline)
            manager = self._manager(
                temp_dir,
                acquisition_controller=controller,
                warmup_cooldown_seconds=5.0,
                leading_idle_seconds=1.0,
                sleep_time=4.0,
                trailing_idle_seconds=2.0,
                safety_margin_seconds=3.0,
                sleep_fn=lambda seconds: timeline.append(("sleep", seconds)),
            )
            write_manifest = manager._write_manifest

            def record_write(manifest, path):
                self.assertFalse(controller.active)
                timeline.append(("write", manifest["status"]))
                write_manifest(manifest, path)

            manager._write_manifest = record_write
            manifest = manager.execute()

            self.assertEqual(manifest["acquisition"]["status"], "COMPLETE")
            self.assertEqual(manifest["quality_status"], "OK")
            self.assertEqual(controller.stop_calls, 1)
            ready_write = timeline.index(("write", "READY"))
            start = timeline.index(("start", None))
            pre_sync = timeline.index(("sync", "pre_acquisition"))
            leading_idle = timeline.index(("sleep", 1.0))
            trailing_idle = timeline.index(("sleep", 2.0))
            safety_margin = timeline.index(("sleep", 3.0))
            stop = timeline.index(("stop", None))
            post_sync = timeline.index(("sync", "post_acquisition"))
            complete_write = timeline.index(("write", "COMPLETE"))
            self.assertLess(ready_write, start)
            self.assertLess(pre_sync, start)
            self.assertLess(start, leading_idle)
            self.assertLess(leading_idle, trailing_idle)
            self.assertLess(trailing_idle, safety_margin)
            self.assertLess(safety_margin, stop)
            self.assertLess(stop, post_sync)
            self.assertLess(stop, complete_write)

    def test_clock_alignment_translates_bursts_into_logger_elapsed_time(self):
        manager = self._manager(
            None,
            sampling_rate_hz=100.0,
            max_clock_uncertainty_fraction=0.10,
        )
        manifest = {
            "acquisition": {
                "started_monotonic_seconds": 1090.0,
                "capture_elapsed_seconds": 30.0,
            },
            "measurement": {
                "cycles": [
                    {
                        "start_event": {"monotonic_seconds": 100.0},
                        "end_event": {"monotonic_seconds": 102.0},
                    }
                ]
            },
        }

        def sync_round(round_name, midpoint, offset, uncertainty):
            return {
                "status": "COMPLETE",
                "round": round_name,
                "method": "minimum_round_trip",
                "selected_sample": {
                    "runner_midpoint_monotonic_seconds": midpoint,
                    "controller_minus_runner_seconds": offset,
                    "round_trip_seconds": 2 * uncertainty,
                    "uncertainty_seconds": uncertainty,
                },
            }

        alignment = manager._finalize_clock_alignment(
            manifest,
            sync_round("pre_acquisition", 90.0, 1000.0, 0.0005),
            sync_round("post_acquisition", 110.0, 1000.1, 0.0005),
        )

        cycle = manifest["measurement"]["cycles"][0]
        self.assertTrue(alignment["classification_eligible"])
        self.assertIsNone(alignment["fallback_reason"])
        self.assertAlmostEqual(
            cycle["start_event"]["aligned_elapsed_seconds"], 10.05
        )
        self.assertAlmostEqual(
            cycle["end_event"]["aligned_elapsed_seconds"], 12.06
        )

        alignment = manager._finalize_clock_alignment(
            manifest,
            sync_round("pre_acquisition", 90.0, 1000.0, 0.0011),
            sync_round("post_acquisition", 110.0, 1000.1, 0.0011),
        )
        self.assertFalse(alignment["classification_eligible"])
        self.assertEqual(
            alignment["fallback_reason"], "CLOCK_UNCERTAINTY_EXCEEDED"
        )

    def test_acquisition_start_failure_completes_workload_for_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = []
            controller = FakeAcquisitionController(
                timeline,
                start_error=RuntimeError("serial board unavailable"),
            )
            manifest = self._manager(
                temp_dir,
                acquisition_controller=controller,
            ).execute()

            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["quality_status"], "REVIEW")
            self.assertIn("ACQUISITION_FAILED", manifest["quality_flags"])
            self.assertEqual(manifest["acquisition"]["status"], "FAILED")
            self.assertEqual(
                manifest["measurement"]["total_executed_inferences"],
                240,
            )
            self.assertEqual(controller.stop_calls, 1)

    def test_workload_failure_stops_acquisition_before_failed_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = []
            controller = FakeAcquisitionController(timeline)
            manager = self._manager(
                temp_dir,
                runner_cls=FailingMeasuredRunner,
                acquisition_controller=controller,
            )
            write_manifest = manager._write_manifest

            def record_write(manifest, path):
                self.assertFalse(controller.active)
                timeline.append(("write", manifest["status"]))
                write_manifest(manifest, path)

            manager._write_manifest = record_write

            with self.assertRaisesRegex(RuntimeError, "measured workload failed"):
                manager.execute()

            manifest_path = next(Path(temp_dir).glob("*.json"))
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "FAILED")
            self.assertEqual(len(persisted["measurement"]["cycles"]), 1)
            self.assertEqual(controller.stop_calls, 1)
            self.assertLess(
                timeline.index(("stop", None)),
                timeline.index(("write", "FAILED")),
            )


if __name__ == "__main__":
    unittest.main()
