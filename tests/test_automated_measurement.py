import json
import queue
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import automated_measurement as automated


def acquisition_event(event, campaign_id, **fields):
    return {
        "event": event,
        "campaign_id": campaign_id,
        "monotonic_seconds": 123.0,
        "wall_time_utc": "2026-07-23T12:00:00+00:00",
        **fields,
    }


class QueueStream:
    def __init__(self):
        self.lines = queue.Queue()

    def push(self, payload):
        self.lines.put(json.dumps(payload) + "\n")

    def finish(self):
        self.lines.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.lines.get()
        if line is None:
            raise StopIteration
        return line


class FakeProcess:
    def __init__(self, command, campaign_id, ready=True, exit_on_terminate=True):
        self.command = command
        self.campaign_id = campaign_id
        self.stdout = QueueStream()
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.exit_on_terminate = exit_on_terminate
        if ready:
            self.stdout.push(
                acquisition_event(
                    "ACQUISITION_READY",
                    campaign_id,
                    status="RUNNING",
                    csv_path="/tmp/campaign.csv",
                    sample_count=1,
                )
            )

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.terminate_calls += 1
        if self.exit_on_terminate:
            self.stdout.push(
                acquisition_event(
                    "ACQUISITION_COMPLETE",
                    self.campaign_id,
                    status="COMPLETE",
                    csv_path="/tmp/campaign.csv",
                    sample_count=25,
                    capture_elapsed_seconds=2.5,
                    deadline_misses=0,
                )
            )
            self.returncode = 0
            self.stdout.finish()

    def kill(self):
        self.killed = True
        self.kill_calls += 1
        self.returncode = -9
        self.stdout.finish()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode

    def exit_unexpectedly(self, returncode=1):
        self.returncode = returncode
        self.stdout.finish()


class FakeProcessFactory:
    def __init__(self, ready=True, exit_on_terminate=True):
        self.ready = ready
        self.exit_on_terminate = exit_on_terminate
        self.process = None

    def __call__(self, command, **_kwargs):
        campaign_id = command[command.index("--campaign-id") + 1]
        self.process = FakeProcess(
            command,
            campaign_id,
            ready=self.ready,
            exit_on_terminate=self.exit_on_terminate,
        )
        return self.process


class ProcessControllerTests(unittest.TestCase):
    def _controller(self, output_directory, factory, **overrides):
        settings = {
            "output_directory": output_directory,
            "port": "/dev/serial/by-id/ti-scb",
            "baud": 115200,
            "address": 0x40,
            "shunt_ohms": 0.012,
            "max_expected_current_a": 5.0,
            "sampling_rate_hz": 10.0,
            "serial_timeout_seconds": 0.01,
            "startup_timeout_seconds": 0.05,
            "stop_timeout_seconds": 0.05,
            "process_factory": factory,
        }
        settings.update(overrides)
        return automated.Ina226ProcessController(**settings)

    def test_same_stem_command_and_clean_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = FakeProcessFactory()
            controller = self._controller(temp_dir, factory)
            description = controller.describe(
                "campaign-123",
                {"sampling_rate_hz": 10.0},
            )

            self.assertEqual(
                Path(description["csv_path"]).name,
                "campaign-123.csv",
            )
            started = controller.start()
            self.assertEqual(started["status"], "RUNNING")
            self.assertIsNone(controller.check_health())
            completed = controller.stop()

            self.assertEqual(completed["status"], "COMPLETE")
            self.assertEqual(completed["sample_count"], 25)
            self.assertTrue(factory.process.terminated)
            self.assertEqual(controller.stop(), completed)
            command = factory.process.command
            self.assertEqual(
                float(command[command.index("--interval-ms") + 1]),
                100.0,
            )
            self.assertNotIn("--samples", command)
            self.assertNotIn("--duration-s", command)
            self.assertNotIn("--overwrite", command)

    def test_unexpected_child_exit_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = FakeProcessFactory()
            controller = self._controller(temp_dir, factory)
            controller.describe("campaign-123", {"sampling_rate_hz": 10.0})
            controller.start()
            factory.process.exit_unexpectedly()

            with self.assertRaisesRegex(
                automated.AcquisitionProcessError,
                "exited unexpectedly",
            ):
                controller.check_health()

    def test_startup_timeout_terminates_child(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = FakeProcessFactory(ready=False)
            controller = self._controller(
                temp_dir,
                factory,
                startup_timeout_seconds=0.01,
            )
            controller.describe("campaign-123", {"sampling_rate_hz": 10.0})

            with self.assertRaisesRegex(
                automated.AcquisitionProcessError,
                "Timed out",
            ):
                controller.start()
            self.assertTrue(factory.process.terminated)
            self.assertIsNone(controller.stop())

    def test_stop_timeout_kills_child_and_fails_capture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = FakeProcessFactory(exit_on_terminate=False)
            controller = self._controller(
                temp_dir,
                factory,
                stop_timeout_seconds=0.01,
            )
            controller.describe("campaign-123", {"sampling_rate_hz": 10.0})
            controller.start()

            with self.assertRaisesRegex(
                automated.AcquisitionProcessError,
                "did not stop cleanly",
            ):
                controller.stop()
            self.assertTrue(factory.process.killed)
            with self.assertRaisesRegex(
                automated.AcquisitionProcessError,
                "did not stop cleanly",
            ):
                controller.stop()
            self.assertEqual(factory.process.terminate_calls, 1)
            self.assertEqual(factory.process.kill_calls, 1)


class CommandResultTests(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            backend="cpu",
            model="Linear_64_64.pt",
            output_directory=Path("measurements/runs"),
            port="/dev/ttyACM0",
            baud=115200,
            address=0x40,
            shunt_ohms=0.012,
            max_expected_current_a=5.0,
            timeout_s=2.0,
            acquisition_startup_timeout_s=30.0,
            acquisition_stop_timeout_s=None,
            number_of_cycles=2,
            sleep_time=0.0,
            inferences_per_cycle=None,
            target_burst_seconds=1.0,
            sampling_rate_hz=10.0,
            min_active_samples=10,
            warmup_inferences=5,
            warmup_seconds=0.0,
            warmup_cooldown_seconds=0.0,
            calibration_initial_inferences=2,
            calibration_target_seconds=0.1,
            calibration_repetitions=4,
            max_relative_mad=0.15,
            max_calibration_inferences=1000,
            leading_idle_seconds=0.0,
            trailing_idle_seconds=0.0,
            safety_margin_seconds=0.0,
        )

    def _run_main(self, execute_result=None, execute_error=None):
        args = self._args()
        manifest = execute_result or {
            "campaign_id": "campaign-123",
            "status": "FAILED",
            "manifest_path": "/tmp/campaign-123.json",
            "acquisition": {
                "status": "NOT_STARTED",
                "csv_path": "/tmp/campaign-123.csv",
            },
        }
        manager = Mock()
        manager.last_manifest = manifest
        if execute_error is None:
            manager.execute.return_value = manifest
        else:
            manager.execute.side_effect = execute_error

        parser = Mock()
        parser.parse_args.return_value = args
        with (
            patch.object(automated, "build_argument_parser", return_value=parser),
            patch.object(automated, "Ina226ProcessController"),
            patch.object(automated, "select_runner", return_value=object),
            patch.object(automated, "RunManager", return_value=manager),
            patch("builtins.print") as output,
        ):
            exit_code = automated.main()
        report = json.loads(output.call_args.args[0])
        return exit_code, report

    def test_main_exit_codes_distinguish_all_outcomes(self):
        clean_manifest = {
            "campaign_id": "campaign-clean",
            "status": "COMPLETE",
            "quality_status": "OK",
            "acquisition": {"status": "COMPLETE", "sample_count": 10},
            "measurement": {"cycles": []},
        }
        degraded_manifest = {
            "campaign_id": "campaign-review",
            "status": "COMPLETE",
            "quality_status": "REVIEW",
            "acquisition": {"status": "FAILED"},
            "measurement": {"cycles": []},
        }

        clean_code, clean_report = self._run_main(clean_manifest)
        degraded_code, degraded_report = self._run_main(degraded_manifest)
        failed_code, failed_report = self._run_main(
            execute_error=RuntimeError("model failed")
        )
        interrupted_code, interrupted_report = self._run_main(
            execute_error=KeyboardInterrupt()
        )

        self.assertEqual((clean_code, clean_report["exit_code"]), (0, 0))
        self.assertEqual(
            (degraded_code, degraded_report["exit_code"]),
            (2, 2),
        )
        self.assertEqual((failed_code, failed_report["exit_code"]), (1, 1))
        self.assertEqual(
            (interrupted_code, interrupted_report["exit_code"]),
            (130, 130),
        )

    def test_select_runner_reports_missing_backend_dependency(self):
        unavailable_runner_module = SimpleNamespace()
        with patch.dict(sys.modules, {"runner": unavailable_runner_module}):
            with self.assertRaisesRegex(RuntimeError, "require PyTorch"):
                automated.select_runner("cpu")
            with self.assertRaisesRegex(RuntimeError, "requires tflite_runtime"):
                automated.select_runner("tpu")


if __name__ == "__main__":
    unittest.main()