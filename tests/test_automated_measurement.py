import json
import queue
import subprocess
import sys
import tempfile
import textwrap
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


class CapturingInput:
    def __init__(self):
        self.lines = []

    def write(self, line):
        self.lines.append(line)
        return len(line)

    def flush(self):
        return None


class BrokenStream:
    def __iter__(self):
        return self

    def __next__(self):
        raise OSError("SSH stdout failed")


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


class FakeSshProcess:
    def __init__(self, command, events, returncode=0):
        self.command = command
        self.stdin = CapturingInput()
        self.stdout = QueueStream()
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        for event in events:
            self.stdout.push(event)
        self.stdout.finish()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeSshProcessFactory:
    def __init__(self, events, returncode=0):
        self.events = events
        self.returncode = returncode
        self.process = None

    def __call__(self, command, **_kwargs):
        self.process = FakeSshProcess(
            command,
            self.events,
            returncode=self.returncode,
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


class SshExperimentControllerTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        settings = {
            "backend": "cuda",
            "model": "Models/CPU/Linear/Linear_64_64.pt",
            "number_of_cycles": 2,
            "sleep_time": 0.7,
            "inferences_per_cycle": None,
            "target_burst_seconds": 0.0,
            "sampling_rate_hz": 100.0,
            "min_active_samples": 50,
            "warmup_inferences": 20,
            "warmup_seconds": 0.0,
            "warmup_cooldown_seconds": None,
            "calibration_initial_inferences": 10,
            "calibration_target_seconds": 0.5,
            "calibration_repetitions": 4,
            "calibration_sizing_max_attempts": 3,
            "calibration_duration_tolerance": 0.20,
            "max_relative_mad": 0.15,
            "burst_duration_margin": 1.2,
            "validation_repetitions": 3,
            "validation_max_rounds": 3,
            "validation_safety_margin": 1.1,
            "validation_cooldown_seconds": None,
            "clock_sync_exchanges": 10,
            "max_clock_uncertainty_fraction": 0.10,
            "max_calibration_inferences": 1_000_000,
            "leading_idle_seconds": 5.0,
            "trailing_idle_seconds": 5.0,
            "safety_margin_seconds": 2.0,
        }
        settings.update(overrides)
        return SimpleNamespace(**settings)

    @staticmethod
    def _events(campaign_id, manifest):
        return [
            acquisition_event("WARMUP_START", campaign_id),
            acquisition_event(
                "READY",
                campaign_id,
                sampling_rate_hz=100.0,
                capture_seconds=15.0,
                capture_samples=1500,
            ),
            acquisition_event(
                "ACQUISITION_START_REQUEST",
                campaign_id,
                sampling_rate_hz=100.0,
                capture_seconds=15.0,
                capture_samples=1500,
            ),
            acquisition_event("BURST_START", campaign_id, cycle=1),
            acquisition_event("BURST_END", campaign_id, cycle=1),
            acquisition_event("ACQUISITION_STOP_REQUEST", campaign_id),
            acquisition_event("COMPLETE", campaign_id),
            acquisition_event(
                "RUN_MANIFEST",
                campaign_id,
                manifest=manifest,
            ),
        ]

    def _controller(self, acquisition, factory, **overrides):
        settings = {
            "runner_host": "bench@inference-host",
            "remote_directory": "/srv/benchmark",
            "remote_python": "python3",
            "remote_manifest_directory": "measurements_board",
            "ssh_options": ["ProxyJump=relay@jump-host"],
            "acquisition_controller": acquisition,
            "startup_timeout_seconds": 1.0,
            "process_factory": factory,
        }
        settings.update(overrides)
        return automated.SshExperimentController(**settings)

    def test_remote_run_controls_local_logger_and_returns_manifest(self):
        campaign_id = "campaign-remote-123"
        manifest = {
            "campaign_id": campaign_id,
            "status": "COMPLETE",
            "quality_status": "OK",
            "manifest_path": "/home/jetson/measurements_jetson/campaign.json",
            "acquisition": {"status": "COMPLETE", "sample_count": 25},
            "measurement": {"cycles": []},
        }
        factory = FakeSshProcessFactory(self._events(campaign_id, manifest))
        acquisition = Mock()
        acquisition.describe.return_value = {
            "csv_path": f"/tmp/{campaign_id}.csv",
            "sampling_rate_hz": 100.0,
        }
        acquisition.start.return_value = {"status": "RUNNING", "sample_count": 1}
        acquisition.check_health.return_value = None
        acquisition.stop.return_value = {"status": "COMPLETE", "sample_count": 25}
        controller = self._controller(acquisition, factory)

        returned = controller.execute(self._args())

        self.assertEqual(returned, manifest)
        acquisition.describe.assert_called_once()
        acquisition.start.assert_called_once_with()
        acquisition.stop.assert_called_once_with()
        replies = [
            json.loads(line)
            for line in factory.process.stdin.lines
        ]
        self.assertEqual(
            [reply["command"] for reply in replies],
            ["ACQUISITION_STARTED", "ACQUISITION_STOPPED"],
        )
        self.assertTrue(all(
            reply["campaign_id"] == campaign_id for reply in replies
        ))
        command = factory.process.command
        self.assertEqual(command[0:4], ["ssh", "-T", "-o", "BatchMode=yes"])
        self.assertIn("ProxyJump=relay@jump-host", command)
        self.assertEqual(command[-2], "bench@inference-host")
        self.assertIn("--stdio_acquisition", command[-1])
        self.assertIn("--backend cuda", command[-1])
        self.assertIn("--burst-duration-margin 1.2", command[-1])
        self.assertIn("--validation-repetitions 3", command[-1])
        self.assertIn("--validation-safety-margin 1.1", command[-1])
        self.assertIn("--clock-sync-exchanges 10", command[-1])
        self.assertIn("--max-clock-uncertainty-fraction 0.1", command[-1])

    def test_clock_sync_request_is_timestamped_and_echoes_request_id(self):
        factory = FakeSshProcessFactory([])
        monotonic_values = iter([20.002, 20.003])
        controller = self._controller(
            Mock(),
            factory,
            monotonic_fn=lambda: next(monotonic_values),
        )
        controller.process = factory(["ssh"])

        controller._handle_event(
            acquisition_event(
                "CLOCK_SYNC_REQUEST",
                "campaign-sync",
                request_id="pre:1",
                round="pre",
            ),
            "",
        )

        response = json.loads(controller.process.stdin.lines[0])
        self.assertEqual(response["command"], "CLOCK_SYNC_RESPONSE")
        self.assertEqual(response["request_id"], "pre:1")
        self.assertEqual(
            response["result"]["controller_received_monotonic_seconds"],
            20.002,
        )
        self.assertEqual(
            response["result"]["controller_sent_monotonic_seconds"],
            20.003,
        )

    def test_local_logger_start_failure_is_reported_to_jetson(self):
        campaign_id = "campaign-remote-failed"
        manifest = {
            "campaign_id": campaign_id,
            "status": "COMPLETE",
            "quality_status": "REVIEW",
            "acquisition": {"status": "FAILED"},
            "measurement": {"cycles": []},
        }
        factory = FakeSshProcessFactory(self._events(campaign_id, manifest))
        acquisition = Mock()
        acquisition.describe.return_value = {
            "csv_path": f"/tmp/{campaign_id}.csv",
        }
        acquisition.start.side_effect = RuntimeError("TI-SCB unavailable")
        acquisition.stop.return_value = None
        controller = self._controller(acquisition, factory)

        returned = controller.execute(self._args())

        self.assertEqual(returned["quality_status"], "REVIEW")
        replies = [json.loads(line) for line in factory.process.stdin.lines]
        self.assertEqual(replies[0]["result"]["status"], "FAILED")
        self.assertIn("TI-SCB unavailable", replies[0]["result"]["failure"]["message"])
        self.assertEqual(replies[1]["result"]["status"], "FAILED")

    def test_runtime_logger_failure_survives_successful_cleanup(self):
        acquisition = Mock()
        controller = self._controller(
            acquisition,
            FakeSshProcessFactory([]),
        )
        controller.acquisition_result = {
            "status": "RUNNING",
            "csv_path": "/tmp/campaign.csv",
        }
        acquisition.check_health.side_effect = RuntimeError("serial disconnected")
        acquisition.stop.return_value = {
            "status": "COMPLETE",
            "sample_count": 8,
        }

        controller._check_local_acquisition()
        result = controller._stop_local_acquisition()

        self.assertEqual(result["status"], "FAILED")
        self.assertIn("serial disconnected", result["failure"]["message"])

    def test_remote_manifest_is_saved_next_to_local_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = {
                "campaign_id": "campaign-remote-123",
                "manifest_path": (
                    "/home/jetson/measurements_jetson/campaign-remote-123.json"
                ),
                "acquisition": {
                    "csv_path": str(
                        Path(temp_dir) / "campaign-remote-123.csv"
                    )
                },
            }

            path = automated.persist_local_manifest(manifest, temp_dir)
            automated.persist_local_manifest(manifest, temp_dir)
            persisted = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(path.name, "campaign-remote-123.json")
            self.assertEqual(persisted["manifest_path"], str(path.resolve()))
            self.assertNotIn("remote_manifest_path", persisted)

    def test_manifest_is_fetched_when_stream_ends_before_manifest_event(self):
        campaign_id = "campaign-fallback"
        failed_manifest = {
            "campaign_id": campaign_id,
            "status": "FAILED",
            "acquisition": {"status": "FAILED"},
        }
        factory = FakeSshProcessFactory(
            [acquisition_event("WARMUP_START", campaign_id)],
            returncode=1,
        )
        fetch = Mock(
            return_value=SimpleNamespace(
                returncode=0,
                stdout=json.dumps(failed_manifest),
            )
        )
        controller = self._controller(Mock(), factory, run_factory=fetch)

        returned = controller.execute(self._args())

        self.assertEqual(returned, failed_manifest)
        fetch.assert_called_once()
        self.assertIn(
            "measurements_board/campaign-fallback.json",
            fetch.call_args.args[0][-1],
        )

    def test_remote_manifest_is_deleted_with_same_ssh_route(self):
        run = Mock(return_value=SimpleNamespace(returncode=0))
        controller = self._controller(Mock(), Mock(), run_factory=run)

        controller.delete_remote_manifest("campaign-cleanup")

        command = run.call_args.args[0]
        self.assertIn("ProxyJump=relay@jump-host", command)
        self.assertEqual(command[-2], "bench@inference-host")
        self.assertIn(
            "rm -f -- measurements_board/campaign-cleanup.json",
            command[-1],
        )

    def test_remote_cleanup_requires_safe_campaign_id(self):
        run = Mock()
        controller = self._controller(Mock(), Mock(), run_factory=run)

        with self.assertRaisesRegex(
            automated.RemoteExperimentError,
            "Invalid campaign ID",
        ):
            controller.delete_remote_manifest("../other-file")

        run.assert_not_called()

    def test_reader_thread_failure_is_reported_directly(self):
        process = FakeSshProcess(["ssh"], [], returncode=1)
        process.stdout = BrokenStream()
        factory = Mock(return_value=process)
        controller = self._controller(Mock(), factory)

        with self.assertRaisesRegex(
            automated.RemoteExperimentError,
            "SSH stdout failed",
        ):
            controller.execute(self._args())

    def test_failed_manifest_with_zero_ssh_exit_is_rejected(self):
        campaign_id = "campaign-status-mismatch"
        manifest = {
            "campaign_id": campaign_id,
            "status": "FAILED",
        }
        factory = FakeSshProcessFactory(
            [
                acquisition_event("WARMUP_START", campaign_id),
                acquisition_event(
                    "RUN_MANIFEST",
                    campaign_id,
                    manifest=manifest,
                ),
            ],
            returncode=0,
        )
        controller = self._controller(Mock(), factory)

        with self.assertRaisesRegex(
            automated.RemoteExperimentError,
            "failed manifest with SSH exit code 0",
        ):
            controller.execute(self._args())

    def test_real_subprocess_completes_bidirectional_handshake(self):
        remote_program = textwrap.dedent(
            """
            import json
            import sys
            import time

            campaign_id = "campaign-real-pipes"

            def emit(event, **fields):
                print(json.dumps({
                    "event": event,
                    "campaign_id": campaign_id,
                    "monotonic_seconds": time.monotonic(),
                    "wall_time_utc": "2026-07-23T12:00:00+00:00",
                    **fields,
                }), flush=True)

            def synchronize(round_name):
                request_id = round_name + ":1"
                emit(
                    "CLOCK_SYNC_REQUEST",
                    round=round_name,
                    request_id=request_id,
                )
                response = json.loads(sys.stdin.readline())
                assert response["command"] == "CLOCK_SYNC_RESPONSE"
                assert response["request_id"] == request_id

            synchronize("pre_acquisition")
            emit(
                "ACQUISITION_START_REQUEST",
                sampling_rate_hz=100.0,
                capture_seconds=1.0,
                capture_samples=100,
            )
            started = json.loads(sys.stdin.readline())
            assert started["command"] == "ACQUISITION_STARTED"
            emit("BURST_START", cycle=1)
            emit("BURST_END", cycle=1)
            emit("ACQUISITION_STOP_REQUEST")
            stopped = json.loads(sys.stdin.readline())
            assert stopped["command"] == "ACQUISITION_STOPPED"
            synchronize("post_acquisition")
            manifest = {
                "campaign_id": campaign_id,
                "status": "COMPLETE",
                "quality_status": "OK",
                "acquisition": {"status": "COMPLETE", "sample_count": 10},
                "measurement": {"cycles": []},
            }
            emit("COMPLETE")
            emit("RUN_MANIFEST", manifest=manifest)
            """
        )

        def process_factory(_command, **kwargs):
            return subprocess.Popen(
                [sys.executable, "-u", "-c", remote_program],
                **kwargs,
            )

        acquisition = Mock()
        acquisition.describe.return_value = {
            "csv_path": "/tmp/campaign-real-pipes.csv",
        }
        acquisition.start.return_value = {"status": "RUNNING"}
        acquisition.check_health.return_value = None
        acquisition.stop.return_value = {
            "status": "COMPLETE",
            "sample_count": 10,
        }
        controller = self._controller(
            acquisition,
            process_factory,
        )

        manifest = controller.execute(self._args())

        self.assertEqual(manifest["campaign_id"], "campaign-real-pipes")
        self.assertEqual(manifest["status"], "COMPLETE")
        acquisition.start.assert_called_once_with()
        acquisition.stop.assert_called_once_with()


class ArgumentParserTests(unittest.TestCase):
    def test_measurement_defaults_match_campaign_profile(self):
        args = automated.build_argument_parser().parse_args([
            "--backend", "cuda",
            "--model", "Models/CUDA/Linear/Linear_64_64.pt",
            "--output-directory", "measurements/runs/test",
            "--shunt-ohms", "0.012",
            "--max-expected-current-a", "5.0",
        ])

        self.assertEqual(args.number_of_cycles, 100)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.sleep_time, 3.0)
        self.assertEqual(args.target_burst_seconds, 0.0)
        self.assertEqual(args.sampling_rate_hz, 100.0)
        self.assertEqual(args.min_active_samples, 100)
        self.assertEqual(args.warmup_inferences, 20)
        self.assertEqual(args.calibration_initial_inferences, 20)
        self.assertEqual(args.calibration_target_seconds, 1.0)
        self.assertEqual(args.calibration_repetitions, 8)
        self.assertEqual(args.trailing_idle_seconds, 30.0)


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
            connection_config=None,
            runner_host=None,
            jump_host=None,
            remote_directory=".",
            remote_python="python3",
            remote_manifest_directory="measurements_jetson",
            ssh_option=[],
            ssh_connect_timeout_s=10.0,
            remote_startup_timeout_s=60.0,
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
            patch.object(automated, "build_local_manager", return_value=manager),
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

    def test_jump_host_builds_required_ssh_options(self):
        self.assertEqual(
            automated.build_ssh_options(
                "relay@jump-host",
                10.0,
                [],
            ),
            [
                "ProxyJump=relay@jump-host",
                "ConnectTimeout=10",
                "ConnectionAttempts=1",
            ],
        )

    def test_connection_config_supplies_remote_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "hosts.json"
            config_path.write_text(
                json.dumps(
                    {
                        "runner_host": "bench@inference-host",
                        "jump_host": "relay@jump-host",
                        "remote_directory": "/srv/benchmark",
                        "remote_python": "/srv/venv/bin/python",
                        "remote_manifest_directory": "manifests",
                        "ssh_connect_timeout_s": 7.0,
                        "ssh_options": ["ServerAliveInterval=15"],
                    }
                ),
                encoding="utf-8",
            )
            args = self._args()
            args.connection_config = config_path
            args.remote_directory = None
            args.remote_python = None
            args.remote_manifest_directory = None
            args.ssh_connect_timeout_s = None

            configured = automated.apply_connection_config(args)

            self.assertEqual(configured.runner_host, "bench@inference-host")
            self.assertEqual(configured.jump_host, "relay@jump-host")
            self.assertEqual(configured.remote_directory, "/srv/benchmark")
            self.assertEqual(configured.remote_python, "/srv/venv/bin/python")
            self.assertEqual(configured.ssh_connect_timeout_s, 7.0)
            self.assertEqual(
                configured.ssh_option,
                ["ServerAliveInterval=15"],
            )

    def test_cli_host_overrides_connection_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "hosts.json"
            config_path.write_text(
                json.dumps({"runner_host": "config@inference-host"}),
                encoding="utf-8",
            )
            args = self._args()
            args.connection_config = config_path
            args.runner_host = "override@inference-host"

            configured = automated.apply_connection_config(args)

            self.assertEqual(
                configured.runner_host,
                "override@inference-host",
            )

    def test_main_uses_ssh_runner_and_persists_remote_manifest(self):
        args = self._args()
        args.runner_host = "bench@inference-host"
        args.jump_host = "relay@jump-host"
        manifest = {
            "campaign_id": "campaign-remote-main",
            "status": "COMPLETE",
            "quality_status": "OK",
            "acquisition": {"status": "COMPLETE", "sample_count": 12},
            "measurement": {"cycles": []},
        }
        parser = Mock()
        parser.parse_args.return_value = args
        local_acquisition = Mock()
        remote = Mock()
        remote.execute.return_value = manifest

        with (
            patch.object(automated, "build_argument_parser", return_value=parser),
            patch.object(
                automated,
                "Ina226ProcessController",
                return_value=local_acquisition,
            ),
            patch.object(
                automated,
                "SshExperimentController",
                return_value=remote,
            ) as remote_cls,
            patch.object(automated, "persist_local_manifest") as persist,
            patch.object(
                automated,
                "build_local_manager",
                side_effect=AssertionError("local runner must not be used"),
            ),
            patch("builtins.print") as output,
        ):
            exit_code = automated.main()

        self.assertEqual(exit_code, 0)
        remote.execute.assert_called_once_with(args)
        persist.assert_called_once_with(manifest, args.output_directory)
        remote.delete_remote_manifest.assert_called_once_with(
            "campaign-remote-main"
        )
        self.assertIs(
            remote_cls.call_args.kwargs["acquisition_controller"],
            local_acquisition,
        )
        self.assertEqual(
            remote_cls.call_args.kwargs["ssh_options"],
            [
                "ProxyJump=relay@jump-host",
                "ConnectTimeout=10",
                "ConnectionAttempts=1",
            ],
        )
        report = json.loads(output.call_args.args[0])
        self.assertEqual(report["campaign_id"], "campaign-remote-main")
        self.assertEqual(manifest["orchestration"]["mode"], "ssh")
        self.assertEqual(
            manifest["orchestration"]["runner_role"],
            "inference_host",
        )
        self.assertEqual(
            manifest["orchestration"]["acquisition_role"],
            "controller_host",
        )
        self.assertNotIn("runner_host", manifest["orchestration"])
        self.assertNotIn("acquisition_host", manifest["orchestration"])

    def test_remote_cleanup_runs_only_after_local_manifest_is_durable(self):
        args = self._args()
        args.keep_remote_manifest = False
        manifest = {"campaign_id": "campaign-cleanup"}
        remote = Mock()
        calls = []

        with patch.object(
            automated,
            "persist_local_manifest",
            side_effect=lambda *_: calls.append("persist") or Path("local.json"),
        ):
            remote.delete_remote_manifest.side_effect = lambda *_: calls.append(
                "delete"
            )
            path = automated.prepare_remote_manifest(manifest, args, remote)

        self.assertEqual(path, Path("local.json"))
        self.assertEqual(calls, ["persist", "delete"])

    def test_remote_manifest_is_retained_when_requested(self):
        args = self._args()
        args.keep_remote_manifest = True
        manifest = {"campaign_id": "campaign-retained"}
        remote = Mock()

        with patch.object(
            automated,
            "persist_local_manifest",
            return_value=Path("local.json"),
        ):
            automated.prepare_remote_manifest(manifest, args, remote)

        remote.delete_remote_manifest.assert_not_called()

    def test_remote_manifest_is_retained_if_local_persistence_fails(self):
        args = self._args()
        args.keep_remote_manifest = False
        manifest = {"campaign_id": "campaign-recovery"}
        remote = Mock()

        with (
            patch.object(
                automated,
                "persist_local_manifest",
                side_effect=OSError("disk full"),
            ),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            automated.prepare_remote_manifest(manifest, args, remote)

        remote.delete_remote_manifest.assert_not_called()

    def test_remote_cleanup_failure_keeps_successful_local_copy(self):
        args = self._args()
        args.keep_remote_manifest = False
        manifest = {"campaign_id": "campaign-recovery"}
        remote = Mock()
        remote.delete_remote_manifest.side_effect = automated.RemoteExperimentError(
            "cleanup SSH failed"
        )

        with (
            patch.object(
                automated,
                "persist_local_manifest",
                return_value=Path("local.json"),
            ),
            patch("builtins.print") as output,
        ):
            path = automated.prepare_remote_manifest(manifest, args, remote)

        self.assertEqual(path, Path("local.json"))
        self.assertIn("remote recovery copy retained", output.call_args.args[0])

    def test_main_persists_received_manifest_when_ssh_reports_error(self):
        args = self._args()
        args.runner_host = "bench@inference-host"
        manifest = {
            "campaign_id": "campaign-remote-error",
            "status": "FAILED",
            "acquisition": {"status": "FAILED"},
        }
        parser = Mock()
        parser.parse_args.return_value = args
        remote = Mock()
        remote.manifest = manifest
        remote.execute.side_effect = automated.RemoteExperimentError(
            "SSH transport failed"
        )

        with (
            patch.object(automated, "build_argument_parser", return_value=parser),
            patch.object(automated, "Ina226ProcessController"),
            patch.object(
                automated,
                "SshExperimentController",
                return_value=remote,
            ),
            patch.object(automated, "prepare_remote_manifest") as persist,
            patch("builtins.print") as output,
        ):
            exit_code = automated.main()

        self.assertEqual(exit_code, 1)
        persist.assert_called_once_with(manifest, args, remote)
        report = json.loads(output.call_args.args[0])
        self.assertEqual(report["campaign_id"], "campaign-remote-error")


if __name__ == "__main__":
    unittest.main()