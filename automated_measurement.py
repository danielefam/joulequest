#!/usr/bin/env python3
"""Run one model campaign with unattended INA226 acquisition."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from run_manager import RunManager


class AcquisitionProcessError(RuntimeError):
    """Raised when the INA226 logger cannot provide a valid capture."""


class Ina226ProcessController:
    def __init__(
        self,
        output_directory,
        port,
        baud,
        address,
        shunt_ohms,
        max_expected_current_a,
        sampling_rate_hz,
        serial_timeout_seconds,
        startup_timeout_seconds,
        stop_timeout_seconds=None,
        python_executable=None,
        logger_script=None,
        process_factory=subprocess.Popen,
        monotonic_fn=time.monotonic,
    ):
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if startup_timeout_seconds <= 0:
            raise ValueError("acquisition startup timeout must be positive")
        if stop_timeout_seconds is not None and stop_timeout_seconds <= 0:
            raise ValueError("acquisition stop timeout must be positive")

        self.output_directory = Path(output_directory)
        self.port = port
        self.baud = baud
        self.address = address
        self.shunt_ohms = shunt_ohms
        self.max_expected_current_a = max_expected_current_a
        self.sampling_rate_hz = sampling_rate_hz
        self.interval_ms = 1000.0 / sampling_rate_hz
        self.serial_timeout_seconds = serial_timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.stop_timeout_seconds = (
            4 * serial_timeout_seconds + self.interval_ms / 1000.0 + 1.0
            if stop_timeout_seconds is None
            else stop_timeout_seconds
        )
        self.python_executable = python_executable or sys.executable
        self.logger_script = Path(logger_script or Path(__file__).with_name(
            "ina226_serial_logger.py"
        ))
        self.process_factory = process_factory
        self.monotonic_fn = monotonic_fn

        self.campaign_id = None
        self.csv_path = None
        self.process = None
        self.reader_thread = None
        self.events = queue.Queue()
        self.ready_event = None
        self.completion_event = None
        self.failure_event = None
        self.stopping = False
        self.stop_result = None
        self.stop_error = None

    def describe(self, campaign_id, plan):
        if plan["sampling_rate_hz"] != self.sampling_rate_hz:
            raise ValueError(
                "RunManager and INA226 sampling rates must be identical"
            )
        self.campaign_id = campaign_id
        self.csv_path = (
            self.output_directory / f"{campaign_id}.csv"
        ).resolve()
        return {
            "status": "PENDING",
            "csv_path": str(self.csv_path),
            "requested_port": self.port,
            "baud": self.baud,
            "i2c_address": self.address,
            "shunt_ohms": self.shunt_ohms,
            "max_expected_current_a": self.max_expected_current_a,
            "sampling_rate_hz": self.sampling_rate_hz,
            "requested_interval_ms": self.interval_ms,
            "serial_timeout_seconds": self.serial_timeout_seconds,
        }

    def _command(self):
        if self.campaign_id is None or self.csv_path is None:
            raise RuntimeError("describe() must be called before start()")
        command = [
            self.python_executable,
            str(self.logger_script),
            "--output",
            str(self.csv_path),
            "--baud",
            str(self.baud),
            "--address",
            hex(self.address),
            "--shunt-ohms",
            str(self.shunt_ohms),
            "--max-expected-current-a",
            str(self.max_expected_current_a),
            "--interval-ms",
            str(self.interval_ms),
            "--timeout-s",
            str(self.serial_timeout_seconds),
            "--campaign-id",
            self.campaign_id,
            "--json-events",
        ]
        if self.port is not None:
            command.extend(["--port", self.port])
        return command

    def _read_stdout(self):
        for raw_line in self.process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                self.events.put(
                    AcquisitionProcessError(
                        f"INA226 logger emitted invalid JSON: {line!r}"
                    )
                )
                continue
            self.events.put(payload)

    def _next_event(self, timeout):
        try:
            event = self.events.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if isinstance(event, Exception):
            raise event
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise AcquisitionProcessError(
                "INA226 logger emitted an invalid event payload"
            )
        event_campaign_id = event.get("campaign_id")
        if event_campaign_id != self.campaign_id:
            raise AcquisitionProcessError(
                "INA226 logger event campaign ID does not match the experiment"
            )
        return event

    @staticmethod
    def _event_result(payload, boundary_name):
        result = {
            key: value
            for key, value in payload.items()
            if key not in {
                "event",
                "campaign_id",
                "monotonic_seconds",
                "wall_time_utc",
            }
        }
        result[f"{boundary_name}_event"] = {
            "monotonic_seconds": payload["monotonic_seconds"],
            "wall_time_utc": payload["wall_time_utc"],
        }
        return result

    @staticmethod
    def _failure_from_event(payload):
        failure_type = payload.get("failure_type", "AcquisitionProcessError")
        failure_message = payload.get(
            "failure_message",
            "INA226 logger reported an unspecified failure",
        )
        return AcquisitionProcessError(
            f"{failure_type}: {failure_message}"
        )

    def _drain_after_exit(self):
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=0.2)

    def _terminate_after_start_failure(self):
        if self.process is None:
            return
        self.stopping = True
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=self.stop_timeout_seconds)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._drain_after_exit()
        self.process = None

    def start(self):
        try:
            self.process = self.process_factory(
                self._command(),
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            if self.process.stdout is None:
                raise AcquisitionProcessError(
                    "INA226 logger stdout pipe was not created"
                )
            self.reader_thread = threading.Thread(
                target=self._read_stdout,
                name="ina226-event-reader",
                daemon=True,
            )
            self.reader_thread.start()

            deadline = self.monotonic_fn() + self.startup_timeout_seconds
            while self.monotonic_fn() < deadline:
                event = self._next_event(
                    min(0.1, deadline - self.monotonic_fn())
                )
                if event is not None:
                    if event["event"] == "ACQUISITION_READY":
                        self.ready_event = event
                        return self._event_result(event, "ready")
                    if event["event"] == "ACQUISITION_FAILED":
                        self.failure_event = event
                        raise self._failure_from_event(event)
                    if event["event"] == "ACQUISITION_COMPLETE":
                        self.completion_event = event
                        raise AcquisitionProcessError(
                            "INA226 logger stopped before becoming ready"
                        )
                if self.process.poll() is not None:
                    self._drain_after_exit()
                    event = self._next_event(0)
                    if event is not None and event["event"] == "ACQUISITION_FAILED":
                        self.failure_event = event
                        raise self._failure_from_event(event)
                    raise AcquisitionProcessError(
                        "INA226 logger exited before becoming ready "
                        f"with code {self.process.returncode}"
                    )
            raise AcquisitionProcessError(
                "Timed out waiting for the INA226 logger's first CSV sample"
            )
        except Exception:
            self._terminate_after_start_failure()
            raise

    def _handle_runtime_event(self, event):
        if event["event"] == "ACQUISITION_FAILED":
            self.failure_event = event
            raise self._failure_from_event(event)
        if event["event"] == "ACQUISITION_COMPLETE":
            self.completion_event = event
            if not self.stopping:
                raise AcquisitionProcessError(
                    "INA226 logger stopped before acquisition was requested to stop"
                )

    def check_health(self):
        if self.process is None:
            return None
        while True:
            event = self._next_event(0)
            if event is None:
                break
            self._handle_runtime_event(event)
        if self.process.poll() is not None:
            self._drain_after_exit()
            while True:
                event = self._next_event(0)
                if event is None:
                    break
                self._handle_runtime_event(event)
            raise AcquisitionProcessError(
                "INA226 logger exited unexpectedly "
                f"with code {self.process.returncode}"
            )
        return None

    def stop(self):
        if self.stop_result is not None:
            return dict(self.stop_result)
        if self.stop_error is not None:
            raise self.stop_error
        try:
            return self._stop_once()
        except Exception as error:
            self.stop_error = error
            raise

    def _stop_once(self):
        if self.process is None:
            return None

        self.stopping = True
        if self.process.poll() is None:
            self.process.terminate()
        deadline = self.monotonic_fn() + self.stop_timeout_seconds
        timed_out = False
        while self.monotonic_fn() < deadline:
            event = self._next_event(
                min(0.1, deadline - self.monotonic_fn())
            )
            if event is not None:
                self._handle_runtime_event(event)
            if self.process.poll() is not None:
                self._drain_after_exit()
                while True:
                    event = self._next_event(0)
                    if event is None:
                        break
                    self._handle_runtime_event(event)
                break
        else:
            timed_out = True

        if timed_out or self.process.poll() is None:
            self.process.kill()
            self.process.wait()
            self._drain_after_exit()
            raise AcquisitionProcessError(
                "INA226 logger did not stop cleanly before the timeout"
            )
        if self.failure_event is not None:
            raise self._failure_from_event(self.failure_event)
        if self.completion_event is None:
            raise AcquisitionProcessError(
                "INA226 logger exited without an ACQUISITION_COMPLETE event"
            )
        if self.process.returncode != 0:
            raise AcquisitionProcessError(
                f"INA226 logger exited with code {self.process.returncode}"
            )

        self.stop_result = self._event_result(
            self.completion_event,
            "completion",
        )
        return dict(self.stop_result)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run one model experiment with automated INA226 acquisition.",
    )
    parser.add_argument("--backend", choices=["tpu", "cpu", "cuda"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x40)
    parser.add_argument("--shunt-ohms", type=float, required=True)
    parser.add_argument("--max-expected-current-a", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--acquisition-startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--acquisition-stop-timeout-s", type=float, default=None)
    parser.add_argument("--number_of_cycles", type=int, default=5)
    parser.add_argument("--sleep_time", type=float, default=10.0)
    parser.add_argument("--inferences_per_cycle", type=int, default=None)
    parser.add_argument("--target_burst_seconds", type=float, default=10.0)
    parser.add_argument("--sampling_rate_hz", type=float, default=10.0)
    parser.add_argument("--min_active_samples", type=int, default=50)
    parser.add_argument("--warmup_inferences", type=int, default=110)
    parser.add_argument("--warmup_seconds", type=float, default=0.0)
    parser.add_argument("--warmup_cooldown_seconds", type=float, default=None)
    parser.add_argument("--calibration_initial_inferences", type=int, default=10)
    parser.add_argument("--calibration_target_seconds", type=float, default=0.5)
    parser.add_argument("--calibration_repetitions", type=int, default=5)
    parser.add_argument("--max_relative_mad", type=float, default=0.15)
    parser.add_argument("--max_calibration_inferences", type=int, default=1_000_000)
    parser.add_argument("--leading_idle_seconds", type=float, default=5.0)
    parser.add_argument("--trailing_idle_seconds", type=float, default=5.0)
    parser.add_argument("--safety_margin_seconds", type=float, default=2.0)
    return parser


def select_runner(backend):
    import runner

    if backend == "tpu":
        runner_cls = getattr(runner, "TFLiteTPURunner", None)
        if runner_cls is None:
            raise RuntimeError(
                "The tpu backend requires tflite_runtime and the Edge TPU "
                "runtime; activate the BANERA TensorFlow/TPU environment"
            )
        return runner_cls

    runner_cls = getattr(runner, "TorchRunner", None)
    if runner_cls is None:
        raise RuntimeError(
            "The cpu and cuda backends require PyTorch; activate the "
            "BANERA PyTorch environment"
        )
    return runner_cls


def build_result_report(
    manifest,
    output_directory,
    elapsed_seconds,
    exit_code,
    error=None,
    command_started_at_utc=None,
):
    manifest = manifest or {}
    campaign_id = manifest.get("campaign_id")
    acquisition = manifest.get("acquisition", {})
    measurement = manifest.get("measurement", {})
    cycles = measurement.get("cycles", [])
    manifest_path = manifest.get("manifest_path")
    csv_path = acquisition.get("csv_path")
    if campaign_id is not None:
        if manifest_path is None:
            manifest_path = str(
                (Path(output_directory) / f"{campaign_id}.json").resolve()
            )
        if csv_path is None:
            csv_path = str(
                (Path(output_directory) / f"{campaign_id}.csv").resolve()
            )
    report = {
        "event": "AUTOMATED_MEASUREMENT_RESULT",
        "campaign_id": campaign_id,
        "status": manifest.get("status", "FAILED"),
        "quality_status": manifest.get("quality_status"),
        "acquisition_status": acquisition.get("status", "NOT_STARTED"),
        "acquisition_failure": acquisition.get("failure"),
        "exit_code": exit_code,
        "command_started_at_utc": command_started_at_utc,
        "command_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_elapsed_seconds": elapsed_seconds,
        "measured_cycle_seconds": sum(
            cycle.get("elapsed_seconds", 0.0) for cycle in cycles
        ),
        "acquisition_elapsed_seconds": acquisition.get(
            "capture_elapsed_seconds"
        ),
        "sample_count": acquisition.get("sample_count"),
        "manifest_path": manifest_path,
        "csv_path": csv_path,
    }
    if error is not None:
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return report


def main():
    args = build_argument_parser().parse_args()
    command_started = time.monotonic()
    command_started_at_utc = datetime.now(timezone.utc).isoformat()
    manager = None

    try:
        controller = Ina226ProcessController(
            output_directory=args.output_directory,
            port=args.port,
            baud=args.baud,
            address=args.address,
            shunt_ohms=args.shunt_ohms,
            max_expected_current_a=args.max_expected_current_a,
            sampling_rate_hz=args.sampling_rate_hz,
            serial_timeout_seconds=args.timeout_s,
            startup_timeout_seconds=args.acquisition_startup_timeout_s,
            stop_timeout_seconds=args.acquisition_stop_timeout_s,
        )
        manager = RunManager(
            runner_cls=select_runner(args.backend),
            model_path=args.model,
            number_of_cycles=args.number_of_cycles,
            sleep_time=args.sleep_time,
            backend=args.backend,
            inferences_per_cycle=args.inferences_per_cycle,
            target_burst_seconds=args.target_burst_seconds,
            sampling_rate_hz=args.sampling_rate_hz,
            min_active_samples=args.min_active_samples,
            warmup_inferences=args.warmup_inferences,
            warmup_seconds=args.warmup_seconds,
            warmup_cooldown_seconds=args.warmup_cooldown_seconds,
            calibration_initial_inferences=args.calibration_initial_inferences,
            calibration_target_seconds=args.calibration_target_seconds,
            calibration_repetitions=args.calibration_repetitions,
            max_relative_mad=args.max_relative_mad,
            max_calibration_inferences=args.max_calibration_inferences,
            leading_idle_seconds=args.leading_idle_seconds,
            trailing_idle_seconds=args.trailing_idle_seconds,
            safety_margin_seconds=args.safety_margin_seconds,
            acquisition_controller=controller,
            manifest_directory=args.output_directory,
        )
        manifest = manager.execute()
    except KeyboardInterrupt as error:
        exit_code = 130
        report = build_result_report(
            None if manager is None else manager.last_manifest,
            args.output_directory,
            time.monotonic() - command_started,
            exit_code,
            error,
            command_started_at_utc,
        )
    except Exception as error:
        exit_code = 1
        report = build_result_report(
            None if manager is None else manager.last_manifest,
            args.output_directory,
            time.monotonic() - command_started,
            exit_code,
            error,
            command_started_at_utc,
        )
    else:
        acquisition_status = manifest.get("acquisition", {}).get("status")
        if manifest.get("status") != "COMPLETE":
            exit_code = 1
        elif acquisition_status == "COMPLETE":
            exit_code = 0
        else:
            exit_code = 2
        report = build_result_report(
            manifest,
            args.output_directory,
            time.monotonic() - command_started,
            exit_code,
            command_started_at_utc=command_started_at_utc,
        )

    print(json.dumps(report, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())