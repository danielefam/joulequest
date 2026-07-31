#!/usr/bin/env python3
"""Run one model campaign with unattended INA226 acquisition."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import queue
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class AcquisitionProcessError(RuntimeError):
    """Raised when the INA226 logger cannot provide a valid capture."""


class RemoteExperimentError(RuntimeError):
    """Raised when the remote board experiment cannot be controlled over SSH."""


CONNECTION_CONFIG_KEYS = {
    "runner_host",
    "jump_host",
    "remote_directory",
    "remote_python",
    "remote_manifest_directory",
    "ssh_connect_timeout_s",
    "ssh_options",
}
CONNECTION_DEFAULTS = {
    "remote_directory": ".",
    "remote_python": "python3",
    "remote_manifest_directory": "measurements_board",
    "ssh_connect_timeout_s": 10.0,
}


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

    def synchronize_clock(self, round_name, exchange_count):
        now = self.monotonic_fn()
        return {
            "status": "COMPLETE",
            "round": round_name,
            "method": "same_host_monotonic",
            "requested_exchange_count": exchange_count,
            "valid_exchange_count": 1,
            "selected_sample": {
                "request_id": f"{round_name}:same-host",
                "runner_sent_monotonic_seconds": now,
                "controller_received_monotonic_seconds": now,
                "controller_sent_monotonic_seconds": now,
                "runner_received_monotonic_seconds": now,
                "runner_midpoint_monotonic_seconds": now,
                "controller_minus_runner_seconds": 0.0,
                "round_trip_seconds": 0.0,
                "uncertainty_seconds": 0.0,
            },
            "errors": [],
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


class SshExperimentController:
    """Run RunManager on the remote board while acquisition remains local."""

    def __init__(
        self,
        runner_host,
        remote_directory,
        remote_python,
        remote_manifest_directory,
        ssh_options,
        acquisition_controller,
        startup_timeout_seconds=60.0,
        process_factory=subprocess.Popen,
        run_factory=subprocess.run,
        monotonic_fn=time.monotonic,
    ):
        if startup_timeout_seconds <= 0:
            raise ValueError("remote startup timeout must be positive")
        self.runner_host = runner_host
        self.remote_directory = remote_directory
        self.remote_python = remote_python
        self.remote_manifest_directory = remote_manifest_directory
        self.ssh_options = list(ssh_options or [])
        self.acquisition_controller = acquisition_controller
        self.startup_timeout_seconds = startup_timeout_seconds
        self.process_factory = process_factory
        self.run_factory = run_factory
        self.monotonic_fn = monotonic_fn
        self.process = None
        self.reader_thread = None
        self.lines = queue.Queue()
        self.campaign_id = None
        self.manifest = None
        self.acquisition_result = None
        self.acquisition_stopped = False

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def _remote_arguments(self, args):
        arguments = [
            "run_manager.py",
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--number_of_cycles",
            str(args.number_of_cycles),
            "--sleep_time",
            str(args.sleep_time),
            "--target_burst_seconds",
            str(args.target_burst_seconds),
            "--sampling_rate_hz",
            str(args.sampling_rate_hz),
            "--min_active_samples",
            str(args.min_active_samples),
            "--warmup_inferences",
            str(args.warmup_inferences),
            "--warmup_seconds",
            str(args.warmup_seconds),
            "--calibration_initial_inferences",
            str(args.calibration_initial_inferences),
            "--calibration_target_seconds",
            str(args.calibration_target_seconds),
            "--calibration_repetitions",
            str(args.calibration_repetitions),
            "--calibration-sizing-max-attempts",
            str(args.calibration_sizing_max_attempts),
            "--calibration-duration-tolerance",
            str(args.calibration_duration_tolerance),
            "--max_relative_mad",
            str(args.max_relative_mad),
            "--burst-duration-margin",
            str(args.burst_duration_margin),
            "--validation-repetitions",
            str(args.validation_repetitions),
            "--validation-max-rounds",
            str(args.validation_max_rounds),
            "--validation-safety-margin",
            str(args.validation_safety_margin),
            "--clock-sync-exchanges",
            str(args.clock_sync_exchanges),
            "--max-clock-uncertainty-fraction",
            str(args.max_clock_uncertainty_fraction),
            "--max_calibration_inferences",
            str(args.max_calibration_inferences),
            "--leading_idle_seconds",
            str(args.leading_idle_seconds),
            "--trailing_idle_seconds",
            str(args.trailing_idle_seconds),
            "--safety_margin_seconds",
            str(args.safety_margin_seconds),
            "--manifest_directory",
            self.remote_manifest_directory,
            "--stdio_acquisition",
        ]
        if args.inferences_per_cycle is not None:
            arguments.extend(
                ["--inferences_per_cycle", str(args.inferences_per_cycle)]
            )
        if args.warmup_cooldown_seconds is not None:
            arguments.extend(
                [
                    "--warmup_cooldown_seconds",
                    str(args.warmup_cooldown_seconds),
                ]
            )
        if args.validation_cooldown_seconds is not None:
            arguments.extend(
                [
                    "--validation-cooldown-seconds",
                    str(args.validation_cooldown_seconds),
                ]
            )
        return arguments

    def build_command(self, args):
        remote_tokens = [
            *shlex.split(self.remote_python),
            *self._remote_arguments(args),
        ]
        remote_command = (
            f"cd {shlex.quote(self.remote_directory)} && exec "
            + " ".join(shlex.quote(token) for token in remote_tokens)
        )
        command = ["ssh", "-T", "-o", "BatchMode=yes"]
        for option in self.ssh_options:
            command.extend(["-o", option])
        command.extend([self.runner_host, remote_command])
        return command

    def _ssh_prefix(self):
        command = ["ssh", "-T", "-o", "BatchMode=yes"]
        for option in self.ssh_options:
            command.extend(["-o", option])
        command.append(self.runner_host)
        return command

    def _fetch_remote_manifest(self):
        if self.campaign_id is None:
            return None
        remote_path = posixpath.join(
            self.remote_manifest_directory,
            f"{self.campaign_id}.json",
        )
        remote_command = (
            f"cd {shlex.quote(self.remote_directory)} && "
            f"cat -- {shlex.quote(remote_path)}"
        )
        try:
            result = self.run_factory(
                [*self._ssh_prefix(), remote_command],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        try:
            manifest = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        return manifest if isinstance(manifest, dict) else None

    def delete_remote_manifest(self, campaign_id):
        if (
            not isinstance(campaign_id, str)
            or not campaign_id
            or posixpath.basename(campaign_id) != campaign_id
            or campaign_id in (".", "..")
        ):
            raise RemoteExperimentError("Invalid campaign ID for remote cleanup")
        remote_path = posixpath.join(
            self.remote_manifest_directory,
            f"{campaign_id}.json",
        )
        remote_command = (
            f"cd {shlex.quote(self.remote_directory)} && "
            f"rm -f -- {shlex.quote(remote_path)}"
        )
        try:
            result = self.run_factory(
                [*self._ssh_prefix(), remote_command],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RemoteExperimentError(
                "Could not remove the remote manifest"
            ) from error
        if result.returncode != 0:
            raise RemoteExperimentError(
                "Could not remove the remote manifest "
                f"(SSH exit code {result.returncode})"
            )

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        except Exception as error:
            self.lines.put(error)
        finally:
            self.lines.put(None)

    def _send_result(self, command, result, request_id=None):
        payload = {
            "command": command,
            "campaign_id": self.campaign_id,
            "result": result,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        self.process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        self.process.stdin.flush()

    def _handle_clock_sync_request(self, event):
        controller_received_seconds = self.monotonic_fn()
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RemoteExperimentError(
                "Clock synchronization request has no request ID"
            )
        controller_sent_seconds = self.monotonic_fn()
        self._send_result(
            "CLOCK_SYNC_RESPONSE",
            {
                "controller_received_monotonic_seconds": (
                    controller_received_seconds
                ),
                "controller_sent_monotonic_seconds": controller_sent_seconds,
            },
            request_id=request_id,
        )

    def _failure_result(self, error, base=None):
        if (
            base is not None
            and base.get("status") == "FAILED"
            and isinstance(base.get("failure"), dict)
        ):
            return dict(base)
        return {
            **(base or {}),
            "status": "FAILED",
            "failed_at_utc": self._utc_now(),
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    def _start_local_acquisition(self, event):
        description = self.acquisition_controller.describe(
            self.campaign_id,
            event,
        )
        try:
            started = self.acquisition_controller.start()
            self.acquisition_result = {**description, **started}
        except Exception as error:
            self.acquisition_result = self._failure_result(error, description)
        self._send_result("ACQUISITION_STARTED", self.acquisition_result)

    def _check_local_acquisition(self):
        if (
            self.acquisition_result is None
            or self.acquisition_result.get("status") == "FAILED"
            or self.acquisition_stopped
        ):
            return
        try:
            result = self.acquisition_controller.check_health()
            if result is not None:
                self.acquisition_result.update(result)
        except Exception as error:
            self.acquisition_result = self._failure_result(
                error,
                self.acquisition_result,
            )

    def _stop_local_acquisition(self):
        if self.acquisition_stopped:
            return self.acquisition_result
        acquisition_already_failed = (
            self.acquisition_result is not None
            and self.acquisition_result.get("status") == "FAILED"
        )
        try:
            stopped = self.acquisition_controller.stop()
            if stopped is not None and not acquisition_already_failed:
                self.acquisition_result = {
                    **(self.acquisition_result or {}),
                    **stopped,
                }
        except Exception as error:
            self.acquisition_result = self._failure_result(
                error,
                self.acquisition_result,
            )
        self.acquisition_stopped = True
        return self.acquisition_result or {
            "status": "FAILED",
            "failure": {
                "type": "AcquisitionProcessError",
                "message": "Local acquisition was never started",
            },
        }

    def _handle_event(self, event, raw_line):
        event_name = event.get("event")
        event_campaign_id = event.get("campaign_id")
        if not isinstance(event_name, str):
            raise RemoteExperimentError("Remote JSON payload has no event name")
        if not isinstance(event_campaign_id, str) or not event_campaign_id:
            raise RemoteExperimentError(
                f"Remote event {event_name} has no campaign_id"
            )
        if self.campaign_id is None:
            self.campaign_id = event_campaign_id
        elif event_campaign_id != self.campaign_id:
            raise RemoteExperimentError(
                "Remote event campaign ID does not match the experiment"
            )

        if event_name == "CLOCK_SYNC_REQUEST":
            self._handle_clock_sync_request(event)
        elif event_name == "ACQUISITION_START_REQUEST":
            self._start_local_acquisition(event)
        elif event_name == "ACQUISITION_STOP_REQUEST":
            result = self._stop_local_acquisition()
            self._send_result("ACQUISITION_STOPPED", result)
        elif event_name == "RUN_MANIFEST":
            manifest = event.get("manifest")
            if not isinstance(manifest, dict):
                raise RemoteExperimentError(
                    "Remote board returned an invalid manifest payload"
                )
            self.manifest = manifest
        else:
            print(raw_line.rstrip(), flush=True)
        self._check_local_acquisition()

    def execute(self, args):
        self.process = self.process_factory(
            self.build_command(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RemoteExperimentError("SSH process pipes were not created")
        self.reader_thread = threading.Thread(
            target=self._read_stdout,
                name="remote-board-event-reader",
            daemon=True,
        )
        self.reader_thread.start()
        startup_deadline = self.monotonic_fn() + self.startup_timeout_seconds
        received_event = False
        reader_finished = False

        try:
            while True:
                try:
                    raw_line = self.lines.get(timeout=0.1)
                except queue.Empty:
                    raw_line = ""
                if isinstance(raw_line, Exception):
                    raise RemoteExperimentError(
                        f"Failed to read remote board stdout: {raw_line}"
                    ) from raw_line
                if raw_line is None:
                    reader_finished = True
                elif raw_line:
                    line = raw_line.strip()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        print(
                            f"[remote-board] {line}",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        received_event = True
                        self._handle_event(event, raw_line)

                if not received_event and self.monotonic_fn() >= startup_deadline:
                    raise RemoteExperimentError(
                        "Timed out waiting for the first remote board event"
                    )
                if reader_finished and self.process.poll() is not None:
                    break

            return_code = self.process.wait()
            if self.manifest is None:
                self.manifest = self._fetch_remote_manifest()
            if self.manifest is None:
                raise RemoteExperimentError(
                    "Remote board exited without returning its JSON manifest "
                    f"(SSH exit code {return_code})"
                )
            manifest_status = self.manifest.get("status")
            if manifest_status not in ("COMPLETE", "FAILED"):
                raise RemoteExperimentError(
                    "Remote board returned invalid manifest status "
                    f"{manifest_status!r}"
                )
            if manifest_status == "COMPLETE" and return_code != 0:
                raise RemoteExperimentError(
                    "Remote board reported a complete campaign but SSH exited with "
                    f"code {return_code}"
                )
            if manifest_status == "FAILED" and return_code == 0:
                raise RemoteExperimentError(
                    "Remote board returned a failed manifest with SSH exit code 0"
                )
            return self.manifest
        finally:
            if not self.acquisition_stopped and self.acquisition_result is not None:
                self._stop_local_acquisition()
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self.reader_thread is not None:
                self.reader_thread.join(timeout=0.2)
            for stream_name in ("stdin", "stdout"):
                stream = getattr(self.process, stream_name, None)
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except OSError:
                        pass


def persist_local_manifest(manifest, output_directory):
    campaign_id = manifest.get("campaign_id")
    if not campaign_id:
        raise RemoteExperimentError("Remote board manifest has no campaign_id")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"{campaign_id}.json"
    manifest.pop("remote_manifest_path", None)
    manifest["manifest_path"] = str(path.resolve())
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
    return path


def prepare_remote_manifest(manifest, args, remote_controller=None):
    manifest.setdefault(
        "orchestration",
        {
            "mode": "ssh",
            "runner_role": "inference_host",
            "acquisition_role": "controller_host",
        },
    )
    path = persist_local_manifest(manifest, args.output_directory)
    if (
        remote_controller is not None
        and not getattr(args, "keep_remote_manifest", False)
    ):
        try:
            remote_controller.delete_remote_manifest(manifest.get("campaign_id"))
        except RemoteExperimentError as error:
            print(f"warning: {error}; remote recovery copy retained", file=sys.stderr)
    return path


def load_connection_config(path):
    if path is None:
        return {}
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Connection config does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Connection config is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Connection config must contain one JSON object")
    unknown_keys = sorted(set(payload) - CONNECTION_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(
            "Unknown connection config keys: " + ", ".join(unknown_keys)
        )
    for key in (
        "runner_host",
        "jump_host",
        "remote_directory",
        "remote_python",
        "remote_manifest_directory",
    ):
        if key in payload and payload[key] is not None:
            if not isinstance(payload[key], str) or not payload[key].strip():
                raise ValueError(f"Connection config {key} must be a nonempty string")
    if "ssh_connect_timeout_s" in payload:
        timeout = payload["ssh_connect_timeout_s"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(
                "Connection config ssh_connect_timeout_s must be positive"
            )
    if "ssh_options" in payload:
        options = payload["ssh_options"]
        if not isinstance(options, list) or not all(
            isinstance(option, str) and option.strip() for option in options
        ):
            raise ValueError(
                "Connection config ssh_options must be a list of strings"
            )
    return payload


def apply_connection_config(args):
    config = load_connection_config(args.connection_config)
    for key in (
        "runner_host",
        "jump_host",
        "remote_directory",
        "remote_python",
        "remote_manifest_directory",
        "ssh_connect_timeout_s",
    ):
        command_value = getattr(args, key)
        if command_value is None:
            setattr(args, key, config.get(key, CONNECTION_DEFAULTS.get(key)))
    config_options = list(config.get("ssh_options", []))
    args.ssh_option = [*config_options, *(args.ssh_option or [])]
    return args


def build_ssh_options(jump_host, connect_timeout_seconds, extra_options):
    options = list(extra_options or [])
    option_names = {
        option.partition("=")[0].strip().lower()
        for option in options
    }
    if jump_host is not None and "proxyjump" not in option_names:
        options.insert(0, f"ProxyJump={jump_host}")
    if "connecttimeout" not in option_names:
        options.append(f"ConnectTimeout={connect_timeout_seconds:g}")
    if "connectionattempts" not in option_names:
        options.append("ConnectionAttempts=1")
    return options


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
    parser.add_argument("--calibration-sizing-max-attempts", type=int, default=3)
    parser.add_argument("--calibration-duration-tolerance", type=float, default=0.20)
    parser.add_argument("--max_relative_mad", type=float, default=0.15)
    parser.add_argument("--burst-duration-margin", type=float, default=1.2)
    parser.add_argument("--validation-repetitions", type=int, default=3)
    parser.add_argument("--validation-max-rounds", type=int, default=3)
    parser.add_argument("--validation-safety-margin", type=float, default=1.1)
    parser.add_argument("--validation-cooldown-seconds", type=float, default=None)
    parser.add_argument("--clock-sync-exchanges", type=int, default=10)
    parser.add_argument(
        "--max-clock-uncertainty-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--max_calibration_inferences", type=int, default=1_000_000)
    parser.add_argument("--leading_idle_seconds", type=float, default=5.0)
    parser.add_argument("--trailing_idle_seconds", type=float, default=5.0)
    parser.add_argument("--safety_margin_seconds", type=float, default=2.0)
    parser.add_argument(
        "--connection-config",
        type=Path,
        default=None,
        help="Ignored local JSON file containing SSH host settings",
    )
    parser.add_argument(
        "--runner-host",
        default=None,
        help="SSH destination running inference",
    )
    parser.add_argument(
        "--jump-host",
        default=None,
        help="SSH jump host used to reach the inference host",
    )
    parser.add_argument(
        "--remote-directory",
        default=None,
        help="Remote directory containing run_manager.py",
    )
    parser.add_argument(
        "--remote-python",
        default=None,
        help="Remote Python executable or command prefix",
    )
    parser.add_argument(
        "--remote-manifest-directory",
        default=None,
        help="Manifest directory on the inference host",
    )
    parser.add_argument(
        "--keep-remote-manifest",
        action="store_true",
        help="Retain the inference-host manifest after saving its local copy",
    )
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Additional ssh -o option; may be repeated",
    )
    parser.add_argument(
        "--ssh-connect-timeout-s",
        type=float,
        default=None,
        help="SSH TCP connection timeout for each host",
    )
    parser.add_argument(
        "--remote-startup-timeout-s",
        type=float,
        default=60.0,
        help="Maximum wait for the first event from the inference host",
    )
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


def build_local_manager(args, acquisition_controller):
    from run_manager import RunManager

    return RunManager(
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
        calibration_sizing_max_attempts=args.calibration_sizing_max_attempts,
        calibration_duration_tolerance=args.calibration_duration_tolerance,
        max_relative_mad=args.max_relative_mad,
        burst_duration_margin=args.burst_duration_margin,
        validation_repetitions=args.validation_repetitions,
        validation_max_rounds=args.validation_max_rounds,
        validation_safety_margin=args.validation_safety_margin,
        validation_cooldown_seconds=args.validation_cooldown_seconds,
        clock_sync_exchanges=args.clock_sync_exchanges,
        max_clock_uncertainty_fraction=args.max_clock_uncertainty_fraction,
        max_calibration_inferences=args.max_calibration_inferences,
        leading_idle_seconds=args.leading_idle_seconds,
        trailing_idle_seconds=args.trailing_idle_seconds,
        safety_margin_seconds=args.safety_margin_seconds,
        acquisition_controller=acquisition_controller,
        manifest_directory=args.output_directory,
    )


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
    args = apply_connection_config(build_argument_parser().parse_args())
    command_started = time.monotonic()
    command_started_at_utc = datetime.now(timezone.utc).isoformat()
    manager = None
    remote_controller = None
    manifest = None

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
        if args.runner_host is not None:
            remote_controller = SshExperimentController(
                runner_host=args.runner_host,
                remote_directory=args.remote_directory,
                remote_python=args.remote_python,
                remote_manifest_directory=args.remote_manifest_directory,
                ssh_options=build_ssh_options(
                    args.jump_host,
                    args.ssh_connect_timeout_s,
                    args.ssh_option,
                ),
                acquisition_controller=controller,
                startup_timeout_seconds=args.remote_startup_timeout_s,
            )
            manifest = remote_controller.execute(args)
            prepare_remote_manifest(manifest, args, remote_controller)
        else:
            manager = build_local_manager(args, controller)
            manifest = manager.execute()
    except KeyboardInterrupt as error:
        exit_code = 130
        manifest = (
            manager.last_manifest
            if manager is not None
            else None if remote_controller is None else remote_controller.manifest
        )
        if args.runner_host is not None and manifest is not None:
            try:
                prepare_remote_manifest(manifest, args, remote_controller)
            except OSError:
                pass
        report = build_result_report(
            manifest,
            args.output_directory,
            time.monotonic() - command_started,
            exit_code,
            error,
            command_started_at_utc,
        )
    except Exception as error:
        exit_code = 1
        manifest = (
            manager.last_manifest
            if manager is not None
            else None if remote_controller is None else remote_controller.manifest
        )
        if args.runner_host is not None and manifest is not None:
            try:
                prepare_remote_manifest(manifest, args, remote_controller)
            except OSError:
                pass
        report = build_result_report(
            manifest,
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