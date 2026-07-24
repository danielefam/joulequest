"""Plan and execute clean, measurable inference bursts.

The workload has three separate phases:

1. warm-up removes first-use backend, allocator, kernel, and cache overhead;
2. calibration estimates steady-state inference latency before acquisition;
3. measurement executes only useful bursts recorded by the INA226EVM.

"""

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CapturePlan:
    """Expected duration and sample count for one INA226 capture."""

    capture_seconds: float
    capture_samples: int


@dataclass(frozen=True)
class CalibrationResult:
    """class that stores the timing information obtained before the real measurement begins."""
    # batch is referred to one repeated group of forward passes, to not be confused with tensor batch

    stable_latency_seconds: float # estimated time for one steady-state inference
    median_batch_seconds: float 
    batch_inferences: int # number of inferences executed in each calibration batch.
    retained_batches: int # number of batches used in the calculation
    discarded_batches: int
    sizing_pilot_inferences: int
    total_executed_inferences: int
    relative_mad: float
    coefficient_of_variation: float


def calculate_required_burst_seconds(target_burst_seconds, min_active_samples, sampling_rate_hz):
    """Return the burst duration required by time and sample constraints.

    T_samples = min_active_samples / sampling_rate_hz seconds
    I want my burst to last at least target_burst_seconds. 
    sampling_rate_hz could be slow and therefore we may need more time.
    Therefore we will take a maximum between a target we have in seconds and 
    a minimum amount of sample that we want to take for each burst
    """
    return max(
        float(target_burst_seconds),
        float(min_active_samples) / float(sampling_rate_hz),
        # units of measurement [#samples] / ([#samples]/[s]) = [s]
    )


def calculate_inference_count(stable_latency_seconds, target_burst_seconds, min_active_samples, sampling_rate_hz):
    """Smallest number of inferences per burst expected to produce a useful burst"""
    required_seconds = calculate_required_burst_seconds(
        target_burst_seconds,
        min_active_samples,
        sampling_rate_hz,
    )
    return max(1, math.ceil(required_seconds / stable_latency_seconds))


def calculate_capture_plan(number_of_cycles, estimated_burst_seconds, sleep_time_seconds,
                leading_idle_seconds, trailing_idle_seconds, safety_margin_seconds, sampling_rate_hz,):
    """Calculate total recording time"""

    capture_seconds = (
        leading_idle_seconds
        + number_of_cycles * estimated_burst_seconds
        + (number_of_cycles - 1) * sleep_time_seconds
        + trailing_idle_seconds
        + safety_margin_seconds
    )
    return CapturePlan(
        capture_seconds=capture_seconds,
        capture_samples=math.ceil(capture_seconds * sampling_rate_hz),
    )


class StdioAcquisitionController:
    """Coordinate acquisition owned by the process controlling stdin/stdout."""

    def __init__(self, input_stream=None, output_stream=None):
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.campaign_id = None
        self.plan = None
        self.stop_result = None

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, event, **fields):
        payload = {
            "event": event,
            "monotonic_seconds": time.monotonic(),
            "wall_time_utc": self._utc_now(),
            **fields,
        }
        print(
            json.dumps(payload, sort_keys=True),
            file=self.output_stream,
            flush=True,
        )

    def _read_result(self, expected_command):
        line = self.input_stream.readline()
        if not line:
            raise RuntimeError(
                f"Acquisition controller disconnected before {expected_command}"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Acquisition controller sent invalid JSON"
            ) from error
        if payload.get("command") != expected_command:
            raise RuntimeError(
                "Expected acquisition command "
                f"{expected_command}, received {payload.get('command')!r}"
            )
        if payload.get("campaign_id") != self.campaign_id:
            raise RuntimeError("Acquisition command campaign ID does not match")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Acquisition command result must be a dictionary")
        return result

    def describe(self, campaign_id, plan):
        self.campaign_id = campaign_id
        self.plan = dict(plan)
        return {
            "status": "PENDING",
            "control_protocol": "stdio_json_v1",
            "acquisition_role": "controller_host",
        }

    def start(self):
        self._emit(
            "ACQUISITION_START_REQUEST",
            campaign_id=self.campaign_id,
            **self.plan,
        )
        return self._read_result("ACQUISITION_STARTED")

    def check_health(self):
        return None

    def stop(self):
        if self.stop_result is not None:
            return dict(self.stop_result)
        if self.campaign_id is None:
            return None
        self._emit(
            "ACQUISITION_STOP_REQUEST",
            campaign_id=self.campaign_id,
        )
        self.stop_result = self._read_result("ACQUISITION_STOPPED")
        return dict(self.stop_result)


class RunManager:
    """Run warm-up, calibration, and measured cycles in that exact order.

    Empirical evidence shows that the first cold burst is significantly slower than the subsequent ones.
    A warming burst is then triggered
    """
    def __init__(
        self,
        runner_cls,
        model_path,
        number_of_cycles=5,
        sleep_time=10.0,
        backend="cuda",
        inferences_per_cycle=None,
        target_burst_seconds=10.0,
        sampling_rate_hz=10.0,
        min_active_samples=50,
        warmup_inferences=110,
        warmup_seconds=0.0,
        warmup_cooldown_seconds=None,
        calibration_initial_inferences=10,
        calibration_target_seconds=0.5,
        calibration_repetitions=5,
        max_relative_mad=0.15,
        max_calibration_inferences=1_000_000,
        leading_idle_seconds=5.0,
        trailing_idle_seconds=5.0,
        safety_margin_seconds=2.0,
        wait_for_acquisition=False,
        acquisition_controller=None,
        manifest_directory=None,
        sleep_fn=time.sleep,
        input_fn=input,
    ):
        self.runner_cls = runner_cls
        self.model_path = model_path
        self.number_of_cycles = number_of_cycles
        self.sleep_time = sleep_time
        self.backend = backend
        self.inferences_per_cycle = inferences_per_cycle
        self.target_burst_seconds = target_burst_seconds
        self.sampling_rate_hz = sampling_rate_hz
        self.min_active_samples = min_active_samples
        self.warmup_inferences = warmup_inferences
        self.warmup_seconds = warmup_seconds
        self.warmup_cooldown_seconds = (
            sleep_time
            if warmup_cooldown_seconds is None
            else warmup_cooldown_seconds
        )
        self.calibration_initial_inferences = calibration_initial_inferences
        self.calibration_target_seconds = calibration_target_seconds
        self.calibration_repetitions = calibration_repetitions
        self.max_relative_mad = max_relative_mad
        self.max_calibration_inferences = max_calibration_inferences
        self.leading_idle_seconds = leading_idle_seconds
        self.trailing_idle_seconds = trailing_idle_seconds
        self.safety_margin_seconds = safety_margin_seconds
        self.wait_for_acquisition = wait_for_acquisition
        self.acquisition_controller = acquisition_controller
        self.manifest_directory = manifest_directory
        self.sleep_fn = sleep_fn
        self.input_fn = input_fn
        self.last_manifest = None
        
        if self.calibration_repetitions < 2:
            raise ValueError(
                "calibration_repetitions must include one discarded and "
                "at least one retained batch"
            )
        if self.max_relative_mad < 0:
            raise ValueError("max_relative_mad cannot be negative")
        if self.wait_for_acquisition and self.acquisition_controller is not None:
            raise ValueError(
                "wait_for_acquisition cannot be combined with automated acquisition"
            )

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def _emit_event(self, event, verbose=True, **fields):
        payload = {
            "event": event,
            "monotonic_seconds": time.monotonic(),
            "wall_time_utc": self._utc_now(),
            **fields,

        }
        if verbose:
            print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    #gpt5.6
    def _new_campaign_id(self):
        model_stem = Path(self.model_path).stem.replace(" ", "-")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{model_stem}_{uuid.uuid4().hex[:8]}"

    def _run_warmup(self, runner):
        """Discard cold-start work until both warm-up minima are satisfied."""
        total_inferences = 0
        total_seconds = 0.0
        batches = 0
        next_batch_inferences = self.warmup_inferences

        while (total_inferences < self.warmup_inferences or total_seconds < self.warmup_seconds):
            result = runner.run_burst(next_batch_inferences)
            total_inferences += result.executed_inferences
            total_seconds += result.elapsed_seconds
            batches += 1

            remaining_seconds = max(0.0, self.warmup_seconds - total_seconds)
            remaining_inferences = max(0, self.warmup_inferences - total_inferences)
            count_for_remaining_time = math.ceil(remaining_seconds / result.latency_seconds)
            next_batch_inferences = min(self.max_calibration_inferences,
                                        max(1, remaining_inferences, count_for_remaining_time))

        return {
            "purpose": "discard first-use backend and hardware overhead",
            "included_in_measurement": False,
            "batches": batches,
            "executed_inferences": total_inferences,
            "elapsed_seconds": total_seconds,
        }

    def _calibrate(self, runner):
        pilot = runner.run_burst(self.calibration_initial_inferences)
        scaled_count = math.ceil(
            self.calibration_initial_inferences * self.calibration_target_seconds / pilot.elapsed_seconds
        )
        batch_inferences = min(
            self.max_calibration_inferences,
            max(self.calibration_initial_inferences, scaled_count),
        )

        batches = [runner.run_burst(batch_inferences) for _ in range(self.calibration_repetitions)]

        retained = batches[1:]  # discard the first one
        #latency_seconds is derived over one single inference
        latencies = [result.latency_seconds for result in retained] 
        stable_latency = statistics.median(latencies)
        median_batch =statistics.median(result.elapsed_seconds for result in retained)
        median_absolute_deviation = statistics.median(
            abs(latency - stable_latency) for latency in latencies
        )
        relative_mad = median_absolute_deviation / stable_latency
        coefficient_of_variation = (
            statistics.pstdev(latencies) / statistics.mean(latencies)
        )
        if (
            relative_mad > self.max_relative_mad
            or coefficient_of_variation > self.max_relative_mad
        ):
            raise RuntimeError(
                "Calibration did not stabilize: "
                f"relative MAD={relative_mad:.6g}, "
                f"coefficient of variation={coefficient_of_variation:.6g}, "
                f"limit={self.max_relative_mad:.6g}"
            )

        return CalibrationResult(
            stable_latency_seconds=stable_latency,
            median_batch_seconds = median_batch,
            batch_inferences=batch_inferences,
            retained_batches=len(retained),
            discarded_batches=1,
            sizing_pilot_inferences=pilot.executed_inferences,
            total_executed_inferences=(
                pilot.executed_inferences
                + sum(result.executed_inferences for result in batches)
            ),
            relative_mad=relative_mad,
            coefficient_of_variation=coefficient_of_variation,
        )

    def _select_inference_count(self, calibration):
        if self.inferences_per_cycle is None:
            return calculate_inference_count(
                calibration.stable_latency_seconds,
                self.target_burst_seconds,
                self.min_active_samples,
                self.sampling_rate_hz,
            ), "automatic"

        # number of inferences * latency for ONE inference * sampling rate
        estimated_samples = (
            self.inferences_per_cycle
            * calibration.stable_latency_seconds
            * self.sampling_rate_hz
        )
        if estimated_samples < self.min_active_samples:
            raise ValueError(
                "The --inferences_per_cycle override is expected to produce "
                f"{estimated_samples} samples, below the minimum active samples ({self.min_active_samples})"
            )
        return self.inferences_per_cycle, "manual_override"

    def _manifest_path(self, campaign_id):
        if self.manifest_directory is None:
            return None
        return Path(self.manifest_directory) / f"{campaign_id}.json"

    #gpt5.6
    @staticmethod
    def _write_manifest(manifest, path):
        """Atomically replace the campaign manifest to avoid partial JSON."""
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest["manifest_path"] = str(path.resolve())
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)

    def _build_measurement_plan(self, calibration):
        """Build the immutable plan used by the INA226 operator."""
        inference_count, selection_mode = self._select_inference_count(calibration)
        estimated_burst_seconds = (
            inference_count * calibration.stable_latency_seconds
        )
        capture_plan = calculate_capture_plan(
            self.number_of_cycles,
            estimated_burst_seconds,
            self.sleep_time,
            self.leading_idle_seconds,
            self.trailing_idle_seconds,
            self.safety_margin_seconds,
            self.sampling_rate_hz
        )
        plan = {
            "selection_mode": selection_mode,
            "inferences_per_cycle": inference_count,
            "number_of_cycles": self.number_of_cycles,
            "target_burst_seconds": self.target_burst_seconds,
            "estimated_burst_seconds": estimated_burst_seconds,
            "sampling_rate_hz": self.sampling_rate_hz,
            "min_active_samples": self.min_active_samples,
            "estimated_active_samples_per_cycle": (
                estimated_burst_seconds * self.sampling_rate_hz
            ),
            "sleep_time_seconds": self.sleep_time,
            "leading_idle_seconds": self.leading_idle_seconds,
            "trailing_idle_seconds": self.trailing_idle_seconds,
            "safety_margin_seconds": self.safety_margin_seconds,
            **asdict(capture_plan)
        }
        return plan

    def _prepare_acquisition(self, runner, campaign_id, manifest):
        """Run excluded work and write the manifest at the READY boundary."""
        self._emit_event("WARMUP_START", campaign_id=campaign_id)
        manifest["warmup"] = self._run_warmup(runner)
        manifest["warmup"]["cooldown_seconds"] = (
            self.warmup_cooldown_seconds
        )
        self.sleep_fn(self.warmup_cooldown_seconds)
        self._emit_event(
            "WARMUP_END",
            verbose=True,
            campaign_id=campaign_id,
            **manifest["warmup"],
        )

        self._emit_event("CALIBRATION_START", campaign_id=campaign_id)
        calibration = self._calibrate(runner)
        manifest["calibration"] = {
            **asdict(calibration),
            "included_in_measurement": False,
        }
        self._emit_event(
            "CALIBRATION_END",
            verbose=True,
            campaign_id=campaign_id,
            **manifest["calibration"],
        )

        plan = self._build_measurement_plan(calibration)
        manifest["plan"] = plan
        manifest["status"] = "READY"
        return plan

    def _run_measured_cycle(
        self,
        runner,
        campaign_id,
        cycle,
        inference_count,
        estimated_burst_seconds,
    ):
        """Prepare one fresh state and record exactly one measured cycle."""
        # Preparation is outside the timer. This ensures BURST_START marks the
        # beginning of the forward-pass interval, not randomization overhead.
        runner.prepare_burst()
        start_event = self._emit_event(
            "BURST_START",
            campaign_id=campaign_id,
            cycle=cycle,
            requested_inferences=inference_count,
        )
        result = runner.run_prepared_burst(inference_count)
        actual_samples = result.elapsed_seconds * self.sampling_rate_hz
        duration_ratio = result.elapsed_seconds / estimated_burst_seconds
        quality_flags = []
        if actual_samples < self.min_active_samples:
            quality_flags.append("UNDER_RESOLVED")
        if duration_ratio < 0.5 or duration_ratio > 2.0:
            quality_flags.append("DURATION_ANOMALY")

        end_event = self._emit_event(
            "BURST_END",
            campaign_id=campaign_id,
            cycle=cycle,
            requested_inferences=result.requested_inferences,
            executed_inferences=result.executed_inferences,
            elapsed_seconds=result.elapsed_seconds,
            estimated_ina226_samples=actual_samples,
            quality_flags=quality_flags,
        )
        return {
            "cycle": cycle,
            "requested_inferences": result.requested_inferences,
            "executed_inferences": result.executed_inferences,
            "elapsed_seconds": result.elapsed_seconds,
            "estimated_ina226_samples": actual_samples,
            "quality_flags": quality_flags,
            "start_event": start_event,
            "end_event": end_event,
        }

    def _run_measurement(self, runner, campaign_id, plan, manifest):
        """Execute measured cycles with leading/trailing idle guards."""
        measurement = manifest["measurement"]
        self.sleep_fn(self.leading_idle_seconds)
        self._check_acquisition_health(manifest)

        for cycle in range(1, self.number_of_cycles + 1):
            cycle_record = self._run_measured_cycle(
                runner,
                campaign_id,
                cycle,
                plan["inferences_per_cycle"],
                plan["estimated_burst_seconds"],
            )
            measurement["cycles"].append(cycle_record)
            measurement["total_executed_inferences"] += (
                cycle_record["executed_inferences"]
            )
            self._check_acquisition_health(manifest)

            # There is no idle interval after the final measured cycle.
            if cycle < self.number_of_cycles:
                self.sleep_fn(self.sleep_time)
                self._check_acquisition_health(manifest)

        self.sleep_fn(self.trailing_idle_seconds)
        self._check_acquisition_health(manifest)
        return measurement

    def _describe_acquisition(self, campaign_id, plan):
        description = self.acquisition_controller.describe(campaign_id, plan)
        if not isinstance(description, dict):
            raise TypeError("acquisition describe() must return a dictionary")
        return {"status": "PENDING", **description}

    def _merge_acquisition_result(self, manifest, result):
        if result is None:
            return
        if not isinstance(result, dict):
            raise TypeError("acquisition lifecycle methods must return dictionaries")
        acquisition = manifest.setdefault("acquisition", {})
        previous_status = acquisition.get("status")
        previous_failure = acquisition.get("failure")
        acquisition.update(result)
        if previous_status == "FAILED":
            acquisition["status"] = previous_status
            acquisition["failure"] = previous_failure

    def _mark_acquisition_failed(self, manifest, error):
        acquisition = manifest.setdefault("acquisition", {})
        acquisition["status"] = "FAILED"
        acquisition.setdefault("failed_at_utc", self._utc_now())
        acquisition.setdefault(
            "failure",
            {
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def _start_acquisition(self, manifest):
        try:
            result = self.acquisition_controller.start()
            self._merge_acquisition_result(manifest, result)
        except Exception as error:
            self._mark_acquisition_failed(manifest, error)

    def _check_acquisition_health(self, manifest):
        if (
            self.acquisition_controller is None
            or manifest["acquisition"].get("status") == "FAILED"
        ):
            return
        try:
            result = self.acquisition_controller.check_health()
            self._merge_acquisition_result(manifest, result)
        except Exception as error:
            self._mark_acquisition_failed(manifest, error)

    def _stop_acquisition(self, manifest):
        try:
            result = self.acquisition_controller.stop()
            self._merge_acquisition_result(manifest, result)
        except Exception as error:
            self._mark_acquisition_failed(manifest, error)

    def _complete_manifest(self, manifest, manifest_path, campaign_id):
        """Add quality status, persist the final manifest, and emit COMPLETE."""
        all_flags = [
            flag
            for cycle in manifest["measurement"]["cycles"]
            for flag in cycle["quality_flags"]
        ]
        if manifest.get("acquisition", {}).get("status") == "FAILED":
            all_flags.append("ACQUISITION_FAILED")
        manifest["quality_status"] = "OK" if not all_flags else "REVIEW"
        manifest["quality_flags"] = sorted(set(all_flags))
        manifest["status"] = "COMPLETE"
        manifest["completed_at_utc"] = self._utc_now()
        self._write_manifest(manifest, manifest_path)
        self._emit_event(
            "COMPLETE",
            campaign_id=campaign_id,
            quality_status=manifest["quality_status"],
            total_executed_inferences=manifest["measurement"][
                "total_executed_inferences"
            ],
        )

    def execute(self):
        """Execute one campaign and return its complete manifest dictionary."""
        campaign_id = self._new_campaign_id()
        manifest_path = self._manifest_path(campaign_id)
        manifest = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "status": "STARTING",
            "created_at_utc": self._utc_now(),
            "model_path": self.model_path,
            "backend": self.backend,
            "workload_policy": {
                "parameters": (
                    "fixed_model" if self.backend == "tpu" else "fresh_per_burst"
                ),
                "input": "fresh_per_burst",
            },
        }
        self.last_manifest = manifest
        runner = None
        acquisition_stopped = False

        try:
            runner = self.runner_cls(self.model_path, self.backend)
            runner.prepare()

            plan = self._prepare_acquisition(runner, campaign_id, manifest)
            if self.acquisition_controller is not None:
                manifest["acquisition"] = self._describe_acquisition(
                    campaign_id,
                    plan,
                )
            self._write_manifest(manifest, manifest_path)
            self._emit_event("READY", campaign_id=campaign_id, **plan)

            if self.acquisition_controller is not None:
                self._start_acquisition(manifest)
            elif self.wait_for_acquisition:
                self.input_fn("Start INA226 acquisition, then press Enter to begin the leading idle interval...")

            manifest["measurement"] = {
                "cycles": [],
                "total_executed_inferences": 0,
            }
            self._run_measurement(
                runner,
                campaign_id,
                plan,
                manifest,
            )

            if self.acquisition_controller is not None:
                self.sleep_fn(self.safety_margin_seconds)
                self._check_acquisition_health(manifest)
                self._stop_acquisition(manifest)
                acquisition_stopped = True
            elif self.wait_for_acquisition:
                self.input_fn("Stop INA226 acquisition, then press Enter to write the manifest...")
            self._complete_manifest(manifest, manifest_path, campaign_id)
            return manifest

        except (Exception, KeyboardInterrupt) as error:
            if self.acquisition_controller is not None and not acquisition_stopped:
                self._stop_acquisition(manifest)
                acquisition_stopped = True
            manifest["status"] = "FAILED"
            manifest["failed_at_utc"] = self._utc_now()
            manifest["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            self._write_manifest(manifest, manifest_path)
            self._emit_event(
                "FAILED",
                verbose=True,
                campaign_id=campaign_id,
                failure_type=type(error).__name__,
                failure_message=str(error),
            )
            raise
        finally:
            if self.acquisition_controller is not None and not acquisition_stopped:
                self._stop_acquisition(manifest)
            if runner is not None:
                runner.close()


def build_argument_parser():
    """Build the documented command-line interface for one campaign."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--backend", choices=["tpu", "cpu", "cuda"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--number_of_cycles", type=int, default=5)
    parser.add_argument("--sleep_time", type=float, default=10.0)
    parser.add_argument("--inferences_per_cycle", type=int, default=None)
    parser.add_argument("--target_burst_seconds", type=float, default=10.0, help="desired cycle duration")
    parser.add_argument("--sampling_rate_hz", type=float, default=10.0)
    parser.add_argument("--min_active_samples", type=int, default=50, help="desired minimum number of samples per cycle")
    parser.add_argument("--warmup_inferences", type=int, default=110)
    parser.add_argument("--warmup_seconds", type=float, default=0.0)
    parser.add_argument("--warmup_cooldown_seconds", type=float, default=None, help="it uses --sleep_time if omitted")
    parser.add_argument("--calibration_initial_inferences", type=int, default=10)
    parser.add_argument("--calibration_target_seconds", type=float, default=0.5)
    parser.add_argument("--calibration_repetitions", type=int, default=5, help="at least 2 because the first one in discarded")
    parser.add_argument("--max_relative_mad", type=float, default=0.15, help="maximum accepted relative MAD and coefficient of variation")
    parser.add_argument("--max_calibration_inferences", type=int, default=1000000)
    parser.add_argument("--leading_idle_seconds", type=float, default=5.0)
    parser.add_argument("--trailing_idle_seconds", type=float, default=5.0)
    parser.add_argument("--safety_margin_seconds", type=float, default=2.0)
    acquisition_group = parser.add_mutually_exclusive_group()
    acquisition_group.add_argument("--wait_for_acquisition", action="store_true", help="Pause after READY so manual INA226 acquisition can be started")
    acquisition_group.add_argument("--stdio_acquisition", action="store_true", help="Coordinate acquisition with the SSH client using JSON stdin/stdout")
    parser.add_argument("--manifest_directory", default=None, help="Directory receiving one JSON manifest per campaign")
    return parser


def main():
    args = build_argument_parser().parse_args()

    if args.backend == "tpu":
        from runner import TFLiteTPURunner
        runner_cls = TFLiteTPURunner
    else:
        from runner import TorchRunner
        runner_cls = TorchRunner

    acquisition_controller = (
        StdioAcquisitionController() if args.stdio_acquisition else None
    )
    manager = RunManager(
        runner_cls=runner_cls,
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
        wait_for_acquisition=args.wait_for_acquisition,
        acquisition_controller=acquisition_controller,
        manifest_directory=args.manifest_directory,
    )
    try:
        manifest = manager.execute()
    except (Exception, KeyboardInterrupt):
        if args.stdio_acquisition and manager.last_manifest is not None:
            print(
                json.dumps(
                    {
                        "event": "RUN_MANIFEST",
                        "campaign_id": manager.last_manifest.get("campaign_id"),
                        "manifest": manager.last_manifest,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        raise
    if args.stdio_acquisition:
        print(
            json.dumps(
                {
                    "event": "RUN_MANIFEST",
                    "campaign_id": manifest["campaign_id"],
                    "manifest": manifest,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
