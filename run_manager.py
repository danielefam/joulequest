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
    stability_limit: float
    is_stable: bool
    sizing_attempts: tuple
    sizing_converged: bool


def calculate_required_burst_seconds(
    target_burst_seconds,
    min_active_samples,
    sampling_rate_hz,
    burst_duration_margin=1.2,
):
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
    ) * float(burst_duration_margin)


def calculate_inference_count(
    stable_latency_seconds,
    target_burst_seconds,
    min_active_samples,
    sampling_rate_hz,
    burst_duration_margin=1.2,
):
    """Smallest number of inferences per burst expected to produce a useful burst"""
    required_seconds = calculate_required_burst_seconds(
        target_burst_seconds,
        min_active_samples,
        sampling_rate_hz,
        burst_duration_margin,
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


def calculate_clock_sync_sample(
    runner_sent_seconds,
    controller_received_seconds,
    controller_sent_seconds,
    runner_received_seconds,
):
    """Calculate one controller-minus-runner monotonic clock observation."""
    timestamps = (
        float(runner_sent_seconds),
        float(controller_received_seconds),
        float(controller_sent_seconds),
        float(runner_received_seconds),
    )
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError("Clock synchronization timestamps must be finite")
    t1, t2, t3, t4 = timestamps
    if t4 < t1 or t3 < t2:
        raise ValueError("Clock synchronization timestamps are out of order")
    round_trip_seconds = (t4 - t1) - (t3 - t2)
    if round_trip_seconds < 0:
        raise ValueError("Clock synchronization round trip cannot be negative")
    return {
        "runner_sent_monotonic_seconds": t1,
        "controller_received_monotonic_seconds": t2,
        "controller_sent_monotonic_seconds": t3,
        "runner_received_monotonic_seconds": t4,
        "runner_midpoint_monotonic_seconds": (t1 + t4) / 2.0,
        "controller_minus_runner_seconds": (
            (t2 - t1) + (t3 - t4)
        ) / 2.0,
        "round_trip_seconds": round_trip_seconds,
        "uncertainty_seconds": round_trip_seconds / 2.0,
    }


class StdioAcquisitionController:
    """Coordinate acquisition owned by the process controlling stdin/stdout."""

    _MIN_ADAPTIVE_CLOCK_SYNC_EXCHANGES = 3
    _CLOCK_SYNC_EARLY_STOP_MARGIN = 0.5

    def __init__(
        self,
        input_stream=None,
        output_stream=None,
        monotonic_fn=time.monotonic,
        max_clock_uncertainty_fraction=None,
    ):
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.monotonic_fn = monotonic_fn
        self.max_clock_uncertainty_fraction = max_clock_uncertainty_fraction
        self.campaign_id = None
        self.plan = None
        self.stop_result = None

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, event, **fields):
        payload = {
            "event": event,
            "monotonic_seconds": self.monotonic_fn(),
            "wall_time_utc": self._utc_now(),
            **fields,
        }
        print(
            json.dumps(payload, sort_keys=True),
            file=self.output_stream,
            flush=True,
        )
        return payload

    def _read_result(self, expected_command, expected_request_id=None):
        line = self.input_stream.readline()
        received_monotonic_seconds = self.monotonic_fn()
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
        if (
            expected_request_id is not None
            and payload.get("request_id") != expected_request_id
        ):
            raise RuntimeError("Clock synchronization request ID does not match")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Acquisition command result must be a dictionary")
        return result, received_monotonic_seconds

    def describe(self, campaign_id, plan):
        self.campaign_id = campaign_id
        self.plan = dict(plan)
        return {
            "status": "PENDING",
            "control_protocol": "stdio_json_v2",
            "acquisition_role": "controller_host",
        }

    def _clock_sync_early_stop_threshold(self):
        if self.plan is None or self.max_clock_uncertainty_fraction is None:
            return None
        try:
            sample_period_seconds = 1.0 / float(self.plan["sampling_rate_hz"])
            threshold_seconds = (
                float(self.max_clock_uncertainty_fraction)
                * sample_period_seconds
                * self._CLOCK_SYNC_EARLY_STOP_MARGIN
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        if not math.isfinite(threshold_seconds) or threshold_seconds <= 0:
            return None
        return threshold_seconds

    @classmethod
    def _clock_sync_can_stop_early(cls, samples, threshold_seconds):
        if (
            threshold_seconds is None
            or len(samples) < cls._MIN_ADAPTIVE_CLOCK_SYNC_EXCHANGES
        ):
            return False
        uncertainties = [sample["uncertainty_seconds"] for sample in samples]
        offsets = [
            sample["controller_minus_runner_seconds"] for sample in samples
        ]
        return (
            max(uncertainties) <= threshold_seconds
            and max(offsets) - min(offsets) <= threshold_seconds
        )

    def synchronize_clock(self, round_name, exchange_count):
        samples = []
        errors = []
        attempted_exchange_count = 0
        early_stop_threshold = self._clock_sync_early_stop_threshold()
        for exchange_index in range(exchange_count):
            attempted_exchange_count = exchange_index + 1
            request_id = f"{round_name}:{exchange_index + 1}"
            request = self._emit(
                "CLOCK_SYNC_REQUEST",
                campaign_id=self.campaign_id,
                round=round_name,
                request_id=request_id,
            )
            try:
                result, runner_received_seconds = self._read_result(
                    "CLOCK_SYNC_RESPONSE",
                    expected_request_id=request_id,
                )
                sample = calculate_clock_sync_sample(
                    request["monotonic_seconds"],
                    result["controller_received_monotonic_seconds"],
                    result["controller_sent_monotonic_seconds"],
                    runner_received_seconds,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                errors.append(
                    {
                        "request_id": request_id,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                continue
            sample["request_id"] = request_id
            samples.append(sample)
            if not errors and self._clock_sync_can_stop_early(
                samples,
                early_stop_threshold,
            ):
                break

        progress = {
            "requested_exchange_count": exchange_count,
            "attempted_exchange_count": attempted_exchange_count,
            "valid_exchange_count": len(samples),
            "stopped_early": attempted_exchange_count < exchange_count,
            "early_stop_threshold_seconds": early_stop_threshold,
        }

        if not samples:
            return {
                "status": "UNAVAILABLE",
                "round": round_name,
                "method": "minimum_round_trip",
                **progress,
                "errors": errors,
            }
        selected = min(samples, key=lambda sample: sample["round_trip_seconds"])
        return {
            "status": "COMPLETE",
            "round": round_name,
            "method": "minimum_round_trip",
            **progress,
            "selected_sample": selected,
            "errors": errors,
        }

    def start(self):
        self._emit(
            "ACQUISITION_START_REQUEST",
            campaign_id=self.campaign_id,
            **self.plan,
        )
        result, _ = self._read_result("ACQUISITION_STARTED")
        return result

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
        self.stop_result, _ = self._read_result("ACQUISITION_STOPPED")
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
        batch_size=1,
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
        calibration_sizing_max_attempts=3,
        calibration_duration_tolerance=0.20,
        max_relative_mad=0.15,
        burst_duration_margin=1.2,
        validation_repetitions=3,
        validation_max_rounds=3,
        validation_safety_margin=1.1,
        validation_cooldown_seconds=None,
        clock_sync_exchanges=10,
        max_clock_uncertainty_fraction=0.50,
        max_calibration_inferences=1_000_000,
        leading_idle_seconds=5.0,
        trailing_idle_seconds=5.0,
        safety_margin_seconds=2.0,
        wait_for_acquisition=False,
        acquisition_controller=None,
        manifest_directory=None,
        sleep_fn=time.sleep,
        input_fn=input,
        monotonic_fn=time.monotonic,
    ):
        self.runner_cls = runner_cls
        self.model_path = model_path
        self.number_of_cycles = number_of_cycles
        self.sleep_time = sleep_time
        self.backend = backend
        self.batch_size = batch_size
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
        self.calibration_sizing_max_attempts = calibration_sizing_max_attempts
        self.calibration_duration_tolerance = calibration_duration_tolerance
        self.max_relative_mad = max_relative_mad
        self.burst_duration_margin = burst_duration_margin
        self.validation_repetitions = validation_repetitions
        self.validation_max_rounds = validation_max_rounds
        self.validation_safety_margin = validation_safety_margin
        self.validation_cooldown_seconds = (
            sleep_time
            if validation_cooldown_seconds is None
            else validation_cooldown_seconds
        )
        self.clock_sync_exchanges = clock_sync_exchanges
        self.max_clock_uncertainty_fraction = (
            max_clock_uncertainty_fraction
        )
        self.max_calibration_inferences = max_calibration_inferences
        self.leading_idle_seconds = leading_idle_seconds
        self.trailing_idle_seconds = trailing_idle_seconds
        self.safety_margin_seconds = safety_margin_seconds
        self.wait_for_acquisition = wait_for_acquisition
        self.acquisition_controller = acquisition_controller
        self.manifest_directory = manifest_directory
        self.sleep_fn = sleep_fn
        self.input_fn = input_fn
        self.monotonic_fn = monotonic_fn
        self.last_manifest = None
        
        if self.calibration_repetitions < 2:
            raise ValueError(
                "calibration_repetitions must include one discarded and "
                "at least one retained batch"
            )
        if isinstance(self.batch_size, bool) or not isinstance(
            self.batch_size, int
        ) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.calibration_target_seconds <= 0:
            raise ValueError("calibration_target_seconds must be positive")
        if self.calibration_sizing_max_attempts < 1:
            raise ValueError("calibration_sizing_max_attempts must be positive")
        if not 0 <= self.calibration_duration_tolerance < 1:
            raise ValueError(
                "calibration_duration_tolerance must be in the range [0, 1)"
            )
        if self.max_relative_mad < 0:
            raise ValueError("max_relative_mad cannot be negative")
        if self.burst_duration_margin < 1:
            raise ValueError("burst_duration_margin must be at least 1")
        if self.validation_repetitions < 1:
            raise ValueError("validation_repetitions must be positive")
        if self.validation_max_rounds < 1:
            raise ValueError("validation_max_rounds must be positive")
        if self.validation_safety_margin < 1:
            raise ValueError("validation_safety_margin must be at least 1")
        if self.validation_cooldown_seconds < 0:
            raise ValueError("validation_cooldown_seconds cannot be negative")
        if self.clock_sync_exchanges < 1:
            raise ValueError("clock_sync_exchanges must be positive")
        if self.max_clock_uncertainty_fraction <= 0:
            raise ValueError(
                "max_clock_uncertainty_fraction must be positive"
            )
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
            "monotonic_seconds": self.monotonic_fn(),
            "wall_time_utc": self._utc_now(),
            **fields,

        }
        if verbose:
            print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    def _new_campaign_id(self):
        model_stem = Path(self.model_path).stem.replace(" ", "-")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return (
            f"{timestamp}_{model_stem}_bs{self.batch_size}_"
            f"{uuid.uuid4().hex[:8]}"
        )

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

        sizing_results = []
        sizing_attempts = []
        sizing_converged = False
        for attempt in range(1, self.calibration_sizing_max_attempts + 1):
            sizing_result = runner.run_burst(batch_inferences)
            sizing_results.append(sizing_result)
            target_ratio = (
                sizing_result.elapsed_seconds / self.calibration_target_seconds
            )
            sizing_attempts.append(
                {
                    "attempt": attempt,
                    "inferences": batch_inferences,
                    "elapsed_seconds": sizing_result.elapsed_seconds,
                    "target_ratio": target_ratio,
                }
            )
            if (
                abs(target_ratio - 1.0) <= self.calibration_duration_tolerance
                or batch_inferences == self.max_calibration_inferences
            ):
                sizing_converged = (
                    abs(target_ratio - 1.0)
                    <= self.calibration_duration_tolerance
                )
                break

            adjusted_count = min(
                self.max_calibration_inferences,
                max(
                    self.calibration_initial_inferences,
                    math.ceil(
                        batch_inferences
                        * self.calibration_target_seconds
                        / sizing_result.elapsed_seconds
                    ),
                ),
            )
            if adjusted_count == batch_inferences:
                break
            if attempt == self.calibration_sizing_max_attempts:
                break
            batch_inferences = adjusted_count

        batches = [
            sizing_results[-1],
            *[
                runner.run_burst(batch_inferences)
                for _ in range(self.calibration_repetitions - 1)
            ],
        ]

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
        is_stable = not (
            relative_mad > self.max_relative_mad
            or coefficient_of_variation > self.max_relative_mad
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
                + sum(
                    result.executed_inferences for result in sizing_results
                )
                + sum(
                    result.executed_inferences for result in batches[1:]
                )
            ),
            relative_mad=relative_mad,
            coefficient_of_variation=coefficient_of_variation,
            stability_limit=self.max_relative_mad,
            is_stable=is_stable,
            sizing_attempts=tuple(sizing_attempts),
            sizing_converged=sizing_converged,
        )

    def _select_inference_count(self, calibration):
        if self.inferences_per_cycle is None:
            return calculate_inference_count(
                calibration.stable_latency_seconds,
                self.target_burst_seconds,
                self.min_active_samples,
                self.sampling_rate_hz,
                self.burst_duration_margin,
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

    def _validate_inference_count(
        self,
        runner,
        inference_count,
        selection_mode,
        required_burst_seconds,
    ):
        rounds = []
        total_executed_inferences = 0

        for round_number in range(1, self.validation_max_rounds + 1):
            results = []
            for _ in range(self.validation_repetitions):
                self.sleep_fn(self.validation_cooldown_seconds)
                results.append(runner.run_burst(inference_count))

            durations = [result.elapsed_seconds for result in results]
            total_executed_inferences += sum(
                result.executed_inferences for result in results
            )
            minimum_duration = min(durations)
            median_duration = statistics.median(durations)
            passed = minimum_duration >= required_burst_seconds
            rounds.append(
                {
                    "round": round_number,
                    "inferences_per_burst": inference_count,
                    "durations_seconds": durations,
                    "minimum_duration_seconds": minimum_duration,
                    "median_duration_seconds": median_duration,
                    "passed": passed,
                }
            )
            if passed:
                return inference_count, {
                    "included_in_measurement": False,
                    "required_burst_seconds": required_burst_seconds,
                    "repetitions_per_round": self.validation_repetitions,
                    "cooldown_seconds": self.validation_cooldown_seconds,
                    "safety_margin": self.validation_safety_margin,
                    "rounds": rounds,
                    "adjusted": len(rounds) > 1,
                    "passed": True,
                    "final_minimum_burst_seconds": minimum_duration,
                    "final_median_burst_seconds": median_duration,
                    "total_executed_inferences": total_executed_inferences,
                }

            if selection_mode == "manual_override":
                raise ValueError(
                    "The --inferences_per_cycle override produced a minimum "
                    f"validated duration of {minimum_duration} seconds, below "
                    f"the required duration ({required_burst_seconds} seconds)"
                )

            adjusted_count = max(
                inference_count + 1,
                math.ceil(
                    inference_count
                    * required_burst_seconds
                    / minimum_duration
                    * self.validation_safety_margin
                ),
            )
            if adjusted_count > self.max_calibration_inferences:
                raise RuntimeError(
                    "Validated inference count exceeds "
                    f"max_calibration_inferences ({self.max_calibration_inferences})"
                )
            inference_count = adjusted_count

        raise RuntimeError(
            "Unable to validate a burst duration that satisfies the active "
            f"sample requirement after {self.validation_max_rounds} rounds"
        )

    def _build_measurement_plan(
        self,
        calibration,
        inference_count,
        selection_mode,
        validation,
    ):
        """Build the immutable plan used by the INA226 operator."""
        required_burst_seconds = calculate_required_burst_seconds(
            self.target_burst_seconds,
            self.min_active_samples,
            self.sampling_rate_hz,
            self.burst_duration_margin,
        )
        estimated_burst_seconds = validation["final_median_burst_seconds"]
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
            "required_burst_seconds": required_burst_seconds,
            "burst_duration_margin": self.burst_duration_margin,
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
            "quality_flags": [
                *([] if calibration.is_stable else ["CALIBRATION_UNSTABLE"]),
                *(
                    []
                    if calibration.sizing_converged
                    else ["CALIBRATION_SIZING_NOT_CONVERGED"]
                ),
            ],
        }
        self._emit_event(
            "CALIBRATION_END",
            verbose=True,
            campaign_id=campaign_id,
            **manifest["calibration"],
        )

        required_burst_seconds = calculate_required_burst_seconds(
            self.target_burst_seconds,
            self.min_active_samples,
            self.sampling_rate_hz,
            self.burst_duration_margin,
        )
        inference_count, selection_mode = self._select_inference_count(
            calibration
        )
        inference_count, validation = self._validate_inference_count(
            runner,
            inference_count,
            selection_mode,
            required_burst_seconds,
        )
        manifest["burst_validation"] = validation
        plan = self._build_measurement_plan(
            calibration,
            inference_count,
            selection_mode,
            validation,
        )
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
            verbose=False,
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
            verbose=False,
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

    def _synchronize_clock_round(self, round_name):
        synchronize = getattr(
            self.acquisition_controller,
            "synchronize_clock",
            None,
        )
        if not callable(synchronize):
            return {
                "status": "UNAVAILABLE",
                "round": round_name,
                "method": "unavailable",
                "requested_exchange_count": self.clock_sync_exchanges,
                "valid_exchange_count": 0,
                "errors": [
                    {
                        "type": "ClockSynchronizationUnavailable",
                        "message": (
                            "Acquisition controller does not support clock "
                            "synchronization"
                        ),
                    }
                ],
            }
        try:
            result = synchronize(round_name, self.clock_sync_exchanges)
            if not isinstance(result, dict):
                raise TypeError(
                    "clock synchronization result must be a dictionary"
                )
            return result
        except Exception as error:
            return {
                "status": "UNAVAILABLE",
                "round": round_name,
                "method": "unavailable",
                "requested_exchange_count": self.clock_sync_exchanges,
                "valid_exchange_count": 0,
                "errors": [
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                ],
            }

    @staticmethod
    def _validated_sync_values(round_result):
        if not isinstance(round_result, dict):
            raise ValueError("Clock synchronization round is missing")
        if round_result.get("status") != "COMPLETE":
            raise ValueError("Clock synchronization round is unavailable")
        sample = round_result.get("selected_sample")
        if not isinstance(sample, dict):
            raise ValueError("Clock synchronization sample is missing")
        required = (
            "runner_midpoint_monotonic_seconds",
            "controller_minus_runner_seconds",
            "round_trip_seconds",
            "uncertainty_seconds",
        )
        values = {}
        for field in required:
            value = float(sample[field])
            if not math.isfinite(value):
                raise ValueError(
                    f"Clock synchronization field {field} must be finite"
                )
            values[field] = value
        if values["round_trip_seconds"] < 0 or values["uncertainty_seconds"] < 0:
            raise ValueError("Clock synchronization uncertainty cannot be negative")
        return values

    @staticmethod
    def _translate_measurement_cycles(
        cycles,
        capture_elapsed,
        pre_midpoint,
        post_midpoint,
        pre_offset,
        post_offset,
        logger_start,
    ):
        translated_cycles = 0
        bounds_valid = capture_elapsed is not None and capture_elapsed > 0
        for cycle in cycles:
            translated = []
            for event_name in ("start_event", "end_event"):
                event = cycle.get(event_name, {})
                try:
                    runner_seconds = float(event["monotonic_seconds"])
                except (KeyError, TypeError, ValueError):
                    bounds_valid = False
                    break
                position = (
                    (runner_seconds - pre_midpoint)
                    / (post_midpoint - pre_midpoint)
                )
                offset = pre_offset + position * (post_offset - pre_offset)
                controller_seconds = runner_seconds + offset
                elapsed_seconds = controller_seconds - logger_start
                if not math.isfinite(elapsed_seconds):
                    bounds_valid = False
                    break
                event["aligned_controller_monotonic_seconds"] = controller_seconds
                event["aligned_elapsed_seconds"] = elapsed_seconds
                translated.append(elapsed_seconds)
            if len(translated) != 2:
                continue
            start_elapsed, end_elapsed = translated
            if not 0 <= start_elapsed < end_elapsed <= capture_elapsed:
                bounds_valid = False
                continue
            translated_cycles += 1
        return translated_cycles, bounds_valid

    def _finalize_clock_alignment(self, manifest, pre_sync, post_sync):
        threshold_seconds = (
            self.max_clock_uncertainty_fraction / self.sampling_rate_hz
        )
        alignment = {
            "protocol_version": 2,
            "status": "UNAVAILABLE",
            "method": "minimum_round_trip",
            "pre_sync": pre_sync,
            "post_sync": post_sync,
            "uncertainty_threshold_fraction_of_sample_period": (
                self.max_clock_uncertainty_fraction
            ),
            "uncertainty_threshold_seconds": threshold_seconds,
            "classification_eligible": False,
            "fallback_reason": "CLOCK_SYNC_UNAVAILABLE",
        }
        manifest.setdefault("acquisition", {})["clock_alignment"] = alignment

        try:
            pre_values = self._validated_sync_values(pre_sync)
            post_values = self._validated_sync_values(post_sync)
        except (KeyError, TypeError, ValueError) as error:
            alignment["fallback_reason"] = "CLOCK_SYNC_METADATA_INVALID"
            alignment["validation_error"] = str(error)
            return alignment

        pre_midpoint = pre_values["runner_midpoint_monotonic_seconds"]
        post_midpoint = post_values["runner_midpoint_monotonic_seconds"]
        if post_midpoint <= pre_midpoint:
            alignment["fallback_reason"] = "CLOCK_SYNC_ROUNDS_OUT_OF_ORDER"
            return alignment

        acquisition = manifest["acquisition"]
        try:
            logger_start = float(acquisition["started_monotonic_seconds"])
        except (KeyError, TypeError, ValueError):
            alignment["fallback_reason"] = "LOGGER_MONOTONIC_ORIGIN_UNAVAILABLE"
            return alignment
        if not math.isfinite(logger_start):
            alignment["fallback_reason"] = "LOGGER_MONOTONIC_ORIGIN_INVALID"
            return alignment

        pre_offset = pre_values["controller_minus_runner_seconds"]
        post_offset = post_values["controller_minus_runner_seconds"]
        uncertainty_seconds = max(
            pre_values["uncertainty_seconds"],
            post_values["uncertainty_seconds"],
        )
        alignment.update(
            {
                "status": "COMPLETE",
                "method": (
                    "same_host_monotonic"
                    if pre_sync.get("method") == "same_host_monotonic"
                    and post_sync.get("method") == "same_host_monotonic"
                    else "minimum_round_trip_with_linear_drift"
                ),
                "uncertainty_seconds": uncertainty_seconds,
                "offset_drift_seconds": post_offset - pre_offset,
                "logger_started_monotonic_seconds": logger_start,
            }
        )

        capture_elapsed = acquisition.get("capture_elapsed_seconds")
        try:
            capture_elapsed = float(capture_elapsed)
        except (TypeError, ValueError):
            capture_elapsed = None
        cycles = manifest.get("measurement", {}).get("cycles", [])
        translated_cycles, bounds_valid = self._translate_measurement_cycles(
            cycles=cycles,
            capture_elapsed=capture_elapsed,
            pre_midpoint=pre_midpoint,
            post_midpoint=post_midpoint,
            pre_offset=pre_offset,
            post_offset=post_offset,
            logger_start=logger_start,
        )
        expected_cycles = len(cycles)
        alignment["translated_cycle_count"] = translated_cycles
        if not bounds_valid or translated_cycles != expected_cycles:
            alignment["fallback_reason"] = "ALIGNED_BURST_BOUNDS_INVALID"
            return alignment
        if uncertainty_seconds > threshold_seconds:
            alignment["fallback_reason"] = "CLOCK_UNCERTAINTY_EXCEEDED"
            return alignment
        alignment["classification_eligible"] = True
        alignment["fallback_reason"] = None
        return alignment

    def _complete_manifest(self, manifest, manifest_path, campaign_id):
        """Add quality status, persist the final manifest, and emit COMPLETE."""
        all_flags = [
            flag
            for cycle in manifest["measurement"]["cycles"]
            for flag in cycle["quality_flags"]
        ]
        all_flags.extend(manifest.get("calibration", {}).get("quality_flags", []))
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

    @staticmethod
    def _input_batch_size(runner):
        batch_size = getattr(runner, "input_batch_size", 1)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("input_batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("input_batch_size must be a positive integer")
        return batch_size

    def execute(self):
        """Execute one campaign and return its complete manifest dictionary."""
        campaign_id = self._new_campaign_id()
        manifest_path = self._manifest_path(campaign_id)
        manifest = {
            "schema_version": 2,
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
            runner = self.runner_cls(
                self.model_path,
                self.backend,
                batch_size=self.batch_size,
            )
            runner.prepare()
            manifest["input_batch_size"] = self._input_batch_size(runner)

            plan = self._prepare_acquisition(runner, campaign_id, manifest)
            if self.acquisition_controller is not None:
                manifest["acquisition"] = self._describe_acquisition(
                    campaign_id,
                    plan,
                )
            self._write_manifest(manifest, manifest_path)
            self._emit_event("READY", campaign_id=campaign_id, **plan)

            pre_sync = None
            if self.acquisition_controller is not None:
                pre_sync = self._synchronize_clock_round("pre_acquisition")
                manifest["acquisition"]["clock_alignment"] = {
                    "protocol_version": 2,
                    "status": "PENDING",
                    "pre_sync": pre_sync,
                }
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
                post_sync = self._synchronize_clock_round("post_acquisition")
                self._finalize_clock_alignment(
                    manifest,
                    pre_sync,
                    post_sync,
                )
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
    parser.add_argument("--batch-size", type=int, default=1)
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
    parser.add_argument("--calibration-sizing-max-attempts", type=int, default=3, help="maximum adaptive attempts used to reach the calibration batch target")
    parser.add_argument("--calibration-duration-tolerance", type=float, default=0.20, help="accepted relative difference from the calibration batch target")
    parser.add_argument("--max_relative_mad", type=float, default=0.15, help="maximum accepted relative MAD and coefficient of variation")
    parser.add_argument("--burst-duration-margin", type=float, default=1.2, help="safety factor applied to automatically planned burst duration")
    parser.add_argument("--validation-repetitions", type=int, default=3, help="excluded final-count validation bursts per round")
    parser.add_argument("--validation-max-rounds", type=int, default=3, help="maximum validation and correction rounds")
    parser.add_argument("--validation-safety-margin", type=float, default=1.1, help="extra count margin applied after failed validation")
    parser.add_argument("--validation-cooldown-seconds", type=float, default=None, help="pause before each validation burst; uses --sleep_time if omitted")
    parser.add_argument("--clock-sync-exchanges", type=int, default=10, help="maximum clock exchanges per pre/post round; precise links stop after at least three")
    parser.add_argument("--max-clock-uncertainty-fraction", type=float, default=0.50, help="maximum alignment uncertainty as a fraction of one sample period")
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

    acquisition_controller = None
    if args.stdio_acquisition:
        acquisition_controller = StdioAcquisitionController(
            max_clock_uncertainty_fraction=(
                args.max_clock_uncertainty_fraction
            )
        )
    manager = RunManager(
        runner_cls=runner_cls,
        model_path=args.model,
        number_of_cycles=args.number_of_cycles,
        sleep_time=args.sleep_time,
        backend=args.backend,
        batch_size=args.batch_size,
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
