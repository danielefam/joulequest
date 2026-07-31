# `run_manager.py` Update

**Document date:** 2026-07-31
**Scope:** Changes made to `run_manager.py` only

## 1. Purpose of the update

`run_manager.py` originally supported a manual acquisition workflow. It prepared
the inference runner, excluded warm-up and calibration from measurement, emitted
a `READY` event, and optionally waited for the user to start and stop INA226
acquisition.

The updated file keeps that workflow available while adding a programmatic
acquisition boundary. An external controller can now start acquisition after
calibration, allow the measured campaign to run, stop acquisition after the
final baseline interval, and receive the completed manifest without requiring
an operator to press Enter.

The update does not move INA226 register access or CSV generation into
`run_manager.py`. This file remains responsible for inference preparation,
adaptive planning, measured-cycle execution, event generation, manifest
generation, and acquisition lifecycle coordination.

## 2. Measurement semantics preserved

The update preserves the existing three workload phases:

1. **Warm-up:** removes cold-start effects and remains outside acquisition.
2. **Calibration:** estimates steady-state latency and remains outside
   acquisition.
3. **Measurement:** contains the useful inference cycles recorded by the power
   logger.

The following rules remain unchanged:

- warm-up and calibration have `included_in_measurement: false`;
- the measured inference count is initially selected from calibration, then
  validated before acquisition unless an explicit valid override is supplied;
- `runner.prepare_burst()` remains outside the timed active interval;
- only `runner.run_prepared_burst()` contributes to measured-cycle duration;
- leading and trailing idle intervals remain available for baseline analysis;
- idle sleep occurs only between measured cycles, never after the last cycle;
- the safety margin is acquisition-only idle time, not useful workload;
- actual requested/executed inference counts and elapsed times are retained per
  cycle.

## 3. New acquisition-controller interface

`RunManager` now accepts an optional `acquisition_controller` argument. The
controller follows four operations:

```python
describe(campaign_id, plan) -> dict
start() -> dict
check_health() -> dict | None
stop() -> dict | None
```

`RunManager` merges returned dictionaries into `manifest["acquisition"]`.
Controller exceptions do not silently disappear: they set acquisition status to
`FAILED` and store failure type, message, and UTC failure time.

An acquisition failure is monotonic. Once acquisition is marked `FAILED`, a
later cleanup result cannot replace it with `COMPLETE`.

Manual and automated acquisition are mutually exclusive. Constructing a
`RunManager` with both `wait_for_acquisition=True` and an acquisition controller
raises `ValueError`.

## 4. Automated lifecycle ordering

With an acquisition controller, `RunManager.execute()` performs this sequence:

1. Create the campaign ID and the initial `STARTING` manifest.
2. Construct and prepare the inference runner.
3. Run warm-up and its cooldown.
4. Run calibration and validate timing stability.
5. Validate the planned inference count with excluded bursts and correct it if
  automatic sizing is too short.
6. Build the adaptive measurement plan from the validated count.
7. Add the acquisition description with status `PENDING`.
8. Atomically write the `READY` manifest.
9. Emit the `READY` event.
10. Start acquisition and wait for the controller result.
11. Record leading idle.
12. Execute all measured cycles and inter-cycle idle intervals.
13. Record trailing idle.
14. Record `safety_margin_seconds` of additional idle.
15. Stop acquisition and wait for cleanup.
16. Atomically write the final manifest.
17. Emit `COMPLETE`.

This ordering guarantees that model setup, warm-up, and calibration happen
before acquisition. It also guarantees that the final manifest write happens
after acquisition has stopped, preventing final JSON file I/O from entering the
power trace.

If workload execution raises an exception or receives `KeyboardInterrupt`,
`RunManager` first attempts to stop acquisition, then writes a `FAILED`
manifest, emits `FAILED`, closes the runner, and re-raises the original error.

## 5. `StdioAcquisitionController`

The updated file includes `StdioAcquisitionController`, used when
`run_manager.py` is controlled through stdin/stdout by another process. It does
not open a serial port and does not know any network or host names. It exchanges
newline-delimited JSON messages with the controlling process.

`StdioAcquisitionController.check_health()` intentionally returns `None`. The
stdio protocol is request/response based and does not consume unsolicited
status messages while a measured burst is running. If acquisition fails after
the start acknowledgement, the controlling process reports that failure in the
eventual `ACQUISITION_STOPPED` result; `RunManager` then records
`ACQUISITION_FAILED` and completes the workload with `quality_status: REVIEW`.

### Start handshake

`run_manager.py` writes:

```json
{
  "event": "ACQUISITION_START_REQUEST",
  "campaign_id": "<campaign_id>",
  "sampling_rate_hz": 100.0,
  "capture_seconds": 23.3,
  "capture_samples": 2331
}
```

The controller replies on stdin:

```json
{
  "command": "ACQUISITION_STARTED",
  "campaign_id": "<campaign_id>",
  "result": {
    "status": "RUNNING",
    "sample_count": 1
  }
}
```

The leading idle interval begins only after this reply is received.

### Stop handshake

After trailing idle and the safety margin, `run_manager.py` writes:

```json
{
  "event": "ACQUISITION_STOP_REQUEST",
  "campaign_id": "<campaign_id>"
}
```

The controller replies:

```json
{
  "command": "ACQUISITION_STOPPED",
  "campaign_id": "<campaign_id>",
  "result": {
    "status": "COMPLETE",
    "sample_count": 2405
  }
}
```

The final manifest is written only after this stop reply.

The protocol validates command names, campaign IDs, JSON structure, and result
types. A disconnect, malformed JSON message, wrong command, or mismatched
campaign ID raises an error handled by the normal failure path.

## 6. CLI additions

The acquisition-boundary options are:

| Option | Meaning |
| --- | --- |
| `--wait_for_acquisition` | Preserve the manual Enter-based start/stop workflow |
| `--stdio_acquisition` | Coordinate acquisition through JSON stdin/stdout |

They belong to one mutually exclusive argument group.

Adaptive sizing also accepts the following controls:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--calibration-sizing-max-attempts` | `3` | Maximum attempts to make a calibration batch reach its target duration |
| `--calibration-duration-tolerance` | `0.20` | Accepted relative error from the calibration batch target duration |
| `--validation-repetitions` | `3` | Excluded final-count validation bursts per validation round |
| `--validation-max-rounds` | `3` | Maximum validation and automatic correction rounds |
| `--validation-safety-margin` | `1.1` | Extra count factor after a failed automatic validation round |
| `--validation-cooldown-seconds` | `sleep_time` | Pause before each validation burst when explicitly set, otherwise inherited from `--sleep_time` |

The remote-controlled mode is selected with:

```bash
python run_manager.py \
  --backend cuda \
  --model Models/CUDA/Linear/Linear_64_64.pt \
  --manifest_directory measurement_manifests \
  --stdio_acquisition
```

`--stdio_acquisition` is intended to be started by an orchestrating process,
not entered interactively in a terminal, because it blocks while waiting for
the JSON start and stop replies.

## 7. Calibration stability and duration validation

Calibration starts with a sizing pilot and scales toward
`calibration_target_seconds`. It then measures and rescales the calibration
batch up to `calibration_sizing_max_attempts` times. Sizing converges when the
observed duration is within `calibration_duration_tolerance` of the target. The
final sizing attempt is reused as the first discarded full calibration batch;
the retained batches estimate steady-state latency.

The update adds `max_relative_mad`, with default value `0.15`. For retained
per-inference latencies it computes:

```text
stable latency = median(latencies)
relative MAD = median(abs(latency - stable latency)) / stable latency
coefficient of variation = population standard deviation / mean
```

When either relative MAD or coefficient of variation is greater than
`max_relative_mad`, calibration is retained with `is_stable: false` and
`CALIBRATION_UNSTABLE`. Measurement continues and completes with
`quality_status: REVIEW`; no captured cycle is discarded for this condition.

The accepted metrics are stored in the manifest under `calibration`:

```json
{
  "relative_mad": 0.0041,
  "coefficient_of_variation": 0.0052
}
```

This validation prevents an unstable latency estimate from silently selecting
an unreliable measured inference count.

Failure to reach the calibration duration target does not discard a capture by
itself. It records `CALIBRATION_SIZING_NOT_CONVERGED`, alongside any
`CALIBRATION_UNSTABLE` flag, so that the completed campaign receives
`quality_status: REVIEW`.

### Final-count validation

The latency median produces only an initial `inferences_per_cycle`. Before
`READY` and before acquisition starts, `RunManager` executes
`validation_repetitions` excluded bursts with that exact count. Each validation
burst receives a fresh workload state and is separated by the validation
cooldown. The shortest observed duration must meet $T_{required}$.

For an automatically selected count that fails validation, the next count is:

$$
N_{next}=\left\lceil
N_{current}\frac{T_{required}}{T_{validation,min}}m_v
\right\rceil
$$

where $m_v$ is `validation_safety_margin`. The corrected count is measured
again, up to `validation_max_rounds`. This protects the capture from a faster
GPU operating regime caused, for example, by power, DVFS, or thermal changes.

An explicit `--inferences_per_cycle` override is never silently modified. It
raises `ValueError` before acquisition if its shortest validation burst is too
short. An automatic count that cannot pass validation also fails before
acquisition instead of creating known under-resolved regions.

## 8. Adaptive inference and capture planning

The automatic count continues to use:

$$
T_{required}=1.2\max\left(T_{target},\frac{N_{samples,min}}{f_s}\right)
$$

$$
N_{initial}=\max\left(1,\left\lceil
\frac{T_{required}}{t_{inference}}\right\rceil\right)
$$

The final value of $N_{inferences}$ is the count that passes the pre-acquisition
validation described above. It remains fixed for every measured cycle.

The capture plan remains:

$$
T_{capture}=T_{leading}+C\,T_{burst}+(C-1)T_{sleep}
+T_{trailing}+T_{margin}
$$

$$
N_{capture}=\left\lceil T_{capture}f_s\right\rceil
$$

An explicit `--inferences_per_cycle` override remains supported, but it is
rejected both when its calibration estimate would produce fewer than
`min_active_samples` and when validation measures an insufficient minimum burst
duration.

## 9. Manifest updates

New automated manifests use `schema_version: 2`. Schema v2 adds a logger
monotonic origin, pre/post minimum-RTT synchronization rounds, uncertainty and
drift metadata, and `aligned_elapsed_seconds` on every burst boundary. Existing
schema-v1 files remain readable but processing classifies them with Otsu.

The stdio protocol is `stdio_json_v2`. Ten `CLOCK_SYNC_REQUEST` /
`CLOCK_SYNC_RESPONSE` exchanges run before acquisition and ten after it. The
alignment gate is inclusive:

$$
u_{clock} \leq \frac{0.10}{f_s}
$$

Above the gate, or when synchronization metadata is incomplete, processing
uses Otsu plus hysteresis and retains the capture.

### Workload policy

The initial manifest now records how inputs and parameters change:

```json
{
  "workload_policy": {
    "parameters": "fresh_per_burst",
    "input": "fresh_per_burst"
  }
}
```

For the TPU compatibility path, parameters are reported as `fixed_model` while
input remains `fresh_per_burst`.

### Acquisition metadata

When a controller is present, the manifest receives an `acquisition` object.
Its exact hardware/sample fields come from the external controller, while
`RunManager` owns status merging and failure recording.

### Partial measurement retention

`manifest["measurement"]` is initialized before the first measured cycle.
Each completed cycle is appended immediately. If a later cycle fails, the
`FAILED` manifest retains all cycles that completed before the failure.

### Status and quality

The campaign status flow is:

```text
STARTING -> READY -> COMPLETE
                  -> FAILED
```

Cycle quality flags remain:

- `UNDER_RESOLVED`: actual estimated samples are below `min_active_samples`;
- `DURATION_ANOMALY`: actual duration is below half or above twice the
  calibration estimate.

Calibration can additionally contribute `CALIBRATION_UNSTABLE`. Poor clock
alignment is recorded as a classifier fallback reason, not as a campaign
quality failure.

`CALIBRATION_SIZING_NOT_CONVERGED` is also a calibration quality flag. It means
the sizing attempts did not reach the requested calibration duration; the
separate final-count validation still has to pass before acquisition can begin.

### Calibration and validation metadata

The calibration section now records the excluded sizing attempts:

```json
{
  "sizing_attempts": [
    {"attempt": 1, "inferences": 640, "elapsed_seconds": 0.52},
    {"attempt": 2, "inferences": 1231, "elapsed_seconds": 1.01}
  ],
  "sizing_converged": true
}
```

The top-level `burst_validation` object records every excluded final-count
round, its durations, whether it passed, the final minimum and median duration,
and the total excluded inference count. These bursts are not included in
`measurement.total_executed_inferences` or power/energy normalization.

If acquisition fails while the workload completes, the campaign remains
`COMPLETE`, gains `ACQUISITION_FAILED`, and receives
`quality_status: REVIEW`. A workload exception instead produces campaign status
`FAILED`.

## 10. Event output

All lifecycle events are emitted as one-line JSON records with both monotonic
and UTC timestamps:

```text
WARMUP_START
WARMUP_END
CALIBRATION_START
CALIBRATION_END
READY
BURST_START
BURST_END
COMPLETE or FAILED
```

The stdio acquisition mode additionally emits:

```text
ACQUISITION_START_REQUEST
ACQUISITION_STOP_REQUEST
RUN_MANIFEST
```

After `RunManager.execute()` returns, `main()` emits `RUN_MANIFEST` containing
the complete manifest. If execution fails, it emits the current failed manifest
before re-raising the error. This allows the controlling process to preserve
the manifest without reading the remote filesystem during the normal path.

## 11. Atomic writes and cleanup

Manifest writes remain atomic:

1. serialize to `<campaign_id>.json.tmp`;
2. replace `<campaign_id>.json` with `os.replace()`.

This prevents readers from observing partially written JSON.

Runner cleanup remains in `finally`, so `runner.close()` is called after normal
completion, workload failure, acquisition failure, or interruption. Acquisition
stop is idempotently guarded by `acquisition_stopped` to avoid duplicate normal
stop requests.

## 12. Backward compatibility

The following modes remain supported:

- runner-only execution without acquisition control;
- manual acquisition with `--wait_for_acquisition`;
- explicit `--inferences_per_cycle` campaigns;
- iterative calibration sizing and excluded final-count validation;
- CPU, CUDA, and TPU runner selection;
- the existing event names and manifest schema version;
- use of `RunManager` as a Python class with injected `sleep_fn` and `input_fn`
  for deterministic tests.

The new automated mode is opt-in through an injected controller or
`--stdio_acquisition`.

## 13. Test coverage

`tests/test_run_manager.py` covers the updated contracts without requiring
measurement hardware:

- required burst duration and rounded inference count;
- capture planning with exactly `C - 1` inter-cycle sleeps;
- warm-up, calibration, and measured-work accounting;
- workload policy metadata;
- rejection of a too-short manual inference override;
- rejection of unstable calibration;
- iterative calibration correction after a pilot regime change;
- final-count correction when validation observes a faster execution regime;
- complete lifecycle event ordering;
- acquisition start/stop ordering around the capture window;
- safety margin before acquisition stop;
- acquisition-start failure producing `COMPLETE/REVIEW`;
- workload failure preserving completed cycles;
- acquisition cleanup before writing a failed manifest;
- stdio start/stop handshake validation;
- final `RUN_MANIFEST` emission for the controlling process.

The focused validation command is:

```bash
python -m unittest tests.test_run_manager -v
```

## 14. Summary

The `run_manager.py` update turns the previous manual acquisition boundary into
a reusable, testable controller interface while preserving the original
measurement protocol. Warm-up and calibration remain excluded, inference count
selection remains adaptive, measured cycles remain the only useful workload,
idle baselines remain available, acquisition stops before final manifest I/O,
and all planned and actual execution information remains recorded.