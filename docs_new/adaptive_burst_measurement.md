# Adaptive Clean-Burst Measurement Protocol

**Implementation status:** automated direct-serial acquisition with validated burst sizing, 2026-07-31
**Validated priority:** PyTorch CPU/CUDA
**Compatibility only:** Edge TPU remains available but is not part of the current validation campaign.

## 1. Objective

The runner must produce rectangular, measurable active regions without repeating the historical fixed count of 100,000 inferences for every layer. Fast layers need more repetitions than slow layers, but every measured region must be long enough to contain useful INA226 samples.

The implementation separates all work into three phases:

1. **Warm-up (not measured):** removes cold-start effects.
2. **Calibration (not measured):** estimates stable per-inference latency.
3. **Measured cycles:** executes only the count selected from calibration.

Warm-up and calibration counts are written to the JSON manifest with `included_in_measurement: false`. They must never be used for power-region count or energy normalization.

## 2. Why the first burst is discarded

The first forward burst can be much slower because it may include:

- CUDA context and kernel initialization;
- framework memory allocation;
- cache population;
- backend or driver initialization;
- one-time execution-path setup.

For this reason, the first burst is intentionally retained as a warm-up operation and discarded. This is not an accidental duplicate measurement. By default, `warmup_inferences=110` and `warmup_seconds=0`, which means exactly one discarded burst, matching the proven original workaround. A cooldown then runs before calibration; by default it equals `sleep_time` (10 s). An experiment can set a positive `warmup_seconds` to require longer warm-up. In that mode, the first burst's observed latency sizes the next warm-up batch so that fast layers do not require thousands of tiny Python-level batches. Every warm-up batch and the cooldown remain outside acquisition and measured totals.

Before every burst, the runner generates a new random input and a new model
parameter state. This preparation happens outside the timed interval, so input
generation and parameter initialization are not counted as layer-forward
energy. All inferences inside one burst use that burst's input and parameters;
the next warm-up, calibration, or measured burst receives a new pair.

This protocol measures the average cost of several randomly initialized
workload states. It is appropriate for the current fixed-shape `Linear` and
`Conv2d` benchmarks. Models with data-dependent control flow, dynamic shapes,
sparsity, or caching may require a separate input/parameter policy.

This fresh-parameter policy applies to the active PyTorch CPU/CUDA runner.
Compiled TFLite/Edge TPU models cannot replace their parameters at runtime;
the TPU runner remains compatibility-only and can regenerate inputs but not
weights.

## 3. Meaning of `T`

`T` means **time**, expressed in seconds:

- $T_{target}$: requested duration of one measured active region.
- $T_{samples}$: duration required to collect the minimum number of INA226 samples.
- $T_{required}$: effective minimum burst duration.
- $T_{capture}$: total planned acquisition duration.

The effective burst requirement is:

$$
T_{required}=m\max\left(T_{target},\frac{N_{samples,min}}{f_s}\right)
$$

where:

- $N_{samples,min}$ is `min_active_samples`;
- $f_s$ is `sampling_rate_hz` in samples per second.
- $m$ is `burst_duration_margin`, with default value $1.2$.

After calibration estimates steady-state per-inference latency $t_{inference}$, the measured inference count is:

$$
N_{inferences}=\max\left(1,\left\lceil\frac{T_{required}}{t_{inference}}\right\rceil\right)
$$

The default values are $T_{target}=10$ s, $N_{samples,min}=50$, and $f_s=10$ Hz. These are initial experimental values, not final constants. Representative 5 s, 10 s, and 20 s campaigns must be compared before freezing a profile.

## 4. Stable calibration

Calibration happens after warm-up and before INA226 acquisition.

1. Run a small sizing pilot (`calibration_initial_inferences`).
2. Scale the batch count toward `calibration_target_seconds`.
3. Measure and rescale the batch up to `calibration_sizing_max_attempts` times
   until its duration is within `calibration_duration_tolerance` of the target.
4. Run `calibration_repetitions` batches, changing parameters and input before
   each batch.
5. Reuse the final sizing attempt as the first discarded calibration batch.
6. Compute per-inference latency from the retained batches.
7. Use the median latency as the initial stable estimate.
8. Compare relative median absolute deviation and coefficient of variation with
  `max_relative_mad`.

An unstable calibration no longer aborts the campaign. Its median latency still
sizes the burst, `CALIBRATION_UNSTABLE` is recorded, and the completed capture
is retained with `quality_status: REVIEW`.

The default calibration target is 0.5 s per batch, with five batches total. The first is discarded and four are retained. The batch count is capped by `max_calibration_inferences` to avoid an unbounded sizing decision. Failure to converge within the sizing attempts records `CALIBRATION_SIZING_NOT_CONVERGED`.

## 5. Pre-acquisition burst validation

The latency median produces an initial inference count, but it is not trusted
as the final plan. Before `READY`, the runner executes three excluded bursts
with that exact count. Each burst receives a fresh workload state and the same
cooldown used between measured cycles. The shortest observed duration must
satisfy $T_{required}$.

When validation fails, automatic sizing applies:

$$
N_{next}=\left\lceil
N_{current}\frac{T_{required}}{T_{validation,min}}m_v
\right\rceil
$$

where $m_v$ is `validation_safety_margin`, default $1.1$. The corrected count is
validated again, for at most `validation_max_rounds` rounds. A manual override
is never silently changed: it fails before acquisition when its measured
duration is insufficient. Failure to validate an automatic count also stops
before acquisition rather than producing knowingly under-resolved regions.

The manifest records all excluded attempts under `calibration.sizing_attempts`
and `burst_validation.rounds`. The final count remains fixed for every measured
cycle, so energy normalization and cycle comparability are unchanged.

## 6. Sampling-rate policy

The current verified campaign rate is 10 Hz. The runner accepts `sampling_rate_hz` so the rate can be changed after hardware testing, but it does not automatically select a different sensor rate for every layer.

Preferred policy:

1. Characterize the highest sustainable INA226EVM/GUI or direct-bridge rate.
2. Verify requested versus observed rate, jitter, and dropped samples.
3. Select one fixed validated rate for a hardware campaign.
4. Adapt the inference count per model.

A fixed campaign rate keeps filter configuration and measurements comparable. Dynamic sampling rates should be introduced only if bench validation proves that no single fixed rate supports both the fastest and slowest workloads.

## 7. Automated acquisition sequence

Use `automated_measurement.py` on the PC physically connected to the TI-SCB.
The inference host is selected with `--runner-host` and reached through
non-interactive SSH. No interaction is required after the command starts.

1. The PC opens an SSH process running `run_manager.py --stdio_acquisition` on
  the Jetson.
2. The Jetson creates the runner and loads or builds the selected model.
3. The Jetson executes warm-up, cooldown, and calibration while acquisition is
   stopped.
4. The Jetson selects `inferences_per_cycle`, writes its manifest in `READY`
   state, and emits `READY`.
5. The hosts perform ten monotonic clock exchanges and retain the minimum-RTT
   observation from the pre-acquisition round.
6. The Jetson emits `ACQUISITION_START_REQUEST`, and the PC launches its local
   `ina226_serial_logger.py` child process.
7. The PC configures and verifies the INA226 calibration register, creates the
   CSV, flushes the first complete sample, and acknowledges the Jetson.
8. Only after that acknowledgement does the Jetson begin
   `leading_idle_seconds`.
9. The Jetson executes measured cycles with idle intervals only between cycles.
10. The Jetson records `trailing_idle_seconds`, then an additional idle
    `safety_margin_seconds`.
11. The Jetson emits `ACQUISITION_STOP_REQUEST` and waits. The PC requests
   cooperative logger shutdown and waits for CSV and serial-port cleanup.
12. The PC acknowledges the stopped logger, and the hosts perform a second
    minimum-RTT synchronization round to measure drift across the capture.
13. The Jetson translates burst events into logger elapsed time, writes its
    schema-v2 manifest, and sends it through SSH.
14. The PC saves that manifest beside the CSV and prints the final
    `AUTOMATED_MEASUREMENT_RESULT` JSON record.
15. After the local manifest has been saved atomically, the PC removes the
    remote manifest to conserve storage on the inference board.

```bash
python automated_measurement.py \
  --connection-config measurement_hosts.local.json \
  --backend cuda \
  --model Models/CUDA/Linear/Linear_8192_8192.pt \
  --port /dev/serial/by-id/usb-Texas_Instruments_Generic_Bulk_Device_12345678-if01 \
  --output-directory measurements/runs \
  --shunt-ohms 0.012 \
  --max-expected-current-a 5.0 \
  --sampling_rate_hz 10
```

The automated command derives `interval_ms = 1000 / sampling_rate_hz`; planning
and acquisition therefore cannot silently use different requested rates. The
observed rate and deadline misses are recorded separately because serial and
sensor timing can prevent the requested rate from being achieved.

The Jetson needs `run_manager.py`, `runner.py`, and `base_runner.py`. The PC
needs `automated_measurement.py` and `ina226_serial_logger.py`. The TI serial
device path and local output directory are PC paths; the model, remote working
directory, and remote manifest directory are Jetson paths. SSH must be
non-interactive; use repeatable `--ssh-option` arguments for options such as a
custom identity file. Store host-specific settings in the ignored
`measurement_hosts.local.json`, created from `measurement_hosts.example.json`.

The remote manifest is a transfer/recovery file rather than the authoritative
archive. It is deleted only after the same campaign ID has been persisted in
the PC output directory. Use `--keep-remote-manifest` when a board-side copy is
needed for diagnostics. A failed local write or failed cleanup SSH command
leaves the remote file in place for recovery and does not overwrite the local
measurement result.

### Manual fallback

Use `--wait_for_acquisition` while acquisition still uses the TI GUI.

1. Start the runner.
2. The runner prepares the model and input.
3. It executes warm-up and calibration while INA226 acquisition is stopped.
4. It emits a JSON `READY` event and writes a manifest in `READY` state.
5. Configure the TI GUI with the `sampling_rate_hz` and at least the reported `capture_samples`.
6. Start collection in the GUI.
7. Press Enter in the runner terminal.
8. The runner waits `leading_idle_seconds` before the first measured burst.
9. It emits `BURST_START` and `BURST_END` for each cycle.
10. It sleeps only between cycles, never after the final cycle.
11. It waits `trailing_idle_seconds` after the final burst.
12. Stop/export the GUI capture after the planned capture duration and pair it with the JSON manifest.

Example CPU campaign with automatic count:

```bash
python run_manager.py \
  --backend cuda \
  --model Models/CUDA/Linear/Linear_64_64.pt \
  --number_of_cycles 5 \
  --sleep_time 5 \
  --sampling_rate_hz 10 \
  --target_burst_seconds 10 \
  --min_active_samples 50 \
  --wait_for_acquisition
```

An explicit `--inferences_per_cycle` remains available for controlled experiments. The runner rejects an override when its estimated duration would contain fewer than `min_active_samples`.

## 8. Capture planning

The `READY` event reports acquisition duration and sample count using:

$$
T_{capture}=T_{leading}+C\,T_{burst}+(C-1)T_{sleep}+T_{trailing}+T_{margin}
$$

where $C$ is the number of measured cycles. There are only $C-1$ inter-cycle sleeps because no sleep follows the final burst.

The requested sample count is:

$$
N_{capture}=\left\lceil T_{capture}f_s\right\rceil
$$

The default idle guards are 5 s before and after the measured cycles, with a 2 s
acquisition safety margin. Automated acquisition records that safety margin as
additional idle baseline before stopping. The manual workflow leaves margin
handling to the operator.

## 9. Structured output and manifest

Console events are one-line JSON records:

- `WARMUP_START` / `WARMUP_END`;
- `CALIBRATION_START` / `CALIBRATION_END`;
- `READY`;
- `BURST_START` / `BURST_END`;
- `COMPLETE` or `FAILED`.

Each event contains monotonic and UTC timestamps. Each `BURST_END` reports requested count, executed count, elapsed time, estimated INA226 samples, and quality flags.

Automated artifacts are colocated and use the campaign ID as their exact stem:

```text
<output-directory>/<campaign_id>.csv
<output-directory>/<campaign_id>.json
```

Automated manifests use `schema_version: 2` and include an `acquisition` object
with the authoritative CSV path, requested serial/sensor settings, actual port,
readiness and completion timestamps, capture duration, sample count, achieved
rate, deadline misses, clock alignment, stop reason, status, and failure
details. A manifest also distinguishes:

- excluded warm-up work;
- excluded calibration work;
- the `fresh_per_burst` parameter and input policy;
- planned measured count and capture configuration;
- actual measured count and duration for every cycle;
- campaign status and quality flags.

`UNDER_RESOLVED` means an actual burst was shorter than the minimum sample requirement. `DURATION_ANOMALY` means its duration was less than half or more than twice the calibration estimate. Such campaigns complete with `quality_status: REVIEW` and must not enter the final energy lookup table without investigation.

`ACQUISITION_FAILED` means the workload completed but the INA226 capture did not.
The manifest remains `COMPLETE` with `quality_status: REVIEW`, any partial CSV is
retained, and the automated command exits with code 2. Workload failures retain
completed cycle records and produce a `FAILED` manifest only after logger
cleanup.

Automated command exit codes are:

| Code | Meaning |
| ---: | --- |
| `0` | Workload and acquisition completed cleanly |
| `2` | Workload completed, but acquisition requires review or failed |
| `1` | Workload or orchestration failed |
| `130` | User interruption after cleanup |

## 9. CLI parameter reference

| Parameter                          |             Unit |                   Default | Included in measured energy? | Purpose                                                   |
| ---------------------------------- | ---------------: | ------------------------: | ---------------------------- | --------------------------------------------------------- |
| `number_of_cycles`               |           cycles |                         5 | Yes                          | Number of useful active regions                           |
| `sleep_time`                     |                s |                        10 | No                           | Idle separation between measured cycles                   |
| `inferences_per_cycle`           |       inferences |                 automatic | Yes                          | Optional manual measured-count override                   |
| `target_burst_seconds`           |                s |                        10 | Planning only                | Desired active-region duration                            |
| `sampling_rate_hz`               |               Hz |                        10 | Metadata                     | INA226 campaign sampling rate                             |
| `min_active_samples`             |          samples |                        50 | Planning/QC                  | Minimum useful samples per region                         |
| `warmup_inferences`              | inferences/batch |                       110 | No                           | Minimum discarded cold-start workload                     |
| `warmup_seconds`                 |                s |                         0 | No                           | Optional minimum duration; zero keeps one discarded burst |
| `warmup_cooldown_seconds`        |                s |            `sleep_time` | No                           | Idle cooldown after warm-up and before calibration        |
| `calibration_initial_inferences` |       inferences |                        10 | No                           | Calibration sizing pilot                                  |
| `calibration_target_seconds`     |                s |                       0.5 | No                           | Desired timer-dominating batch duration                   |
| `calibration_repetitions`        |          batches |                         5 | No                           | Full timing batches, including one discard                |
| `max_relative_mad`               |            ratio |                      0.15 | No                           | MAD/CV review threshold; never discards a capture         |
| `burst_duration_margin`          |            ratio |                       1.2 | Planning only                | Headroom above target/minimum sample duration             |
| `clock_sync_exchanges`           |        exchanges |                        10 | No                           | Samples in each pre/post clock round                      |
| `max_clock_uncertainty_fraction` | sample-period ratio |                    0.50 | Classification only          | Above this value processing uses Otsu                     |
| `max_calibration_inferences`     |       inferences |                 1,000,000 | No                           | Safety cap for a calibration batch                        |
| `leading_idle_seconds`           |                s |                         5 | Baseline only                | Idle baseline before first useful burst                   |
| `trailing_idle_seconds`          |                s |                         5 | Baseline only                | Idle baseline after final useful burst                    |
| `safety_margin_seconds`          |                s |                         2 | Baseline only                | Final measured idle window used for energy baseline       |
| `wait_for_acquisition`           |             flag |                     false | No                           | Pause at `READY` for manual GUI start                     |
| `manifest_directory`             |             path |                      none | N/A                          | Manual-run JSON campaign metadata destination             |

Every burst receives fresh parameters and input; this is part of the runner
protocol rather than a command-line option. The preparation is excluded from
the burst timer, while the complete sequence of forward passes is included.

## 10. Current limitations

- Schema-v2 automated acquisition translates remote monotonic events into the
   logger elapsed-time domain. If uncertainty exceeds 50% of one sample period,
  or metadata is missing, processing uses Otsu plus hysteresis without
  discarding the capture.
- Schema-v1 and manual captures do not have a verified monotonic mapping and
  therefore use the same Otsu fallback.
- TPU execution follows the common runner API but has not been validated in this implementation phase.
