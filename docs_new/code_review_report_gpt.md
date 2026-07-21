# energyBANERA Robustness Review

**Date:** 2026-07-12
**Last updated:** 2026-07-16
**Status:** Partially remediated — this report retains open and newly identified issues
**Scope:** Primarily `run_manager.py`, `data_processing.py`, runner/model-builder contracts, hardware acquisition, and NAS lookup-table output

## 1. Purpose

energyBANERA measures the empirical power consumption of neural-network layers on edge hardware and produces an offline lookup table containing metrics such as `power_avg_W` and `energy_avg_J`. A NAS algorithm will use these values in an objective of the form:

$$
L_{NAS} = L_{acc}(\theta) + \lambda C_{energy}(\alpha)
$$

Because an invalid measurement can alter architecture selection, the processing pipeline must fail closed: incomplete, anomalous, or statistically insufficient campaigns must be explicitly rejected instead of emitting plausible-looking values or `NaN`.

This document records the issues found during the review. It does not implement fixes; they will be addressed separately.

---

## 2. Executive summary

The current implementation is useful for exploratory measurements but is not yet sufficiently robust to serve as an unattended producer of a deterministic NAS energy lookup table.

The highest-risk remaining findings are:

1. INA226EVM acquisition and target inference are not programmatically synchronized.
2. No implemented acquisition loop handles USB disconnections, malformed reads, or device re-enumeration.
3. Per-inference energy still uses a hard-coded divisor and is not yet linked to the executed inference count recorded by the campaign manifest.
4. Final energy aggregation still applies active duration twice.
5. Otsu segmentation assumes a stationary bimodal signal and can confuse fan or thermal power changes with inference.
6. Empty or statistically insufficient campaigns can emit `NaN` or fail through uncontrolled exceptions.
7. The reporting path has no explicit quality/status contract and can write invalid metrics into downstream artifacts.
8. Adaptive burst sizing introduces new validation and calibration-stability risks described in Section 3.1.

### Remediation completed since the initial review

- `RunManager` now distinguishes discarded warm-up, discarded calibration, and measured bursts.
- It creates a per-campaign ID, writes an atomic JSON manifest when configured, and emits structured `WARMUP_*`, `CALIBRATION_*`, `READY`, `BURST_*`, `COMPLETE`, and `FAILED` events.
- The measurement plan can derive `inferences_per_cycle` from calibrated steady-state latency, target burst duration, and the requested INA226 sample count.
- `InferenceRunner` now supplies a shared burst-oriented lifecycle: `prepare_burst()`, `run_burst()`, `run_prepared_burst()`, synchronization, `BurstResult`, and `close()`.

These changes make the target workload traceable and remove the duplicate timing burst. They do **not** yet make the sensor acquisition, timestamp reconciliation, or data-processing calculation trustworthy enough for NAS publication.

### Empirical evidence

The current `data_processing.get_average_power()` implementation was evaluated over all 186 CSV captures under `Data/`, using the report configuration of 10 Hz, median kernel 11, low-pass cutoff 0.1 Hz, and rolling window 30.

Results:

| Outcome                     | Files |
| --------------------------- | ----: |
| Finite, non-negative output |   178 |
| Non-finite output           |     8 |
| Raised exception            |     0 |
| Negative power/energy       |     0 |

The eight non-finite results contained `NaN` for both `power_var_W2` and `energy_var_J2`, caused by sample variance with `ddof=1` when only one active region was detected.

Affected captures at the time of review:

- `Data/Jetson_nano_power_record/Conv_3_0/Conv_128_256_3_0.csv`
- `Data/Jetson_nano_power_record/Conv_3_0/Conv_8_1024_3_0.csv`
- `Data/Jetson_nano_power_record/Linear/Lirobustness_review_gptnear_8192_8192.csv`
- `Data/new_measures/jetson_nano/Conv_3_0/Conv_1_512_3_0.csv`
- `Data/new_measures/jetson_nano/Conv_3_0/Conv_8_512_3_0.csv`
- `Data/new_measures/jetson_nano/Conv_3_0_nograd/Conv_1_512_3_0.csv`
- `Data/new_measures/jetson_nano/Linear_nograd/Linear_4096_4096.csv`
- `Data/new_measures/jetson_nano/Linear_nograd/Linear_4096_8192.csv`

---

# 3. Hardware I/O robustness and error handling

## HW-01 — INA226EVM acquisition fault tolerance is not implemented

**Severity:** Critical
**Files:** `run_manager.py`, proposed acquisition system
**Status:** Open

`run_manager.py` controls only target-side inference. It does not open, read, monitor, or recover the INA226EVM USB connection. The direct-acquisition design remains a proposal in `docs_new/ina226_direct_acquisition_design.md`.

Consequences:

- A USB reset or cable/device interruption is not visible to the inference runner.
- Inference may continue while acquisition has stopped.
- A truncated CSV may be treated as a complete campaign.
- Missing samples and acquisition gaps are not represented explicitly.
- The target and host cannot coordinate a controlled pause or abort.

This is especially important for transient voltage problems on high-current boards such as Raspberry Pi 5.

### Required remediation

Implement a host-side acquisition state machine:

`IDLE → ARMING → RECORDING → RECOVERING → FINALIZING → COMPLETE/FAILED`

For each sensor read:

1. Catch timeout, malformed-frame, device-removal, and OS I/O errors separately.
2. Retry a bounded number of times with backoff.
3. On persistent failure, close and reopen the transport, wait for USB re-enumeration, and repeat INA226 initialization/calibration.
4. Verify a plausible sample after reconnection.
5. Record every missing sequence explicitly; do not interpolate across transport failures.
6. Mark the campaign failed if a gap overlaps an inference or baseline interval.
7. Preserve partial data for diagnosis, then rerun the complete layer campaign.
8. Finalize the campaign manifest atomically in a `finally` path.

## HW-02 — Samples do not have a robust acquisition-health contract

**Severity:** High
**Status:** Open

The legacy CSV contains a sample number and electrical measurements, but no per-sample monotonic timestamp, read status, retry count, or gap marker.

Each future sample should include:

- monotonic timestamp;
- wall-clock timestamp;
- sequence number;
- power, bus voltage, current, and optionally shunt voltage;
- read status and retry count;
- campaign ID and burst ID.

The acquisition system should validate the physical relationship $P \approx VI$ to help detect calibration or protocol-decoding failures.

## HW-03 — Target-side failures are only partially finalized

**Severity:** High
**File:** `run_manager.py`
**Status:** Open

`RunManager` now catches failures, emits a `FAILED` event, records a failure object in the manifest, and calls `runner.close()` from `finally`. This resolves the original uncontrolled target-side failure path **when a manifest directory is configured**.

It remains incomplete because the acquisition process is not informed or stopped, no abort/cancellation state exists, and a manifest is not persisted when `manifest_directory` is omitted. CUDA errors, out-of-memory conditions, model errors, or device failures still cannot be reconciled with a sensor trace automatically.

`RunManager` needs:

- structured lifecycle events;
- per-cycle status;
- recoverable versus fatal error classification;
- host-requested cancellation;
- a guaranteed final `COMPLETE`, `ABORTED`, or `FAILED` event;
- a returned result object instead of output being communicated only by `print()`.

## HW-04 — No reliable host/target handshake exists

**Severity:** Critical
**Files:** `run_manager.py`, future acquisition orchestrator
**Status:** Open

Human-controlled acquisition and SSH/terminal inference introduce variable delay from:

- SSH process startup;
- Python imports;
- model loading;
- CUDA or accelerator initialization;
- scheduler delay;
- manual GUI interaction.

At 10 Hz, the sampling period alone is 100 ms. Short bursts cannot be aligned accurately without event markers.

### Required synchronization pattern

1. Host opens and validates the INA226EVM.
2. Host begins recording and confirms multiple valid idle samples.
3. Target loads the model, prepares input, warms up, and emits `READY <campaign_id>`.
4. Host records a fixed leading-idle guard interval.
5. Host sends `GO <campaign_id> <burst_id>`.
6. Target emits `BURST_START` immediately before inference.
7. Target synchronizes the accelerator and emits `BURST_END` immediately after inference.
8. Host records a trailing-idle guard interval.
9. Target reports the actual inference count and cycle status.
10. Host reconciles expected bursts, target events, samples, and gaps before accepting the campaign.

Monotonic clocks should be used on both machines. A recognizable short-long-short inference beacon can provide a secondary alignment check. A hardware GPIO trigger would be stronger if supported later.

## HW-05 — Burst duration may be below the useful resolution

**Severity:** High
**Status:** Open

At 10 Hz, an active region represented by one or two samples cannot support reliable filtering, segmentation, baseline subtraction, or variance estimation. `RunManager` now selects a count from calibrated latency and enforces a planned lower bound through `target_burst_seconds` and `min_active_samples`. It also flags a measured burst as `UNDER_RESOLVED` when its estimated sample count is too low.

This is only a planning safeguard: it uses a configured sampling rate rather than actual sensor timestamps, and it does not reject the campaign when a cycle is under-resolved.

A campaign quality gate must reject bursts that are too short or contain too few valid active samples.

## ~~HW-06 — Cycle timing executes the workload twice~~

**Severity:** High
**File:** `run_manager.py`
**Status:** Resolved in the current `RunManager` design.

The old `measure_cycle_inference_time()` path has been replaced by explicit warm-up and calibration bursts. Calibration is recorded as excluded work in the manifest and the measured loop begins only after the configured warm-up cooldown and leading idle interval. No second, unlabelled timing call remains.

---

## 3.1 New risks from adaptive burst sizing

## RUN-01 — Automatic burst sizing must be regression-tested on every edit

**Severity:** Critical
**Files:** `run_manager.py`
**Status:** Open

The automatic path must always return `(inference_count, "automatic")` when `inferences_per_cycle` is omitted. A regression in this branch can make the default CLI configuration return `None`, which then fails while `_build_measurement_plan()` unpacks the result. Add a unit test for both automatic and manual selection paths before relying on a default campaign command.

## RUN-02 — Adaptive sizing lacks input-domain validation

**Severity:** High
**Files:** `run_manager.py`
**Status:** Open

`sampling_rate_hz`, `target_burst_seconds`, `min_active_samples`, cycle count, idle durations, calibration counts, and maximum counts are accepted without a complete positivity and consistency check. Invalid values can cause division by zero, negative capture durations, an empty measurement loop, or a zero-sized calibration batch. Validate these inputs in the constructor or argument parser before creating a runner.

## RUN-03 — Calibration can be unrepresentative of measured bursts

**Severity:** High
**Files:** `run_manager.py`, `base_runner.py`, `runner.py`
**Status:** Open

The measured count is sized from a short calibration batch. CPU frequency scaling, CUDA/TPU clock changes, thermal throttling, allocator behaviour, or different parameter/input states can make measured bursts substantially longer or shorter than planned. The current duration-ratio flag is useful, but it is based on host-side execution time and marks the campaign only as `REVIEW`; it neither resizes a retry nor rejects the affected burst.

Require a calibration stability diagnostic, record operating telemetry where available, and reject or rerun cycles outside an explicitly justified duration tolerance.

## RUN-04 — Event timestamps cannot yet define INA226 integration bounds

**Severity:** High
**Files:** `run_manager.py`, future acquisition service
**Status:** Open

`BURST_START` and `BURST_END` contain target-process monotonic timestamps, while the INA226 CSV currently has no compatible monotonic timestamps. The events therefore improve traceability but cannot yet be used to integrate the matching sensor samples, particularly when acquisition and workload run on different machines. The acquisition daemon must persist its own monotonic timestamps and establish a handshake or clock-offset reconciliation.

---

# 4. Signal-processing resiliency

## DSP-01 — Raw input is not validated before filtering

**Severity:** Critical
**File:** `data_processing.py`
**Status:** Open

The pipeline does not verify:

- presence of required columns;
- numeric data types;
- finite power values;
- monotonic sample numbers or timestamps;
- duplicates or missing samples;
- physically plausible electrical ranges;
- minimum capture length;
- effective sampling rate and jitter.

Invalid data can enter filters and produce an exception or, more dangerously, a plausible but invalid result.

A formal validation stage must run before any filtering. Gaps overlapping active or baseline windows must invalidate the campaign.

## DSP-02 — Sampling frequency and integration interval are inconsistent

**Severity:** Critical
**File:** `data_processing.py`
**Status:** Open

The module-level `sampling_interval` is fixed at 0.1 s, corresponding to 10 Hz. However, `get_average_power()` defaults to `fs=5`, while the command-line path and current report call use 10 Hz.

This can make filter coefficients and energy integration use inconsistent time bases.

Required direction:

- derive timing from recorded timestamps;
- validate $\Delta t_i=t_{i+1}-t_i$ and sample jitter;
- integrate irregular samples directly or resample explicitly;
- store one immutable acquisition configuration in the campaign manifest;
- remove the independent module-level timing assumption.

## DSP-03 — Median-filter boundary padding can create artificial transitions

**Severity:** Medium
**File:** `data_processing.py`
**Status:** Open

`scipy.signal.medfilt` zero-pads boundaries. Since board power is not near zero, edge padding can generate artificial transients that propagate into later filters.

Use a median filter with reflected/nearest boundary handling, or remove enough edge samples and document the discarded interval.

## DSP-04 — Low-pass filtering lacks finite-value and length guards

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

`filtfilt` requires a minimum signal length and valid finite values. Short or contaminated captures are not checked before filtering.

The pipeline must calculate its required minimum length and reject captures that cannot safely pass through the configured filter.

## DSP-05 — Centered rolling mean contaminates region boundaries

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

A centered rolling window spreads active values into adjacent idle samples and idle values into active samples. Baseline windows selected immediately beside detected boundaries are therefore contaminated.

Any retained centered smoother requires explicit guard bands at least as wide as its boundary contamination. Alternatively, use event-defined intervals and avoid smoothing for numerical integration.

## DSP-06 — Otsu assumes a stationary, bimodal signal

**Severity:** Critical
**File:** `data_processing.py`
**Status:** Open

A fan activation, DVFS transition, thermal throttling, cooling event, or unrelated background process can create a second power mode. Otsu can classify that state as inference even if no inference is occurring.

The code calculates a median expected active duration, but the duration-based filtering is commented out and the calculated value is unused.

### Required segmentation hierarchy

Primary segmentation should use target-emitted burst markers. Signal analysis should refine or validate those boundaries rather than discover the entire campaign blindly.

When markers are unavailable, use a guarded fallback:

1. Estimate a slowly varying idle baseline with a rolling median or low quantile.
2. Calculate residual power:

   $$
   r(t)=P(t)-\widehat{P}_{idle}(t)
   $$
3. Estimate robust idle noise:

   $$
   \sigma_{robust}=1.4826\operatorname{MAD}(r)
   $$
4. Use hysteresis thresholds for entering and leaving the active state.
5. Enforce expected minimum and maximum duration.
6. Enforce expected region count and idle separation.
7. Reject baseline windows with a change point, excessive slope, or excessive variance.

Otsu should only be accepted when both classes contain enough samples, class separation has adequate SNR, region count matches the manifest, durations agree with target events, and baseline drift remains within tolerance.

## DSP-07 — Idle baseline windows are too broad and non-local

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

The current calculation combines entire idle regions before and after an active region. Long idle intervals may contain unrelated thermal, fan, or background states.

Use fixed-width local windows immediately before and after each burst, excluding smoothing guard samples. Estimate them with robust medians or trimmed means. If both are valid, interpolate the idle baseline across the active interval:

$$
\widehat{P}_{idle}(t)=P_{pre}+\frac{t-t_{start}}{t_{end}-t_{start}}(P_{post}-P_{pre})
$$

Reject a burst when pre/post baselines differ beyond a configured tolerance.

## DSP-08 — Empty smoothed data is not handled

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

After filtering and rolling-window removal, the cleaned signal can be empty. The code accesses the first and last activity values without checking this condition.

Expected outcome: explicit `INSUFFICIENT_SAMPLES`, not an indexing error.

## DSP-09 — No-active-region campaigns are not handled

**Severity:** Critical
**File:** `data_processing.py`
**Status:** Open

A constant or low-SNR capture can produce no active regions. Aggregation then operates on empty lists and zero total duration.

Expected outcome: explicit `NO_ACTIVE_REGION`, with all numerical NAS metrics unavailable.

## DSP-10 — Missing idle segments can use undefined or stale variables

**Severity:** Critical
**File:** `data_processing.py`
**Status:** Open

`intra_idle_avg` and `intra_idle_var` are assigned only when at least one idle segment exists. If no idle segment exists for the first region, an unbound-local error can occur. If it happens after another iteration, Python may reuse the previous iteration's values, silently applying the wrong baseline.

Every active region must have independently initialized baseline state. A missing valid baseline must reject that region.

## DSP-11 — One-sample regions produce undefined variance

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

Pandas sample variance is undefined for a one-sample active or idle segment. This can propagate `NaN` through per-region calculations.

Minimum region sizes must be enforced before calculating statistics.

## DSP-12 — Cleaned and raw positional bounds are mixed

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

Transitions are positions in `df_clean`, but the last endpoint and some idle bounds use lengths from the original dataframe. Centered rolling-window `NaN` removal makes those coordinate systems different. The result can overestimate duration and select incorrect slices.

All region computations must use a single coordinate system and retain a mapping to original sample IDs/timestamps.

## DSP-13 — Filtered samples are strongly autocorrelated

**Severity:** Medium
**File:** `data_processing.py`
**Status:** Open

Median filtering, low-pass filtering, and rolling averaging make adjacent samples highly dependent. Variance formulas based on independent observations overstate the effective sample count and understate uncertainty.

Use effective sample size estimation or block bootstrap methods when calculating uncertainty from filtered samples.

## DSP-14 — Thermal and frequency state are not recorded

**Severity:** High
**Status:** Open

Thermal throttling can alter power, throughput, and burst duration. Fan behavior can alter the idle baseline. Without temperature, clock, and throttling telemetry, these effects cannot be distinguished reliably after the campaign.

Future manifests should record available board telemetry such as temperature, accelerator/CPU/GPU frequency, throttling status, fan state, and power mode. Campaigns should either hold these conditions constant or stratify results by operating state.

---

# 5. Metric correctness and NAS sanitization

## MET-01 — Per-inference energy uses a hard-coded divisor

**Severity:** Critical
**Files:** `data_processing.py`, `run_manager.py`
**Status:** Open

Per-region energy is divided by `100000.0`, but `run_manager.py` defaults to 110 inferences per cycle. The nearby comment also refers to 10,000 rather than 100,000.

The actual executed inference count must be recorded for every burst and supplied to processing.

For burst $i$:

$$
E_i=\frac{\int_{t_{start,i}}^{t_{end,i}}\left(P(t)-\widehat{P}_{idle}(t)\right)dt}{N_i}
$$

For irregular timestamps, use trapezoidal integration rather than an assumed interval.

## MET-02 — Final energy aggregation applies duration twice

**Severity:** Critical
**File:** `data_processing.py`
**Status:** Open

Per-region `energy_J` already contains active duration. Final aggregation multiplies it by active duration again and divides by total duration. The effective contribution is proportional to $d_i^2$, over-weighting long or throttled bursts.

A campaign-wide per-inference estimate should normally be:

$$
E_{avg}=\frac{\sum_i E_{net,i}}{\sum_i N_i}
$$

This issue is also acknowledged in `docs_new/formule_calcolo_risultati.md`.

## MET-03 — `power_var_W2` semantics are ambiguous

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

The implementation calculates both:

- within-region active-plus-idle variance;
- between-region variance of net mean power.

Only the between-region variance is returned as `power_var_W2`. The name does not indicate which uncertainty it represents.

Report distinct metrics:

- `power_between_burst_var_W2`;
- `energy_between_burst_var_J2`;
- `power_mean_standard_error_W`;
- `energy_mean_standard_error_J`;
- confidence intervals;
- accepted and rejected burst counts.

## MET-04 — Variance of a difference of means is not calculated correctly

**Severity:** High
**File:** `data_processing.py`
**Status:** Open

Adding raw active and idle sample variances is not the uncertainty of the difference between their means. Under independence, the basic form is:

$$
\operatorname{Var}(\bar P_A-\bar P_I)=\frac{s_A^2}{n_{A,eff}}+\frac{s_I^2}{n_{I,eff}}
$$

Because filtered samples are autocorrelated, effective sample counts or block bootstrap estimates are required.

## MET-05 — Sample variance is undefined for one detected burst

**Severity:** Critical
**File:** `data_processing.py`
**Status:** Open

`np.var(..., ddof=1)` returns `NaN` when only one active region is available. This produced non-finite output in eight existing CSV captures.

Policy decision required:

- reject campaigns with fewer than two accepted bursts when variance is mandatory; or
- allow a mean-only diagnostic result but mark variance unavailable and prohibit NAS ingestion.

## MET-06 — Negative or implausible net metrics are not quality-gated

**Severity:** High
**Status:** Open

A shifted or contaminated baseline can produce negative power or energy, while a calibration problem can produce implausibly large values. No board-specific physical limits or signal-to-noise checks are applied before export.

Every accepted result must satisfy finite-value, non-negative, plausibility, and SNR constraints.

## MET-07 — The result has no explicit success/failure schema

**Severity:** Critical
**Files:** `data_processing.py`, `data_report.py`
**Status:** Open

The processing function returns only four numerical values. It cannot distinguish a valid measurement from a degraded or rejected one.

Every result should include:

- `status`: `OK`, `REJECTED`, or `FAILED`;
- `failure_code` and `failure_message`;
- campaign ID;
- typed layer specification;
- board, backend, precision, and power mode;
- expected, detected, accepted, and rejected region counts;
- requested and executed inference counts;
- sample count, gap count, and effective sampling rate;
- baseline drift and SNR diagnostics;
- algorithm/configuration version;
- numerical metrics only when status is `OK`.

Suggested failure codes:

- `INVALID_SCHEMA`
- `NONFINITE_INPUT`
- `ACQUISITION_GAP`
- `INSUFFICIENT_SAMPLES`
- `NO_ACTIVE_REGION`
- `REGION_COUNT_MISMATCH`
- `MISSING_IDLE_BASELINE`
- `BASELINE_DRIFT`
- `LOW_SNR`
- `THERMAL_THROTTLING`
- `NONFINITE_RESULT`
- `NEGATIVE_NET_ENERGY`

## MET-08 — Reporting does not isolate failed files

**Severity:** High
**File:** `data_report.py`
**Status:** Open

The report loop assumes every CSV succeeds and directly appends numerical values. A malformed file can abort the complete report, while a statistically invalid file can write `NaN` into the workbook.

The report should catch failures per file, preserve failed rows with status and reason, continue processing other files, and write accepted and rejected measurements to separate sheets or tables.

## MET-09 — NAS ingestion does not have a fail-closed gate

**Severity:** Critical
**Status:** Open

Before lookup-table publication, require:

- `status == OK`;
- all required metrics finite;
- no active or baseline interval overlapping an acquisition gap;
- minimum accepted burst count;
- expected region count reconciliation;
- physical plausibility and SNR checks;
- known processing version and campaign metadata.

The NAS loader should reject invalid rows rather than impute or silently accept missing values.

---

# 6. Object-oriented contracts and extensibility

## ~~OO-01 — Abstract runner signature does not match implementations~~

**Severity:** Critical
**Files:** `base_runner.py`, `runner.py`
**Status:** Resolved.

`InferenceRunner.run_inference()` now accepts an inference count and the base class owns the burst-oriented timing contract. `run_burst()` prepares a fresh workload outside the timer, `run_prepared_burst()` times exactly the requested work, and it returns a typed `BurstResult` with requested/executed counts and elapsed duration.

## ~~OO-02 — TPU runner does not implement required cycle timing~~

**Severity:** Critical
**Files:** `run_manager.py`, `runner.py`
**Status:** Resolved by the common base implementation.

Both runners inherit the timing path from `InferenceRunner`; the TPU runner only needs to implement the actual `run_inference()` operation. This removes the old unconditional call to the missing TPU-specific timing method.

## ~~OO-03 — Optional capabilities are detected through `hasattr()`~~

**Severity:** Medium
**File:** `run_manager.py`
**Status:** Resolved for the runner lifecycle.

The base class now declares safe default methods and `prepare_burst()` invokes the lifecycle explicitly. The intended lifecycle is:

- `prepare()`
- `warmup()`
- `generate_input(seed)`
- `run_burst(inference_count) -> BurstResult`
- `synchronize()`
- `close()`

## OO-04 — Conditional class definitions make imports environment-dependent

**Severity:** Medium
**File:** `runner.py`
**Status:** Open

Runner classes are defined only when their optional dependency is importable. Importing a named runner can therefore fail differently across host environments.

Prefer stable class definitions whose constructors raise a clear dependency/backend error, or use an explicit backend registry/factory.

## OO-05 — Base model builder is shaped only for Linear layers

**Severity:** High
**Files:** `base_model_builder.py`, `linear_model_builder.py`
**Status:** Open

The base constructor accepts only `input_size` and `output_size`, which does not represent the declared Conv2d dimensions. Its documentation is TPU-specific despite a PyTorch implementation.

Use typed specifications such as:

- `LinearSpec(in_features, out_features)`;
- `Conv2dSpec(in_channels, out_channels, image_size, kernel_size, padding, stride, dilation, groups)`.

A builder should accept a generic `LayerSpec` and return a `ModelArtifact` containing path, backend, precision, quantization metadata, and build status.

## OO-06 — Conv2d output channels are hard-coded

**Severity:** High
**File:** `runner.py`
**Status:** Open

The Conv2d construction fixes `out_channels=1`. The current filename/specification does not include output channels, limiting the lookup table and making the layer definition incomplete for general NAS use.

The layer schema and naming convention must explicitly define every parameter that changes compute and energy cost.

## OO-07 — Model filename parsing is not validated

**Severity:** High
**File:** `runner.py`
**Status:** Open

Layer configuration is inferred by splitting the filename and indexing positional parameters. Missing, malformed, or extra fields can cause unclear errors or incorrect model construction.

Use a validated `LayerSpec` manifest. The filename should be an identifier, not the authoritative schema.

## OO-08 — TPU compilation failures are swallowed

**Severity:** High
**Files:** `linear_model_builder.py`, `model_builder_manager.py`
**Status:** Open

Compilation errors are printed and not propagated. The manager can continue as if the model was built successfully.

Builders must return structured success/failure artifacts or raise typed exceptions. A failed artifact must never enter a campaign plan.

## OO-09 — `RunManager` mixes orchestration, timing, warm-up, and presentation

**Severity:** Medium
**File:** `run_manager.py`
**Status:** Open

`RunManager` constructs runners, mutates models, generates inputs, warms up, measures time, executes cycles, sleeps, and prints status. This makes synchronization, testing, cancellation, logging, and alternative acquisition backends difficult.

Separate target workload execution from host campaign orchestration and event/report presentation.

## OO-10 — Metric extensibility is dictionary-based and tightly coupled

**Severity:** High
**Files:** `data_processing.py`, `data_report.py`
**Status:** Open

Metrics are returned as an untyped dictionary with fixed keys. Adding thermal delta, carbon footprint, uncertainty, or quality diagnostics risks breaking report consumers.

Recommended separation:

- `InferenceRunner`: executes target workload;
- `AcquisitionBackend`: samples INA226 or another sensor;
- `CampaignOrchestrator`: controls synchronization and lifecycle;
- `SignalProcessor`: filters and segments;
- `MetricCalculator`: computes empirical metrics;
- `QualityGate`: accepts or rejects campaigns/bursts;
- `MetricPlugin`: adds derived or auxiliary metrics;
- `ResultRepository`: stores versioned lookup-table records.

Examples:

- `EnergyMetric`: integrates baseline-corrected power;
- `ThermalDeltaMetric`: consumes target temperature telemetry;
- `CarbonMetric`: calculates

  $$
  CO_{2e}=E_{kWh}I_{grid}
  $$
- `PerformanceMetric`: records latency and throughput.

Carbon intensity is contextual metadata, not a responsibility of the inference runner.

---

# 7. Recommended remediation order

## P0 — Correctness blockers

- [ ] Remove hard-coded inference normalization and consume per-burst `executed_inferences` from the manifest.
- [ ] Correct final energy aggregation.
- [ ] Reject zero-region and one-region campaigns appropriately.
- [ ] Reject all non-finite inputs and outputs.
- [ ] Fix cleaned/raw index and duration handling.
- [ ] Fix missing/stale idle baseline variables.
- [ ] Make sampling frequency and integration timing consistent.
- [x] ~~Remove duplicate timing execution.~~
- [x] ~~Align the abstract runner API with implementations.~~
- [x] ~~Add timing support to every runner through a common base implementation.~~
- [ ] Add explicit result/QC status required by NAS ingestion.
- [ ] Add tests for automatic and manual adaptive count selection, invalid configuration, and calibration instability.

## P1 — Trustworthy campaign segmentation

- [x] ~~Add target-side campaign and burst manifests.~~ Target persistence still depends on configuring `manifest_directory`; sensor data are not yet attached.
- [x] ~~Emit target `READY`, `BURST_START`, and `BURST_END` events.~~
- [ ] Use event-defined active windows as the primary segmentation source.
- [ ] Add local pre/post baseline windows with guard bands.
- [ ] Add expected count, duration, SNR, drift, and gap quality gates.
- [ ] Make Otsu a validated fallback rather than the sole authority.
- [ ] Record temperature, clock, throttling, fan, and power-mode telemetry.

## P2 — Hardware automation

- [ ] Implement the INA226EVM acquisition daemon.
- [ ] Add bounded read retries and USB reopen/reinitialization.
- [ ] Persist timestamps, sequence numbers, calibration, and read health.
- [ ] Coordinate host acquisition and target inference through a handshake.
- [ ] Abort and rerun campaigns with gaps overlapping measurement windows.
- [ ] Produce atomic per-campaign and campaign-level manifests.

## P3 — Extensibility and maintainability

- [ ] Introduce typed `LayerSpec`, `CampaignSpec`, `BurstResult`, and `MeasurementResult` objects.
- [ ] Replace filename-based model configuration with validated manifests.
- [ ] Separate workload execution, acquisition, DSP, QC, metrics, and storage.
- [ ] Add metric plugins for thermal and carbon information.
- [ ] Return structured model build artifacts and propagate compilation failures.
- [ ] Add unit, synthetic-signal, and hardware fault-injection tests.

---

# 8. Required test coverage for later fixes

## Synthetic DSP tests

- constant idle signal;
- one valid active burst;
- multiple equal bursts;
- active-first and active-last captures;
- no idle before/after a burst;
- fan step without inference;
- fan step during inference;
- linearly drifting idle baseline;
- sudden baseline change;
- short capture below filter requirements;
- one-sample active/idle segments;
- missing, duplicate, and non-monotonic samples;
- `NaN`, infinity, and non-numeric power values;
- irregular timestamps and sample gaps;
- low-SNR workload;
- thermal-throttling pattern with increasing burst duration;
- incorrect expected burst count;
- acquisition gap overlapping an active interval.

## Runner contract tests

- every registered backend can be constructed through the same interface;
- every backend supports prepare, warm-up, timed burst, synchronization, and close;
- requested and executed inference counts match;
- timing executes exactly one burst;
- CUDA synchronization occurs before end timestamps;
- errors produce structured failure results;
- cancellation always produces a final event.

## Hardware fault-injection tests

- unplug and reconnect the INA226EVM during idle;
- unplug and reconnect during inference;
- inject malformed and delayed USB frames;
- force target SSH disconnection;
- force target process failure;
- simulate host acquisition process restart;
- verify partial captures cannot be published to the NAS lookup table.

---

# 9. Acceptance criteria for NAS publication

A measurement row may enter the energy lookup table only when:

1. The campaign status is `OK`.
2. All required metrics are finite and physically plausible.
3. Requested and executed inference counts reconcile.
4. Expected and accepted burst counts reconcile.
5. At least the configured minimum number of bursts is accepted.
6. No active or baseline interval overlaps an acquisition gap.
7. Baseline drift, SNR, duration, and thermal-state checks pass.
8. Board, backend, precision, layer specification, sensor calibration, sampling configuration, and processing version are present.
9. The source capture and QC manifest remain traceable from the lookup-table row.
10. Failed campaigns remain visible in a separate rejection log and are never silently omitted or converted to `NaN` rows.

---

# 10. Final assessment

The existing separation between model builders, runners, and data processing is a useful starting point, but the interfaces and quality gates are not yet strong enough for unattended hardware benchmarking or deterministic NAS consumption.

The most urgent concern is not only that processing can crash. It can also return plausible numerical means with invalid uncertainty, incorrect per-inference normalization, or incorrectly segmented fan/thermal regions. The lookup table must therefore remain a controlled research artifact until the P0 correctness items and fail-closed QC contract are implemented.
