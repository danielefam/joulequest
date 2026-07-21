# energyBANERA — Full Codebase Review Report

**Date:** 2026-07-08
**Scope:** all Python scripts in the repository root and `draft/`
**Context:** energy benchmarking framework measuring per-layer power on edge boards (Jetson Nano, Coral Dev Board, Raspberry Pi 5) via INA226EVM @ 10 Hz, producing an energy lookup table (`energy_avg_J`, `power_avg_W`) for NAS optimization.

Severity legend:
- **[Critical]** — crashes, or silently corrupts measurement data
- **[High]** — wrong results under realistic conditions
- **[Medium]** — fragility, maintainability, contract violations
- **[Low]** — cosmetic, dead code, hygiene

---

## Summary Table (top issues)

| # | File | Issue | Severity |
|---|------|-------|----------|
| 1 | `data_processing.py` | Stale idle stats reused across regions (silent data corruption) | Critical |
| 2 | `data_processing.py` | NaN variance with 0 or 1 detected regions → NaN enters NAS lookup table | Critical |
| 3 | `runner.py` | Capital-letter model names (`Linear_64_64`) crash with `UnboundLocalError` | Critical |
| 4 | `run_manager.py` + `runner.py` | `--backend tpu` crashes: `TFLiteTPURunner` has no `generate_input()` | Critical |
| 5 | `data_processing.py` / `run_manager.py` | Magic number `100000` duplicated in two files; comment says 10000 | Critical |
| 6 | `data_processing.py` | `fs=5` in `get_average_power` default vs. real 10 Hz sampling | High |
| 7 | `data_processing.py` | Otsu threshold on unimodal data fabricates fake active regions | High |
| 8 | `runner.py` | Missing `torch.cuda.synchronize()` → burst end not guaranteed | High |
| 9 | `runner.py` | No `torch.inference_mode()` → autograd overhead included in energy | High |
| 10 | `data_processing.py` | Index-space mismatch `len(df)` vs `len(df_clean)` | High |
| 11 | `data_processing.py` | Duration from row count, not timestamps (USB sample drops undetected) | High |
| 12 | `data_report.py` | No per-file error handling; conv filename parsing broken | High |
| 13 | `base_runner.py` / `base_model_builder.py` | ABC contracts don't match implementations | Medium |
| 14 | `model_builder_manager.py` | `__main__` never calls `build_all()` — prints "done" doing nothing | Medium |
| 15 | `draft/ActivePeriod.py` | Imports a module that doesn't exist in the repo | Medium |

---

## 1. `data_processing.py` (signal-processing core)

### [Critical] 1.1 Stale idle statistics reused across regions
In `compute_means_variances`, `intra_idle_avg` / `intra_idle_var` are assigned **only inside** `if idle_segments:`. If a region has no non-empty idle neighbours:

- 1st region → `NameError` (crash), or
- later region → the loop **silently reuses the previous region's idle stats**, corrupting `power_offset_W` and `energy_J` for that region with no warning.

**Fix:** add an `else` branch that flags/skips the region (and logs it), or fail the file explicitly.

### [Critical] 1.2 NaN propagation into the NAS lookup table
- `regions == []` → `np.mean([])` → `NaN` + RuntimeWarning only.
- Exactly 1 region → `np.var([x], ddof=1)` → `NaN` for `power_var_W2` and `energy_var_J2` even on a "successful" run.

These NaNs flow into `data_report.py` and the Excel lookup table with no trace, and will break a downstream NAS optimizer that assumes finite values.

**Fix:** return an explicit status (`ok` / `rejected` + reason), require a minimum number of regions (≥ 3 for a meaningful variance), and never emit NaN — rejected files should go to a separate "failed" log/sheet.

### [Critical] 1.3 Magic number `100000` hardcoded, duplicated, and misdocumented
`intra_energy_avg = intra_power_avg * duration / 100000.0` — the comment says *"divided by 10000"*, the code divides by 100000, and the actual inference count is hardcoded independently in `run_manager.py` (`count=100000`). If either side changes, every energy value is silently scaled wrong.

**Fix:** make the inference count a parameter recorded with the measurement (sidecar/manifest or filename) and read it during processing.

### [High] 1.4 Sampling frequency inconsistency (`fs=5` vs 10 Hz)
- `sampling_interval = 0.1` (i.e. 10 Hz) is the module-level truth.
- `get_average_power(..., fs=5, ...)` defaults to 5 Hz, and `data_report.py` calls it with `fs=5`.
- The CLI `main()` defaults to `fs=10`.

The Butterworth normalized cutoff `cutoff / (0.5*fs)` is therefore wrong in one of the two paths (filter cutoff differs by 2×). **Fix:** derive `fs = 1 / sampling_interval` in one place; remove the duplicated default.

### [High] 1.5 Otsu's threshold fabricates regions on failed campaigns
`threshold_otsu` **always** returns a threshold, even on a unimodal (all-idle) trace — it will split measurement noise in half and "find" fake active regions. Baseline steps from fan activation / thermal throttling create a third mode that also confuses a global Otsu.

**Fix:** after computing the threshold, verify bimodality: minimum active-idle contrast in W and a minimum normalized separation ((mean_hi − mean_lo)/std). Reject files that fail. Additionally validate `len(regions)` against the expected `nb_run` from the campaign.

### [High] 1.6 Duration computed from row count, not time
`duration = (end - start) * sampling_interval` assumes zero dropped samples. If the USB logger stutters (plausible on the Pi 5 near 5 A), rows go missing, duration and thus energy are underestimated silently. **Fix:** compute duration from the `Sample` column (or timestamps), and validate monotonicity/gap-free spacing in `load_data`.

### [High] 1.7 Index-space mismatch: `len(df)` vs `len(df_clean)`
Segmentation runs on `df_clean = df["smoothed"].dropna()` (rolling `center=True` drops ~window/2 samples at each edge), but:

- `end_indices = np.append(end_indices, len(df.values))` — uses the **unclean** length;
- `next_start = len(df) - 1` — same problem.

Off-by-window errors at the trace tail; the last idle slice can be wrong or empty. **Fix:** use `len(df_clean)` consistently.

### [High] 1.8 Mixed positional/label indexing
`df_clean.iloc[start:end]` (positional) is mixed with `df_clean[prev_end:start]` (label-based on an integer index with gaps after `dropna`). Behaviour is subtle and version-dependent in pandas. **Fix:** use `.iloc` everywhere.

### [Medium] 1.9 Statistical issues
- `intra_power_var = intra_activ_var + intra_idle_var` is the variance of a single sample difference; the variance of the **mean estimate** would be `activ_var/n_active + idle_var/n_idle`. The current value overstates uncertainty by ~2 orders of magnitude.
- `extra_energy_avg = Σ(duration·energy)/Σ(duration)` double-weights duration (energy already scales with duration). A plain mean across regions is unbiased since every region runs the same inference count.
- Samples are strongly autocorrelated after median + Butterworth + rolling mean; intra-region variances treat them as independent. The region-to-region variance (`power_var_W2`) is the statistically sound estimator — prefer it.

### [Medium] 1.10 Robustness gaps
- No validation that the CSV has columns `Sample` / `EVM1 POWER Results (W)` → raw `KeyError` on schema drift.
- `filtfilt` with `order=5` raises a cryptic error on short traces (`padlen`); no length guard.
- `medfilt` requires an odd kernel; an even `-k` argument crashes.
- `expected_duration` (median of durations) is computed and never used; the fan-region filter is commented out — the known fan problem is currently unhandled.
- `-d/--data` help string says *"File path of .tflite file"* — copy-paste error; it's a CSV.
- `args.kernel_size`, `args.freq`, `args.cutoff` are parsed without `type=`, then manually cast — use `type=int` / `type=float` in `add_argument`.
- `get_average_power` mutates the caller's DataFrame in place (adds 3 columns) — surprising side effect.
- `seaborn` imported and never used.

---

## 2. `run_manager.py` (campaign orchestrator on the board)

### [Critical] 2.1 TPU backend crashes at startup
`execute()` calls `runner.generate_input()` unconditionally, but `TFLiteTPURunner` doesn't implement it → `AttributeError` for `--backend tpu`. The Coral workflow documented in the protocol cannot run with the current code.

### [Critical] 2.2 Hardcoded `count=100000` coupled to `data_processing.py`
See issue 1.3. There is also no `--count` CLI flag, even though the protocol TIPS section recommends fewer inferences for demanding layers — doing so today would silently desynchronize the energy-per-inference division.

### [High] 2.3 No error handling in the run loop
One CUDA OOM / delegate failure at run 3/10 kills the whole campaign while the INA226 keeps logging → a CSV with the wrong number of bursts and no record of why. **Fix:** try/except per run, log failures, exit with a summary (and non-zero exit code) so the number of expected bursts is known to the processing side.

### [High] 2.4 No host/target synchronization
Nothing couples the SSH-triggered script and the Windows-side logger; the operator must start collection manually first. If logging starts late, the first burst has no leading idle baseline. **Fix options:** initial settle-sleep contract (≥ rolling-window seconds of guaranteed idle at start/end), a power "beacon" pattern (short distinctive bursts) before the campaign, or a host-orchestrated launcher.

### [Medium] 2.5 Hygiene
- `numpy`, `importlib.util`, `gc` imported and unused.
- Commented-out dead code (`#self.runner = runner_cls(model_path)`, `del runner`, `gc.collect()`).
- The `else: raise ValueError(...)` branch is unreachable (`argparse` `choices` already restricts values).
- `--sleep_time` is `int`; fractional seconds impossible (minor).

---

## 3. `runner.py` (inference runners)

### [Critical] 3.1 Case-sensitive layer-type parsing → `UnboundLocalError`
`_extract_layer_info` compares `parts[0]` against lowercase `"linear"`/`"conv"`, but the measurement protocol instructs `--model Linear_64_64` (capital L). Capitalized names fall to the `else` branch (`type: "Linear"`), then `_build_model` matches nothing and hits `return model` with `model` unassigned → crash. **Fix:** `layer_type = parts[0].lower()` and raise a clear `ValueError` for unknown types in `_build_model`.

### [High] 3.2 Missing `torch.cuda.synchronize()`
CUDA execution is asynchronous — the loop only enqueues kernels. Without a synchronize before `time.sleep`, the GPU is still draining the queue while the script believes the run has ended: burst boundaries in the script's timeline are wrong, logs/timestamps are wrong, and short `sleep_time` values can smear adjacent bursts together. The call is present but commented out. **Fix:** `if self.device.type == "cuda": torch.cuda.synchronize()` after the inner loop.

### [High] 3.3 No `torch.no_grad()` / `inference_mode()` during inference
Model parameters have `requires_grad=True`, so each of the 100 000 forwards builds and discards an autograd graph. The measured energy therefore includes autograd bookkeeping, not pure inference. **Fix:** wrap the loop in `with torch.inference_mode():`.

### [High] 3.4 `TFLiteTPURunner.run_inference` issues
- Regenerates its own random input each call, unlike `TorchRunner` (inconsistent contract).
- `input_shape[1]` assumes a 2-D input tensor — will break for conv models on TPU.
- `scale, zero_point` are used unguarded; a non-quantized model gives `scale == 0` → division by zero.
- `time.sleep(1)` inside `run_inference` sits inside the "active" window (same problem in `TorchRunner`); sleeping is the manager's job.

### [Medium] 3.5 Contract/design issues
- `TFLiteTPURunner.__init__` accepts `device` and silently ignores it (`super().__init__(model_path)`).
- `_load_model(self, from_state_dict=False, device="cpu")` in `TorchRunner` shadows the base signature `_load_model(self)`.
- The `model_path` for Torch is not a path at all — the file is never read (weights are random). Functional for energy measurement, but the name is misleading and `torch.load` would fail if `from_state_dict=True` since no file exists.
- Printing the full input/output tensors after every run floods the console for 8192-wide layers.
- Conditional class definitions behind `_tf_available` / `_torch_available` make `from runner import TorchRunner` fail with a confusing `ImportError` on the wrong device — a registry with an explicit error message would be clearer.
- File header comment says `# tflite_tpu_runner.py` but the file is `runner.py`.

---

## 4. `base_runner.py` (ABC)

- [Medium] The abstract `run_inference(self, input_data)` does not match either implementation (`count`, `repeat`) — Liskov violation; the ABC no longer describes the real contract.
- [Medium] `generate_input()` is required by `RunManager` but not declared abstract — this is exactly why bug 2.1 (TPU crash) went unnoticed.
- [Low] `import time` unused; large block of commented-out `benchmark()` dead code.

**Fix:** align abstract signatures with reality and add `generate_input` as an abstract method (a default no-op implementation for runners that self-generate inputs is acceptable).

---

## 5. `base_model_builder.py` (ABC)

- [Medium] `__init__(self, input_size, output_size)` is Linear-specific. A conv builder needs `(in_channels, image_size, kernel_size, padding)` and cannot honour the base signature — the planned `conv_model_builder.py` (protocol "Future work") will not fit. Notably, both existing subclasses already **don't call** `super().__init__()`, confirming the contract doesn't fit even today.

**Fix:** accept a generic spec (dict or a small `LayerSpec` dataclass with `type` + `params` + a single canonical `filename_stem()`/`from_filename()`), which would also eliminate the *three* independent filename-parsing implementations (`runner.py`, `data_report.py`, builders).

---

## 6. `linear_model_builder.py`

- [Medium] `compile_model_for_tpu` catches all exceptions and only **prints** — a failed edge TPU compilation looks like success to `ModelBuilderManager.build_all()`, which continues happily. Return a success flag or re-raise, and aggregate failures.
- [Medium] `build_and_compile()` signatures differ between the two subclasses (`(self)` vs `(self, quantize=False)`) — another ABC drift.
- [Medium] Neither subclass calls `super().__init__()` (see §5).
- [Low] Directory defaults (`Models/TPU/Linear`) are relative to CWD — running from another directory scatters model folders.
- [Low] Large commented-out quantization block; `subprocess` import fine, but the `command` list should validate that `self.tflite_model_path` exists before invoking the compiler.

---

## 7. `model_builder_manager.py`

- [Medium] **The `__main__` block never calls `manager.build_all()`** — it constructs the manager and prints `'done'`. Following the protocol PDF ("go check model_builder_manager.py") produces zero models while claiming success.
- [Medium] The `TypeError` message references `TPUBaseModelBuilder`, but the actual base class is `BaseModelBuilder` — stale message.
- [Medium] `build_all()` has no per-spec error handling: one failed build aborts the remaining specs (with 64 combinations, that's expensive).
- [Low] Docstring says "ModelFactoryManager" (stale name); ~20 lines of commented-out legacy functions.
- [Low] Importing `TFTPULinearModelBuilder` at module level makes the whole file crash on machines without TensorFlow, even if the user wanted the Torch builder.

---

## 8. `data_report.py`

- [High] **Conv filename parsing is broken:** `in_size, out_size = parts[1], parts[2]` assumes the Linear naming scheme. For `Conv_1_32_3_0.csv` it treats `in_channels=1, image_size=32` as in/out sizes, and the layer label uses `parts[3]/parts[4]` — the "out_size" used in the matrix is actually the image size. The matrix sheet built via `net_power_dict` is then semantically wrong for conv data.
- [High] **No per-file error handling:** one malformed CSV (or a non-CSV file in the folder — `os.listdir` doesn't filter by extension) aborts the entire report run. Wrap each file in try/except, collect failures, and write them to a "rejected" sheet.
- [High] NaN results from `data_processing` (see 1.2) are written to Excel silently — poisoning the NAS lookup table.
- [Medium] `detailed_rows` / `net_power_dict` are module-level globals mutated inside `main()` — works, but fragile if the module is imported.
- [Medium] Hardcoded personal default paths (`/home/frederic/...`) as argparse defaults.
- [Medium] `--plot_folder` directory is never created (`os.makedirs` missing) → `show_data` save crashes on first run.
- [Medium] Matrix hyperlink code is half-dead: the `HYPERLINK` formula is commented out and the loop rewrites the same plain values; the lookup label is hardcoded to `Linear(...)` so conv rows never match.
- [Low] Sheet name says "Average Power Matrix" but stores `energy_avg_J`; the index label `"In\Out"` contains an unescaped backslash (works, but fragile); `numpy` imported unused; output filename `linear_power_report.xlsx` is misleading for conv campaigns and silently overwrites previous reports.

---

## 9. `draft/` scripts (exploratory — lower bar, but noted)

Common to all: hardcoded absolute paths from a previous user's machine (`/home/frederic/...`), no CLI arguments, `seaborn` imported unused (in most), duplicated filter code instead of importing `data_processing`.

### `draft/ActivePeriod.py`
- [Medium] `import tpu_linear_data_processing_old as tpu` — **module does not exist anywhere in the repo**; the script cannot run at all.
- Contains the original (working) version of the fan-region duration filter (`max_duration = 1.5 × median`) that was lost/commented out when migrated into `data_processing.py` — worth recovering.
- Same index-space bug family as `data_processing.py`: `end_indices` extended with `len(power)` (raw length) while thresholding `df["smoothed"]` (which has NaN edges — here not even dropna'd: `is_active = df["smoothed"] > threshold` keeps NaN rows as `False`, subtly different behaviour from the main pipeline).

### `draft/medLowPassFiltering.py`
- Uses the naive global above/below-threshold split (no region segmentation) — superseded by `data_processing.py`. `fs = 5` hardcoded here is likely where the wrong `fs=5` default in `get_average_power` came from.
- Last print says `"Final Results: average:"` but prints the **variance difference** (copy-paste label bug); also `variance_active - variance_idle` can be negative — not a meaningful variance.

### `draft/lowPassFiltering.py`, `draft/medianFiltering.py`
- Plot-only explorations; fine as drafts. `medianFiltering.py` uses `kernel_size = 43` vs the pipeline's 11 — if 43 was found better empirically, that knowledge is now lost.
- `.dropna()` on filter outputs that never contain NaN (`filtfilt`/`medfilt` return dense arrays) — harmless cargo cult.

### `draft/plotResults.py`
- `window_size = 1` rolling mean is a no-op.
- X-axis plots `df["Sample"]` but the label says "Time (s)" — mislabeled axis.
- Points to `new_measures/Conv_1_32_3_0.csv` which may not exist in the repo — will crash if missing.

**Recommendation:** either delete `draft/` or move it to an `experiments/` folder with a README stating it is not part of the pipeline; recover the fan-filter and the kernel-size finding into the main pipeline first.

---

## 10. Cross-cutting recommendations (priority order)

1. **Fix the silent-corruption bugs first** (1.1, 1.2, 1.7): they invalidate data you may already be collecting.
2. **Single source of truth for the inference count** (1.3/2.2): add `--count` to `run_manager.py`, record it (manifest JSON next to the CSV or encoded in the filename), and read it in `data_processing.py`.
3. **Make measurement failures explicit**: campaign manifest (model name, `nb_run`, `count`, per-run status) + processing-side validation (`len(regions) == nb_run`, bimodality check, SNR check). Reject loudly, never emit NaN.
4. **Fix the Jetson measurement validity issues** before the next campaign: lowercase parsing (3.1), `torch.cuda.synchronize()` (3.2), `torch.inference_mode()` (3.3).
5. **Repair the ABCs** (`generate_input` abstract, aligned `run_inference` signature, generic `LayerSpec`) so the planned conv TPU builder and future boards don't repeat these breakages.
6. **One filename parser** (`LayerSpec.from_filename` / `.filename_stem()`) used by builders, runners and the report — removes three inconsistent implementations, including the broken conv parsing in `data_report.py`.
7. **Unify constants**: `sampling_interval`, `fs`, filter defaults defined once (a small `config.py` or constants at the top of `data_processing.py` only).

---

## 11. Possible Improvements

Enhancements beyond bug fixing — none of these are defects, but each would raise the quality, credibility, or capability of the framework.

### 11.1 Measurement methodology

- **Warm-up runs.** The first burst of a campaign includes one-time costs (CUDA context creation, cuDNN autotuning, memory allocator warm-up, cold caches). Run 1–2 unmeasured warm-up bursts before the recorded runs, or mark run 1 for exclusion in post-processing.
- **Lock device clocks before measuring.** On the Jetson Nano, run `sudo jetson_clocks` and pin the `nvpmodel` power mode, and record both in the campaign manifest. Without this, DVFS (dynamic frequency scaling) makes power depend on the board's mood, inflating inter-region variance.
- **Record thermal state.** Log `tegrastats` (Jetson) or `vcgencmd measure_temp` (Pi) alongside the campaign. This lets you correlate baseline drift with temperature and later add a thermal-delta green metric with zero changes to the measurement code.
- **Seed the random inputs** (`torch.manual_seed`, `np.random.seed`) and store the seed in the manifest, so a campaign is exactly reproducible.
- **Direct INA226 acquisition over I2C** (already listed as future work in the protocol). Replacing the TI web GUI with a small Python acquisition daemon (e.g., `smbus2` from a companion device) removes the human from the loop, gives programmatic start/stop, real timestamps per sample, and eliminates the manual CSV rename step — the single largest source of operator error in the current protocol.
- **Sync beacon.** Emit a deterministic power pattern (e.g., three 0.5 s bursts, 1 s apart) at campaign start so the processing side gains an absolute time anchor and a free verification of the 10 Hz time base.

### 11.2 Signal processing

- **Change-point detection as an alternative to global Otsu.** Algorithms like PELT (`ruptures` library) or a two-state HMM segment step signals robustly and handle baseline drift natively — a better fit for a signal that is by construction a square wave plus noise.
- **Windowed/rolling Otsu or baseline detrending** for long campaigns where fan/thermal events shift the idle level mid-file.
- **Use `scipy.signal.sosfiltfilt`** with second-order sections instead of `(b, a)` polynomials — numerically safer for an order-5 Butterworth.
- **Rationalize the filter chain.** Median + Butterworth + rolling mean is triple smoothing with three tuning knobs; a calibration notebook that compares combinations on golden traces would justify (or simplify) the chain, and preserve empirical findings such as the `kernel_size = 43` experiment currently stranded in `draft/medianFiltering.py`.
- **Report uncertainty honestly:** bootstrap confidence intervals across regions, and always include `n_regions` in the output so the NAS can weight low-confidence entries.

### 11.3 Architecture

- **`LayerSpec` dataclass** as the single codec between layer parameters and filenames (`from_filename` / `filename_stem`), shared by builders, runners, and the report generator.
- **Runner/builder registries** (`RUNNERS: dict[str, type]`) populated according to installed frameworks, replacing the conditional class definitions — unknown backends then fail with an actionable message listing what is available.
- **Metric plugin interface** so new green metrics (carbon footprint = energy × grid carbon intensity; thermal delta; cost) can be added without touching segmentation code.
- **`logging` module with levels** instead of `print`, writing a per-campaign log file next to the CSV.
- **Central configuration** (`config.py` or a small YAML) for sampling interval, filter defaults, inference count, and paths; type hints throughout.

### 11.4 Data outputs

- **Machine-readable lookup table** (Parquet or CSV + JSON metadata) as the primary artifact for the NAS, with the Excel report kept as a human-facing view. Include schema version, board ID, campaign date, and code version in the metadata.
- **Status column on every row** (`ok` / `rejected` + reason) and a separate rejected sheet, so failed measurements are visible and re-runnable rather than silently absent.
- **QC plots** with the threshold and detected regions overlaid on the trace, so a human can validate segmentation at a glance.

### 11.5 Testing and tooling

- **Unit tests on synthetic traces** (square wave + Gaussian noise + drift + fan step) asserting that segmentation recovers the known ground-truth power/energy — this makes every future change to `data_processing.py` safe.
- **Regression tests on golden CSVs** from `Data/` with frozen expected outputs.
- **Pinned dependency versions** in the two requirements files (currently unpinned, so environments drift across boards).
- **Packaging** (`pyproject.toml` with console entry points, e.g. `banera-run`, `banera-report`) instead of loose scripts copied to each board.
- **Lint/CI** (ruff + pytest via GitHub Actions) to keep the codebase healthy as new boards and layer types are added.
- **Progress feedback** (`tqdm`) in `data_report.py` batch loops, which iterate over dozens of files.
