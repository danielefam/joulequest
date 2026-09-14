<h1>JouleQuest<br><sub><sub>Explore the energy behind every run</sub></sub></h1>

This project runs small AI-model tests and measures their power use with an
INA226EVM connected through a TI-SCB serial device.

## Automated measurement

`automated_measurement.py` is the primary entry point for one unattended
experiment in the two-host setup:

```text
PC connected to TI-SCB              Jetson reached through SSH
automated_measurement.py  ------->  run_manager.py
ina226_serial_logger.py              runner.py
                                      base_runner.py
```

The PC starts the Jetson experiment over SSH. The Jetson performs model setup,
excluded warm-up, and excluded calibration, then asks the PC to start local
INA226 acquisition. After the measured cycles and idle guards, the Jetson asks
the PC to stop acquisition before writing its final manifest. No interaction is
required after starting the command.

Automated runs use monotonic clock exchanges immediately before and after the
capture. The resulting schema-v2 manifest translates each remote burst into
the logger's `Elapsed Time (s)` domain. Processing uses those intervals only
when alignment uncertainty is at most 50% of one sample period; otherwise it
keeps the capture and automatically uses Otsu plus hysteresis.

```bash
python automated_measurement.py \
  --connection-config measurement_hosts.local.json \
  --backend cuda \
  --port /dev/serial/by-id/usb-Texas_Instruments_Generic_Bulk_Device_12345678-if01 \
  --output-directory measurements/runs \
  --shunt-ohms 0.012 \
  --max-expected-current-a 5.0 \
  --model Models/CUDA/Linear/Linear_8192_8192.pt
```

`--model` and `--remote-directory` refer to paths on the Jetson. `--port` and
`--output-directory` refer to the PC connected to the TI-SCB. Omit `--port`
when exactly one TI-SCB device is connected and can be auto-detected. Each
command creates an unambiguous local pair:

```text
measurements/runs/<campaign_id>.csv
measurements/runs/<campaign_id>.json
```

Acquisition begins only after model loading, warm-up, and adaptive calibration.
The CSV contains the leading idle baseline, measured cycles, inter-cycle idle,
trailing idle baseline, and `safety_margin_seconds`. The logger process exits
before the final manifest is written, so manifest I/O is not present in the
capture. The Jetson also retains its original manifest under
`--remote-manifest-directory` until the PC has saved its local copy. The PC then
removes the remote manifest to conserve board storage, so no manual `scp` or
cleanup is needed after a successful command. Use `--keep-remote-manifest` to
retain the Jetson copy for diagnostics. If local persistence or cleanup SSH
fails, the remote recovery copy is not removed.

SSH must work without a password prompt because automation uses
`BatchMode=yes`. Copy the public template, fill in the local values, and keep
that file out of Git:

```bash
cp measurement_hosts.example.json measurement_hosts.local.json
$EDITOR measurement_hosts.local.json
```

`measurement_hosts.local.json` contains `runner_host`, `jump_host`, remote
paths, and optional SSH settings. It is listed in `.gitignore`. Use the values
from that local file for a visible preflight:

```bash
ssh -J JUMP_USER@JUMP_HOST \
  -o BatchMode=yes \
  BENCH_USER@INFERENCE_HOST \
  'printf "SSH_OK: "; hostname'
```

The command returns `0` for a clean campaign, `2` when model execution completes
but acquisition is invalid, `1` for an experiment failure, and `130` after a
cleaned-up user interruption. Its final stdout line is a JSON result containing
the campaign status, timing, sample count, and both output paths.

## Manual fallback

`run_manager.py` remains available for runner-only or manual GUI acquisition.

It runs the test in three steps:

1. Warm-up: prepares the computer. This part is not measured.
2. Calibration: finds a suitable number of model runs. This part is not measured.
3. Measurement: runs the real test cycles.

A simple CUDA example is:

```bash
python run_manager.py --backend cuda --model Models/CUDA/Linear/Linear_64_64.pt --batch-size 1 --wait_for_acquisition
```

Use a model file that exists on your computer. The model name tells the program its shape. For example, `Linear_64_64.pt` means a linear layer with 64 inputs and 64 outputs.

## Important files

- `run_manager.py`: starts and manages the measurement.
- `automated_measurement.py`: runs on the PC and coordinates SSH plus local acquisition.
- `base_runner.py`: defines the common timed-burst runner contract.
- `runner.py`: runs the model on CPU, CUDA, or Edge TPU.
- `ina226_serial_logger.py`: saves measurements directly from the INA226 device to a CSV file.
- `banera_pt_requirements.txt`: saved Python environment for PyTorch CPU/CUDA work.
- `banera_tf_requirements.txt`: saved Python environment for TensorFlow/Edge TPU work.
- `tests/`: automated checks for the code.
- `docs_new/`: longer project notes and measurement documentation.

JouleQuest owns measurement processing and lookup-table generation.
Differentiable Linear, Conv2d, and attention interpolation over the resulting
CSV is provided by the independent [`joulegrad`](https://github.com/danielefam/joulegrad) Python API (installable via `pip install git+https://github.com/danielefam/joulegrad.git`). See
[docs_new/joulegrad_energy_estimation.md](docs_new/joulegrad_energy_estimation.md).

## Configuration

Choose the backend with `--backend`:

- `cpu`: normal processor. This is the easiest option.
- `cuda`: NVIDIA GPU. Use this only when CUDA and PyTorch are installed.
- `tpu`: Edge TPU. This needs the Edge TPU software and hardware.

These are the most useful settings:

| Setting                         | Meaning                                                  | Default                     |
| ------------------------------- | -------------------------------------------------------- | --------------------------- |
| `--number_of_cycles`          | Number of real measurement cycles                        | `5`                       |
| `--batch-size`                | Input samples processed by each model forward pass       | `1`                       |
| `--sleep_time`                | Pause, in seconds, between cycles                        | `10`                      |
| `--target_burst_seconds`      | Desired duration, in seconds, of each real cycle         | `10`                      |
| `--sampling_rate_hz`          | INA226 samples per second                                | `10`                      |
| `--burst-duration-margin`     | Safety factor for automatic burst sizing                 | `1.2`                     |
| `--calibration-sizing-max-attempts` | Attempts to reach the calibration batch duration   | `3`                       |
| `--calibration-duration-tolerance` | Accepted relative calibration duration error        | `0.20`                    |
| `--validation-repetitions`   | Excluded final-count validation bursts per round          | `3`                       |
| `--validation-max-rounds`    | Maximum validation and correction rounds                  | `3`                       |
| `--validation-safety-margin` | Extra inference-count margin after failed validation      | `1.1`                     |
| `--validation-cooldown-seconds` | Pause before each validation burst; defaults to sleep  | unset                     |
| `--clock-sync-exchanges`      | Maximum exchanges per pre/post round; precise links stop after at least 3 | `10`                      |
| `--max-clock-uncertainty-fraction` | Maximum uncertainty as a sample-period fraction    | `0.50`                    |
| `--wait_for_acquisition`      | Stops and waits for you to start manual INA226 recording | off                         |
| `--runner-host`               | SSH destination running inference                        | none (single-host fallback) |
| `--jump-host`                 | SSH host used to reach the inference host                | none                        |
| `--connection-config`         | Ignored JSON containing SSH/remote settings              | none                        |
| `--remote-directory`          | Jetson directory containing `run_manager.py`             | `.`                       |
| `--remote-manifest-directory` | Manifest directory on the Jetson                         | `measurements_jetson`     |
| `--keep-remote-manifest`      | Keep the Jetson copy after local persistence             | off                         |

The automated command accepts the same adaptive workload controls but does not
accept `--wait_for_acquisition`, logger duration/sample limits, or overwrite.
Its `--sampling_rate_hz` value controls both burst planning and the logger
interval.

Example with shorter cycles:

```bash
python run_manager.py \
  --backend cpu \
  --model Models/CPU/Linear/Linear_64_64.pt \
  --number_of_cycles 1 \
  --sleep_time 0 \
  --target_burst_seconds 0.5 \
  --sampling_rate_hz 100 \
  --min_active_samples 50 \
  --warmup_inferences 5 \
  --warmup_seconds 0 \
  --warmup_cooldown_seconds 0 \
  --calibration_initial_inferences 2 \
  --calibration_target_seconds 0.1 \
  --calibration_repetitions 2 \
  --calibration-sizing-max-attempts 1 \
  --validation-repetitions 1 \
  --validation-max-rounds 3 \
  --leading_idle_seconds 0 \
  --trailing_idle_seconds 0 \
  --safety_margin_seconds 0 \
  --wait_for_acquisition
```

## Python environment

On the PC connected to the TI-SCB, install pyserial:

```bash
python -m pip install -r Docs/INA226EVM/requirements.txt
```

Result processing also requires pandas, scikit-image, and matplotlib:

```bash
python -m pip install pandas scikit-image matplotlib
```

On the Jetson, use its JetPack-compatible PyTorch environment for CPU or CUDA.
Copy only the inference-side scripts:

```bash
scp run_manager.py runner.py base_runner.py \
  -o ProxyJump=JUMP_USER@JUMP_HOST \
  BENCH_USER@INFERENCE_HOST:/path/to/JouleQuest/
```

Keep these scripts on the PC:

```text
automated_measurement.py
ina226_serial_logger.py
```

The no-SSH single-host fallback remains available by omitting `--runner-host`;
that machine then needs all five runtime scripts and both backend and serial
dependencies.

## Complete experiment campaign

`automated_measurement.py` intentionally runs one experiment. To schedule the
complete Linear/Conv matrix for one physical board, use the separate batch
launcher:

```bash
./run_measurement_campaign.sh --board BOARD_LABEL --suite all
```

The launcher invokes `automated_measurement.py` once per model, stores results
under `measurements/runs/BOARD_LABEL/`, stops on the first error by default,
and skips models that already have a `COMPLETE` manifest. It never switches
boards. After it finishes, change the physical board and start a new invocation
with a new board label and local connection/current settings.

Inspect all generated commands without starting SSH, inference, or acquisition:

```bash
./run_measurement_campaign.sh --board BOARD_LABEL --suite all --dry-run
```

Add an idle cooling interval between different layer experiments when needed:

```bash
./run_measurement_campaign.sh \
  --board BOARD_LABEL \
  --suite all \
  --experiment-cooldown-seconds 60
```

`Ctrl+C` stops the whole campaign even with `--continue-on-error`; restarting
with the same board label skips experiments that already have a `COMPLETE`
manifest.

Campaign parameters and matrices can be edited at the beginning of the script
or overridden with environment variables. See
`docs_new/experiment_campaign.md` for the complete matrix and examples.
The original Linear/Conv matrices remain intact. The launcher additionally
schedules 201 pruning-oriented basic-layer points: 57 low-feature Linear
combinations, 80 small-spatial `3x3` Conv points, and a 64-point `1x1` Conv
grid. Reusing the same board label skips already completed original points.

## Process campaign results

Process the colocated CSV/JSON pairs and regenerate the per-measurement PDFs and
summary with:

```bash
MPLBACKEND=Agg python processing_report/process_and_visualize.py
```

To process one measurement with optional trimming, use:

```bash
python processing_report/data_processing.py measurements/runs/MEASUREMENT.csv \
  --tail-trim-percentage 10 \
  --discard-initial-samples \
  --initial-trim-percentage 10
```

By default, data processing discards the lowest 10% and highest 10% power samples
in each active region, as well as the first 10% of samples in time order.
Trimming is skipped when a cycle has fewer than 10 inferences.

By default, the script reads `measurements/runs/jetson_nano_base`, finds each
manifest beside its same-stem CSV, uses the achieved sampling rate recorded for
that acquisition, and processes only manifests with `status: COMPLETE`.
Incomplete CSV captures are reported and skipped rather than producing empty
statistics.

Schema-v2 manifests use synchronized elapsed-time intervals when their
uncertainty passes the strict gate. Older schema-v1 data and poor/missing sync
metadata use Otsu plus hysteresis and remain in the output. Energy is integrated
with actual elapsed-time deltas after subtracting the median power from the
final measured `safety_margin_seconds` window. If that window is unavailable,
the processor retains the result and reports a classified-idle fallback.

```bash
python processing_report/process_and_visualize.py \
  --data-dir measurements/runs/BOARD_LABEL \
  --plot-dir measurements/Plot/BOARD_LABEL \
  --output-name BOARD_LABEL.csv \
  --max-clock-uncertainty-fraction 0.50
```

The generated summary CSV (saved in `measurements/lookup_summaries/summaries/`) includes model
and campaign identity, quality status, achieved sampling rate, inference count,
expected and detected active regions, classifier source, clock uncertainty/fallback
reason, idle-baseline source/statistics, threshold, power mean/variance, and
energy mean/variance.

## Energy lookup table

Create a table for the downstream energy-inference program from one or more
processed board summaries:

```bash
python -m processing_report.build_energy_lookup_table \
  measurements/lookup_summaries/summaries/pi5.csv \
  --output measurements/lookup_summaries/pi5_energy_lookup.csv
```

The output has one row per measured layer configuration. Linear rows use
`input_features` and `output_features`; convolution rows use
`input_channels`, `output_channels`, `input_image_size`, `kernel_size`,
`stride`, and `padding`. `energy_mean_mJ` is the mean energy per input sample,
while `measurement_count` and `energy_stddev_mJ` retain repeatability data for
the inference program. Only complete, quality-approved, positive, internally
consistent campaigns are included.

## More help

For the complete measurement procedure, read `docs_new/adaptive_burst_measurement.md`.
