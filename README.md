# energyBANERA

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

```bash
python automated_measurement.py \
  --connection-config measurement_hosts.local.json \
  --backend cuda \
  --model Models/CPU/Linear/Linear_8192_8192.pt \
  --port /dev/serial/by-id/usb-Texas_Instruments_Generic_Bulk_Device_12345678-if01 \
  --output-directory measurements/runs \
  --shunt-ohms 0.012 \
  --max-expected-current-a 5.0
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
`--remote-manifest-directory`; the PC receives a copy automatically, so no
manual `scp` is needed after a successful command.

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

A simple CPU example is:

```bash
python run_manager.py --backend cpu --model Models/CPU/Linear/Linear_64_64.pt --wait_for_acquisition
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

## Configuration

Choose the backend with `--backend`:

- `cpu`: normal processor. This is the easiest option.
- `cuda`: NVIDIA GPU. Use this only when CUDA and PyTorch are installed.
- `tpu`: Edge TPU. This needs the Edge TPU software and hardware.

These are the most useful settings:

| Setting | Meaning | Default |
| --- | --- | --- |
| `--number_of_cycles` | Number of real measurement cycles | `5` |
| `--sleep_time` | Pause, in seconds, between cycles | `10` |
| `--target_burst_seconds` | Desired duration, in seconds, of each real cycle | `10` |
| `--sampling_rate_hz` | INA226 samples per second | `10` |
| `--wait_for_acquisition` | Stops and waits for you to start manual INA226 recording | off |
| `--runner-host` | SSH destination running inference | none (single-host fallback) |
| `--jump-host` | SSH host used to reach the inference host | none |
| `--connection-config` | Ignored JSON containing SSH/remote settings | none |
| `--remote-directory` | Jetson directory containing `run_manager.py` | `.` |
| `--remote-manifest-directory` | Manifest directory on the Jetson | `measurements_jetson` |

The automated command accepts the same adaptive workload controls but does not
accept `--wait_for_acquisition`, logger duration/sample limits, or overwrite.
Its `--sampling_rate_hz` value controls both burst planning and the logger
interval.

Example with shorter cycles:

```bash
python run_manager.py \
  --backend cpu \
  --model Models/CPU/Linear/Linear_64_64.pt \
  --number_of_cycles 3 \
  --sleep_time 5 \
  --target_burst_seconds 5 \
  --wait_for_acquisition
```

## Python environment

On the PC connected to the TI-SCB, install pyserial:

```bash
python -m pip install -r Docs/INA226EVM/requirements.txt
```

On the Jetson, use its JetPack-compatible PyTorch environment for CPU or CUDA.
Copy only the inference-side scripts:

```bash
scp run_manager.py runner.py base_runner.py \
  -o ProxyJump=JUMP_USER@JUMP_HOST \
  BENCH_USER@INFERENCE_HOST:/path/to/energyBANERA/
```

Keep these scripts on the PC:

```text
automated_measurement.py
ina226_serial_logger.py
```

The no-SSH single-host fallback remains available by omitting `--runner-host`;
that machine then needs all five runtime scripts and both backend and serial
dependencies.

## More help

For the complete measurement procedure, read `docs_new/adaptive_burst_measurement.md`.