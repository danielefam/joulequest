# energyBANERA

This project runs small AI-model tests and measures their power use with an
INA226EVM connected through a TI-SCB serial device.

## Automated measurement

`automated_measurement.py` is the primary entry point for one unattended
experiment. It prepares the selected model, performs excluded warm-up and
calibration, starts direct INA226 acquisition, executes the measured cycles,
stops acquisition, and writes the paired artifacts without prompting.

```bash
python automated_measurement.py \
  --backend cpu \
  --model measurements/Data/Linear/Linear_8192_8192.pt \
  --port /dev/serial/by-id/usb-Texas_Instruments_Generic_Bulk_Device_12345678-if01 \
  --output-directory measurements/runs \
  --shunt-ohms 0.012 \
  --max-expected-current-a 5.0
```

Omit `--port` when exactly one TI-SCB device is connected and it can be
auto-detected. Each command creates an unambiguous pair:

```text
measurements/runs/<campaign_id>.csv
measurements/runs/<campaign_id>.json
```

Acquisition begins only after model loading, warm-up, and adaptive calibration.
The CSV contains the leading idle baseline, measured cycles, inter-cycle idle,
trailing idle baseline, and `safety_margin_seconds`. The logger process exits
before the final manifest is written, so manifest I/O is not present in the
capture.

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
- `automated_measurement.py`: coordinates one runner and serial acquisition process.
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

Use the PyTorch requirements for CPU or CUDA work:

```bash
conda create --name banera_pt --file banera_pt_requirements.txt
conda activate banera_pt
python -m pip install -r Docs/INA226EVM/requirements.txt
```

Use the TensorFlow requirements only for Edge TPU work:

```bash
conda create --name banera_tf --file banera_tf_requirements.txt
conda activate banera_tf
python -m pip install -r Docs/INA226EVM/requirements.txt
```

## More help

For the complete measurement procedure, read `docs_new/adaptive_burst_measurement.md`.