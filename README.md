# energyBANERA

This project runs small AI-model tests and can measure their power use with an INA226 device.

## Main file

The main file is `run_manager.py`.

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
```

Use the TensorFlow requirements only for Edge TPU work:

```bash
conda create --name banera_tf --file banera_tf_requirements.txt
conda activate banera_tf
```

## More help

For the complete measurement procedure, read `docs_new/adaptive_burst_measurement.md`.