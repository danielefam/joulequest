# energyBANERA - Project Documentation

> **BANERA** = **B**enchmarking **A**rchitectural **NE**tworks for **R**esource-**A**ware inference

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation: Power vs Accuracy Tradeoff in NAS](#2-motivation-power-vs-accuracy-tradeoff-in-nas)
3. [Measurement Hardware](#3-measurement-hardware)
4. [Raspberry Pi 5 Notes](#4-raspberry-pi-5-notes)
5. [Software Architecture](#5-software-architecture)
6. [End-to-End Workflow](#6-end-to-end-workflow)
7. [CSV Data Format](#7-csv-data-format)
8. [Signal Processing Pipeline](#8-signal-processing-pipeline)
9. [Quick Start Guide](#9-quick-start-guide)
10. [Directory Structure](#10-directory-structure)
11. [Dependencies and Environments](#11-dependencies-and-environments)

---

## 1. Project Overview

energyBANERA is an **energy benchmarking** framework designed to measure the power consumption of individual neural network layers on embedded and edge hardware. The main objective is to build an **energy lookup table** that maps each layer configuration (type and dimensions) to its real cost in terms of:

- **Average power** dissipated during inference [W]
- **Energy per single inference** [J]
- **Measurement variance** (stability indicator)

These measurements can be injected into **Neural Architecture Search (NAS)** algorithms that must balance model accuracy against energy cost on the target device.

---

## 2. Motivation: Power vs Accuracy Tradeoff in NAS

### Problem Statement

In NAS, the optimal architecture is selected by minimizing a composite objective:

$$
\mathcal{L}_{NAS} = \mathcal{L}_{acc}(\theta) + \lambda \cdot \mathcal{C}_{energy}(\alpha)
$$

where:
- $\mathcal{L}_{acc}$ is the accuracy loss term
- $\mathcal{C}_{energy}$ is the architecture energy cost
- $\alpha$ is the vector of architectural choices (layer type, dimensions, etc.)
- $\lambda$ controls the tradeoff strength

### How energyBANERA Helps

The framework measures $\mathcal{C}_{energy}(\alpha)$ **empirically** for each candidate layer choice. Instead of relying on proxies such as FLOPs, MACs, or parameter count, it provides real hardware-level measurements.

### NAS + BANERA Workflow

```text
[NAS Search Space]
        |
        v
[Architectural choice alpha]
        |
        v
[BANERA Lookup Table]  <-- built offline with this framework
        |
        v
[Energy cost C_energy(alpha)]
        |
        v
[NAS optimizer] -- balances accuracy vs energy --> [Optimal Architecture]
```

### Measured Layer Types

| Layer Type | Parameters | Tested Hardware |
|------------|------------|-----------------|
| `Linear`   | `(in_features, out_features)` | Coral Dev Board, Jetson Nano |
| `Conv2d`   | `(in_channels, image_size, kernel_size, padding)` | Jetson Nano |

Linear sizes currently covered: **64, 128, 256, 512, 1024, 2048, 4096, 8192** (all input x output combinations).

---

## 3. Measurement Hardware

### 3.1 Instrument: INA226EVM

The **Texas Instruments INA226EVM** is an evaluation board built around INA226, a high-precision current and power monitor with I2C interface.

**Key features:**
- Simultaneous measurement of shunt voltage, bus voltage, current, and power
- Current resolution depends on shunt resistor value
- Sampling used in this project: **10 Hz** (0.1 s per sample)
- USB/I2C connection to host PC

**Recorded CSV fields:**

| CSV Column | Unit | Description |
|------------|------|-------------|
| `Sample` | - | Sample index |
| `EVM1 VSHUNT Results (mV)` | mV | Shunt voltage drop |
| `EVM1 VBUS Results (V)` | V | Supply bus voltage |
| `EVM1 CURRENT Results (A)` | A | Computed current |
| `EVM1 POWER Results (W)` | W | Computed power (`P = V_bus x I`) |

### 3.2 Target Devices

#### Google Coral Dev Board (Edge TPU)
- ML processor: Edge TPU
- Runtime backend: TFLite runtime + Edge TPU delegate (`libedgetpu.so.1`)
- Required model type: INT8 quantized TFLite compiled by `edgetpu_compiler`
- OS: Mendel Linux
- Docs: https://coral.ai/docs/

#### NVIDIA Jetson Nano
- GPU: 128-core Maxwell
- Runtime backend: PyTorch (CPU or CUDA)
- Required model type: `.pt` PyTorch state dict

---

## 4. Raspberry Pi 5 Notes

Raspberry Pi 5 is a valid BANERA measurement target for PyTorch `Linear` models. It behaves similarly to the Jetson CPU path from the software point of view (`--backend cpu`), but power delivery constraints are stricter.

### 4.1 Recommended Power Setup

- Use a stable USB-C PSU capable of **5V up to 5A**.
- Use a short and good-quality USB-C cable (poor cables are a common instability source).
- Avoid powering the board from a PC USB port during measurements.
- If the board disconnects, LED color changes unexpectedly, or HDMI output disappears, first suspect power instability.

### 4.2 INA226EVM Configuration for Pi 5

- In INA226EVM software, set **Max Expected Current = 5A**.
- Keep the same sampling approach used in this project (10 Hz, long inference bursts) for consistency with existing datasets.

### 4.3 Where to Store Pi 5 Measurements

Use a dedicated folder to keep data separated from Coral/Jetson:

```text
Data/
`- Raspberry_pi5_power_record/
        `- Linear/
```

Filename convention stays unchanged:

- `Linear_{in_features}_{out_features}.csv`
- Example: `Linear_512_512.csv`

### 4.4 Pi 5 Measurement Command

```bash
python run_manager.py --backend cpu --model Models/CPU/Linear/Linear_128_256.pt --nb_run 5 --sleep_time 10
```

### 4.5 Quick Stability Checklist (Pi 5)

1. Boot with minimal peripherals connected.
2. Verify HDMI and SSH are both stable at idle before running bursts.
3. Run one small model first (`Linear_64_64`) as a smoke test.
4. Only then launch the full campaign (`64, 512, 1024, 2048, ...`).

---

## 5. Software Architecture

### High-Level Class Diagram

```text
BaseModelBuilder (ABC)
|- TFTPULinearModelBuilder     # Coral Edge TPU - Linear layers
`- TorchLinearModelBuilder     # CPU/CUDA (Jetson) - Linear layers

InferenceRunner (ABC)
|- TFLiteTPURunner             # Edge TPU inference via TFLite
`- TorchRunner                 # Inference via PyTorch

ModelBuilderManager            # Batch model generation
RunManager                     # Controlled inference sessions

data_processing.py             # Signal processing over power CSV traces
data_report.py                 # Excel reporting and aggregation
```

### Module Responsibilities

| File | Responsibility |
|------|----------------|
| `base_model_builder.py` | Abstract builder interface |
| `base_runner.py` | Abstract inference runner interface |
| `linear_model_builder.py` | Build, quantize, compile Linear layers |
| `model_builder_manager.py` | Build full model collections over a search grid |
| `runner.py` | Concrete TPU and Torch inference runners |
| `run_manager.py` | Session orchestration for repeatable measurements |
| `data_processing.py` | Signal filtering and statistical extraction |
| `data_report.py` | Report aggregation and Excel export |

---

## 6. End-to-End Workflow

### Phase 1 - Model Preparation

```text
Define search space (size lists)
        |
        v
ModelBuilderManager.build_all()
        |
        +-- [TF] build_model() -> Keras Dense
        |         convert_to_tflite_quantized() -> INT8 TFLite
        |         compile_model_for_tpu() -> edgetpu_compiler -> compiled .tflite
        |
        `-- [Torch] build_model() -> nn.Linear / nn.Conv2d
                    save_model() -> .pt state dict
```

### Phase 2 - Power Acquisition

```text
INA226EVM in series with target device power rail
        |
        v
run_manager.py --backend [tpu|cpu|cuda] --model <path> --nb_run N --sleep_time T
        |
        v
RunManager.execute():
    for i in range(nb_run):
        runner.run_inference(count=100000, repeat=1)
        sleep(sleep_time)
        |
        v
INA226EVM records power trace -> CSV
```

Each burst corresponds to `count x repeat` inferences. Comparing burst vs idle isolates the ML contribution.

### Phase 3 - Processing and Report

```text
data_report.py --data_folder <csv_folder> --plot_folder <plot_folder>
        |
        v
For each CSV:
    data_processing.get_average_power(df)
        -> {power_avg_W, power_var_W2, energy_avg_J, energy_var_J2}
        |
        v
Excel export:
    - Average Power Matrix
    - Detailed Stats
    - SVG plot per layer
```

---

## 7. CSV Data Format

### Row Example

```text
Sample,EVM1 VSHUNT Results (mV),EVM1 VBUS Results (V),EVM1 CURRENT Results (A),EVM1 POWER Results (W)
0,2.8,4.97,0.23329327610872677,1.1582260371959945
1,2.7775,4.97,0.23146208869814022,1.1513590844062949
```

### Filename Convention

Files are named so scripts can infer layer configuration automatically.

| Layer Type | Example | Pattern |
|------------|---------|---------|
| Linear | `Linear_128_256.csv` | `Linear_{in_features}_{out_features}.csv` |
| Conv | `Conv_64_128_3_0.csv` | `Conv_{in_channels}_{image_size}_{kernel_size}_{padding}.csv` |

### Data Folders

```text
Data/
|- Coral_dev_board_power_record/
|  `- Linear/               # Edge TPU measurements
`- Jetson_nano_power_record/
   |- Linear/               # Jetson Linear measurements
   `- Conv_3_0/             # Conv2d kernel 3x3 padding 0
```

---

## 8. Signal Processing Pipeline

The main entry point is `data_processing.get_average_power()`.

### 7.1 Load Data

```python
df = pd.read_csv(csv_path)
# Main column: 'EVM1 POWER Results (W)'
```

### 7.2 Median Filter

```python
df["median_filtered"] = medfilt(df['EVM1 POWER Results (W)'], kernel_size=11)
```

Purpose: remove impulse outliers and spikes.

### 7.3 Butterworth Low-Pass Filter

```python
b, a = butter(order=5, Wn=cutoff/nyq, btype='low')
df["lowpass_filtered"] = filtfilt(b, a, df["median_filtered"])
```

Purpose: keep slow trend, suppress high-frequency noise.

### 7.4 Rolling Mean Smoothing

```python
df["smoothed"] = df["lowpass_filtered"].rolling(window=30, center=True).mean()
```

Purpose: stabilize plateau segmentation.

### 7.5 Otsu Threshold for Idle vs Active

$$
T_{Otsu} = \arg\min_T \left[ w_0(T)\sigma_0^2(T) + w_1(T)\sigma_1^2(T) \right]
$$

```python
threshold = threshold_otsu(df['smoothed'].dropna().values)
```

### 7.6 Active Region Detection

```text
is_active[i] = smoothed[i] > threshold
change_points = diff(is_active)
start_indices = where(change_points == +1)
end_indices   = where(change_points == -1)
```

### 7.7 Region-Level Power and Energy

For each active region $r$:

$$
P_{offset,r} = \bar{P}_{active,r} - \bar{P}_{idle,r}
$$

$$
\sigma^2_{P,r} = \sigma^2_{active,r} + \sigma^2_{idle,r}
$$

$$
E_r = P_{offset,r} \cdot \Delta t_r \cdot \frac{1}{100000}
$$

The $1/100000$ factor matches `count=100000` inferences per burst.

### 7.8 Aggregate Metrics Across Regions

$$
\bar{P}_{final} = \frac{1}{N} \sum_r P_{offset,r}
$$

$$
\bar{E}_{final} = \frac{\sum_r \Delta t_r \cdot E_r}{\sum_r \Delta t_r}
$$

**Returned structure:**

```python
results = {
    "power_avg_W": float,
    "power_var_W2": float,
    "energy_avg_J": float,
    "energy_var_J2": float,
}
```

---

## 9. Quick Start Guide

### 8.1 Environment Setup

**Coral / TensorFlow path:**
```bash
conda create --name banera_tf --file banera_tf_requirements.txt
conda activate banera_tf
```

**Jetson / PyTorch path:**
```bash
conda create --name banera_pt --file banera_pt_requirements.txt
conda activate banera_pt
```

### 8.2 Build Models (One-Time)

```python
from model_builder_manager import ModelBuilderManager
from linear_model_builder import TFTPULinearModelBuilder
from itertools import product

sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
specs = list(product(sizes, sizes))

manager = ModelBuilderManager(TFTPULinearModelBuilder, specs)
manager.build_all()
```

### 8.3 Run Measurement Sessions

```bash
# Coral Edge TPU
python run_manager.py --backend tpu --model Models/TPU/Compiled/Linear_128_256_edgetpu.tflite --nb_run 5 --sleep_time 10

# Jetson CPU
python run_manager.py --backend cpu --model Models/CPU/Linear/Linear_128_256.pt --nb_run 5 --sleep_time 10

# Jetson CUDA
python run_manager.py --backend cuda --model Models/CPU/Linear/Linear_128_256.pt --nb_run 5 --sleep_time 10

# Raspberry Pi 5 (CPU)
python run_manager.py --backend cpu --model Models/CPU/Linear/Linear_128_256.pt --nb_run 5 --sleep_time 10
```

### 8.4 Analyze Single CSV

```bash
python data_processing.py -d Data/Coral_dev_board_power_record/Linear/Linear_128_256.csv -k 11 -fs 5 -cutoff 0.1 -w 30
```

### 8.5 Build Folder-Level Excel Report

```bash
python data_report.py --data_folder Data/Jetson_nano_power_record/Linear --plot_folder Plot/Jetson_nano_plots/Linear
```

---

## 10. Directory Structure

```text
energyBANERA-main/
|- base_model_builder.py
|- base_runner.py
|- linear_model_builder.py
|- model_builder_manager.py
|- runner.py
|- run_manager.py
|- data_processing.py
|- data_report.py
|- banera_tf_requirements.txt
|- banera_pt_requirements.txt
|- Data/
|- Docs/
|- draft/
|- Tuto/
`- docs_new/
   |- README.md
        `- studio_guide.md
```

---

## 11. Dependencies and Environments

### TensorFlow / Edge TPU Environment

Main packages:
- `tensorflow`
- `tflite_runtime`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-image`
- `openpyxl` or `xlsxwriter`

### PyTorch / CUDA Environment

Main packages:
- `torch`
- `torchvision`
- `numpy`

### External Tool Required

- `edgetpu_compiler` for Coral Edge TPU deployment.

---

## Practical Notes for Electronic Measurements

### INA226EVM Wiring Concept

```text
DC Supply
   |
   +---[INA226EVM IN+]---[shunt R]---[INA226EVM IN-]---> Target VDD
   |
   +----------------------------------------------------> Target GND
```

### Sampling Considerations

- Effective logging used in this project: 10 Hz
- Fast per-inference transients are averaged by long bursts (100000 inferences)
- This is suitable for robust mean energy estimation per inference

### Recommended Measurement Campaign

1. Measure each `(in_size, out_size)` with at least `nb_run=5`
2. Keep `sleep_time=10` to create a visible idle baseline
3. Inspect generated plots to detect anomalies
4. Import `energy_avg_J` in the NAS energy regularization term
