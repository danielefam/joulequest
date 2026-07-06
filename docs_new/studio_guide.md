# Operational Guide: Study Order and Start Measurements Today

This guide is designed to let you:
1. Learn the codebase in the right order.
2. Be operational this afternoon (about 4 hours from now).
3. Collect a first valid measurement campaign for NAS power vs accuracy tradeoff.

## 1) Recommended Code Study Order

Read files in this exact order.

### Step A - Understand the Base Interfaces (15-20 min)
1. `base_model_builder.py`
2. `base_runner.py`

Goal:
- Understand the two core abstract contracts: model building and inference.
- Memorize key methods: `build_and_compile`, `_load_model`, `run_inference`.

Why first:
- Everything else builds on these interfaces.

### Step B - Understand Model Creation (25-35 min)
1. `linear_model_builder.py`
2. `model_builder_manager.py`

Goal:
- Separate the two flows:
  - TensorFlow/TFLite for Coral Edge TPU
  - PyTorch for CPU/CUDA
- Understand model naming: `Linear_in_out`
- Understand output directories: `Models/TPU/...` and `Models/CPU/...`

Key points:
- TF branch: quantization and Edge TPU compilation
- Torch branch: `.pt` state dict save path

### Step C - Understand Inference and Benchmark Loop (30-40 min)
1. `runner.py`
2. `run_manager.py`

Goal:
- Understand synthetic input generation
- Understand inference burst structure:
  - `run_inference(count=100000, repeat=1)`
  - pause between runs via `sleep_time`
- Understand backend switch:
  - `tpu`
  - `cpu`
  - `cuda`

Critical details:
- `runner.py` infers layer type from filename (`Linear` or `Conv`)
- `run_manager.py` is the practical entry point for acquisition

### Step D - Understand Energy Metric Extraction (45-60 min)
1. `data_processing.py`

Goal:
- Understand filter pipeline:
  - median filter
  - Butterworth low-pass
  - rolling average
  - Otsu threshold
- Understand active/idle segmentation and power offset formula
- Understand how `energy_avg_J` is computed per inference

Critical details:
- `k`, `fs`, `cutoff`, `window` strongly affect robustness
- The divisor `100000` is consistent with burst `count=100000`

### Step E - Understand NAS-Ready Outputs (20-30 min)
1. `data_report.py`
2. Real CSVs in `Data/...`

Goal:
- Understand Excel report generation
- Understand the in_size x out_size matrix used as energy lookup

Expected outcome:
- You know exactly which script starts acquisition
- You know exactly which script performs analysis
- You know where to read per-layer energy for NAS integration

## 2) Time Plan: Start This Afternoon (T+4h)

Assume T0 is now and measurement starts in about 4 hours.

### Block 1 (T0 -> T0+60 min): Minimum Effective Study
- 0-20 min: Step A
- 20-55 min: Step B
- 55-60 min: quick personal recap (5 lines):
  - How do I create a model?
  - How do I run inference?
  - Where do I read average energy?

Deliverable:
- You have a clear end-to-end mental map.

### Block 2 (T0+60 -> T0+120 min): Environment and Smoke Test
- Choose initial platform:
  - Jetson (recommended if immediately available)
  - Coral (if runtime + compiler are already ready)
- Verify environment dependencies
- Run one quick test with a single model

Deliverable:
- At least one full run without runtime errors.

### Block 3 (T0+120 -> T0+180 min): Electrical Setup
- Connect INA226EVM in series on target power rail
- Verify stable idle reading for 2-3 minutes
- Define CSV naming before acquisition

Deliverable:
- Clean idle trace and naming convention ready.

### Block 4 (T0+180 -> T0+240 min): First Measurement Campaign
- Run 3-5 representative Linear configurations:
  - small: 64x64
  - medium: 512x512
  - medium-high: 1024x1024
  - high: 2048x2048
  - optional: 4096x4096
- For each configuration:
  - `nb_run=5`
  - `sleep_time=10`
  - dedicated CSV logging

Deliverable:
- First dataset usable today for tradeoff analysis.

## 3) Practical Procedure for Today (Checklist)

### 3.1 Pre-Flight Checklist (15 min)
- Stable power supply with sufficient current budget
- INA226EVM recognized by host PC
- Enough disk space for CSV logs
- Target device reachable (SSH or local terminal)
- Test model available:
  - `Linear_128_128.pt` or
  - `Linear_128_128_edgetpu.tflite`

### 3.2 Inference Execution Commands

#### Jetson/CPU
```bash
python run_manager.py --backend cpu --model Models/CPU/Linear/Linear_128_128.pt --nb_run 5 --sleep_time 10
```

#### Jetson/CUDA
```bash
python run_manager.py --backend cuda --model Models/CPU/Linear/Linear_128_128.pt --nb_run 5 --sleep_time 10
```

#### Coral/TPU
```bash
python run_manager.py --backend tpu --model Models/TPU/Compiled/Linear_128_128_edgetpu.tflite --nb_run 5 --sleep_time 10
```

Operational note:
- Start INA226 logging just before command launch
- Stop logging shortly after the final burst

### 3.3 Recommended CSV Naming

Mandatory naming convention:
- `Linear_in_out.csv`
- Example: `Linear_512_512.csv`

Reason:
- `data_processing.py` and `data_report.py` expect this format.

### 3.4 Immediate Post-Acquisition Analysis

Single CSV:
```bash
python data_processing.py -d Data/Coral_dev_board_power_record/Linear/Linear_512_512.csv -k 11 -fs 5 -cutoff 0.1 -w 30
```

Folder report:
```bash
python data_report.py --data_folder Data/Jetson_nano_power_record/Linear --plot_folder Plot/Jetson_nano_plots/Linear
```

## 4) Minimum Campaign for Today

If time is tight, run this set only:
1. `Linear_64_64`
2. `Linear_512_512`
3. `Linear_1024_1024`
4. `Linear_2048_2048`

For each model:
- 5 bursts (`nb_run=5`)
- 100000 inferences per burst (already in code)
- 10 s idle pause between bursts

Estimated duration for 4 models:
- Setup and file handling: 20-30 min
- Acquisition: 90-130 min (hardware-dependent)
- Quick analysis: 20-30 min
Total: about 3-3.5 hours

## 5) Measurement Quality Criteria

Consider a measurement valid if:
1. Idle and active regions are clearly separable
2. The 5 runs produce coherent average energy (no variance explosion)
3. No abnormal supply jumps between runs
4. Energy trend grows with layer size (reasonable scaling)

Red flags:
- Flat trace without visible bursts
- Huge isolated non-repeatable spikes
- Larger model consistently appearing much cheaper than a smaller one without architectural reason

## 6) Tradeoff Strategy for Tomorrow

With today's data you can already define a first energy-aware cost:

`NAS_Cost = Accuracy_Loss + lambda * Layer_Energy`

where `Layer_Energy` comes from `energy_avg_J` in your lookup table.

Practical steps:
1. Build table: `layer -> energy_avg_J`
2. During NAS, sum selected layer costs
3. Sweep at least 3 lambda values (low, medium, high)
4. Plot accuracy vs energy Pareto frontier

## 7) If Something Blocks Today: Quick Fallbacks

Fallback A (recommended):
- Measure only Jetson CPU with 2 models: `64x64` and `1024x1024`
- Generate a minimal report and verify full pipeline end-to-end

Fallback B:
- If Edge TPU compilation fails today, do not block campaign
- Continue with CPU/CUDA and add TPU measurements tomorrow

## 8) Outputs You Should Have by Tonight

Final checklist:
1. At least 3 new correctly named CSV files
2. At least 1 plot per measured model
3. One Excel report with per-layer average energy
4. A short personal table with:
   - Layer
   - `energy_avg_J`
   - `power_avg_W`
   - signal quality notes

If you complete this checklist, you are ready to integrate real measured energy cost into your NAS workflow.
