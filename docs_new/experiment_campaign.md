# One-board Experiment Campaign

**Document date:** 2026-07-24  
**Launcher:** `run_measurement_campaign.sh`

## 1. Separation of responsibilities

The measurement tools retain two distinct responsibilities:

- `automated_measurement.py` runs exactly one model experiment and produces one
  same-stem CSV/JSON pair;
- `run_measurement_campaign.sh` selects model specifications and invokes
  `automated_measurement.py` once for every required experiment.

The batch launcher operates on exactly one physical board per invocation. It
does not switch SSH targets or electrical settings during a campaign. After a
campaign finishes, power down the setup, change the board, update the ignored
connection configuration and board-specific current range, then invoke the
launcher again with a different board label.

## 2. Default experiment matrix

### Linear layers

Input and output sizes both use:

```text
64 128 256 512 1024 2048 4096 8192
```

The Cartesian product contains:

$$
8 \times 8 = 64\text{ Linear experiments}
$$

Each generated model path follows:

```text
Models/CUDA/Linear/Linear_<input_size>_<output_size>.pt
```

Example:

```text
Models/CUDA/Linear/Linear_128_2048.pt
```

### Convolutional layers

The protocol table displays:

- **rows:** input-channel count;
- **columns:** square image size.

The naming convention does not follow visual row/column position. It follows
the runner parser exactly:

```text
Conv_<input_channels>_<image_size>_<kernel_size>_<padding>.pt
```

Input-channel values (table rows):

```text
1 2 4 8 16 32 64 128 256 512
```

Image-size values (table columns):

```text
32 64 128 256 512 1024
```

Kernel/padding pairs:

```text
3:0 3:1 5:0 5:1
```

The Cartesian product contains:

$$
10 \times 6 \times 4 = 240\text{ Conv experiments}
$$

For example, the table cell at input-channel row `2`, image-size column `64`,
with kernel `5` and padding `1`, becomes:

```text
Models/CUDA/Conv/Conv_2_64_5_1.pt
```

The complete default suite therefore contains:

$$
64 + 240 = 304\text{ experiments per board}
$$

## 3. Basic commands

Run the complete matrix:

```bash
./run_measurement_campaign.sh --board BOARD_LABEL --suite all
```

Run only Linear experiments:

```bash
./run_measurement_campaign.sh --board BOARD_LABEL --suite linear
```

Run only Conv experiments:

```bash
./run_measurement_campaign.sh --board BOARD_LABEL --suite conv
```

List the commands without starting acquisition or inference:

```bash
./run_measurement_campaign.sh --board BOARD_LABEL --suite all --dry-run
```

`BOARD_LABEL` is a local, filesystem-safe label. It determines the result
directory but does not select the SSH host. The SSH destination and remote paths
come from the ignored `measurement_hosts.local.json` file.

## 4. Configurable parameters

The configuration block at the beginning of the script contains all normal
campaign parameters:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CONNECTION_CONFIG` | `measurement_hosts.local.json` | Ignored SSH/remote configuration |
| `OUTPUT_ROOT` | `measurements/runs` | Parent directory for board campaigns |
| `BACKEND` | `cuda` | `cpu`, `cuda`, or `tpu` |
| `MODEL_ROOT` | `Models/CUDA` | Model/specification root on the inference host |
| `LINEAR_MODEL_DIRECTORY` | `${MODEL_ROOT}/Linear` | Linear model directory |
| `CONV_MODEL_DIRECTORY` | `${MODEL_ROOT}/Conv` | Conv model directory |
| `MODEL_SUFFIX` | `.pt` | Filename suffix |
| `NUMBER_OF_CYCLES` | `10` | Measured cycles per experiment |
| `SLEEP_TIME` | `0.7` s | Idle time between measured cycles |
| `TARGET_BURST_SECONDS` | `0` s | Time requirement in addition to sample requirement |
| `SAMPLING_RATE_HZ` | `100` Hz | Requested INA226 sampling rate |
| `MIN_ACTIVE_SAMPLES` | `50` | Minimum active samples per cycle |
| `WARMUP_INFERENCES` | `20` | Excluded warm-up count |
| `WARMUP_SECONDS` | `0` s | Optional excluded warm-up duration |
| `WARMUP_COOLDOWN_SECONDS` | empty | Use `SLEEP_TIME`, or set an explicit cooldown |
| `CALIBRATION_INITIAL_INFERENCES` | `10` | Calibration sizing pilot count |
| `CALIBRATION_TARGET_SECONDS` | `0.5` s | Target full calibration-batch duration |
| `CALIBRATION_REPETITIONS` | `4` | Full calibration batches including one discard |
| `MAX_RELATIVE_MAD` | `0.15` | Maximum accepted relative MAD and CV |
| `MAX_CALIBRATION_INFERENCES` | `1000000` | Calibration batch safety cap |
| `LEADING_IDLE_SECONDS` | `5` s | Baseline before the first measured cycle |
| `TRAILING_IDLE_SECONDS` | `5` s | Baseline after the final measured cycle |
| `SAFETY_MARGIN_SECONDS` | `2` s | Additional acquisition-only idle |
| `SHUNT_OHMS` | `0.012` ohm | Installed shunt resistance |
| `MAX_EXPECTED_CURRENT_A` | `5.0` A | Board/workload current range |
| `INA226_PORT` | empty | Auto-detect one TI-SCB; set to force a port |

The matrix variables are:

```bash
LINEAR_SIZES="64 128 256 512 1024 2048 4096 8192"
CONV_INPUT_CHANNELS="1 2 4 8 16 32 64 128 256 512"
CONV_IMAGE_SIZES="32 64 128 256 512 1024"
CONV_KERNEL_PADDING="3:0 3:1 5:0 5:1"
```

Values can be modified in the script or overridden for one invocation. For
example, a CPU board with a lower current range can use:

```bash
BACKEND=cpu MAX_EXPECTED_CURRENT_A=3.0 \
  ./run_measurement_campaign.sh --board BOARD_LABEL --suite all
```

A reduced validation matrix can use:

```bash
LINEAR_SIZES="64 128" \
CONV_INPUT_CHANNELS="1 2" \
CONV_IMAGE_SIZES="32 64" \
CONV_KERNEL_PADDING="3:0" \
  ./run_measurement_campaign.sh --board TEST_LABEL --suite all --dry-run
```

## 5. Result organization

Each board receives a separate directory:

```text
measurements/runs/<board_label>/
├── <campaign_id>.csv
├── <campaign_id>.json
├── campaign_summary.tsv
└── logs/
    └── <model_name>.log
```

Every model experiment retains its campaign-ID CSV/JSON association. The TSV
summary records UTC completion time, model path, status, exit code, and log path.
After each JSON manifest is stored locally, its transfer copy is removed from
the inference board to prevent a long campaign from accumulating board-side
files. A failed local write or cleanup connection leaves that remote copy in
place for recovery. Run `automated_measurement.py` directly with
`--keep-remote-manifest` only when board-side retention is required.

## 6. Failure and resume policy

By default, the launcher stops after the first nonzero
`automated_measurement.py` exit code. This prevents a disconnected sensor or
incorrect board setup from invalidating hundreds of later experiments.

To continue after failures:

```bash
./run_measurement_campaign.sh \
  --board BOARD_LABEL \
  --suite all \
  --continue-on-error
```

The launcher still exits nonzero after the matrix if one or more experiments
failed, allowing a calling terminal or scheduler to detect an incomplete
campaign.

When restarted with the same board label, the launcher scans existing JSON
manifests and skips a model when it finds both:

```text
status == COMPLETE
model_path == generated model path
```

This allows a long campaign to resume without repeating successful models.
Use `--repeat-completed` to intentionally run every model again.

## 7. One board at a time

After the final scheduled experiment, the script exits and prints a message
indicating that the board can be changed. Before starting the next board:

1. stop and power down the current setup safely;
2. connect the next board through the INA226 measurement path;
3. update `measurement_hosts.local.json` if the inference destination or remote
   environment changed;
4. set the correct `BACKEND` and `MAX_EXPECTED_CURRENT_A`;
5. choose a new `BOARD_LABEL`;
6. run a reduced dry-run or smoke campaign before starting all 304 experiments.

Example:

```bash
BACKEND=cpu MAX_EXPECTED_CURRENT_A=3.0 \
  ./run_measurement_campaign.sh --board NEXT_BOARD --suite all
```

The script never changes board settings automatically and never mixes multiple
boards in one invocation.

## 8. TPU naming customization

The default paths follow the PyTorch naming convention. For an existing set of
compiled linear Edge TPU models, override the backend, directory, and suffix:

```bash
BACKEND=tpu \
LINEAR_MODEL_DIRECTORY=Compiled \
MODEL_SUFFIX=_edgetpu.tflite \
  ./run_measurement_campaign.sh --board CORAL_LABEL --suite linear
```

This generates names such as:

```text
Compiled/Linear_64_64_edgetpu.tflite
```

Conv TPU campaigns should be enabled only when matching compiled models exist;
set `CONV_MODEL_DIRECTORY` and `MODEL_SUFFIX` to their actual naming layout.

## 9. Preflight checklist

Before a real full campaign:

```bash
bash -n run_measurement_campaign.sh
./run_measurement_campaign.sh --board TEST_LABEL --suite all --dry-run
```

Also verify:

- `measurement_hosts.local.json` points to the intended single board;
- SSH works non-interactively;
- the local TI-SCB serial device is visible and unused by another process;
- shunt resistance and maximum current are correct for the board;
- the remote Python environment supports the selected backend;
- sufficient storage is available for hundreds of CSV/JSON pairs;
- the largest Conv specifications are feasible on the selected board.

Large Conv layers may exceed available memory. The default stop-on-error policy
will preserve all completed experiments and stop at the first failed model so
the matrix or board settings can be adjusted before resuming.