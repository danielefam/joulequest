# Complete Terminal Acquisition Report

**Scope:** INA226EVM measurements acquired from a Linux terminal in this
repository.

**Source files:** `ina226_serial_logger.py`, `automated_measurement.py`,
`run_manager.py`, and `run_measurement_campaign.sh`.

**Important:** This document is intentionally local working documentation. The
repository ignores `docs_new/*` except for a small allowlist of published
documents. Therefore this file is excluded from Git status, commits, and Git
history by the existing `.gitignore`; no ignore rule was added for it.

## 1. Acquisition architectures

There are two terminal workflows:

1. **Direct acquisition:** the terminal starts `ina226_serial_logger.py`.
   The logger configures the INA226, polls its registers, calculates values,
   and writes a CSV until a sample limit, duration, `Ctrl+C`, or SIGTERM.
2. **Automated acquisition:** the terminal starts
   `automated_measurement.py` or `run_measurement_campaign.sh`. The local
   computer still owns the INA226 logger and CSV. The inference program runs
   locally or on a remote board over SSH. The workload controller decides when
   to start and stop the measured bursts.

In both workflows the INA226 is connected to the TI Sensor Control Board
(TI-SCB), which appears on Linux as a USB serial device, normally
`/dev/ttyACM0`. The logger uses the TI-SCB's newline-delimited command and JSON
response protocol.

## 2. Physical and terminal prerequisites

The acquisition computer must have:

- the TI-SCB connected by USB;
- the INA226EVM connected to the TI-SCB;
- the correct shunt resistance installed or externally connected;
- Python 3 and the `pyserial` dependency;
- read and write permission for the serial device;
- no other program using the serial device.

Install the Ubuntu-side dependency in an isolated environment:

```bash
cd /path/to/energyBANERA-main
python3 -m venv ~/.venvs/ina226
source ~/.venvs/ina226/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Docs/INA226EVM/requirements.txt
```

The requirements file currently pins `pyserial==3.5`.

Find the board and inspect its USB identity:

```bash
lsusb -d 1cbe:00ab
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
```

The TI-SCB identifier used by the logger is VID `0x1CBE`, PID `0x00AB`. A
stable device path under `/dev/serial/by-id/` is preferable for unattended
runs. An explicit path is always accepted with `--port`; if it is omitted, the
logger accepts exactly one matching TI-SCB port by USB identifier or device
description.

Check access and ownership:

```bash
PORT=/dev/ttyACM0
test -r "$PORT" && test -w "$PORT" && echo "serial access OK"
ls -l "$PORT"
fuser -v "$PORT"
```

If access is denied, add the user to `dialout`, reconnect the terminal session,
and reconnect the board:

```bash
sudo usermod -aG dialout "$USER"
```

Do not use `sudo` to run the logger. Fix the device permission instead. If
`fuser` reports another owner, stop that process before acquisition.

## 3. Direct terminal acquisition

### 3.1 Inspect the command-line interface

From the repository root:

```bash
python ina226_serial_logger.py --help
```

The required measurement parameters are:

| Option | Meaning |
| --- | --- |
| `--output PATH` | CSV destination; parent directories are created |
| `--port PATH` | Serial device; omit for one-board auto-detection |
| `--baud 115200` | Host serial setting used by default |
| `--address 0x40` | INA226 I2C address, accepted as decimal or `0x` integer |
| `--shunt-ohms R` | Installed shunt resistance |
| `--max-expected-current-a I` | Current range used to calculate calibration |
| `--interval-ms T` | Requested host polling period |
| `--samples N` | Stop after `N` rows; `0` means no row limit |
| `--duration-s S` | Stop after `S` seconds; `0` disables this limit |
| `--timeout-s S` | Per-command serial timeout |
| `--overwrite` | Allow replacement of an existing CSV |

The logger refuses to overwrite an existing file unless `--overwrite` is
present. This protects completed measurements from accidental reuse of a
filename.

### 3.2 Run a smoke capture

Use the actual fitted shunt value. The following is only an example for a
`0.012 ohm` shunt and a 5 A expected range:

```bash
mkdir -p measurements/manual
python ina226_serial_logger.py \
  --port /dev/ttyACM0 \
  --address 0x40 \
  --shunt-ohms 0.012 \
  --max-expected-current-a 5.0 \
  --interval-ms 100 \
  --samples 10 \
  --output measurements/manual/ina226_smoke.csv
```

The logger performs the following steps before the first row:

1. Validate positive resistance, positive expected current, positive interval,
   valid limits, and a calibration value representable by 16 bits.
2. Detect or use the requested serial port.
3. Open the port as 8 data bits, no parity, 1 stop bit, with a 115200 baud
   default and a 100 ms serial read timeout.
4. Wait briefly for the TI-SCB and clear stale input bytes.
5. Select the INA226 with `setdevice`.
6. Read configuration register `0x00`.
7. Calculate a calibration value from the requested current and shunt.
8. Write that value to calibration register `0x05`.
9. Read register `0x05` back and abort if it differs from the written value.
10. Open the output CSV using exclusive creation by default and write its
    header.

For `Rshunt = 0.012 ohm` and `Imax = 5 A`, the requested calibration is
`0x0AEC`. The effective Current LSB is recalculated from the value read back
from the device, not assumed from the requested floating-point input.

### 3.3 Run an unattended capture

Use the stable `/dev/serial/by-id/` path and a time limit when the terminal is
not continuously watched:

```bash
python ina226_serial_logger.py \
  --port /dev/serial/by-id/usb-Texas_Instruments_Generic_Bulk_Device_SERIAL-if01 \
  --address 0x40 \
  --shunt-ohms 0.012 \
  --max-expected-current-a 5.0 \
  --interval-ms 10 \
  --duration-s 60 \
  --output measurements/manual/ina226_60s.csv
```

For an SSH session that may disconnect, run the command inside `tmux`:

```bash
tmux new -s ina226
source ~/.venvs/ina226/bin/activate
cd /path/to/energyBANERA-main
# paste the acquisition command here
```

Detach with `Ctrl+B`, then `D`; reattach with:

```bash
tmux attach -t ina226
```

`Ctrl+C` stops the direct logger with exit code `130`. Rows already flushed to
the CSV remain available. A normal finite run exits `0`; argument, port,
protocol, or serial errors print an error and exit `1`.

## 4. Exact serial transaction sequence

Every command sent by `ScbSerial.command()` is ASCII text followed by `\n`.
The logger waits up to `--timeout-s` for a matching acknowledgment and the
expected register result.

For address `0x40`, selection is sent as:

```text
setdevice 0
```

The address is converted to its low nibble: `0x40 -> 0`, `0x41 -> 1`, and so
on. Register addresses and values use hexadecimal text:

```text
rreg 0
wreg 5 aec
rreg 5
rreg 1
rreg 2
rreg 3
rreg 4
```

For each read, the firmware returns acknowledgment, register data, and state
information. A typical valid portion is:

```json
{"register":{"address":2,"value":4000}}
{"evm_state":"idle"}
```

The logger ignores unrelated text and malformed JSON lines, but requires a
matching acknowledgment and a register value for `rreg`. This accommodates
TI-SCB firmware that may echo a command terminator in a way that is not valid
JSON.

At acquisition time, one CSV sample performs four register transactions in
this order:

1. `rreg 1`: shunt voltage;
2. `rreg 2`: bus voltage;
3. `rreg 3`: power;
4. `rreg 4`: current.

The host then timestamps the sample, computes engineering units, writes one
row, flushes the file, increments the sample number, and sleeps until the next
requested deadline. If the deadline was missed, it resets the next deadline
instead of building an unbounded backlog and increments `deadline_misses`.

## 5. Register conversion and calibration

The logger uses these INA226 conversions:

$$
V_{shunt} = int16(R_{01}) \times 2.5\ \mu V
$$

$$
V_{bus} = R_{02} \times 1.25\ mV
$$

$$
Current\_LSB = \frac{0.00512}{CAL \times R_{shunt}}
$$

$$
I_{register} = int16(R_{04}) \times Current\_LSB
$$

$$
P_{register} = R_{03} \times 25 \times Current\_LSB
$$

The requested current scale is selected as:

$$
Requested\ Current\_LSB = \frac{I_{max}}{32768}
$$

The calibration register is the floored value of
`0.00512 / (Requested Current LSB * Rshunt)`. It must be between `1` and
`0xFFFF`. The logger also checks the shunt-voltage full-scale limit of
`0.08192 V`; the requested current must not exceed
`0.08192 / Rshunt`.

Two independent diagnostic values are also recorded:

$$
I_{calculated} = \frac{V_{shunt}}{R_{shunt}}, \qquad
P_{calculated} = V_{bus} I_{calculated}
$$

The register-derived and independently calculated columns are intentionally
both retained so calibration or shunt configuration errors can be detected.

## 6. CSV output and terminal messages

The output has one header followed by one row per completed sample. Columns
include:

- sample number, UTC timestamp, and host monotonic elapsed time;
- I2C address, shunt resistance, expected current, and effective Current LSB;
- raw configuration and requested polling interval;
- register-derived shunt voltage, bus voltage, current, and power;
- independently calculated current and power;
- raw shunt, bus, current, power, and calibration registers.

The logger prints startup details to stderr, including the selected port,
configuration, decoded conversion cycle, calibration, current scale, and CSV
path. It warns about power-down or triggered modes, stale single-channel
values, an interval shorter than the INA226 conversion cycle, and missed
deadlines. At completion it prints the sample count, achieved rate, and
deadline-miss count.

The polling interval is a host request. It does not change the INA226
conversion time or averaging bits in configuration register `0x00`. With the
repository's commonly tested configuration `0x4127`, the sensor conversion
cycle is approximately 2.2 ms. A practical campaign default is
`--interval-ms 10`, or 100 requested samples per second. Always inspect the
achieved rate and `deadline-misses` on the actual host.

## 7. Automated single-experiment acquisition

For a model experiment controlled from one terminal, use
`automated_measurement.py`. It starts the local logger as a child process and
passes `--json-events` so the controller can identify acquisition boundaries.
The following is a reduced example; the model and backend must exist in the
selected environment:

```bash
python automated_measurement.py \
  --backend cpu \
  --model Models/CPU/Linear/Linear_64_64.pt \
  --output-directory measurements/runs/test_board \
  --shunt-ohms 0.012 \
  --max-expected-current-a 5.0 \
  --number_of_cycles 1 \
  --sleep_time 0 \
  --target_burst_seconds 0.5 \
  --sampling_rate_hz 100 \
  --min_active_samples 50 \
  --warmup_inferences 5 \
  --warmup_seconds 0 \
  --calibration_initial_inferences 2 \
  --calibration_target_seconds 0.1 \
  --calibration_repetitions 2 \
  --calibration-sizing-max-attempts 1 \
  --validation-repetitions 1 \
  --validation-max-rounds 3 \
  --leading_idle_seconds 0 \
  --trailing_idle_seconds 0 \
  --safety_margin_seconds 0
```

The automated controller derives the logger interval as
`1000 / sampling_rate_hz`. It does not expose the direct logger's sample or
duration limit because the workload controller owns the measurement boundary.

### 7.1 Local, no-SSH sequence

The local automated path is:

1. Parse and validate model, backend, serial, shunt, current, sampling, and
   timing arguments.
2. Build a `RunManager` and an INA226 process controller.
3. Run warm-up inferences to remove first-use backend, allocator, kernel, and
   cache effects from the measured workload.
4. Run calibration pilots and repeated batches to estimate steady-state
   inference latency.
5. Calculate the required burst duration and inference count from the target
   duration and minimum active sample count.
6. Start the logger child process. The logger initializes the INA226 and emits
   `ACQUISITION_READY` only after its first CSV row has been flushed.
7. Run leading idle time, measured inference bursts, inter-cycle sleeps, and
   trailing idle time according to the capture plan.
8. Stop the logger. It flushes and closes the CSV and emits
   `ACQUISITION_COMPLETE`.
9. Write the JSON manifest beside the CSV and print an
   `AUTOMATED_MEASUREMENT_RESULT` summary.

The first sample is deliberately required before the workload proceeds. This
prevents an inference burst from starting while serial initialization is still
in progress.

### 7.2 Remote inference sequence over SSH

When `runner_host` is configured, the acquisition computer remains the
controller host and the inference board runs `run_manager.py` through a
non-interactive SSH command. The local process constructs a command equivalent
to:

```bash
ssh -T -o BatchMode=yes -o ProxyJump=USER@JUMP_HOST BOARD_USER@BOARD_HOST \
  'cd /remote/energyBANERA && exec python run_manager.py ... --stdio_acquisition'
```

The exact command is generated by the program and includes the model,
backend, burst planning, clock, idle, manifest, and validation arguments.
The JSON-over-stdio handshake is:

1. `run_manager.py` emits `CLOCK_SYNC_REQUEST`; the local controller returns
   `CLOCK_SYNC_RESPONSE` with four monotonic timestamps.
2. The remote manager emits `ACQUISITION_START_REQUEST` with the campaign and
   capture plan.
3. The local controller starts `ina226_serial_logger.py` and waits for
   `ACQUISITION_READY`.
4. The local controller returns `ACQUISITION_STARTED`, including the CSV path,
   first-sample timing, and sampling configuration.
5. The remote manager runs the planned workload and periodically checks the
   local acquisition health.
6. The remote manager emits additional clock-sync requests before and after
   measured rounds so remote inference events can be aligned with the local
   CSV's monotonic time domain.
7. The remote manager emits `ACQUISITION_STOP_REQUEST` after the final measured
   round and trailing idle period.
8. The local controller terminates the logger, waits for
   `ACQUISITION_COMPLETE`, and returns `ACQUISITION_STOPPED`.
9. The remote manager emits the final `RUN_MANIFEST`.
10. The local controller saves the manifest beside the CSV and normally removes
    the remote recovery copy. Use `--keep-remote-manifest` to retain it.

The logger's stdout is reserved for JSON events in this mode. Human-readable
startup and completion messages continue to use stderr, so they cannot corrupt
the event stream.

## 8. Complete campaign from the terminal

First validate the shell script and inspect the generated matrix without
starting acquisition:

```bash
bash -n run_measurement_campaign.sh
./run_measurement_campaign.sh --board test_board --suite linear --dry-run
```

Start a complete board campaign:

```bash
./run_measurement_campaign.sh --board jetson_nano --suite all
```

The launcher performs these steps:

1. Load defaults and environment overrides, including board label, backend,
   model paths, current range, sampling rate, idle periods, and SSH config.
2. Validate all numeric settings and required files.
3. Compute the selected Linear, Conv, LeNet, and ResNet-18 model list.
4. Create `measurements/runs/<board>/logs/` and the summary TSV.
5. For each model, skip a matching existing `COMPLETE` manifest unless
   `--repeat-completed` was supplied.
6. Wait the configured inter-experiment cooldown after the first experiment.
7. Invoke `automated_measurement.py` once for the model and tee its output to
   a model-specific log.
8. Record completion or failure in `campaign_summary.tsv`.
9. Stop on the first failure by default, or continue when
   `--continue-on-error` is supplied.
10. Leave successful CSV/JSON pairs intact so a later invocation can resume.

Useful variants:

```bash
./run_measurement_campaign.sh --board jetson_nano --suite linear
./run_measurement_campaign.sh --board jetson_nano --suite conv
./run_measurement_campaign.sh --board jetson_nano --suite all --continue-on-error
BACKEND=cpu MAX_EXPECTED_CURRENT_A=3.0 \
  ./run_measurement_campaign.sh --board pi5 --suite all
```

The default script requests 100 INA226 samples per second, uses a `0.012 ohm`
shunt and a 5 A range, and places results under
`measurements/runs/<board_label>/`. The connection file
`measurement_hosts.local.json` is intentionally ignored because it contains
machine-specific settings.

## 9. Acquisition artifacts

For one campaign ID, the expected local pair is:

```text
measurements/runs/<board>/<campaign_id>.csv
measurements/runs/<board>/<campaign_id>.json
```

The campaign directory also contains:

```text
measurements/runs/<board>/campaign_summary.tsv
measurements/runs/<board>/logs/<model_stem>.log
```

The CSV is the sample stream. The JSON manifest records the experiment plan,
measurement cycles, acquisition status, calibration metadata, timing events,
quality status, and paths. The TSV is the batch-level resume and status index.

Inspect the first and last CSV rows from the terminal:

```bash
head -n 2 measurements/runs/<board>/<campaign_id>.csv
tail -n 2 measurements/runs/<board>/<campaign_id>.csv
python - <<'PY'
import csv
from pathlib import Path

path = Path("measurements/runs/<board>/<campaign_id>.csv")
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print("rows:", len(rows))
print("first:", rows[0] if rows else "none")
print("last:", rows[-1] if rows else "none")
PY
```

Replace the angle-bracket placeholders before executing the inspection block.

## 10. Validation and processing

Run the focused unit tests after changing acquisition code:

```bash
python -m unittest tests.test_ina226_serial_logger -v
```

The tests cover port detection, Linux help text, permission errors,
calibration, register command formatting, CSV conversions, readiness events,
cooperative stop, row preservation, and serial close behavior.

Process completed campaign files with:

```bash
MPLBACKEND=Agg python processing_report/process_and_visualize.py
```

Before trusting a campaign, verify:

1. the CSV exists and has data rows;
2. the manifest status is `COMPLETE`;
3. the acquisition status is complete;
4. `sample_count` agrees with the CSV row count;
5. calibration and shunt metadata match the physical setup;
6. achieved sampling rate is acceptable;
7. `deadline_misses` is zero or understood;
8. the model log and campaign TSV show no failed experiment.

## 11. Failure behavior and recovery

The logger fails before writing data when it cannot detect or open the port,
receives no valid TI-SCB response, cannot represent the calibration value, or
reads a different calibration value than the one written. It closes the serial
device in a `finally` block.

During automated acquisition, a missing first sample causes startup failure.
An unexpected logger exit, invalid event, campaign ID mismatch, stop timeout,
or nonzero logger exit marks acquisition as failed. The controller attempts to
terminate the child and preserves already flushed CSV rows.

For a failed campaign:

```bash
tail -n 40 measurements/runs/<board>/logs/<model_stem>.log
tail -n 20 measurements/runs/<board>/campaign_summary.tsv
```

Correct the physical, serial, current-range, model, or SSH problem before
resuming. Re-run with the same board label to skip models with a matching
`COMPLETE` manifest. Use `--repeat-completed` only when repeating successful
models is intentional. `Ctrl+C` returns `130`, preserves completed results,
and does not start the next model.

## 12. Git exclusion check

The existing ignore rules contain:

```text
docs_new/*
!docs_new/adaptive_burst_measurement.md
!docs_new/cli_acquisition_report.md
!docs_new/run_manager_update.md
!docs_new/experiment_campaign.md
```

The new report is not one of the exceptions. Confirm the behavior from the
repository root:

```bash
git check-ignore -v docs_new/terminal_acquisition_full_report.md
git status --short --ignored docs_new/terminal_acquisition_full_report.md
```

The first command should identify the `docs_new/*` rule. The second should show
the file as ignored, not as an untracked file. Because ignored files are not
added to commits, this report will not enter Git history unless someone
explicitly forces it with `git add -f`.
