# Headless INA226EVM acquisition through the TI-SCB on Ubuntu

**Initial Windows validation:** 2026-07-21
**Ubuntu migration and validation:** 2026-07-22
**Scope:** Current INA226EVM detachable sensor card used with the TI Sensor Control Board (TI-SCB)

This report interprets "without relying on the CLI" in the question as "without relying on the GUI, using a CLI."

## Executive answer

The TI GUI and Windows USB driver are not required on Ubuntu. The TI-SCB firmware exposes a documented command protocol on its USB CDC ACM serial port, normally `/dev/ttyACM0`:

- `setdevice`, `rreg`, and `wreg` use newline-delimited commands and return JSON on the serial port.
- `collect` is started and stopped on the serial port but sends measurements through the board's USB bulk interface.

TI does not currently publish a standalone INA226EVM command-line executable. The supported interface is the protocol described in section 4.2.4 of the INA226EVM User's Guide (SBOU276). A small host program must send those commands and write the replies to disk.

For this repository, the recommended implementation is the included Ubuntu-compatible serial logger. It writes each sample to CSV immediately and retains no growing sample array, so a 50,000-point run does not have the GUI's memory/plotting failure mode. It can be launched over SSH and kept alive in `tmux`. Use USB bulk mode only when serial request/response polling is too slow.

## 1. Ubuntu hardware and permission check

Ubuntu's in-kernel `cdc_acm` driver supports the TI-SCB serial interface; do not install the Windows-only SBAC253 package. Connect the INA226EVM to the TI-SCB, then connect the TI-SCB over USB and verify VID:PID `1CBE:00AB`:

```bash
lsusb -d 1cbe:00ab
ls -l /dev/serial/by-id/
```

On the Ubuntu host used for this migration, the board appears as:

```text
/dev/ttyACM0
USB VID:PID 1CBE:00AB
/dev/serial/by-id/usb-Texas_Instruments_Generic_Bulk_Device_12345678-if01
```

The `/dev/serial/by-id/` name is preferable in unattended scripts because `/dev/ttyACM0` can change after reconnecting devices. The logger also auto-detects a single `1CBE:00AB` board when `--port` is omitted.

Check access before changing permissions:

```bash
PORT=/dev/ttyACM0
test -r "$PORT" && test -w "$PORT" && echo "serial access OK"
```

On a standard Ubuntu installation, the device is owned by `root:dialout`. If the access check fails, add the SSH user to that group, then completely disconnect and reconnect the SSH session so the new group is applied:

```bash
sudo usermod -aG dialout "$USER"
id -nG
```

Do not run the logger with `sudo`; fix device permissions instead. Only one process should communicate with the board. If the port is busy or commands time out, identify its owner with `fuser -v "$PORT"` and stop that process before starting the logger.

## 2. Recommended recorder

Python 3.10 or newer is required. A small dedicated virtual environment keeps acquisition independent of the PyTorch and TensorFlow environments:

```bash
sudo apt-get install python3-venv tmux
python3 -m venv ~/.venvs/ina226
source ~/.venvs/ina226/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Docs/INA226EVM/requirements.txt
python -m serial.tools.list_ports -v
```

First capture ten samples. `0.01` below is only an example; replace it with the resistance actually fitted to the EVM. SBOU276 instructs the user to fit a 2512 resistor at `R1` or connect an external shunt, so there is no universal value to assume. Using the wrong value scales current and power by the same error.

```bash
python ina226_serial_logger.py \
    --address 0x40 \
    --shunt-ohms 0.01 \
    --interval-ms 100 \
    --samples 10 \
    --output Data/ina226_smoke_test.csv
```

For a long run, omit `--samples`, optionally use `--duration-s`, and stop with `Ctrl+C`:

```bash
python ina226_serial_logger.py \
    --port /dev/ttyACM0 \
    --address 0x40 \
    --shunt-ohms 0.012 \
    --interval-ms 100 \
    --output Data/test/test.csv
```

Replace `/dev/ttyACM0` with the stable `/dev/serial/by-id/...` path for unattended runs. The logger refuses to overwrite an existing file unless `--overwrite` is supplied. Rows are flushed as they are written, and the output includes `Sample` and `EVM1 POWER Results (W)`, which are already consumed by this repository's processing scripts. Address, shunt resistance, configuration register, calibration register, requested interval, raw readings, UTC time, and host elapsed time are retained for reproducibility.

For a capture that must survive an SSH disconnect, start `tmux new -s ina226`, activate the virtual environment, enter the repository, and run the same command. Detach with `Ctrl+B`, then `D`; reconnect later with `tmux attach -t ina226`.

### Live validation

On 2026-07-22, Ubuntu with Linux 6.8 enumerated the connected `1CBE:00AB` TI-SCB as `/dev/ttyACM0`. The logger selected INA226 address `0x40`, read configuration `0x4127`, and captured two rows at a requested 100 ms interval. The CSV contained all expected power columns. Calibration was `0x0000`, so the main current and power columns correctly used the documented shunt-voltage calculation fallback.

The earlier Windows capture on 2026-07-21 read configuration `0x4127`, calibration `0x0AEC`, and captured 10 rows over 0.922 seconds. Together these checks validate serial transport and CSV compatibility on both hosts. They do not certify the connected setup's absolute measurement accuracy; the changed calibration value also demonstrates why every run must record and inspect register `0x05`.

## 3. Manual protocol test

The serial interface can also be exercised interactively with pySerial's terminal. The guide does not prescribe a meaningful baud rate for the USB CDC transport; `115200 8N1` is a conventional host setting for this firmware.

```bash
python -m serial.tools.miniterm /dev/ttyACM0 115200 --eol LF
```

For an INA226 at address `0x40`, enter:

```text
setdevice 0
rreg 1
rreg 2
rreg 3
rreg 4
rreg 5
```

`setdevice` takes only the four least-significant address bits in **decimal**. Thus `0x40` maps to `0`, `0x41` to `1`, and `0x4A` to `10`. Register addresses and values used by `rreg` and `wreg` are hexadecimal.

A read has this documented response shape:

```json
{"acknowledge":"rreg 2"}
{"register":{"address":2,"value":4000}}
{"evm_state":"idle"}
```

The firmware tested for this report echoes the command terminator as a raw control character inside the acknowledgment string, making that particular line invalid JSON. The register and state lines are valid JSON. The included logger tolerates this firmware defect while still checking that the acknowledgment starts with the command it sent. A generic client should not assume that every returned line can be passed directly to a JSON decoder.

Useful registers are:

|  Address | Register      | Interpretation                              |
| -------: | ------------- | ------------------------------------------- |
| `0x00` | Configuration | Averaging, conversion times, operating mode |
| `0x01` | Shunt voltage | Signed 16-bit, 2.5 uV/LSB                   |
| `0x02` | Bus voltage   | Unsigned 16-bit, 1.25 mV/LSB                |
| `0x03` | Power         | Unsigned 16-bit,`25 * Current_LSB` W/LSB  |
| `0x04` | Current       | Signed 16-bit,`Current_LSB` A/LSB         |
| `0x05` | Calibration   | Sets current and power scaling              |

## 4. Scaling and calibration

For raw register values, shunt resistance $R_{shunt}$, and calibration value `CAL`:

$$
V_{shunt} = \operatorname{int16}(R_{01}) \times 2.5\ \mu V
$$

$$
V_{bus} = R_{02} \times 1.25\ mV
$$

$$
Current\_LSB = \frac{0.00512}{CAL \times R_{shunt}}
$$

$$
I = \operatorname{int16}(R_{04}) \times Current\_LSB
$$

$$
P = R_{03} \times 25 \times Current\_LSB
$$

After reset, `CAL` can be zero, in which case the INA226 current and power registers are not usable. The included logger still computes:

$$
I_{calculated} = \frac{V_{shunt}}{R_{shunt}}, \qquad P_{calculated} = V_{bus} I_{calculated}
$$

If `CAL` is nonzero, the main current and power columns use registers `0x04` and `0x03`; the independently calculated values remain in separate diagnostic columns. Read back register `0x05` and record the physical shunt value with every experiment. Do not copy a calibration value from another setup unless its shunt resistance and expected current range are identical.

## 5. Higher-rate USB bulk mode

Serial polling performs several request/response transactions per sample. TI's bulk mode is preferable when that overhead prevents the required rate.

Start collection by sending this ASCII command over the serial port:

```text
collect timerPeriod collectFlags channelAddressIDs numDevices
```

- `timerPeriod` is the MCU timer period in microseconds, as an unsigned 32-bit decimal value.
- `collectFlags` is a bit mask: shunt voltage `8`, bus voltage `4`, current `2`, power `1`; use `15` for all four.
- `channelAddressIDs` concatenates the four low address bits for up to four EVMs, beginning at the least-significant nibble.
- `numDevices` is from `1` through `4`.

Example from TI for two boards at `0x41` and `0x43`, collecting shunt and bus voltage every 2.2 ms:

```text
collect 2200 12 49 2
```

The serial port acknowledges with JSON and enters the `collecting` state. The binary bulk interface then emits one record per selected register:

```text
frameID | deviceNumID | registerAddress | registerSize | data MSB | data LSB
 1 byte |    1 byte   |      1 byte     |    1 byte    | registerSize bytes
```

`frameID` is currently always zero. For the INA226, `registerSize` is two and the data is most-significant byte first. Stop cleanly by sending `stop` on the serial port and wait for `{"evm_state":"idle"}`.

Bulk mode requires a host program that opens interface 0 (`Generic Bulk Device` through `libusb`/PyUSB on Ubuntu), discovers its bulk-IN endpoint, reassembles records across USB transfer boundaries, groups registers into samples, and streams rows to disk. Endpoint addresses should be discovered from USB descriptors rather than hard-coded. A udev rule may be needed to grant non-root access to that interface. The serial logger deliberately uses only the simpler, fully self-describing serial path.

## 6. Sampling limitations

- The requested host interval is not necessarily the sensor update interval. INA226 conversion time and averaging are configured in register `0x00`; in shunt-and-bus continuous mode, one new result takes approximately `(shunt conversion time + bus conversion time) * averaging count`.
- Polling faster than that produces duplicate conversions. Polling slower loses intermediate conversions.
- Ubuntu userspace scheduling and USB do not provide hard real-time timestamps. Use `Elapsed Time (s)` for observed host timing; use the configured MCU period as the nominal spacing in bulk mode.
- Four serial register reads may take longer than a short `--interval-ms`. The logger never queues an expanding backlog; it starts the next sample immediately when behind schedule.
- For transient workloads, bulk mode gives better alignment because the SCB firmware reads the selected result registers as a sample set.

## 7. Alternative: detach the EVM and use I2C

The INA226 card can be detached from the TI-SCB and connected to a separate Linux host's I2C controller. This is often the cleanest long-term solution because standard Linux `i2c-dev` libraries can configure and read the INA226 directly. Do not connect the TI-SCB and another I2C controller simultaneously. Verify logic voltage, pull-ups, common ground, address straps, shunt polarity, the 36 V common-mode limit, current rating, and shunt dissipation before powering the load.

Using the Raspberry Pi or Jetson being measured as the acquisition host changes its own workload and therefore its measured power. Prefer a separate controller or PC when measurement intrusion matters.

## 8. Recommendation

1. Use the included serial logger at the repository's current 10 Hz processing rate and verify its CSV against a short GUI export under a steady load.
2. Confirm the physical shunt resistance and calibration register before comparing absolute watts.
3. Move to the documented bulk stream only if measured serial throughput or sample alignment is insufficient.
4. Keep acquisition and workload control in separate processes, use `tmux` or a service for SSH-launched campaigns, and record configuration, shunt value, address, sample interval, kernel/firmware version, and UTC start time alongside every run.

## Primary sources

- Texas Instruments, [INA226EVM User&#39;s Guide (SBOU276)](https://www.ti.com/lit/sbou276), especially sections 4.2.4.1 and 4.2.4.2.
- Texas Instruments, [INA226 data sheet (SBOS547B)](https://www.ti.com/lit/gpn/ina226), register map and calibration equations.
- Texas Instruments, [TI Sensor Control Board User&#39;s Guide (SLAU839)](https://www.ti.com/lit/slau839).
- Texas Instruments, [PAMB Windows USB Drivers (SBAC253)](https://www.ti.com/tool/download/SBAC253), version 2.4.0.0B released 2025-10-06 at the time of writing. This package is Windows-only and is not needed on Ubuntu.
- Texas Instruments, [INA226EVM product page](https://www.ti.com/tool/INA226EVM).

TI also links `SBOC409` and `SBOC410`, dated 2011, for the older INA226EVM Rev A software and LabVIEW source. Those packages target the legacy SM-USB-DIG design and are not the correct basis for the current TI-SCB protocol described by SBOU276.
