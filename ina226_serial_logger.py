#!/usr/bin/env python3
"""Stream INA226EVM measurements from a TI-SCB serial port to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import serial
from serial.tools import list_ports


TI_SCB_VID = 0x1CBE
TI_SCB_PID = 0x00AB
SHUNT_VOLTAGE_LSB_V = 2.5e-6
BUS_VOLTAGE_LSB_V = 1.25e-3
SHUNT_VOLTAGE_FULL_SCALE_V = 0.08192
CURRENT_REGISTER_STEPS = 1 << 15
CALIBRATION_CONSTANT = 0.00512
MAX_CALIBRATION_VALUE = 0xFFFF
CONVERSION_TIMES_SECONDS = (
    140e-6,
    204e-6,
    332e-6,
    588e-6,
    1.1e-3,
    2.116e-3,
    4.156e-3,
    8.244e-3,
)
AVERAGING_COUNTS = (1, 4, 16, 64, 128, 256, 512, 1024)


class ProtocolError(RuntimeError):
    """Raised when the TI-SCB does not return the documented JSON response."""


def int_auto(value: str) -> int:
    return int(value, 0)


def signed_16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def conversion_cycle_seconds(configuration: int) -> float | None:
    mode = configuration & 0x07
    conversion_seconds = 0.0
    if mode & 0x01:
        conversion_seconds += CONVERSION_TIMES_SECONDS[(configuration >> 3) & 0x07]
    if mode & 0x02:
        conversion_seconds += CONVERSION_TIMES_SECONDS[(configuration >> 6) & 0x07]
    if conversion_seconds == 0:
        return None
    return conversion_seconds * AVERAGING_COUNTS[(configuration >> 9) & 0x07]


def calculate_calibration(max_expected_current_a: float, shunt_ohms: float) -> int:
    requested_current_lsb_a = max_expected_current_a / CURRENT_REGISTER_STEPS
    calibration = math.floor(
        CALIBRATION_CONSTANT / (requested_current_lsb_a * shunt_ohms)
    )
    if not 1 <= calibration <= MAX_CALIBRATION_VALUE:
        raise ValueError(
            "--max-expected-current-a and --shunt-ohms produce a calibration "
            "outside the INA226 16-bit range"
        )
    return calibration


def detect_port(requested_port: str | None) -> str:
    if requested_port:
        return requested_port

    candidates = [
        port.device
        for port in list_ports.comports()
        if (port.vid, port.pid) == (TI_SCB_VID, TI_SCB_PID)
        or "msp432 usb serial port" in (port.description or "").lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ProtocolError(
            "No TI-SCB serial port found. Connect the board, check "
            "'python -m serial.tools.list_ports -v', or pass a port such as "
            "--port /dev/ttyACM0."
        )
    raise ProtocolError(
        f"Multiple TI-SCB serial ports found ({', '.join(candidates)}); pass --port."
    )


class ScbSerial:
    def __init__(self, port: str, baud: int, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=timeout_seconds,
            )
        except serial.SerialException as error:
            raise ProtocolError(
                f"Cannot open {port}: {error}. Ensure no other process owns the "
                "port and check read/write access; standard Ubuntu installations "
                "grant access through the dialout group."
            ) from error
        time.sleep(0.25)
        self._serial.reset_input_buffer()

    def close(self) -> None:
        self._serial.close()

    def command(self, command: str, expect_register: bool = False) -> int | None:
        self._serial.write(f"{command}\n".encode("ascii"))
        self._serial.flush()

        deadline = time.monotonic() + self._timeout_seconds
        acknowledged = False
        register_value: int | None = None
        responses: list[str] = []

        while time.monotonic() < deadline:
            raw_line = self._serial.readline()
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            responses.append(line)
            acknowledgement_prefix = '{"acknowledge":"'
            if line.startswith(acknowledgement_prefix):
                echoed_command = line[len(acknowledgement_prefix) :]
                if echoed_command == command or echoed_command.startswith(f"{command} "):
                    acknowledged = True
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            acknowledgement = payload.get("acknowledge")
            if (
                isinstance(acknowledgement, str)
                and (
                    acknowledgement.strip() == command
                    or acknowledgement.strip().startswith(f"{command} ")
                )
            ):
                acknowledged = True
            register = payload.get("register")
            if isinstance(register, dict) and "value" in register:
                register_value = int(register["value"])
            if "evm_state" in payload and acknowledged:
                if expect_register and register_value is None:
                    break
                return register_value

        detail = " | ".join(responses) if responses else "no response"
        raise ProtocolError(f"TI-SCB command {command!r} failed: {detail}")

    def set_device(self, i2c_address: int) -> None:
        if not 0x40 <= i2c_address <= 0x4F:
            raise ValueError("INA226 I2C address must be between 0x40 and 0x4f")
        self.command(f"setdevice {i2c_address & 0x0F}")

    def read_register(self, address: int) -> int:
        value = self.command(f"rreg {address:x}", expect_register=True)
        if value is None:
            raise ProtocolError(f"No value returned for register 0x{address:02x}")
        return value & 0xFFFF

    def write_register(self, address: int, value: int) -> None:
        if not 0 <= address <= 0xFF:
            raise ValueError("Register address must fit in 8 bits")
        if not 0 <= value <= 0xFFFF:
            raise ValueError("Register value must fit in 16 bits")
        self.command(f"wreg {address:x} {value:x}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream INA226EVM samples through a TI-SCB USB serial port."
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    parser.add_argument(
        "--port",
        help="Serial port, for example /dev/ttyACM0 (auto-detected if omitted)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--address", type=int_auto, default=0x40, help="INA226 I2C address")
    parser.add_argument(
        "--shunt-ohms",
        type=float,
        required=True,
        help="Installed shunt resistance in ohms; verify the EVM resistor marking",
    )
    parser.add_argument(
        "--max-expected-current-a",
        type=float,
        required=True,
        help="Maximum expected current in amperes; sets current/power scaling",
    )
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--samples", type=int, default=0, help="0 records until Ctrl+C")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 disables the limit")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.shunt_ohms <= 0:
        raise ValueError("--shunt-ohms must be positive")
    if args.max_expected_current_a <= 0:
        raise ValueError("--max-expected-current-a must be positive")
    shunt_current_limit_a = SHUNT_VOLTAGE_FULL_SCALE_V / args.shunt_ohms
    if args.max_expected_current_a > shunt_current_limit_a:
        raise ValueError(
            f"--max-expected-current-a exceeds the {shunt_current_limit_a:.6g} A "
            "shunt-voltage limit for the selected resistance"
        )
    calculate_calibration(args.max_expected_current_a, args.shunt_ohms)
    if args.interval_ms <= 0:
        raise ValueError("--interval-ms must be positive")
    if args.samples < 0 or args.duration_s < 0 or args.timeout_s <= 0:
        raise ValueError("sample and duration limits cannot be negative; timeout must be positive")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    port = detect_port(args.port)
    output_mode = "w" if args.overwrite else "x"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Sample",
        "Timestamp UTC",
        "Elapsed Time (s)",
        "I2C Address",
        "Shunt Resistance (ohm)",
        "Maximum Expected Current (A)",
        "Current LSB (A)",
        "Configuration Raw",
        "Requested Interval (ms)",
        "EVM1 SHUNT VOLTAGE Results (V)",
        "EVM1 BUS VOLTAGE Results (V)",
        "EVM1 CURRENT Results (A)",
        "EVM1 POWER Results (W)",
        "Calculated Current (A)",
        "Calculated Power (W)",
        "Shunt Raw",
        "Bus Raw",
        "Current Raw",
        "Power Raw",
        "Calibration Raw",
    ]

    device = ScbSerial(port, args.baud, args.timeout_s)
    try:
        device.set_device(args.address)
        configuration_raw = device.read_register(0x00)
        conversion_cycle_s = conversion_cycle_seconds(configuration_raw)
        requested_calibration = calculate_calibration(
            args.max_expected_current_a, args.shunt_ohms
        )
        device.write_register(0x05, requested_calibration)
        calibration_raw = device.read_register(0x05)
        if calibration_raw != requested_calibration:
            raise ProtocolError(
                "INA226 calibration verification failed: "
                f"wrote 0x{requested_calibration:04x}, "
                f"read 0x{calibration_raw:04x}"
            )
        current_lsb_a = CALIBRATION_CONSTANT / (
            calibration_raw * args.shunt_ohms
        )
        conversion_cycle_text = (
            "inactive"
            if conversion_cycle_s is None
            else f"{conversion_cycle_s * 1000:.6g} ms"
        )
        print(
            f"TI-SCB {port}; INA226 address=0x{args.address:02x}; "
            f"configuration=0x{configuration_raw:04x}; "
            f"conversion-cycle={conversion_cycle_text}; "
            f"max-current={args.max_expected_current_a:.12g} A; "
            f"calibration=0x{calibration_raw:04x}; "
            f"current-lsb={current_lsb_a:.12g} A; output={args.output}",
            file=sys.stderr,
        )
        operating_mode = configuration_raw & 0x07
        if operating_mode in (0, 4):
            print(
                "warning: INA226 configuration is in a power-down mode; "
                "result registers will not update continuously",
                file=sys.stderr,
            )
        elif operating_mode in (1, 2, 3):
            print(
                "warning: INA226 configuration is in a triggered mode; repeated "
                "reads return the same conversion unless the device is retriggered",
                file=sys.stderr,
            )
        else:
            if operating_mode in (5, 6):
                print(
                    "warning: INA226 configuration does not continuously convert "
                    "both shunt and bus voltage; some logged columns will be stale",
                    file=sys.stderr,
                )
            if (
                conversion_cycle_s is not None
                and args.interval_ms < conversion_cycle_s * 1000
            ):
                print(
                    f"warning: requested interval {args.interval_ms:.6g} ms is "
                    f"shorter than the configured INA226 conversion cycle "
                    f"({conversion_cycle_s * 1000:.6g} ms, approximately "
                    f"{1 / conversion_cycle_s:.6g} fresh samples/s)",
                    file=sys.stderr,
                )

        with args.output.open(output_mode, newline="", encoding="utf-8", buffering=1) as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            started = time.monotonic()
            next_deadline = started
            sample = 0
            first_sample_elapsed_s: float | None = None
            last_sample_elapsed_s: float | None = None
            deadline_misses = 0

            while True:
                if args.samples and sample >= args.samples:
                    break
                if args.duration_s and time.monotonic() - started >= args.duration_s:
                    break

                shunt_raw = device.read_register(0x01)
                bus_raw = device.read_register(0x02)
                power_raw = device.read_register(0x03)
                current_raw = device.read_register(0x04)
                timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                elapsed_seconds = time.monotonic() - started
                if first_sample_elapsed_s is None:
                    first_sample_elapsed_s = elapsed_seconds
                last_sample_elapsed_s = elapsed_seconds

                shunt_voltage_v = signed_16(shunt_raw) * SHUNT_VOLTAGE_LSB_V
                bus_voltage_v = bus_raw * BUS_VOLTAGE_LSB_V
                calculated_current_a = shunt_voltage_v / args.shunt_ohms
                calculated_power_w = bus_voltage_v * calculated_current_a
                current_a = signed_16(current_raw) * current_lsb_a
                power_w = power_raw * 25 * current_lsb_a

                writer.writerow(
                    {
                        "Sample": sample,
                        "Timestamp UTC": timestamp,
                        "Elapsed Time (s)": f"{elapsed_seconds:.9f}",
                        "I2C Address": f"0x{args.address:02x}",
                        "Shunt Resistance (ohm)": f"{args.shunt_ohms:.12g}",
                        "Maximum Expected Current (A)": (
                            f"{args.max_expected_current_a:.12g}"
                        ),
                        "Current LSB (A)": f"{current_lsb_a:.12g}",
                        "Configuration Raw": f"0x{configuration_raw:04x}",
                        "Requested Interval (ms)": f"{args.interval_ms:.12g}",
                        "EVM1 SHUNT VOLTAGE Results (V)": f"{shunt_voltage_v:.12g}",
                        "EVM1 BUS VOLTAGE Results (V)": f"{bus_voltage_v:.12g}",
                        "EVM1 CURRENT Results (A)": f"{current_a:.12g}",
                        "EVM1 POWER Results (W)": f"{power_w:.12g}",
                        "Calculated Current (A)": f"{calculated_current_a:.12g}",
                        "Calculated Power (W)": f"{calculated_power_w:.12g}",
                        "Shunt Raw": shunt_raw,
                        "Bus Raw": bus_raw,
                        "Current Raw": current_raw,
                        "Power Raw": power_raw,
                        "Calibration Raw": calibration_raw,
                    }
                )
                output.flush()
                sample += 1

                next_deadline += args.interval_ms / 1000
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline_misses += 1
                    if deadline_misses == 1:
                        print(
                            "warning: the sampling loop passed the next requested "
                            "deadline; the logger reset it instead of building a "
                            "backlog",
                            file=sys.stderr,
                        )
                    next_deadline = time.monotonic()

        rate_text = "n/a"
        if (
            sample > 1
            and first_sample_elapsed_s is not None
            and last_sample_elapsed_s is not None
        ):
            elapsed_span_s = last_sample_elapsed_s - first_sample_elapsed_s
            if elapsed_span_s > 0:
                rate_text = f"{(sample - 1) / elapsed_span_s:.6g} samples/s"
        print(
            f"Captured {sample} samples to {args.output}; achieved-rate={rate_text}; "
            f"deadline-misses={deadline_misses}",
            file=sys.stderr,
        )
        return 0
    finally:
        device.close()


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Capture stopped by user; completed rows remain in the CSV.", file=sys.stderr)
        return 130
    except (OSError, ValueError, ProtocolError, serial.SerialException) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())