import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ina226_serial_logger as logger


class PortDetectionTests(unittest.TestCase):
    @patch.object(logger.list_ports, "comports")
    def test_detects_ubuntu_acm_port_by_usb_identifier(self, comports):
        comports.return_value = [
            SimpleNamespace(
                device="/dev/ttyACM0",
                vid=logger.TI_SCB_VID,
                pid=logger.TI_SCB_PID,
                description="Generic Bulk Device",
            )
        ]

        self.assertEqual(logger.detect_port(None), "/dev/ttyACM0")

    @patch.object(logger.list_ports, "comports", return_value=[])
    def test_missing_port_error_gives_ubuntu_discovery_command(self, _comports):
        with self.assertRaisesRegex(logger.ProtocolError, "/dev/ttyACM0"):
            logger.detect_port(None)

    def test_explicit_stable_device_path_is_preserved(self):
        device_path = (
            "/dev/serial/by-id/"
            "usb-Texas_Instruments_Generic_Bulk_Device_12345678-if01"
        )

        self.assertEqual(logger.detect_port(device_path), device_path)


class UbuntuCliTests(unittest.TestCase):
    def test_help_uses_linux_serial_device_example(self):
        help_text = logger.build_parser().format_help()

        self.assertIn("/dev/ttyACM0", help_text)
        self.assertNotIn("COM4", help_text)

    @patch.object(
        logger.serial,
        "Serial",
        side_effect=logger.serial.SerialException("Permission denied"),
    )
    def test_open_failure_mentions_dialout_membership(self, _serial):
        with self.assertRaisesRegex(logger.ProtocolError, "dialout"):
            logger.ScbSerial("/dev/ttyACM0", 115200, 2.0)


class CalibrationTests(unittest.TestCase):
    def test_raspberry_pi_4_calibration_matches_three_amp_range(self):
        self.assertEqual(logger.calculate_calibration(3.0, 0.012), 0x1234)

    def test_five_amp_calibration_matches_previous_gui_value(self):
        self.assertEqual(logger.calculate_calibration(5.0, 0.012), 0x0AEC)

    def test_calibration_outside_register_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "16-bit range"):
            logger.calculate_calibration(0.001, 0.012)

    def test_write_register_uses_documented_hexadecimal_command(self):
        device = object.__new__(logger.ScbSerial)
        device.command = Mock()

        device.write_register(0x05, 0x1234)

        device.command.assert_called_once_with("wreg 5 1234")

    def test_help_requires_maximum_expected_current(self):
        parser = logger.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--output", "capture.csv", "--shunt-ohms", "0.012"])


class FakeScbSerial:
    instances = []

    def __init__(self, port, baud, timeout_seconds):
        self.port = port
        self.closed = False
        self.calibration = None
        FakeScbSerial.instances.append(self)

    def set_device(self, i2c_address):
        self.i2c_address = i2c_address

    def read_register(self, address):
        values = {
            0x00: 0x0007,
            0x01: 4,
            0x02: 8000,
            0x03: 2,
            0x04: 3,
        }
        if address == 0x05:
            return self.calibration
        return values[address]

    def write_register(self, address, value):
        if address == 0x05:
            self.calibration = value

    def close(self):
        self.closed = True


class SamplingLoopTests(unittest.TestCase):
    def setUp(self):
        FakeScbSerial.instances.clear()

    @staticmethod
    def _args(output, **overrides):
        settings = {
            "output": Path(output),
            "port": "/dev/ttyACM0",
            "baud": 115200,
            "address": 0x40,
            "shunt_ohms": 0.012,
            "max_expected_current_a": 5.0,
            "interval_ms": 0.5,
            "samples": 2,
            "duration_s": 0.0,
            "timeout_s": 2.0,
            "overwrite": False,
            "campaign_id": "campaign-123",
            "json_events": True,
        }
        settings.update(overrides)
        return SimpleNamespace(**settings)

    @patch.object(logger, "ScbSerial", FakeScbSerial)
    def test_csv_schema_conversions_and_first_row_readiness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "capture.csv"
            events = []

            def receive_event(payload):
                if payload["event"] == "ACQUISITION_READY":
                    with output.open(encoding="utf-8") as csv_file:
                        self.assertEqual(len(csv_file.readlines()), 2)
                if payload["event"] == "ACQUISITION_COMPLETE":
                    self.assertTrue(FakeScbSerial.instances[-1].closed)
                events.append(payload)

            result = logger.run(
                self._args(output),
                event_sink=receive_event,
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                [event["event"] for event in events],
                ["ACQUISITION_READY", "ACQUISITION_COMPLETE"],
            )
            self.assertEqual(
                events[0]["started_monotonic_seconds"],
                events[1]["started_monotonic_seconds"],
            )
            self.assertGreater(events[0]["started_monotonic_seconds"], 0)
            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
                self.assertEqual(csv_file.seek(0), 0)
                self.assertEqual(next(csv.reader(csv_file)), logger.CSV_FIELDNAMES)
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(
                float(rows[0]["EVM1 SHUNT VOLTAGE Results (V)"]),
                10e-6,
            )
            self.assertAlmostEqual(
                float(rows[0]["EVM1 BUS VOLTAGE Results (V)"]),
                10.0,
            )
            self.assertAlmostEqual(
                float(rows[0]["Calculated Current (A)"]),
                10e-6 / 0.012,
            )
            self.assertAlmostEqual(
                float(rows[0]["Calculated Power (W)"]),
                10.0 * 10e-6 / 0.012,
            )
            current_lsb = logger.CALIBRATION_CONSTANT / (0x0AEC * 0.012)
            self.assertAlmostEqual(
                float(rows[0]["EVM1 CURRENT Results (A)"]),
                3 * current_lsb,
            )
            self.assertAlmostEqual(
                float(rows[0]["EVM1 POWER Results (W)"]),
                2 * 25 * current_lsb,
            )

    @patch.object(logger, "ScbSerial", FakeScbSerial)
    def test_cooperative_stop_preserves_rows_and_reports_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "capture.csv"
            stop_checks = 0
            events = []

            def stop_requested():
                nonlocal stop_checks
                stop_checks += 1
                return stop_checks > 2

            logger.run(
                self._args(output, samples=0),
                stop_requested=stop_requested,
                event_sink=events.append,
            )

            complete = events[-1]
            self.assertEqual(complete["stop_reason"], "requested")
            self.assertEqual(complete["sample_count"], 2)
            self.assertIsNotNone(complete["achieved_sampling_rate_hz"])
            self.assertIn("deadline_misses", complete)
            self.assertTrue(FakeScbSerial.instances[-1].closed)
            with output.open(newline="", encoding="utf-8") as csv_file:
                self.assertEqual(len(list(csv.DictReader(csv_file))), 2)


if __name__ == "__main__":
    unittest.main()