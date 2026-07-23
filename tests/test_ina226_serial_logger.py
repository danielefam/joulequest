import unittest
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


if __name__ == "__main__":
    unittest.main()