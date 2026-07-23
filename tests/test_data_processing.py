import tempfile
import unittest
from pathlib import Path

from processing_report.artifact_paths import get_manifest_file_path


class ManifestPairingTests(unittest.TestCase):
    def test_colocated_csv_and_json_are_paired_by_exact_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            csv_path = directory / "campaign-123.csv"
            manifest_path = directory / "campaign-123.json"
            csv_path.write_text("Sample\n", encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")

            self.assertEqual(
                get_manifest_file_path(directory, csv_path),
                manifest_path,
            )

    def test_exact_manifest_can_be_in_nested_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            nested = directory / "manifests" / "cpu"
            nested.mkdir(parents=True)
            manifest_path = nested / "campaign-123.json"
            manifest_path.write_text("{}", encoding="utf-8")

            self.assertEqual(
                get_manifest_file_path(directory, "campaign-123.csv"),
                manifest_path,
            )

    def test_missing_manifest_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(
                get_manifest_file_path(temp_dir, "campaign-missing.csv")
            )

    def test_ambiguous_legacy_matches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "campaign-123_first.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (directory / "campaign-123_second.json").write_text(
                "{}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Ambiguous legacy"):
                get_manifest_file_path(directory, "campaign-123.csv")


if __name__ == "__main__":
    unittest.main()