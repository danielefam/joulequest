import tempfile
import unittest
from pathlib import Path

import pandas as pd

from energy_estimator.build_energy_lookup_table import build_lookup_table


class EnergyLookupTableTests(unittest.TestCase):
    def test_aggregates_completed_linear_and_convolution_measurements(self):
        rows = [
            {
                "model_path": "Models/CPU/Linear/Linear_64_64.pt",
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": 1.0,
            },
            {
                "model_path": "Models/CPU/Linear/Linear_64_64.pt",
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": 3.0,
            },
            {
                "model_path": "Models/CPU/Linear/Linear_64_64.pt",
                "status": "COMPLETE",
                "quality_status": "REVIEW",
                "energy_mean_mJ": 5.0,
            },
            {
                "model_path": "Models/CPU/Conv/Conv_3_32_3_1_16.pt",
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": 4.0,
            },
            {
                "model_path": "Models/CUDA/ResNet/ResNetConv_3_64_224_7_2_3.pt",
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": 5.0,
            },
            {
                "model_path": "Models/CPU/Linear/Linear_64_128.pt",
                "status": "FAILED",
                "quality_status": "OK",
                "energy_mean_mJ": 99.0,
            },
            {
                "model_path": "Models/CPU/ReLU/ReLU_1_64.pt",
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": 99.0,
            },
            {
                "model_path": "Models/CUDA/Attention/Attention_128_256_4.pt",
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": 6.0,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            lookup = build_lookup_table([summary_path])

        self.assertEqual(len(lookup), 4)
        linear = lookup[lookup["layer_type"] == "linear"].iloc[0]
        self.assertEqual(linear["measurement_count"], 3)
        self.assertEqual(linear["input_features"], 64)
        self.assertEqual(linear["output_features"], 64)
        self.assertEqual(linear["energy_mean_mJ"], 3.0)
        self.assertAlmostEqual(linear["energy_stddev_mJ"], 2.0)

        conv = lookup[(lookup["input_channels"] == 3) & (lookup["output_channels"] == 16)].iloc[0]
        self.assertEqual(conv["input_image_size"], 32)
        self.assertEqual(conv["kernel_size"], 3)
        self.assertEqual(conv["stride"], 1)
        self.assertEqual(conv["padding"], 1)
        self.assertEqual(conv["energy_mean_mJ"], 4.0)

        resnet_conv = lookup[lookup["input_image_size"] == 224].iloc[0]
        self.assertEqual(resnet_conv["output_channels"], 64)
        self.assertEqual(resnet_conv["stride"], 2)
        self.assertEqual(resnet_conv["padding"], 3)

        attention = lookup[lookup["layer_type"] == "attention"].iloc[0]
        self.assertEqual(attention["sequence_length"], 128)
        self.assertEqual(attention["embed_dim"], 256)
        self.assertEqual(attention["num_heads"], 4)
        self.assertEqual(attention["head_dim"], 64)