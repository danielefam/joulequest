import tempfile
import unittest
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from energy_estimator import EnergyLookup, estimate_model_energy


class EnergyEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lookup_path = Path(self.temp_dir.name) / "lookup.csv"
        rows = []
        for input_features, output_features in product((2, 4), repeat=2):
            rows.append(
                {
                    "layer_type": "linear",
                    "input_features": input_features,
                    "output_features": output_features,
                    "energy_mean_mJ": input_features + 2 * output_features,
                }
            )
        for input_channels, output_channels, image_size in product(
            (1, 3), (2, 4), (2, 4)
        ):
            rows.append(
                {
                    "layer_type": "conv",
                    "input_channels": input_channels,
                    "output_channels": output_channels,
                    "input_image_size": image_size,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "energy_mean_mJ": (
                        input_channels + output_channels + image_size**2 / 100
                    ),
                }
            )
        for sequence_length, embed_dim, head_dim in product((8, 16), repeat=3):
            rows.append(
                {
                    "layer_type": "attention",
                    "sequence_length": sequence_length,
                    "embed_dim": embed_dim,
                    "head_dim": head_dim,
                    "energy_mean_mJ": sequence_length + embed_dim + head_dim,
                }
            )
        pd.DataFrame(rows).to_csv(self.lookup_path, index=False)
        self.lookup = EnergyLookup(self.lookup_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_linear_exact_value_and_mask_gradient(self):
        output_mask = torch.full((4,), 0.75, requires_grad=True)
        energy = self.lookup.linear(2, output_mask.sum())
        self.assertAlmostEqual(float(energy.detach()), 8.0)
        energy.backward()
        self.assertTrue(torch.allclose(output_mask.grad, torch.full((4,), 2.0)))

    def test_conv_uses_trilinear_spatial_area(self):
        input_channels = torch.tensor(2.0, requires_grad=True)
        output_channels = torch.tensor(3.0, requires_grad=True)
        energy = self.lookup.conv2d(
            input_channels,
            output_channels,
            2,
            5,
            kernel_size=3,
            padding=1,
        )
        self.assertAlmostEqual(float(energy.detach()), 5.1)
        energy.backward()
        self.assertAlmostEqual(float(input_channels.grad), 1.0)
        self.assertAlmostEqual(float(output_channels.grad), 1.0)

    def test_attention_uses_three_dimensional_interpolation(self):
        embed_dim = torch.tensor(12.0, requires_grad=True)
        energy = self.lookup.attention(10, embed_dim, 14)
        self.assertAlmostEqual(float(energy.detach()), 36.0)
        energy.backward()
        self.assertAlmostEqual(float(embed_dim.grad), 1.0)

    def test_model_estimator_uses_explicit_input_and_output_masks(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        hidden_mask = torch.full((4,), 0.75, requires_grad=True)
        result = estimate_model_energy(
            model,
            self.lookup,
            masks={
                "0": {"output": hidden_mask},
                "2": {"input": hidden_mask},
            },
        )
        self.assertAlmostEqual(float(result["total_energy_mJ"].detach()), 17.0)
        result["total_energy_mJ"].backward()
        self.assertIsNotNone(hidden_mask.grad)

    def test_missing_interpolation_corner_is_rejected(self):
        data = pd.read_csv(self.lookup_path)
        drop = (
            data["layer_type"].eq("linear")
            & data["input_features"].eq(4)
            & data["output_features"].eq(4)
        )
        data[~drop].to_csv(self.lookup_path, index=False)
        lookup = EnergyLookup(self.lookup_path)
        with self.assertRaisesRegex(ValueError, "missing measurements"):
            lookup.linear(3, 3)


if __name__ == "__main__":
    unittest.main()