"""CSV-backed differentiable energy lookup for measured layer configurations."""

from pathlib import Path

import pandas as pd
import torch

from .interpolation import multilinear_interpolate


class EnergyLookup:
    """Estimate per-inference energy in mJ from a generated lookup CSV."""

    def __init__(self, path, *, out_of_range="error", dtype=torch.float64):
        self.path = Path(path)
        self.out_of_range = out_of_range
        self.dtype = dtype
        self.data = pd.read_csv(self.path)
        required = {"layer_type", "energy_mean_mJ"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"Lookup table is missing columns: {', '.join(sorted(missing))}")
        if self.data["energy_mean_mJ"].isna().any():
            raise ValueError("Lookup table contains missing energy values")

    def _estimate(self, rows, dimensions, coordinates):
        if rows.empty:
            raise ValueError("No compatible measurements are available")
        missing_columns = set(dimensions).difference(rows.columns)
        if missing_columns:
            raise ValueError(
                "Lookup table does not contain dimensions: "
                + ", ".join(sorted(missing_columns))
            )
        duplicate = rows.duplicated(dimensions, keep=False)
        if duplicate.any():
            raise ValueError("Lookup table contains duplicate measurement coordinates")

        axes = [
            torch.tensor(sorted(rows[dimension].unique()), dtype=self.dtype)
            for dimension in dimensions
        ]
        shape = tuple(axis.numel() for axis in axes)
        values = torch.full(shape, torch.nan, dtype=self.dtype)
        axis_indices = [
            {float(value): index for index, value in enumerate(axis.tolist())}
            for axis in axes
        ]
        for row in rows.to_dict(orient="records"):
            index = tuple(
                axis_index[float(row[dimension])]
                for axis_index, dimension in zip(axis_indices, dimensions)
            )
            values[index] = float(row["energy_mean_mJ"])

        tensor_coordinates = [
            value if isinstance(value, torch.Tensor) else float(value)
            for value in coordinates
        ]
        tensor_device = next(
            (value.device for value in tensor_coordinates if isinstance(value, torch.Tensor)),
            values.device,
        )
        return multilinear_interpolate(
            [axis.to(tensor_device) for axis in axes],
            values.to(tensor_device),
            tensor_coordinates,
            out_of_range=self.out_of_range,
        )

    def linear(self, input_features, output_features):
        rows = self.data[self.data["layer_type"].str.lower().eq("linear")]
        return self._estimate(
            rows,
            ["input_features", "output_features"],
            [input_features, output_features],
        )

    def conv2d(
        self,
        input_channels,
        output_channels,
        input_height,
        input_width=None,
        *,
        kernel_size,
        stride=1,
        padding=0,
    ):
        """Trilinearly interpolate over channels and output spatial area."""

        input_width = input_height if input_width is None else input_width
        rows = self.data[self.data["layer_type"].str.lower().eq("conv")]
        for column, value in (
            ("kernel_size", kernel_size),
            ("stride", stride),
            ("padding", padding),
        ):
            rows = rows[rows[column].eq(value)]
        rows = rows.copy()
        measured_output_size = (
            (rows["input_image_size"] + 2 * padding - kernel_size) // stride + 1
        )
        rows["output_spatial_area"] = measured_output_size**2
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        spatial_area = output_height * output_width
        return self._estimate(
            rows,
            ["input_channels", "output_channels", "output_spatial_area"],
            [input_channels, output_channels, spatial_area],
        )

    def attention(self, sequence_length, embed_dim, head_dim, *, rotary=False):
        layer_type = "rotaryattention" if rotary else "attention"
        rows = self.data[self.data["layer_type"].str.lower().eq(layer_type)]
        return self._estimate(
            rows,
            ["sequence_length", "embed_dim", "head_dim"],
            [sequence_length, embed_dim, head_dim],
        )