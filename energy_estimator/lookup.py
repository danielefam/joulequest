"""CSV-backed differentiable energy lookup for measured layer configurations."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .interpolation import multilinear_interpolate, multilinear_interpolate_batch


@dataclass(frozen=True)
class _PreparedGrid:
    axes: tuple
    values: torch.Tensor
    complete: bool


class EnergyLookup:
    """Estimate per-inference energy in mJ from a generated lookup CSV."""

    def __init__(self, path, *, out_of_range="error", dtype=torch.float64):
        self.path = Path(path)
        if out_of_range not in {"clamp", "error", "extrapolate"}:
            raise ValueError(f"Unknown out-of-range policy: {out_of_range}")
        self.out_of_range = out_of_range
        self.dtype = dtype
        self.data = pd.read_csv(self.path)
        required = {"layer_type", "energy_mean_mJ"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"Lookup table is missing columns: {', '.join(sorted(missing))}")
        if self.data["energy_mean_mJ"].isna().any():
            raise ValueError("Lookup table contains missing energy values")
        self._grids = {}
        self._device_grids = {}
        self._prepare_grids()

    @staticmethod
    def _configuration_value(value):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric

    def _build_grid(self, rows, dimensions):
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
        return _PreparedGrid(
            axes=tuple(axes),
            values=values,
            complete=not bool(torch.isnan(values).any().item()),
        )

    def _prepare_grids(self):
        layer_types = self.data["layer_type"].astype(str).str.lower()

        linear_rows = self.data[layer_types.eq("linear")]
        if not linear_rows.empty:
            self._grids[("linear",)] = self._build_grid(
                linear_rows, ["input_features", "output_features"]
            )

        conv_rows = self.data[layer_types.eq("conv")]
        conv_configuration = ["kernel_size", "stride", "padding"]
        if not conv_rows.empty:
            missing = set(conv_configuration).difference(conv_rows.columns)
            if missing:
                raise ValueError(
                    "Lookup table does not contain dimensions: "
                    + ", ".join(sorted(missing))
                )
            for configuration, rows in conv_rows.groupby(
                conv_configuration, dropna=False, sort=False
            ):
                kernel_size, stride, padding = map(
                    self._configuration_value, configuration
                )
                rows = rows.copy()
                measured_output_size = (
                    (rows["input_image_size"] + 2 * padding - kernel_size)
                    // stride
                    + 1
                )
                rows["output_spatial_area"] = measured_output_size**2
                key = ("conv", kernel_size, stride, padding)
                self._grids[key] = self._build_grid(
                    rows,
                    ["input_channels", "output_channels", "output_spatial_area"],
                )

        for layer_type in ("attention", "rotaryattention"):
            rows = self.data[layer_types.eq(layer_type)]
            if not rows.empty:
                self._grids[(layer_type,)] = self._build_grid(
                    rows, ["sequence_length", "embed_dim", "head_dim"]
                )

    def _grid_on_device(self, key, device):
        grid = self._grids.get(key)
        if grid is None:
            raise ValueError("No compatible measurements are available")
        device = torch.device(device)
        cache_key = (key, device.type, device.index)
        cached = self._device_grids.get(cache_key)
        if cached is None:
            cached = _PreparedGrid(
                axes=tuple(axis.to(device=device) for axis in grid.axes),
                values=grid.values.to(device=device),
                complete=grid.complete,
            )
            self._device_grids[cache_key] = cached
        return cached

    def _estimate(self, key, coordinates):

        tensor_coordinates = [
            value if isinstance(value, torch.Tensor) else float(value)
            for value in coordinates
        ]
        tensor_device = next(
            (value.device for value in tensor_coordinates if isinstance(value, torch.Tensor)),
            torch.device("cpu"),
        )
        grid = self._grid_on_device(key, tensor_device)
        return multilinear_interpolate(
            grid.axes,
            grid.values,
            tensor_coordinates,
            out_of_range=self.out_of_range,
            validate_corners=not grid.complete,
        )

    def _estimate_batch(self, key, coordinate_columns):
        tensor_device = next(
            (
                value.device
                for value in coordinate_columns
                if isinstance(value, torch.Tensor)
            ),
            torch.device("cpu"),
        )
        columns = []
        for value in coordinate_columns:
            if isinstance(value, torch.Tensor):
                column = value.to(device=tensor_device, dtype=self.dtype)
            else:
                column = torch.as_tensor(value, device=tensor_device, dtype=self.dtype)
            columns.append(column.reshape(1) if column.ndim == 0 else column.reshape(-1))
        columns = torch.broadcast_tensors(*columns)
        coordinates = torch.stack(columns, dim=1)
        grid = self._grid_on_device(key, tensor_device)
        return multilinear_interpolate_batch(
            grid.axes,
            grid.values,
            coordinates,
            out_of_range=self.out_of_range,
            validate_corners=not grid.complete,
        )

    def linear(self, input_features, output_features):
        return self._estimate(
            ("linear",),
            [input_features, output_features],
        )

    def linear_batch(self, input_features, output_features):
        return self._estimate_batch(
            ("linear",),
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
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        spatial_area = output_height * output_width
        return self._estimate(
            (
                "conv",
                self._configuration_value(kernel_size),
                self._configuration_value(stride),
                self._configuration_value(padding),
            ),
            [input_channels, output_channels, spatial_area],
        )

    def conv2d_batch(
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
        input_width = input_height if input_width is None else input_width
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        spatial_area = output_height * output_width
        return self._estimate_batch(
            (
                "conv",
                self._configuration_value(kernel_size),
                self._configuration_value(stride),
                self._configuration_value(padding),
            ),
            [input_channels, output_channels, spatial_area],
        )

    def attention(self, sequence_length, embed_dim, head_dim, *, rotary=False):
        layer_type = "rotaryattention" if rotary else "attention"
        return self._estimate(
            (layer_type,),
            [sequence_length, embed_dim, head_dim],
        )

    def attention_batch(
        self, sequence_length, embed_dim, head_dim, *, rotary=False
    ):
        layer_type = "rotaryattention" if rotary else "attention"
        return self._estimate_batch(
            (layer_type,),
            [sequence_length, embed_dim, head_dim],
        )