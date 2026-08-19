"""Differentiable interpolation over measured rectilinear energy grids."""

from itertools import product

import torch


def _as_scalar(value, *, device, dtype):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Interpolation coordinates must be scalar tensors")
        return value.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype)


def _bracket(axis, coordinate, out_of_range):
    if axis.numel() == 1:
        differs = bool((coordinate.detach() != axis[0]).item())
        if differs and out_of_range != "clamp":
            raise ValueError(
                f"Coordinate {float(coordinate.detach()):g} cannot be "
                f"interpolated from the only measured value {float(axis[0]):g}"
            )
        return 0, 0, torch.zeros_like(coordinate)

    below = bool((coordinate.detach() < axis[0]).item())
    above = bool((coordinate.detach() > axis[-1]).item())
    if (below or above) and out_of_range == "error":
        raise ValueError(
            f"Coordinate {float(coordinate.detach()):g} is outside "
            f"[{float(axis[0]):g}, {float(axis[-1]):g}]"
        )
    if out_of_range not in {"clamp", "error", "extrapolate"}:
        raise ValueError(f"Unknown out-of-range policy: {out_of_range}")

    lookup = coordinate.detach().clamp(axis[0], axis[-1])
    upper = int(torch.searchsorted(axis, lookup, right=True).clamp(1, axis.numel() - 1))
    lower = upper - 1
    interpolation_coordinate = coordinate
    if out_of_range == "clamp":
        interpolation_coordinate = coordinate.clamp(axis[0], axis[-1])
    weight = (interpolation_coordinate - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, weight


def multilinear_interpolate(
    axes,
    values,
    coordinates,
    *,
    out_of_range="error",
):
    """Interpolate a scalar from an N-dimensional rectilinear grid.

    Axis selection is discrete, while interpolation weights retain gradients
    with respect to tensor coordinates.
    """

    if len(axes) != len(coordinates):
        raise ValueError("One interpolation coordinate is required per axis")
    if values.ndim != len(axes):
        raise ValueError("The value tensor rank must match the number of axes")

    device = values.device
    dtype = values.dtype
    normalized_axes = [axis.to(device=device, dtype=dtype) for axis in axes]
    normalized_coordinates = [
        _as_scalar(value, device=device, dtype=dtype) for value in coordinates
    ]
    brackets = [
        _bracket(axis, coordinate, out_of_range)
        for axis, coordinate in zip(normalized_axes, normalized_coordinates)
    ]

    result = torch.zeros((), device=device, dtype=dtype)
    corner_values = []
    choices = [(0,) if lower == upper else (0, 1) for lower, upper, _ in brackets]
    for corner in product(*choices):
        index = []
        weight = torch.ones((), device=device, dtype=dtype)
        for use_upper, (lower, upper, fraction) in zip(corner, brackets):
            index.append(upper if use_upper else lower)
            if lower != upper:
                weight = weight * (fraction if use_upper else 1 - fraction)
        corner_value = values[tuple(index)]
        corner_values.append(corner_value)
        result = result + weight * corner_value

    if torch.isnan(torch.stack(corner_values)).any():
        raise ValueError("The requested interpolation cell has missing measurements")
    return result