"""Differentiable interpolation over measured rectilinear energy grids."""

import torch


_CORNER_BITS = {}


def _as_scalar(value, *, device, dtype):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Interpolation coordinates must be scalar tensors")
        return value.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype)


def _bracket(axis, coordinate, out_of_range):
    if out_of_range not in {"clamp", "error", "extrapolate"}:
        raise ValueError(f"Unknown out-of-range policy: {out_of_range}")

    if axis.numel() == 1:
        if out_of_range != "clamp" and bool(
            (coordinate.detach() != axis[0]).any().item()
        ):
            raise ValueError(
                "A coordinate cannot be interpolated from the only measured "
                f"value {float(axis[0]):g}"
            )
        index = torch.zeros_like(coordinate, dtype=torch.long)
        return index, index, torch.zeros_like(coordinate)

    if out_of_range == "error":
        below = bool((coordinate.detach() < axis[0]).any().item())
        above = bool((coordinate.detach() > axis[-1]).any().item())
        if below or above:
            raise ValueError(
                f"A coordinate is outside [{float(axis[0]):g}, {float(axis[-1]):g}]"
            )

    lookup = coordinate.detach().clamp(axis[0], axis[-1])
    upper = torch.searchsorted(axis, lookup, right=True).clamp(1, axis.numel() - 1)
    lower = upper - 1
    interpolation_coordinate = coordinate
    if out_of_range == "clamp":
        interpolation_coordinate = coordinate.clamp(axis[0], axis[-1])
    weight = (interpolation_coordinate - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, weight


def _corner_bits(rank, device):
    key = (rank, device.type, device.index)
    bits = _CORNER_BITS.get(key)
    if bits is None:
        corner_ids = torch.arange(1 << rank, device=device)
        dimensions = torch.arange(rank, device=device)
        bits = ((corner_ids[:, None] >> dimensions[None, :]) & 1).bool()
        _CORNER_BITS[key] = bits
    return bits


def multilinear_interpolate(
    axes,
    values,
    coordinates,
    *,
    out_of_range="error",
    validate_corners=True,
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
    coordinate_matrix = torch.stack(normalized_coordinates, dim=0).unsqueeze(0)
    return multilinear_interpolate_batch(
        normalized_axes,
        values,
        coordinate_matrix,
        out_of_range=out_of_range,
        validate_corners=validate_corners,
    ).squeeze(0)


def multilinear_interpolate_batch(
    axes,
    values,
    coordinates,
    *,
    out_of_range="error",
    validate_corners=True,
):
    """Interpolate a batch of points from one N-dimensional grid."""

    if coordinates.ndim != 2:
        raise ValueError("Batched coordinates must have shape (points, dimensions)")
    if len(axes) != coordinates.shape[1]:
        raise ValueError("One interpolation coordinate is required per axis")
    if values.ndim != len(axes):
        raise ValueError("The value tensor rank must match the number of axes")

    device = values.device
    dtype = values.dtype
    normalized_axes = [axis.to(device=device, dtype=dtype) for axis in axes]
    coordinates = coordinates.to(device=device, dtype=dtype)
    brackets = [
        _bracket(axis, coordinates[:, dimension], out_of_range)
        for dimension, axis in enumerate(normalized_axes)
    ]

    bits = _corner_bits(len(brackets), device)
    indices = []
    weight_factors = []
    for dimension, (lower, upper, fraction) in enumerate(brackets):
        use_upper = bits[:, dimension].unsqueeze(0)
        indices.append(
            torch.where(use_upper, upper.unsqueeze(1), lower.unsqueeze(1))
        )
        weight_factors.append(
            torch.where(use_upper, fraction.unsqueeze(1), 1 - fraction.unsqueeze(1))
        )

    corner_values = values[tuple(indices)]
    if validate_corners and bool(torch.isnan(corner_values).any().item()):
        raise ValueError("The requested interpolation cell has missing measurements")
    weights = torch.stack(weight_factors, dim=2).prod(dim=2)
    return (corner_values * weights).sum(dim=1)